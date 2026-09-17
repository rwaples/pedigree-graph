//! Pair emission on the row-streaming engine.
//!
//! [`Engine::emit_row`] hands out the oriented pairs one row owns; this
//! module assembles them into per-category blocks.  Three assemblies are
//! implemented for the slice 12 screening benchmark and differ only in how
//! task output reaches the final arrays:
//!
//! * [`Emitter::Buffered`]: every task returns its chunks, then blocks are
//!   assembled category by category.  One classification pass; temporary
//!   memory is all chunks plus the block being copied.
//! * [`Emitter::TwoPass`]: a counting pass sizes each block exactly and
//!   assigns every task a disjoint slice of it; a second classification pass
//!   fills the slices in place.  No copy of the result, twice the
//!   classification work.
//! * [`Emitter::BoundedWave`]: tasks run in bounded waves whose chunks are
//!   appended to growing blocks and released before the next wave.
//!
//! All three produce byte-identical blocks: graph blocks arrive in
//! canonical-key order because the owner row rises across ordered task
//! ranges and each row's members are sorted, and view blocks are sorted by the
//! view-space key afterwards.

use super::category::{Category, CategorySet, N_CATEGORIES};
use super::engine::{Engine, Workspace};
use super::{MaxDegree, Pedigree, ROWS_PER_TASK};
use crate::error::Error;
use rayon::prelude::*;
use std::sync::Mutex;

/// The pairs of one category, aligned `first[k]` / `second[k]`, in the receiver's rows.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PairBlock {
    pub first: Vec<u32>,
    pub second: Vec<u32>,
}

impl PairBlock {
    pub fn len(&self) -> usize {
        self.first.len()
    }

    pub fn is_empty(&self) -> bool {
        self.first.is_empty()
    }

    fn try_reserve_exact(&mut self, additional: usize) -> Result<(), Error> {
        let total = self.len() + additional;
        self.first
            .try_reserve_exact(additional)
            .and_then(|_| self.second.try_reserve_exact(additional))
            .map_err(|_| allocation_failed("pair_block", total))
    }

    /// Amortised growth, for the wave assembly.
    fn try_reserve(&mut self, additional: usize) -> Result<(), Error> {
        let total = self.len() + additional;
        self.first
            .try_reserve(additional)
            .and_then(|_| self.second.try_reserve(additional))
            .map_err(|_| allocation_failed("pair_block", total))
    }

    fn extend_from(&mut self, other: &PairBlock) {
        self.first.extend_from_slice(&other.first);
        self.second.extend_from_slice(&other.second);
    }

    /// A checksum over the aligned sequence, sensitive to every value and to
    /// its position, computable in NumPy with wrapping `uint64` arithmetic
    /// (`benchmarks/_pair_common.py`).
    pub fn digest(&self) -> u64 {
        const K1: u64 = 0x9E37_79B9_7F4A_7C15;
        const K2: u64 = 0xC2B2_AE3D_27D4_EB4F;
        const K3: u64 = 0x1656_67B1_9E37_79F9;
        self.first
            .iter()
            .zip(&self.second)
            .enumerate()
            .fold(0u64, |h, (k, (&a, &b))| {
                let term = K1
                    .wrapping_mul(u64::from(a))
                    .wrapping_add(K2.wrapping_mul(u64::from(b)))
                    .wrapping_add(K3);
                h.wrapping_add((k as u64 + 1).wrapping_mul(term))
            })
    }
}

/// One block per registry category in registry order; unrequested blocks are empty.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PairBlocks(pub Vec<PairBlock>);

impl PairBlocks {
    fn empty() -> PairBlocks {
        PairBlocks(vec![PairBlock::default(); N_CATEGORIES])
    }

    pub fn get(&self, cat: Category) -> &PairBlock {
        &self.0[cat.index()]
    }

    /// The pairs over every block.
    pub fn total(&self) -> usize {
        self.0.iter().map(PairBlock::len).sum()
    }
}

/// How task output is assembled into blocks.  A benchmark selector for the
/// slice 12 screening, not a public execution mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Emitter {
    Buffered,
    TwoPass,
    /// `tasks_per_thread` row-range tasks per pool thread run between appends.
    BoundedWave {
        tasks_per_thread: usize,
    },
}

fn allocation_failed(operation: &'static str, requested_elements: usize) -> Error {
    Error::AllocationFailed {
        operation,
        requested_elements,
        dtype: "int32",
    }
}

/// Emission workspaces shared by the tasks of one query, never more than there are threads.
struct WorkspacePool {
    n: usize,
    free: Mutex<Vec<Workspace>>,
}

impl WorkspacePool {
    fn new(n: usize) -> WorkspacePool {
        WorkspacePool {
            n,
            free: Mutex::new(Vec::new()),
        }
    }

    fn take(&self) -> Workspace {
        self.free
            .lock()
            .unwrap()
            .pop()
            .unwrap_or_else(|| Workspace::for_pairs(self.n))
    }

    fn give(&self, ws: Workspace) {
        self.free.lock().unwrap().push(ws);
    }
}

/// One query's immutable inputs, shared by every task.
struct Query<'a> {
    engine: Engine<'a>,
    requested: CategorySet,
    view: Option<&'a [i32]>,
    pool: WorkspacePool,
}

impl Query<'_> {
    /// Run `sink` over every owned pair of the rows in `range`.
    fn run(&self, range: (usize, usize), mut sink: impl FnMut(Category, u32, u32)) {
        let mut ws = self.pool.take();
        for row in range.0..range.1 {
            if self.view.is_some_and(|map| map[row] < 0) {
                continue;
            }
            self.engine
                .emit_row(row, &self.requested, self.view, &mut ws, &mut sink);
        }
        self.pool.give(ws);
    }

    /// The pairs of one task, one chunk per category.
    fn chunk(&self, range: (usize, usize)) -> PairBlocks {
        let mut chunk = PairBlocks::empty();
        self.run(range, |cat, a, b| {
            let block = &mut chunk.0[cat.index()];
            block.first.push(a);
            block.second.push(b);
        });
        chunk
    }

    /// The pair count of one task per category.
    fn count(&self, range: (usize, usize)) -> [usize; N_CATEGORIES] {
        let mut counts = [0usize; N_CATEGORIES];
        self.run(range, |cat, _, _| counts[cat.index()] += 1);
        counts
    }
}

fn task_ranges(n: usize) -> Vec<(usize, usize)> {
    (0..n)
        .step_by(ROWS_PER_TASK)
        .map(|s| (s, (s + ROWS_PER_TASK).min(n)))
        .collect()
}

/// The oriented pairs of every requested category, using the current Rayon pool.
///
/// `requested` names the blocks to fill; every category up to `max_degree`
/// is still classified, because a closest-category block depends on the
/// closer ones.  `view` is the int32 view row of every graph row, `-1` for a
/// row outside the view; with it the blocks are in view rows, filtered to
/// pairs with both rows selected, and sorted by the view-space canonical
/// key.  Without it they are in graph rows and canonical-key order.
///
/// # Errors
///
/// [`Error::AllocationFailed`] when a final block or the view-sort scratch
/// cannot be allocated.
pub fn pair_blocks(
    ped: &Pedigree,
    max_degree: MaxDegree,
    requested: CategorySet,
    view: Option<&[i32]>,
    emitter: Emitter,
) -> Result<PairBlocks, Error> {
    let engine = Engine::new(ped, max_degree);
    let n = engine.len();
    if let Some(map) = view {
        assert_eq!(map.len(), n, "view map must have one entry per graph row");
    }
    let query = Query {
        engine,
        requested,
        view,
        pool: WorkspacePool::new(n),
    };
    let ranges = task_ranges(n);
    let mut blocks = match emitter {
        Emitter::Buffered => buffered(&query, &ranges)?,
        Emitter::TwoPass => two_pass(&query, &ranges)?,
        Emitter::BoundedWave { tasks_per_thread } => {
            bounded_wave(&query, &ranges, tasks_per_thread)?
        }
    };
    if let Some(map) = view {
        let n_view = map.iter().copied().max().map_or(0, |m| m as u64 + 1);
        for cat in requested.iter() {
            sort_by_view_key(&mut blocks.0[cat.index()], n_view)?;
        }
    }
    Ok(blocks)
}

fn buffered(query: &Query, ranges: &[(usize, usize)]) -> Result<PairBlocks, Error> {
    let mut chunks: Vec<PairBlocks> = ranges.par_iter().map(|&r| query.chunk(r)).collect();
    let mut out = PairBlocks::empty();
    for cat in query.requested.iter() {
        let i = cat.index();
        let total: usize = chunks.iter().map(|c| c.0[i].len()).sum();
        out.0[i].try_reserve_exact(total)?;
        for chunk in &mut chunks {
            let part = std::mem::take(&mut chunk.0[i]);
            out.0[i].extend_from(&part);
        }
    }
    Ok(out)
}

/// Cut `buf` into consecutive slices of the given sizes.
fn split_sizes(mut buf: &mut [u32], sizes: impl Iterator<Item = usize>) -> Vec<&mut [u32]> {
    sizes
        .map(|size| {
            let (head, tail) = std::mem::take(&mut buf).split_at_mut(size);
            buf = tail;
            head
        })
        .collect()
}

fn two_pass(query: &Query, ranges: &[(usize, usize)]) -> Result<PairBlocks, Error> {
    let counts: Vec<[usize; N_CATEGORIES]> = ranges.par_iter().map(|&r| query.count(r)).collect();
    let mut out = PairBlocks::empty();
    for cat in query.requested.iter() {
        let i = cat.index();
        let total: usize = counts.iter().map(|c| c[i]).sum();
        out.0[i].try_reserve_exact(total)?;
        out.0[i].first.resize(total, 0);
        out.0[i].second.resize(total, 0);
    }
    // Every task owns one disjoint slice of every block.
    let mut slots: Vec<Vec<(&mut [u32], &mut [u32])>> = (0..ranges.len())
        .map(|_| Vec::with_capacity(N_CATEGORIES))
        .collect();
    for (i, block) in out.0.iter_mut().enumerate() {
        let firsts = split_sizes(&mut block.first, counts.iter().map(|c| c[i]));
        let seconds = split_sizes(&mut block.second, counts.iter().map(|c| c[i]));
        for (slot, pair) in slots.iter_mut().zip(firsts.into_iter().zip(seconds)) {
            slot.push(pair);
        }
    }
    slots
        .into_par_iter()
        .zip(ranges.par_iter())
        .for_each(|(mut slot, &range)| {
            let mut cursor = [0usize; N_CATEGORIES];
            query.run(range, |cat, a, b| {
                let i = cat.index();
                slot[i].0[cursor[i]] = a;
                slot[i].1[cursor[i]] = b;
                cursor[i] += 1;
            });
            for (i, (first, _)) in slot.iter().enumerate() {
                assert_eq!(cursor[i], first.len(), "pass two disagrees with pass one");
            }
        });
    Ok(out)
}

fn bounded_wave(
    query: &Query,
    ranges: &[(usize, usize)],
    tasks_per_thread: usize,
) -> Result<PairBlocks, Error> {
    let wave = (rayon::current_num_threads() * tasks_per_thread).max(1);
    let mut out = PairBlocks::empty();
    for group in ranges.chunks(wave) {
        let chunks: Vec<PairBlocks> = group.par_iter().map(|&r| query.chunk(r)).collect();
        for chunk in chunks {
            for cat in query.requested.iter() {
                let i = cat.index();
                out.0[i].try_reserve(chunk.0[i].len())?;
                out.0[i].extend_from(&chunk.0[i]);
            }
        }
    }
    Ok(out)
}

/// Sort a block by the canonical unordered key `min * n + max` in place.
///
/// Each pair is packed with its orientation into one `u64`, sorted, and
/// unpacked, so the scratch is one word per pair.  Keys are unique within a
/// block, so an unstable sort gives the same order as Python's stable one.
fn sort_by_view_key(block: &mut PairBlock, n: u64) -> Result<(), Error> {
    let len = block.len();
    if len < 2 {
        return Ok(());
    }
    let mut packed: Vec<u64> = Vec::new();
    packed
        .try_reserve_exact(len)
        .map_err(|_| Error::AllocationFailed {
            operation: "view_sort_scratch",
            requested_elements: len,
            dtype: "uint64",
        })?;
    packed.extend(block.first.iter().zip(&block.second).map(|(&a, &b)| {
        let (lo, hi) = (a.min(b), a.max(b));
        ((u64::from(lo) * n + u64::from(hi)) << 1) | u64::from(a > b)
    }));
    packed.par_sort_unstable();
    for (k, p) in packed.iter().enumerate() {
        let key = p >> 1;
        let (lo, hi) = ((key / n) as u32, (key % n) as u32);
        let swapped = p & 1 == 1;
        block.first[k] = if swapped { hi } else { lo };
        block.second[k] = if swapped { lo } else { hi };
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::testing::pedigree;
    use super::super::{count_pairs, MaxDegree, PedigreeColumns};
    use super::*;

    const EMITTERS: [Emitter; 3] = [
        Emitter::Buffered,
        Emitter::TwoPass,
        Emitter::BoundedWave {
            tasks_per_thread: 1,
        },
    ];

    fn all(cols: &PedigreeColumns, view: Option<&[i32]>) -> PairBlocks {
        let ped = cols.try_borrow().unwrap();
        let blocks: Vec<PairBlocks> = EMITTERS
            .iter()
            .map(|&e| {
                pair_blocks(&ped, MaxDegree::MAX, CategorySet::up_to_degree(5), view, e).unwrap()
            })
            .collect();
        assert!(blocks.iter().all(|b| *b == blocks[0]), "emitters disagree");
        blocks.into_iter().next().unwrap()
    }

    fn pairs(blocks: &PairBlocks, cat: Category) -> Vec<(u32, u32)> {
        let b = blocks.get(cat);
        b.first
            .iter()
            .copied()
            .zip(b.second.iter().copied())
            .collect()
    }

    #[test]
    fn parent_roles_put_the_offspring_first() {
        let cols = pedigree(&[(-1, -1), (-1, -1), (0, 1), (0, 1)], &[]);
        let got = all(&cols, None);
        assert_eq!(pairs(&got, Category::MO), vec![(2, 0), (3, 0)]);
        assert_eq!(pairs(&got, Category::FO), vec![(2, 1), (3, 1)]);
        assert_eq!(pairs(&got, Category::FS), vec![(2, 3)]);
    }

    #[test]
    fn lineal_categories_put_the_descendant_first() {
        // 0 -> 1 -> 2 -> 3 -> 4 -> 5, each the mother of the next.
        let cols = pedigree(
            &[(-1, -1), (0, -1), (1, -1), (2, -1), (3, -1), (4, -1)],
            &[],
        );
        let got = all(&cols, None);
        assert_eq!(
            pairs(&got, Category::GP),
            vec![(2, 0), (3, 1), (4, 2), (5, 3)]
        );
        assert_eq!(pairs(&got, Category::GGP), vec![(3, 0), (4, 1), (5, 2)]);
        assert_eq!(pairs(&got, Category::GGGP), vec![(4, 0), (5, 1)]);
        assert_eq!(pairs(&got, Category::G3GP), vec![(5, 0)]);
    }

    #[test]
    fn avuncular_categories_put_the_niece_first() {
        // Sibs 2 and 3 (of 0 x 1); 4 is 2's child with 5; 6 is 4's child.
        let cols = pedigree(
            &[
                (-1, -1),
                (-1, -1),
                (0, 1),
                (0, 1),
                (2, 5),
                (-1, -1),
                (4, -1),
            ],
            &[],
        );
        let got = all(&cols, None);
        assert_eq!(pairs(&got, Category::Av), vec![(4, 3)]);
        assert_eq!(pairs(&got, Category::GAv), vec![(6, 3)]);
        // Half sibs 2 and 7 (share mother 0 only); 7 is a half aunt of 4.
        let cols = pedigree(
            &[
                (-1, -1),
                (-1, -1),
                (0, 1),
                (0, 1),
                (2, 5),
                (-1, -1),
                (4, -1),
                (0, 8),
                (-1, -1),
            ],
            &[],
        );
        let got = all(&cols, None);
        assert_eq!(pairs(&got, Category::HAv), vec![(4, 7)]);
        assert_eq!(pairs(&got, Category::HGAv), vec![(6, 7)]);
    }

    #[test]
    fn removed_cousins_put_the_junior_cousin_first() {
        // Founders 0 x 1 have 2 and 3; 2 has 4 (with 5), 3 has 6 (with 7);
        // 4 has 8 (with 9); 8 has 10 (with 11).  6 and 8 are 1C1R, 6 and 10
        // are 1C2R, with 8 and 10 the junior members.
        let cols = pedigree(
            &[
                (-1, -1),
                (-1, -1),
                (0, 1),
                (0, 1),
                (2, 5),
                (-1, -1),
                (3, 7),
                (-1, -1),
                (4, 9),
                (-1, -1),
                (8, 11),
                (-1, -1),
            ],
            &[],
        );
        let got = all(&cols, None);
        assert_eq!(pairs(&got, Category::C1), vec![(4, 6)]);
        assert_eq!(pairs(&got, Category::C1R1), vec![(8, 6)]);
        assert_eq!(pairs(&got, Category::C1R2), vec![(10, 6)]);
        // Half first cousins once removed: 12 is a half sib of 2 (mother 0
        // only) with child 13; 13 and 8 share one great-grand/grandparent.
        let cols = pedigree(
            &[
                (-1, -1),
                (-1, -1),
                (0, 1),
                (0, 1),
                (2, 5),
                (-1, -1),
                (3, 7),
                (-1, -1),
                (4, 9),
                (-1, -1),
                (8, 11),
                (-1, -1),
                (0, 14),
                (12, 15),
                (-1, -1),
                (-1, -1),
            ],
            &[],
        );
        let got = all(&cols, None);
        assert_eq!(pairs(&got, Category::H1C), vec![(4, 13), (6, 13)]);
        assert_eq!(pairs(&got, Category::H1C1R), vec![(8, 13)]);
    }

    #[test]
    fn symmetric_blocks_store_the_lower_row_first_in_graph_and_view_rows() {
        // Sibs 2 and 3 of 0 x 1, their maternal half sib 4, and MZ twins 6
        // and 7 of 0 x 1, which take no part in sibling categories.
        let cols = pedigree(
            &[
                (-1, -1),
                (-1, -1),
                (0, 1),
                (0, 1),
                (0, 5),
                (-1, -1),
                (0, 1),
                (0, 1),
            ],
            &[(6, 7)],
        );
        let got = all(&cols, None);
        assert_eq!(pairs(&got, Category::MZ), vec![(6, 7)]);
        assert_eq!(pairs(&got, Category::FS), vec![(2, 3)]);
        assert_eq!(pairs(&got, Category::MHS), vec![(2, 4), (3, 4)]);
        // A view listing 4 before 2 reverses the graph order.
        let view = [-1, -1, 1, -1, 0, -1, -1, -1];
        let got = all(&cols, Some(&view));
        assert_eq!(pairs(&got, Category::MHS), vec![(0, 1)]);
    }

    #[test]
    fn a_view_keeps_pairs_through_unselected_intermediates_and_sorts_in_view_space() {
        // 0 -> 2 -> 3 -> 4 (mothers); view [4, 0, 3] selects rows 4, 0, 3.
        let cols = pedigree(&[(-1, -1), (-1, -1), (0, 1), (2, -1), (3, -1)], &[]);
        let view = [1, -1, -1, 2, 0];
        let got = all(&cols, Some(&view));
        // (4, 3) is MO in view rows (0, 2); (3, 0) is GP through the
        // unselected 2, view rows (2, 1); (4, 0) is GGP, view rows (0, 1).
        assert_eq!(pairs(&got, Category::MO), vec![(0, 2)]);
        assert_eq!(pairs(&got, Category::GP), vec![(2, 1)]);
        assert_eq!(pairs(&got, Category::GGP), vec![(0, 1)]);
        assert!(got.get(Category::GGGP).is_empty());
    }

    #[test]
    fn a_dual_valid_pair_keeps_the_lower_row_first() {
        // g(0) and h(1) have p(2); g and p have i(3).  Pair (g, i) is MO and
        // GP: the closest category wins and g stays `second`.  Pair (h, i) is
        // GP only, with i the descendant.
        let cols = pedigree(&[(-1, -1), (-1, -1), (0, 1), (0, 2)], &[]);
        let got = all(&cols, None);
        assert_eq!(pairs(&got, Category::MO), vec![(2, 0), (3, 0)]);
        assert_eq!(pairs(&got, Category::FO), vec![(2, 1), (3, 2)]);
        assert_eq!(pairs(&got, Category::GP), vec![(3, 1)]);
    }

    #[test]
    fn graph_blocks_are_strictly_increasing_in_canonical_key() {
        let cols = crate::relationships::testing::random_pedigree(400, 7);
        let got = all(&cols, None);
        let n = cols.mother.len() as u64;
        for cat in Category::ALL {
            let keys: Vec<u64> = pairs(&got, cat)
                .iter()
                .map(|&(a, b)| u64::from(a.min(b)) * n + u64::from(a.max(b)))
                .collect();
            assert!(
                keys.windows(2).all(|w| w[0] < w[1]),
                "{} not sorted",
                cat.code()
            );
        }
    }

    #[test]
    fn block_lengths_equal_counts_and_requests_filter_blocks() {
        let cols = crate::relationships::testing::random_pedigree(400, 7);
        let ped = cols.try_borrow().unwrap();
        let got = all(&cols, None);
        let counts = count_pairs(&ped, MaxDegree::MAX, None);
        for cat in Category::ALL {
            assert_eq!(got.get(cat).len() as u64, counts.get(cat), "{}", cat.code());
        }
        let only: CategorySet = [Category::C1, Category::Av].into_iter().collect();
        let some = pair_blocks(&ped, MaxDegree::MAX, only, None, Emitter::Buffered).unwrap();
        for cat in Category::ALL {
            if only.contains(cat) {
                assert_eq!(some.get(cat), got.get(cat));
            } else {
                assert!(some.get(cat).is_empty());
            }
        }
    }

    #[test]
    fn thread_count_does_not_change_blocks() {
        let cols = crate::relationships::testing::random_pedigree(5000, 11);
        let ped = cols.try_borrow().unwrap();
        let view: Vec<i32> = (0..cols.mother.len() as i32)
            .map(|r| {
                if r % 3 == 0 {
                    -1
                } else {
                    r / 3 * 2 + r % 3 - 1
                }
            })
            .collect();
        let mut seen = Vec::new();
        for threads in [1, 4] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            for emitter in EMITTERS {
                for v in [None, Some(view.as_slice())] {
                    let blocks = pool
                        .install(|| {
                            pair_blocks(
                                &ped,
                                MaxDegree::MAX,
                                CategorySet::up_to_degree(5),
                                v,
                                emitter,
                            )
                        })
                        .unwrap();
                    seen.push((v.is_some(), blocks));
                }
            }
        }
        for (is_view, blocks) in &seen {
            let reference = &seen.iter().find(|(v, _)| v == is_view).unwrap().1;
            assert_eq!(blocks, reference);
        }
        assert!(seen[1].1.total() > 0 && seen[1].1.total() < seen[0].1.total());
    }
}
