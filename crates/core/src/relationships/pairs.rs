//! Pair emission on the row-streaming engine.
//!
//! [`Engine::emit_row`] hands out the oriented pairs one row owns; this
//! module assembles them into per-category blocks.  The two public
//! [`Execution`] modes differ only in how task output reaches the final
//! arrays, and were chosen by the slice 12 benchmarks
//! (`benchmarks/bench_pair_emitters.md`):
//!
//! * [`Execution::Speed`]: every task returns its chunks, then blocks are
//!   assembled category by category.  One classification pass; temporary
//!   memory is all chunks plus the block being copied, about 2.3 times the
//!   payload.
//! * [`Execution::Memory`]: a counting pass sizes each block exactly and
//!   assigns every task a disjoint slice of it; a second classification pass
//!   fills the slices in place.  No copy of the result, twice the
//!   classification work.
//!
//! Both produce byte-identical blocks: graph blocks arrive in canonical-key
//! order because the owner row rises across ordered task ranges and each
//! row's members are sorted, and view blocks are sorted by the view-space
//! key afterwards.

use super::category::{Category, CategorySet, N_CATEGORIES};
use super::engine::{Engine, WorkspacePool};
use super::{task_ranges, MaxDegree, Pedigree};
use crate::alloc::{self, Family};
use crate::error::Error;
use rayon::prelude::*;

/// The pairs of one category, aligned `first[k]` / `second[k]`, in the
/// receiver's rows.  Rows are `i32`, the host's row dtype, so a block moves
/// into a NumPy array without a copy; every row is non-negative.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PairBlock {
    pub first: Vec<i32>,
    pub second: Vec<i32>,
}

impl PairBlock {
    pub fn len(&self) -> usize {
        self.first.len()
    }

    pub fn is_empty(&self) -> bool {
        self.first.is_empty()
    }

    /// Append one pair to a task chunk.
    fn push(&mut self, a: u32, b: u32) -> Result<(), Error> {
        alloc::push(&mut self.first, a as i32, Family::TaskChunk, "int32")?;
        alloc::push(&mut self.second, b as i32, Family::TaskChunk, "int32")
    }

    fn reserve_exact(&mut self, additional: usize) -> Result<(), Error> {
        alloc::reserve_exact(&mut self.first, additional, Family::PairBlock, "int32")?;
        alloc::reserve_exact(&mut self.second, additional, Family::PairBlock, "int32")
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
                    .wrapping_mul(a as u64)
                    .wrapping_add(K2.wrapping_mul(b as u64))
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

/// How task output is assembled into blocks; the `execution` keyword of the
/// public API (ADR 0006 as amended).  Changes resource use, never results.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Execution {
    /// Fastest: buffered chunks, then one copy into exact blocks.
    Speed,
    /// Lowest peak: count, then fill exact blocks in place.
    Memory,
}

impl Execution {
    /// The public spelling, as the Python keyword accepts it.
    pub fn name(self) -> &'static str {
        match self {
            Execution::Speed => "speed",
            Execution::Memory => "memory",
        }
    }

    /// The mode by its public spelling.
    pub fn parse(name: &str) -> Option<Execution> {
        match name {
            "speed" => Some(Execution::Speed),
            "memory" => Some(Execution::Memory),
            _ => None,
        }
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
    fn run(
        &self,
        range: (usize, usize),
        mut sink: impl FnMut(Category, u32, u32) -> Result<(), Error>,
    ) -> Result<(), Error> {
        let mut ws = self.pool.take()?;
        // The workspace goes back even when a row fails, so that the sibling
        // tasks Rayon has already dispatched reuse it instead of allocating
        // another one under the memory pressure that failed this row.
        let mut result = Ok(());
        for row in range.0..range.1 {
            if self.view.is_some_and(|map| map[row] < 0) {
                continue;
            }
            result = self
                .engine
                .emit_row(row, &self.requested, self.view, &mut ws, &mut sink);
            if result.is_err() {
                break;
            }
        }
        self.pool.give(ws);
        result
    }

    /// The pairs of one task, one chunk per category.
    fn chunk(&self, range: (usize, usize)) -> Result<PairBlocks, Error> {
        let mut chunk = PairBlocks::empty();
        self.run(range, |cat, a, b| chunk.0[cat.index()].push(a, b))?;
        Ok(chunk)
    }

    /// The pair count of one task per category.
    fn count(&self, range: (usize, usize)) -> Result<[usize; N_CATEGORIES], Error> {
        let mut counts = [0usize; N_CATEGORIES];
        self.run(range, |cat, _, _| {
            counts[cat.index()] += 1;
            Ok(())
        })?;
        Ok(counts)
    }
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
/// [`Error::AllocationFailed`] from any buffer the pedigree or its pairs
/// size: the engine, a workspace, a row set, a task chunk or table, a final
/// block, or the view-sort scratch.
///
/// [`Error::InvalidViewMap`] when `view` is not a partial permutation: an
/// entry outside `-1 .. n`, or two graph rows mapped to one view row.  The
/// view-space sort relies on the keys being distinct, so a repeated view row
/// would make the order depend on the thread count.
///
/// # Panics
///
/// If `view` does not have one entry per graph row; the host binding checks
/// that before calling.
pub fn pair_blocks(
    ped: &Pedigree,
    max_degree: MaxDegree,
    requested: CategorySet,
    view: Option<&[i32]>,
    execution: Execution,
) -> Result<PairBlocks, Error> {
    let engine = Engine::new(ped, max_degree)?;
    let n = engine.len();
    if let Some(map) = view {
        assert_eq!(map.len(), n, "view map must have one entry per graph row");
        check_view_map(map)?;
    }
    let query = Query {
        engine,
        requested,
        view,
        pool: WorkspacePool::new(n, true),
    };
    let ranges = task_ranges(n);
    let mut blocks = match execution {
        Execution::Speed => buffered(&query, &ranges)?,
        Execution::Memory => two_pass(&query, &ranges)?,
    };
    if let Some(map) = view {
        // Unselected rows carry -1, so the row count is one past the largest
        // selected row.  A map that selects nothing has no rows at all; taking
        // the maximum over the -1s instead would sign-extend to `u64::MAX`.
        let n_view = map
            .iter()
            .copied()
            .filter(|&m| m >= 0)
            .max()
            .map_or(0, |m| m as u64 + 1);
        for cat in requested.iter() {
            sort_by_view_key(&mut blocks.0[cat.index()], n_view)?;
        }
    }
    Ok(blocks)
}

/// Gather every task's result into a table, in task order.
fn task_table<T: Send>(
    ranges: &[(usize, usize)],
    task: impl Fn((usize, usize)) -> Result<T, Error> + Sync,
) -> Result<Vec<T>, Error> {
    let mut table = alloc::with_capacity(ranges.len(), Family::TaskTable, "object")?;
    ranges
        .par_iter()
        .map(|&r| task(r))
        .collect_into_vec(&mut table);
    table.into_iter().collect()
}

fn buffered(query: &Query, ranges: &[(usize, usize)]) -> Result<PairBlocks, Error> {
    let chunks = task_table(ranges, |r| query.chunk(r))?;
    // Transpose the task-by-category table into one column per category.
    // Only `Vec` handles move, so no pair is copied and nothing is freed.
    let mut columns: Vec<Vec<PairBlock>> = (0..N_CATEGORIES).map(|_| Vec::new()).collect();
    for chunk in chunks {
        for (column, block) in columns.iter_mut().zip(chunk.0) {
            column.push(block);
        }
    }
    // Each category owns a disjoint block and a disjoint column, so the
    // copies run together; every part is dropped as soon as it is copied.
    let mut out = PairBlocks::empty();
    out.0
        .par_iter_mut()
        .zip(columns)
        .try_for_each(|(block, parts)| {
            let total: usize = parts.iter().map(PairBlock::len).sum();
            if total == 0 {
                return Ok(());
            }
            block.reserve_exact(total)?;
            for part in parts {
                block.extend_from(&part);
            }
            Ok(())
        })?;
    Ok(out)
}

/// Cut `buf` into consecutive slices of the given sizes.
fn split_sizes(mut buf: &mut [i32], sizes: impl Iterator<Item = usize>) -> Vec<&mut [i32]> {
    sizes
        .map(|size| {
            let (head, tail) = std::mem::take(&mut buf).split_at_mut(size);
            buf = tail;
            head
        })
        .collect()
}

fn two_pass(query: &Query, ranges: &[(usize, usize)]) -> Result<PairBlocks, Error> {
    let counts = task_table(ranges, |r| query.count(r))?;
    let mut out = PairBlocks::empty();
    for cat in query.requested.iter() {
        let i = cat.index();
        let total: usize = counts.iter().map(|c| c[i]).sum();
        out.0[i].reserve_exact(total)?;
        // The fill pass writes every slot, so this zeroing is redundant.
        // Skipping it would need `set_len` over uninitialised memory, and it
        // is not worth that: the zero fill of both arrays at the 300k
        // degree-5 payload measures 0.166 s, most of which is the page
        // faults the fill pass would take anyway.
        out.0[i].first.resize(total, 0);
        out.0[i].second.resize(total, 0);
    }
    // Every task owns one disjoint slice of every block.
    let mut slots: Vec<Vec<(&mut [i32], &mut [i32])>> = (0..ranges.len())
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
        .try_for_each(|(mut slot, &range)| {
            let mut cursor = [0usize; N_CATEGORIES];
            query.run(range, |cat, a, b| {
                let i = cat.index();
                slot[i].0[cursor[i]] = a as i32;
                slot[i].1[cursor[i]] = b as i32;
                cursor[i] += 1;
                Ok(())
            })?;
            for (i, (first, _)) in slot.iter().enumerate() {
                assert_eq!(cursor[i], first.len(), "pass two disagrees with pass one");
            }
            Ok(())
        })?;
    Ok(out)
}

/// Reject a graph-to-view map that is not a partial permutation.
///
/// One pass, one byte per graph row.  Entries are `-1` for an unselected row,
/// else the row's view index, each used once.
fn check_view_map(map: &[i32]) -> Result<(), Error> {
    let mut seen = alloc::filled(false, map.len(), Family::ViewSortScratch, "bool")?;
    for (position, &value) in map.iter().enumerate() {
        if value < 0 {
            continue;
        }
        let view_row = value as usize;
        let reason = if view_row >= map.len() {
            "view rows run from 0 to one below the graph row count"
        } else if std::mem::replace(&mut seen[view_row], true) {
            "two graph rows map to this view row"
        } else {
            continue;
        };
        return Err(Error::InvalidViewMap {
            position,
            value: value as i64,
            reason,
        });
    }
    Ok(())
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
    let mut packed: Vec<u64> = alloc::with_capacity(len, Family::ViewSortScratch, "uint64")?;
    packed.extend(block.first.iter().zip(&block.second).map(|(&a, &b)| {
        let (lo, hi) = (a.min(b) as u64, a.max(b) as u64);
        ((lo * n + hi) << 1) | u64::from(a > b)
    }));
    packed.par_sort_unstable();
    for (k, p) in packed.iter().enumerate() {
        let key = p >> 1;
        let (lo, hi) = ((key / n) as i32, (key % n) as i32);
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

    const EXECUTIONS: [Execution; 2] = [Execution::Speed, Execution::Memory];

    fn all(cols: &PedigreeColumns, view: Option<&[i32]>) -> PairBlocks {
        let ped = cols.try_borrow().unwrap();
        let blocks: Vec<PairBlocks> = EXECUTIONS
            .iter()
            .map(|&e| {
                pair_blocks(&ped, MaxDegree::MAX, CategorySet::up_to_degree(5), view, e).unwrap()
            })
            .collect();
        assert!(
            blocks.iter().all(|b| *b == blocks[0]),
            "executions disagree"
        );
        blocks.into_iter().next().unwrap()
    }

    fn pairs(blocks: &PairBlocks, cat: Category) -> Vec<(i32, i32)> {
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
                .map(|&(a, b)| a.min(b) as u64 * n + a.max(b) as u64)
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
        let counts = count_pairs(&ped, MaxDegree::MAX, None).unwrap();
        for cat in Category::ALL {
            assert_eq!(got.get(cat).len() as u64, counts.get(cat), "{}", cat.code());
        }
        let only: CategorySet = [Category::C1, Category::Av].into_iter().collect();
        let some = pair_blocks(&ped, MaxDegree::MAX, only, None, Execution::Speed).unwrap();
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
            for execution in EXECUTIONS {
                for v in [None, Some(view.as_slice())] {
                    let blocks = pool
                        .install(|| {
                            pair_blocks(
                                &ped,
                                MaxDegree::MAX,
                                CategorySet::up_to_degree(5),
                                v,
                                execution,
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

    #[test]
    fn a_dual_valid_half_avuncular_pair_keeps_the_lower_row_first() {
        // From tests/test_relationship_pairs.py: 4's mother 2 is a paternal
        // half sib of 5 (father 0), and 5's mother 3 is a paternal half sib
        // of 4 (father 1), so (4, 5) is HAv both ways.
        let cols = pedigree(&[(-1, -1), (-1, -1), (-1, 0), (-1, 1), (2, 1), (3, 0)], &[]);
        let got = all(&cols, None);
        assert_eq!(pairs(&got, Category::HAv), vec![(4, 5)]);
    }

    use crate::alloc::Family;

    #[test]
    fn a_view_map_that_repeats_or_overruns_a_row_is_rejected() {
        let cols = crate::relationships::testing::random_pedigree(40, 3);
        let ped = cols.try_borrow().unwrap();
        let cats = CategorySet::up_to_degree(3);
        let run = |map: Vec<i32>| {
            pair_blocks(
                &ped,
                MaxDegree::MAX,
                cats,
                Some(map.as_slice()),
                Execution::Speed,
            )
        };

        // Two graph rows on one view row would give equal sort keys, and an
        // unstable parallel sort would then order them by thread count.
        let mut repeated: Vec<i32> = (0..40)
            .map(|r| if r % 2 == 0 { r / 2 } else { -1 })
            .collect();
        repeated[2] = 0;
        assert!(matches!(
            run(repeated),
            Err(Error::InvalidViewMap { position: 2, .. })
        ));

        let mut overrun: Vec<i32> = vec![-1; 40];
        overrun[7] = 40;
        assert!(matches!(
            run(overrun),
            Err(Error::InvalidViewMap { position: 7, .. })
        ));

        // Selecting nothing is legal and has no pairs.
        assert_eq!(run(vec![-1; 40]).unwrap().total(), 0);
    }

    /// Which families each mode reserves.
    ///
    /// The table is checked in both directions: a family it calls reached
    /// must fail the call, and one it calls unreached must leave the call
    /// succeeding, since the plant is only ever consumed by a reservation.
    /// A family that starts or stops being reserved therefore fails this
    /// test instead of silently dropping a case.
    fn seam_reaches(family: Family, execution: Option<Execution>, view: bool) -> bool {
        match family {
            Family::ViewSortScratch => view,
            Family::TaskChunk => execution == Some(Execution::Speed),
            Family::TaskTable | Family::PairBlock => execution.is_some(),
            // Reserved only by the kinship walk, the kinship matrix DP and
            // the inbreeding, lineage and generation sweeps; their own seam
            // tests cover them.
            Family::KinshipMemo
            | Family::KinshipStack
            | Family::KinshipOutput
            | Family::KinshipRows
            | Family::KinshipCsc
            | Family::KinshipSums
            | Family::KinshipScratch
            | Family::InbreedingWalk
            | Family::LineageSets
            | Family::LineageOutput
            | Family::FounderMeans => false,
            _ => true,
        }
    }

    /// Every allocation family reports `allocation_failed` for counts and
    /// for both executions, on a graph and a view, instead of aborting.
    ///
    /// The seam is process-wide, so each case runs [`seam_child`] in a fresh
    /// copy of this test binary where nothing else can consume the plant.
    #[test]
    fn a_refused_allocation_of_any_family_is_an_error() {
        let exe = std::env::current_exe().unwrap();
        for family in Family::ALL {
            for (mode, execution, view) in SEAM_MODES {
                let expect = if seam_reaches(family, execution, view) {
                    "fail"
                } else {
                    "pass"
                };
                let out = std::process::Command::new(&exe)
                    .args([
                        "--exact",
                        "relationships::pairs::tests::seam_child",
                        "--nocapture",
                    ])
                    .env("PG_SEAM_FAMILY", family.name())
                    .env("PG_SEAM_MODE", mode)
                    .env("PG_SEAM_EXPECT", expect)
                    .output()
                    .unwrap();
                assert!(
                    out.status.success(),
                    "{}/{mode} expected to {expect}:\n{}",
                    family.name(),
                    String::from_utf8_lossy(&out.stderr)
                );
            }
        }
    }

    /// The seam cases: a mode name, the execution (`None` counts), and
    /// whether a view is passed.
    const SEAM_MODES: [(&str, Option<Execution>, bool); 5] = [
        ("count", None, false),
        ("speed", Some(Execution::Speed), false),
        ("memory", Some(Execution::Memory), false),
        ("speed_view", Some(Execution::Speed), true),
        ("memory_view", Some(Execution::Memory), true),
    ];

    /// The body of one seam case; a no-op unless `PG_SEAM_FAMILY` is set.
    #[test]
    fn seam_child() {
        use crate::alloc::fail_next;
        let Ok(name) = std::env::var("PG_SEAM_FAMILY") else {
            return;
        };
        let family = Family::parse(&name).unwrap();
        let mode = std::env::var("PG_SEAM_MODE").unwrap();
        let (_, execution, view) = SEAM_MODES.into_iter().find(|m| m.0 == mode).unwrap();
        let cols = crate::relationships::testing::random_pedigree(300, 5);
        let ped = cols.try_borrow().unwrap();
        let view_map: Vec<i32> = (0..300)
            .map(|r| if r % 2 == 0 { r / 2 } else { -1 })
            .collect();
        let all_cats = CategorySet::up_to_degree(5);
        fail_next(Some(family));
        let result = match execution {
            None => count_pairs(&ped, MaxDegree::MAX, None).map(|c| c.get(Category::FS) as usize),
            Some(execution) => {
                let v = view.then_some(view_map.as_slice());
                pair_blocks(&ped, MaxDegree::MAX, all_cats, v, execution).map(|b| b.total())
            }
        };
        let expect_failure = std::env::var("PG_SEAM_EXPECT").as_deref() == Ok("fail");
        match (expect_failure, result) {
            (true, Err(Error::AllocationFailed { operation, .. })) => {
                assert_eq!(operation, family.name())
            }
            // The plant is consumed only by a reservation of its family, so a
            // call that succeeds proves this mode never reserves it.
            (false, Ok(_)) => {}
            (_, other) => panic!("{name}/{mode} expecting {expect_failure} gave {other:?}"),
        }
        fail_next(None);
        assert!(pair_blocks(&ped, MaxDegree::MAX, all_cats, None, Execution::Speed).is_ok());
    }
}
