//! Per-row ancestor and descendant counts over the parent edges.
//!
//! * [`distinct_ancestor_counts`]: the number of distinct strict ancestors
//!   of each row, an ancestor reached through several paths counted once.
//! * [`descendant_path_counts`]: the number of walks down the pedigree from
//!   each row, its children plus their path counts, which exceeds the
//!   distinct descendant count wherever marriage loops join two paths.
//!
//! Both sweep parents first and index graph rows directly: the graph rows
//! themselves when every parent row already precedes its children, else the
//! stable depth-major order of `crate::kinship::depth_order`.  Counts do
//! not depend on row labels or sweep order, so they are the 0.9.3 Numba
//! kernels' counts exactly.

use crate::alloc::{self, Family};
use crate::error::Error;
use crate::kinship::depth_order::{readable, readable_mut, unread, ParentsFirst};
use crate::relationships::check_column_length;

const SETS: Family = Family::LineageSets;
const OUTPUT: Family = Family::LineageOutput;

/// The parent columns the parents-first sweeps read, borrowed from the host
/// for one call, with the structural depth they sort by when the rows are
/// not already parents-first.  Depth is optional because a pedigree whose
/// parent rows all precede their children is swept as it is and never
/// needs it.
///
/// The parent rows are checked by the sweeps, not here, in passes they
/// already make: with a depth where it is sorted, in graph order as each row
/// is visited.  No sweep makes a pass of its own to check them.
#[derive(Clone, Copy)]
pub struct ParentColumns<'a> {
    mother: &'a [i32],
    father: &'a [i32],
    depth: Option<&'a [i32]>,
}

impl<'a> ParentColumns<'a> {
    /// # Errors
    ///
    /// [`Error::LengthMismatch`] when `father` is not as long as `mother`.
    pub(crate) fn try_new(
        mother: &'a [i32],
        father: &'a [i32],
        depth: Option<&'a [i32]>,
    ) -> Result<ParentColumns<'a>, Error> {
        check_column_length("father", father.len(), mother.len())?;
        Ok(ParentColumns {
            mother,
            father,
            depth,
        })
    }

    pub fn len(&self) -> usize {
        self.mother.len()
    }

    pub fn is_empty(&self) -> bool {
        self.mother.is_empty()
    }

    pub(crate) fn mother(&self) -> &'a [i32] {
        self.mother
    }

    pub(crate) fn father(&self) -> &'a [i32] {
        self.father
    }

    /// The depth-major sort when the depth sorts, else the graph rows.
    pub(crate) fn parents_first(&self, family: Family) -> ParentsFirst {
        ParentsFirst::of(self.mother, self.father, self.depth, family)
    }

    /// A graph-order sweep's result: `out` when every parent it met preceded
    /// its child, else the error [`ParentsFirst::unordered`] finds.
    pub(crate) fn ordered<T>(
        &self,
        (out, ordered): (T, bool),
        depth_error: Option<Error>,
    ) -> Result<T, Error> {
        if ordered {
            Ok(out)
        } else {
            Err(ParentsFirst::unordered(
                self.mother,
                self.father,
                depth_error,
            ))
        }
    }
}

/// Distinct strict ancestors of every graph row, as int32.
///
/// A row with children owns its closed ancestor set (its strict ancestors
/// and itself) as a sorted, exactly sized slice, built by one merge of its
/// parents' sets; a child's count is the length of that merge before the
/// child is added; a row with the previous row's parents reuses its merge.
/// A set is dropped once its row's last child is counted, and a row without
/// children stores nothing.
///
/// # Errors
///
/// When the rows are not parents-first: [`Error::ValueOutOfRange`] at the
/// first `mother`, then `father`, row outside `-1..n`, then
/// [`Error::LengthMismatch`] on `depth` when it is absent or not one per row
/// and [`Error::ValueOutOfRange`] when a row's depth is negative or not
/// above both parents'.  [`Error::AllocationFailed`] for any buffer.
pub fn distinct_ancestor_counts(ped: ParentColumns<'_>) -> Result<Vec<i32>, Error> {
    match ped.parents_first(SETS) {
        ParentsFirst::Rows { depth_error } => ped.ordered(
            ancestors_in::<false>(&ped, 0..ped.len() as u32)?,
            depth_error,
        ),
        ParentsFirst::DepthMajor(sweep) => {
            Ok(ancestors_in::<true>(&ped, sweep.order.iter().copied())?.0)
        }
    }
}

/// The counts, and whether every parent met was readable ([`readable`]).
/// In graph order the sweep stops at the first parent it cannot read: the
/// caller discards it, and a set it would keep alive is not built.
fn ancestors_in<const CHECKED: bool>(
    ped: &ParentColumns<'_>,
    rows: impl Iterator<Item = u32>,
) -> Result<(Vec<i32>, bool), Error> {
    let n = ped.len();
    let (mother, father) = (ped.mother(), ped.father());
    let mut remaining = alloc::filled(0u32, n, SETS)?;
    // Every parent row in range; a sweep reads only those.
    for &p in mother.iter().chain(father) {
        if let Some(r) = usize::try_from(p).ok().and_then(|p| remaining.get_mut(p)) {
            *r += 1;
        }
    }
    let mut sets: Vec<Box<[u32]>> = alloc::filled(Box::default(), n, SETS)?;
    let mut merged: Vec<u32> = Vec::new();
    let mut merged_for = None;
    let mut counts = alloc::filled(0i32, n, OUTPUT)?;
    let mut ordered = true;

    for row in rows {
        let i = row as usize;
        let rows_read = readable::<CHECKED, _>(&remaining, row).len();
        let parents = [mother[i], father[i]].map(|p| {
            let at = p as usize;
            if at < rows_read {
                Some(at)
            } else {
                unread(p, &mut ordered);
                None
            }
        });
        if !CHECKED && !ordered {
            break;
        }
        // A full sib of the previous row reuses its merge.
        if merged_for != Some(parents) {
            let side = |p: Option<usize>| -> &[u32] { p.map_or(&[], |p| &sets[p]) };
            let (a, b) = (side(parents[0]), side(parents[1]));
            merged.clear();
            alloc::reserve(&mut merged, a.len() + b.len() + 1, SETS)?;
            union_into(a, b, &mut merged);
            merged_for = Some(parents);
        }
        counts[i] = merged.len() as i32;

        if remaining[i] > 0 {
            let at = merged.partition_point(|&r| r < row);
            let mut set: Vec<u32> = alloc::with_capacity(merged.len() + 1, SETS)?;
            set.extend_from_slice(&merged[..at]);
            set.push(row);
            set.extend_from_slice(&merged[at..]);
            sets[i] = set.into_boxed_slice();
        }
        for p in parents.into_iter().flatten() {
            remaining[p] -= 1;
            if remaining[p] == 0 {
                sets[p] = Box::default();
            }
        }
    }
    Ok((counts, ordered))
}

/// Append the sorted union of two sorted, duplicate-free slices.
fn union_into(a: &[u32], b: &[u32], out: &mut Vec<u32>) {
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        let (x, y) = (a[i], b[j]);
        out.push(x.min(y));
        i += usize::from(x <= y);
        j += usize::from(y <= x);
    }
    out.extend_from_slice(&a[i..]);
    out.extend_from_slice(&b[j..]);
}

/// Descendant paths from every graph row, as int64.
///
/// # Errors
///
/// [`Error::ArithmeticOverflow`] when a count outgrows int64, as it does
/// after about 63 generations of repeated sib mating;
/// when the rows need sorting, [`Error::LengthMismatch`] or
/// [`Error::ValueOutOfRange`] on `mother`, `father` or `depth` as
/// [`distinct_ancestor_counts`] reports them; and
/// [`Error::AllocationFailed`] for any buffer.
pub fn descendant_path_counts(ped: ParentColumns<'_>) -> Result<Vec<i64>, Error> {
    match ped.parents_first(OUTPUT) {
        ParentsFirst::Rows { depth_error } => ped.ordered(
            descendants_in::<false>(&ped, (0..ped.len() as u32).rev())?,
            depth_error,
        ),
        ParentsFirst::DepthMajor(sweep) => {
            Ok(descendants_in::<true>(&ped, sweep.order.iter().rev().copied())?.0)
        }
    }
}

/// The reverse sweep over `rows`, children before parents, and whether
/// every parent met was readable ([`readable`]).  Overflow is collected
/// branch-free and reported once at the end: every add's carry is kept, so
/// none is missed, and the counts are dropped when one occurred.  An
/// unreadable parent outranks an overflow, which it may have caused.
fn descendants_in<const CHECKED: bool>(
    ped: &ParentColumns<'_>,
    rows: impl Iterator<Item = u32>,
) -> Result<(Vec<i64>, bool), Error> {
    let (mother, father) = (ped.mother(), ped.father());
    let mut counts = alloc::filled(0i64, ped.len(), OUTPUT)?;
    let mut overflowed = false;
    let mut ordered = true;
    for row in rows {
        let i = row as usize;
        let (paths, carry) = counts[i].overflowing_add(1);
        overflowed |= carry;
        let parents = readable_mut::<CHECKED, _>(&mut counts, row);
        for p in [mother[i], father[i]] {
            match parents.get_mut(p as usize) {
                Some(count) => {
                    let (sum, carry) = count.overflowing_add(paths);
                    *count = sum;
                    overflowed |= carry;
                }
                None => unread(p, &mut ordered),
            }
        }
    }
    if overflowed && ordered {
        return Err(Error::ArithmeticOverflow {
            operation: "descendant_path_counts",
            dtype: "int64",
        });
    }
    Ok((counts, ordered))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kinship::KinshipPedigree;
    use crate::topology::structural_depth;
    use std::collections::BTreeSet;

    /// `kernel` swept depth-major; when the rows are parents-first, also in
    /// graph order with no depth, which must agree bit for bit (each row
    /// reads its parents in one order either way).
    fn run<T: PartialEq + std::fmt::Debug>(
        mother: &[i32],
        father: &[i32],
        kernel: fn(ParentColumns<'_>) -> Result<T, Error>,
    ) -> Result<T, Error> {
        let depth = structural_depth(mother, father);
        let sorted = kernel(ParentColumns::try_new(mother, father, Some(&depth)).unwrap());
        if crate::topology::is_topological(mother, father) {
            let graph = kernel(ParentColumns::try_new(mother, father, None).unwrap());
            assert_eq!(graph.as_ref().ok(), sorted.as_ref().ok());
        }
        sorted
    }

    /// Distinct ancestors by explicit set closure, for comparison.
    fn closure(mother: &[i32], father: &[i32]) -> Vec<i32> {
        (0..mother.len())
            .map(|i| {
                let mut seen = BTreeSet::new();
                let mut stack = vec![i];
                while let Some(r) = stack.pop() {
                    for p in [mother[r], father[r]] {
                        if p >= 0 && seen.insert(p) {
                            stack.push(p as usize);
                        }
                    }
                }
                seen.len() as i32
            })
            .collect()
    }

    /// Descendant paths by memoised recursion, for comparison.
    fn paths(mother: &[i32], father: &[i32]) -> Vec<i64> {
        let n = mother.len();
        let mut children = vec![Vec::new(); n];
        for i in 0..n {
            for p in [mother[i], father[i]] {
                if p >= 0 {
                    children[p as usize].push(i);
                }
            }
        }
        fn go(r: usize, children: &[Vec<usize>], memo: &mut [Option<i64>]) -> i64 {
            if let Some(v) = memo[r] {
                return v;
            }
            let v = children[r].iter().map(|&c| 1 + go(c, children, memo)).sum();
            memo[r] = Some(v);
            v
        }
        let mut memo = vec![None; n];
        (0..n).map(|r| go(r, &children, &mut memo)).collect()
    }

    /// The random pedigree with its rows reversed, so parents follow children.
    fn reversed(mother: &[i32], father: &[i32]) -> (Vec<i32>, Vec<i32>) {
        let n = mother.len() as i32;
        let flip = |col: &[i32]| -> Vec<i32> {
            col.iter()
                .rev()
                .map(|&p| if p < 0 { -1 } else { n - 1 - p })
                .collect()
        };
        (flip(mother), flip(father))
    }

    #[test]
    fn adjacent_full_sibs_match_the_brute_force_in_either_row_order() {
        // Each random row becomes a sibship of one to three consecutive rows,
        // its children drawn from the first.
        let c = crate::relationships::testing::random_pedigree(400, 5);
        let (mut mother, mut father, mut first) = (Vec::new(), Vec::new(), Vec::new());
        let moved = |p: i32, first: &[i32]| if p < 0 { -1 } else { first[p as usize] };
        for i in 0..c.mother.len() {
            first.push(mother.len() as i32);
            for _ in 0..1 + i % 3 {
                mother.push(moved(c.mother[i], &first));
                father.push(moved(c.father[i], &first));
            }
        }
        for (m, f) in [(mother.clone(), father.clone()), reversed(&mother, &father)] {
            assert_eq!(
                run(&m, &f, distinct_ancestor_counts).unwrap(),
                closure(&m, &f)
            );
        }
    }

    #[test]
    fn counts_match_the_brute_force_in_either_row_order() {
        for seed in [1, 2, 3] {
            let c = crate::relationships::testing::random_pedigree(500, seed);
            for (m, f) in [
                (c.mother.clone(), c.father.clone()),
                reversed(&c.mother, &c.father),
            ] {
                assert_eq!(
                    run(&m, &f, distinct_ancestor_counts).unwrap(),
                    closure(&m, &f)
                );
                assert_eq!(run(&m, &f, descendant_path_counts).unwrap(), paths(&m, &f));
            }
        }
    }

    #[test]
    fn a_marriage_loop_counts_a_shared_ancestor_once_and_its_paths_twice() {
        // 0 and 1 found; 2 and 3 are full sibs; 4 is their child.
        let (m, f) = ([-1, -1, 0, 0, 2], [-1, -1, 1, 1, 3]);
        assert_eq!(
            run(&m, &f, distinct_ancestor_counts).unwrap(),
            vec![0, 0, 2, 2, 4]
        );
        assert_eq!(
            run(&m, &f, descendant_path_counts).unwrap(),
            vec![4, 4, 1, 1, 0]
        );
    }

    #[test]
    fn one_row_as_both_parents_is_one_ancestor_and_two_paths() {
        let (m, f) = ([-1, 0], [-1, 0]);
        assert_eq!(run(&m, &f, distinct_ancestor_counts).unwrap(), vec![0, 1]);
        assert_eq!(run(&m, &f, descendant_path_counts).unwrap(), vec![2, 0]);
    }

    #[test]
    fn an_empty_pedigree_counts_nothing() {
        assert!(run(&[], &[], distinct_ancestor_counts).unwrap().is_empty());
        assert!(run(&[], &[], descendant_path_counts).unwrap().is_empty());
    }

    /// Two founders, then two rows per generation, each a child of both rows
    /// of the generation above: every row's path count is twice one more
    /// than a row's below it, so it doubles each generation.
    fn sib_mating_ladder(generations: i32) -> (Vec<i32>, Vec<i32>) {
        let (mut m, mut f) = (vec![-1, -1], vec![-1, -1]);
        for g in 0..generations {
            for _ in 0..2 {
                m.push(2 * g);
                f.push(2 * g + 1);
            }
        }
        (m, f)
    }

    #[test]
    fn a_path_count_past_int64_is_an_overflow_error() {
        // g generations below a row give it 2^(g + 1) - 2 paths.
        let (m, f) = sib_mating_ladder(62);
        let counts = run(&m, &f, descendant_path_counts).unwrap();
        assert_eq!(counts[0], i64::MAX - 1);
        let (m, f) = sib_mating_ladder(64);
        assert_eq!(m.len(), 130);
        match run(&m, &f, descendant_path_counts) {
            Err(Error::ArithmeticOverflow { operation, dtype }) => {
                assert_eq!(operation, "descendant_path_counts");
                assert_eq!(dtype, "int64");
            }
            other => panic!("expected an overflow, got {other:?}"),
        }
    }

    #[test]
    fn a_shallow_depth_is_refused_when_the_rows_need_sorting() {
        // Row 0 is the child of rows 1 and 2, so the sweep must sort by depth.
        let (m, f) = ([1, -1, -1], [2, -1, -1]);
        let depth = [0, 0, 0];
        let ped = ParentColumns::try_new(&m, &f, Some(&depth)).unwrap();
        for result in [
            distinct_ancestor_counts(ped).map(|_| ()),
            descendant_path_counts(ped).map(|_| ()),
        ] {
            assert!(matches!(
                result,
                Err(Error::ValueOutOfRange {
                    field: "depth",
                    position,
                    ..
                }) if position == 0.into()
            ));
        }
    }

    #[test]
    fn parents_first_rows_are_swept_without_reading_depth() {
        let (m, f) = ([-1, -1, 0], [-1, -1, 1]);
        let depth = [0, 0, 0];
        for depth in [None, Some(&depth[..])] {
            let ped = ParentColumns::try_new(&m, &f, depth).unwrap();
            assert_eq!(distinct_ancestor_counts(ped).unwrap(), vec![0, 0, 2]);
            assert_eq!(descendant_path_counts(ped).unwrap(), vec![1, 1, 0]);
        }
    }

    #[test]
    fn rows_that_need_sorting_need_a_depth() {
        let (m, f) = ([1, -1, -1], [2, -1, -1]);
        let ped = ParentColumns::try_new(&m, &f, None).unwrap();
        assert!(matches!(
            descendant_path_counts(ped),
            Err(Error::LengthMismatch { field: "depth", .. })
        ));
    }

    /// Which sweep reserves each of the four sweep families.
    const SWEEP_SEAMS: [(Family, &str); 6] = [
        (Family::InbreedingWalk, "inbreeding"),
        (Family::LineageSets, "ancestors"),
        (Family::LineageOutput, "ancestors"),
        (Family::LineageOutput, "descendants"),
        (Family::LineageOutput, "generations"),
        (Family::FounderMeans, "founder_means"),
    ];

    /// Every sweep family reports `allocation_failed` from each sweep that
    /// reserves it, in a fresh process because the seam is process-wide.
    #[test]
    fn a_refused_allocation_of_any_sweep_family_is_an_error() {
        let exe = std::env::current_exe().unwrap();
        for (family, product) in SWEEP_SEAMS {
            let out = std::process::Command::new(&exe)
                .args(["--exact", "lineage::tests::seam_child", "--nocapture"])
                .env("PG_SWEEP_SEAM_FAMILY", family.name())
                .env("PG_SWEEP_SEAM_PRODUCT", product)
                .output()
                .unwrap();
            assert!(
                out.status.success(),
                "{}/{product}:\n{}",
                family.name(),
                String::from_utf8_lossy(&out.stderr)
            );
        }
    }

    /// The body of one seam case; a no-op unless `PG_SWEEP_SEAM_FAMILY` is set.
    #[test]
    fn seam_child() {
        use crate::alloc::fail_next_above;
        use crate::kinship::{equivalent_generations, founder_contribution_means, inbreeding};
        let Ok(name) = std::env::var("PG_SWEEP_SEAM_FAMILY") else {
            return;
        };
        let family = Family::parse(&name).unwrap();
        let product = std::env::var("PG_SWEEP_SEAM_PRODUCT").unwrap();
        let n = 300;
        let c = crate::relationships::testing::random_pedigree(n, 5);
        let depth = structural_depth(&c.mother, &c.father);
        let ped = KinshipPedigree::try_new(&c.mother, &c.father, &c.twin, &depth).unwrap();
        // Parents-first rows with no depth, as Python passes them, so the
        // plant lands in the sweep's own buffers rather than in a sort the
        // rows did not need (whose failure the sweep would rightly ignore).
        let cols = ParentColumns::try_new(&c.mother, &c.father, None).unwrap();
        let cohort: Vec<i32> = depth.iter().map(|&d| d.min(3)).collect();
        let column: Vec<i64> = (0..n)
            .map(|r| {
                if c.mother[r] < 0 && c.father[r] < 0 {
                    r as i64
                } else {
                    -1
                }
            })
            .collect();
        let run = |ped| match product.as_str() {
            "inbreeding" => inbreeding(ped).map(|v| v.len()),
            "ancestors" => distinct_ancestor_counts(cols).map(|v| v.len()),
            "descendants" => descendant_path_counts(cols).map(|v| v.len()),
            "generations" => equivalent_generations(cols).map(|v| v.len()),
            _ => founder_contribution_means(ped, &cohort, 4, &column, n).map(|v| v.len()),
        };
        // The floor aims the plant at an input-sized buffer.
        fail_next_above(Some(family), n);
        match run(ped) {
            Err(Error::AllocationFailed {
                operation,
                requested_elements,
                ..
            }) => {
                assert_eq!(operation, family.name());
                assert!(requested_elements >= n);
            }
            other => panic!("{name}/{product} gave {other:?}"),
        }
        fail_next_above(None, 0);
        assert!(run(ped).is_ok());
    }
}
