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
//! stable depth-major order of [`crate::kinship::depth_order`].  Counts do
//! not depend on row labels or sweep order, so they are the 0.9.3 Numba
//! kernels' counts exactly.

use crate::alloc::{self, Family};
use crate::error::Error;
use crate::kinship::depth_order::ParentsFirst;
use crate::relationships::{check_column_length, check_row_range};

const SETS: Family = Family::LineageSets;
const OUTPUT: Family = Family::LineageOutput;

/// The parent columns the parents-first sweeps read, borrowed from the host
/// for one call, with the structural depth they sort by when the rows are
/// not already parents-first.  Depth is optional because a pedigree whose
/// parent rows all precede their children is swept as it is and never
/// needs it.
#[derive(Clone, Copy)]
pub struct ParentColumns<'a> {
    mother: &'a [i32],
    father: &'a [i32],
    depth: Option<&'a [i32]>,
    /// Every parent row precedes its child, so the rows sweep as they are.
    parents_first: bool,
}

impl<'a> ParentColumns<'a> {
    /// Check `father` has one entry per row and every parent row is `-1` or
    /// a row of the pedigree, and note whether every parent row precedes its
    /// child, in one pass; `depth` is checked where it is read.
    ///
    /// # Errors
    ///
    /// [`Error::LengthMismatch`] and [`Error::ValueOutOfRange`], as the
    /// relationship engine reports them.
    pub fn try_new(
        mother: &'a [i32],
        father: &'a [i32],
        depth: Option<&'a [i32]>,
    ) -> Result<ParentColumns<'a>, Error> {
        let n = mother.len();
        check_column_length("father", father.len(), n)?;
        let (mut in_range, mut parents_first) = (true, true);
        for (row, (&m, &f)) in mother.iter().zip(father).enumerate() {
            let (m, f) = (i64::from(m), i64::from(f));
            in_range &= m >= -1 && f >= -1 && m < n as i64 && f < n as i64;
            parents_first &= m < row as i64 && f < row as i64;
        }
        if !in_range {
            check_row_range("mother", mother, n)?;
            check_row_range("father", father, n)?;
        }
        Ok(ParentColumns {
            mother,
            father,
            depth,
            parents_first,
        })
    }

    /// Columns a graph construction has already validated: every parent row
    /// is `-1` or a row of the pedigree, and `parents_first` is what
    /// construction found.  Only the lengths are re-checked, which keeps a
    /// sweep of a few milliseconds to the one pass the 0.9.3 kernel made; a
    /// broken promise can give wrong counts or a panic, never undefined
    /// behaviour.
    ///
    /// # Errors
    ///
    /// [`Error::LengthMismatch`] on `father`.
    pub fn validated(
        mother: &'a [i32],
        father: &'a [i32],
        depth: Option<&'a [i32]>,
        parents_first: bool,
    ) -> Result<ParentColumns<'a>, Error> {
        check_column_length("father", father.len(), mother.len())?;
        Ok(ParentColumns {
            mother,
            father,
            depth,
            parents_first,
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

    /// The graph rows when they are parents-first, else the depth-major sort.
    ///
    /// # Errors
    ///
    /// As [`ParentsFirst::sorted`], when the rows need sorting.
    pub(crate) fn parents_first(&self, family: Family) -> Result<ParentsFirst, Error> {
        if self.parents_first {
            return Ok(ParentsFirst::Rows);
        }
        ParentsFirst::sorted(self.mother, self.father, self.depth, family)
    }
}

/// Distinct strict ancestors of every graph row, as int32.
///
/// A row with children owns its closed ancestor set (its strict ancestors
/// and itself) as a sorted, exactly sized slice, built by one merge of its
/// parents' sets; a child's count is the length of that merge before the
/// child is added.  A set is dropped once its row's last child is counted,
/// and a row without children stores nothing.
///
/// # Errors
///
/// When the rows need sorting, [`Error::LengthMismatch`] on `depth` when it
/// is absent or not one per row and [`Error::ValueOutOfRange`] when a row's
/// depth is negative or not above both parents'; [`Error::AllocationFailed`]
/// for any buffer.
pub fn distinct_ancestor_counts(ped: ParentColumns<'_>) -> Result<Vec<i32>, Error> {
    match ped.parents_first(SETS)? {
        ParentsFirst::Rows => ancestors_in(&ped, 0..ped.len() as u32),
        ParentsFirst::DepthMajor(sweep) => ancestors_in(&ped, sweep.order.iter().copied()),
    }
}

fn ancestors_in(
    ped: &ParentColumns<'_>,
    rows: impl Iterator<Item = u32>,
) -> Result<Vec<i32>, Error> {
    let n = ped.len();
    let (mother, father) = (ped.mother(), ped.father());
    let mut remaining = alloc::filled(0u32, n, SETS, "uint32")?;
    for &p in mother.iter().chain(father) {
        if p >= 0 {
            remaining[p as usize] += 1;
        }
    }
    let mut sets: Vec<Box<[u32]>> = alloc::filled(Box::default(), n, SETS, "uint32")?;
    let mut merged: Vec<u32> = Vec::new();
    let mut counts = alloc::filled(0i32, n, OUTPUT, "int32")?;

    for row in rows {
        let i = row as usize;
        let side = |p: i32| -> &[u32] {
            if p < 0 {
                &[]
            } else {
                &sets[p as usize]
            }
        };
        let (a, b) = (side(mother[i]), side(father[i]));
        merged.clear();
        alloc::reserve(&mut merged, a.len() + b.len() + 1, SETS, "uint32")?;
        union_into(a, b, &mut merged);
        counts[i] = merged.len() as i32;

        if remaining[i] > 0 {
            let at = merged.partition_point(|&r| r < row);
            let mut set: Vec<u32> = alloc::with_capacity(merged.len() + 1, SETS, "uint32")?;
            set.extend_from_slice(&merged[..at]);
            set.push(row);
            set.extend_from_slice(&merged[at..]);
            sets[i] = set.into_boxed_slice();
        }
        for p in [mother[i], father[i]] {
            if p >= 0 {
                let p = p as usize;
                remaining[p] -= 1;
                if remaining[p] == 0 {
                    sets[p] = Box::default();
                }
            }
        }
    }
    Ok(counts)
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
/// [`Error::ValueOutOfRange`] on `depth` as [`distinct_ancestor_counts`]
/// reports them; and [`Error::AllocationFailed`] for any buffer.
pub fn descendant_path_counts(ped: ParentColumns<'_>) -> Result<Vec<i64>, Error> {
    match ped.parents_first(OUTPUT)? {
        ParentsFirst::Rows => descendants_in(&ped, (0..ped.len() as u32).rev()),
        ParentsFirst::DepthMajor(sweep) => descendants_in(&ped, sweep.order.iter().rev().copied()),
    }
}

/// The reverse sweep over `rows`, children before parents.  Overflow is
/// collected branch-free and reported once at the end: every add's carry is
/// kept, so none is missed, and the counts are dropped when one occurred.
fn descendants_in(
    ped: &ParentColumns<'_>,
    rows: impl Iterator<Item = u32>,
) -> Result<Vec<i64>, Error> {
    let (mother, father) = (ped.mother(), ped.father());
    let mut counts = alloc::filled(0i64, ped.len(), OUTPUT, "int64")?;
    let mut overflowed = false;
    for row in rows {
        let i = row as usize;
        let (paths, carry) = counts[i].overflowing_add(1);
        overflowed |= carry;
        for p in [mother[i], father[i]] {
            if p >= 0 {
                let count = &mut counts[p as usize];
                let (sum, carry) = count.overflowing_add(paths);
                *count = sum;
                overflowed |= carry;
            }
        }
    }
    if overflowed {
        return Err(Error::ArithmeticOverflow {
            operation: "descendant_path_counts",
            dtype: "int64",
        });
    }
    Ok(counts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kinship::KinshipPedigree;
    use crate::topology::structural_depth;
    use std::collections::BTreeSet;

    fn run<T>(
        mother: &[i32],
        father: &[i32],
        kernel: fn(ParentColumns<'_>) -> Result<T, Error>,
    ) -> Result<T, Error> {
        let depth = structural_depth(mother, father);
        kernel(ParentColumns::try_new(mother, father, Some(&depth)).unwrap())
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
                    position: 0,
                    ..
                })
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
        let cols = ParentColumns::try_new(&c.mother, &c.father, Some(&depth)).unwrap();
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
