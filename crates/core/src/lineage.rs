//! Per-row ancestor and descendant counts over the parent edges.
//!
//! * [`distinct_ancestor_counts`]: the number of distinct strict ancestors
//!   of each row, an ancestor reached through several paths counted once.
//! * [`descendant_path_counts`]: the number of walks down the pedigree from
//!   each row, its children plus their path counts, which exceeds the
//!   distinct descendant count wherever marriage loops join two paths.
//!
//! Both sweep the stable depth-major order of
//! [`crate::kinship::depth_order`], which puts every parent before its
//! children, and index graph rows directly.  Counts do not depend on row
//! labels, so they are the 0.9.3 Numba kernels' counts exactly.

use crate::alloc::{self, Family};
use crate::error::Error;
use crate::kinship::depth_order::DepthOrder;
use crate::kinship::KinshipPedigree;

const SETS: Family = Family::LineageSets;
const OUTPUT: Family = Family::LineageOutput;

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
/// [`Error::ValueOutOfRange`] on `depth` when a row's depth is negative or
/// not above both parents', and [`Error::AllocationFailed`] for any buffer.
pub fn distinct_ancestor_counts(ped: KinshipPedigree<'_>) -> Result<Vec<i32>, Error> {
    let n = ped.len();
    let sweep = DepthOrder::build(&ped, SETS)?;
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

    for &row in &sweep.order {
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
/// [`Error::ValueOutOfRange`] on `depth` when a row's depth is negative or
/// not above both parents'; and [`Error::AllocationFailed`] for any buffer.
pub fn descendant_path_counts(ped: KinshipPedigree<'_>) -> Result<Vec<i64>, Error> {
    let sweep = DepthOrder::build(&ped, OUTPUT)?;
    let (mother, father) = (ped.mother(), ped.father());
    let mut counts = alloc::filled(0i64, ped.len(), OUTPUT, "int64")?;
    let overflow = || Error::ArithmeticOverflow {
        operation: "descendant_path_counts",
        dtype: "int64",
    };
    for &row in sweep.order.iter().rev() {
        let i = row as usize;
        let paths = counts[i].checked_add(1).ok_or_else(overflow)?;
        for p in [mother[i], father[i]] {
            if p >= 0 {
                let p = p as usize;
                counts[p] = counts[p].checked_add(paths).ok_or_else(overflow)?;
            }
        }
    }
    Ok(counts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology::structural_depth;
    use std::collections::BTreeSet;

    fn run<T>(
        mother: &[i32],
        father: &[i32],
        kernel: fn(KinshipPedigree<'_>) -> Result<T, Error>,
    ) -> Result<T, Error> {
        let twin = vec![-1; mother.len()];
        let depth = structural_depth(mother, father);
        kernel(KinshipPedigree::try_new(mother, father, &twin, &depth).unwrap())
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
    fn a_shallow_depth_is_refused() {
        let (m, f) = ([-1, -1, 0], [-1, -1, 1]);
        let twin = [-1; 3];
        let depth = [0, 0, 0];
        let ped = KinshipPedigree::try_new(&m, &f, &twin, &depth).unwrap();
        for result in [
            distinct_ancestor_counts(ped).map(|_| ()),
            descendant_path_counts(ped).map(|_| ()),
        ] {
            assert!(matches!(
                result,
                Err(Error::ValueOutOfRange {
                    field: "depth",
                    position: 2,
                    ..
                })
            ));
        }
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
            "ancestors" => distinct_ancestor_counts(ped).map(|v| v.len()),
            "descendants" => descendant_path_counts(ped).map(|v| v.len()),
            "generations" => equivalent_generations(ped).map(|v| v.len()),
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
