//! The two per-generation Ne prerequisites that sweep the parent edges:
//! Maignel's equivalent complete generations and the per-cohort mean
//! founder-genome contributions.
//!
//! EqG sweeps parents first (the graph rows when they already are, else the
//! stable depth-major order of [`super::depth_order`]); the founder means
//! walk depth by depth, graph rows ascending within a depth.  Both evaluate
//! each value by the float64 expression the 0.9.3 NumPy and Numba code
//! used, in the same order.

use super::depth_order::{DepthOrder, ParentsFirst};
use super::pairwise::KinshipPedigree;
use crate::alloc::{self, Family};
use crate::error::Error;
use crate::lineage::ParentColumns;
use crate::relationships::check_column_length;

/// Equivalent complete generations (Maignel, Boichard and Verrier 1996) of
/// every graph row: `sum over known parents p of (1/2 + eqg[p] / 2)`, so a
/// founder is 0 and a row with two founder parents is 1.
///
/// # Errors
///
/// When the rows need sorting, [`Error::LengthMismatch`] or
/// [`Error::ValueOutOfRange`] on `depth` as
/// [`crate::lineage::distinct_ancestor_counts`] reports them;
/// [`Error::AllocationFailed`] for any buffer.
pub fn equivalent_generations(ped: ParentColumns<'_>) -> Result<Vec<f64>, Error> {
    match ped.parents_first(Family::LineageOutput)? {
        ParentsFirst::Rows => generations_in(&ped, 0..ped.len() as u32),
        ParentsFirst::DepthMajor(sweep) => generations_in(&ped, sweep.order.iter().copied()),
    }
}

fn generations_in(
    ped: &ParentColumns<'_>,
    rows: impl Iterator<Item = u32>,
) -> Result<Vec<f64>, Error> {
    let (mother, father) = (ped.mother(), ped.father());
    let mut eqg = alloc::filled(0.0f64, ped.len(), Family::LineageOutput, "float64")?;
    for row in rows {
        let i = row as usize;
        let mut v = 0.0f64;
        for p in [mother[i], father[i]] {
            if p >= 0 {
                v += 0.5 + 0.5 * eqg[p as usize];
            }
        }
        eqg[i] = v;
    }
    Ok(eqg)
}

/// Per cohort, the mean over its rows of each founder genome's expected
/// contribution, row-major `(cohort, genome)` as float64.
///
/// `cohort` is each graph row's cohort in `0..n_cohorts`, or `n_cohorts`
/// for a row in none.  `founder_column` is the genome column in
/// `0..n_genomes` a founder row seeds, or `-1`.
///
/// For each cohort the adjoint of the Mendelian recursion runs backwards:
/// `u` starts at `1 / size` on the cohort's rows, then from the deepest
/// member's depth down to 1 every row of the depth sends `u / 2` to each
/// parent, mothers first then fathers, rows ascending, and is cleared.
/// What reaches the founder rows is summed per genome in ascending row.
///
/// # Errors
///
/// [`Error::LengthMismatch`] when `cohort` or `founder_column` is not one
/// per row, [`Error::ValueOutOfRange`] on either of them, on `n_cohorts`
/// past int32 or on `depth`,
/// [`Error::ArithmeticOverflow`] when `n_cohorts * n_genomes` exceeds the
/// address space, and [`Error::AllocationFailed`] for any buffer.
pub fn founder_contribution_means(
    ped: KinshipPedigree<'_>,
    cohort: &[i32],
    n_cohorts: usize,
    founder_column: &[i64],
    n_genomes: usize,
) -> Result<Vec<f64>, Error> {
    const MEANS: Family = Family::FounderMeans;
    let n = ped.len();
    check_column_length("cohort", cohort.len(), n)?;
    check_column_length("founder_column", founder_column.len(), n)?;
    if n_cohorts > i32::MAX as usize {
        return Err(Error::ValueOutOfRange {
            field: "n_cohorts",
            position: 0,
            value: n_cohorts as i64,
            minimum: 0,
            maximum: i64::from(i32::MAX),
        });
    }
    if let Some(position) = cohort.iter().position(|&b| b < 0 || b as usize > n_cohorts) {
        return Err(Error::ValueOutOfRange {
            field: "cohort",
            position,
            value: i64::from(cohort[position]),
            minimum: 0,
            maximum: n_cohorts as i64,
        });
    }
    if let Some(position) = founder_column
        .iter()
        .position(|&g| g < -1 || g >= n_genomes as i64)
    {
        return Err(Error::ValueOutOfRange {
            field: "founder_column",
            position,
            value: founder_column[position],
            minimum: -1,
            maximum: n_genomes as i64 - 1,
        });
    }
    let len = n_cohorts
        .checked_mul(n_genomes)
        .ok_or(Error::ArithmeticOverflow {
            operation: "founder_means",
            dtype: "float64",
        })?;
    let sweep = DepthOrder::build(ped.mother(), ped.father(), ped.depth(), MEANS)?;
    let (mother, father, depth) = (ped.mother(), ped.father(), ped.depth());

    let mut size = alloc::filled(0usize, n_cohorts + 1, MEANS, "uint64")?;
    let mut deepest = alloc::filled(0usize, n_cohorts + 1, MEANS, "uint64")?;
    for (&b, &d) in cohort.iter().zip(depth) {
        size[b as usize] += 1;
        deepest[b as usize] = deepest[b as usize].max(d as usize);
    }
    let mut means = alloc::filled(0.0f64, len, MEANS, "float64")?;
    let mut u = alloc::filled(0.0f64, n, MEANS, "float64")?;

    for (b, row_means) in means.chunks_exact_mut(n_genomes.max(1)).enumerate() {
        if size[b] == 0 {
            continue;
        }
        u.fill(0.0);
        let share = 1.0 / size[b] as f64;
        for (u_i, &c) in u.iter_mut().zip(cohort) {
            if c as usize == b {
                *u_i = share;
            }
        }
        for d in (1..=deepest[b]).rev() {
            let rows = sweep.rows_at(d);
            for parent in [mother, father] {
                for &row in rows {
                    let p = parent[row as usize];
                    if p >= 0 {
                        u[p as usize] += 0.5 * u[row as usize];
                    }
                }
            }
            for &row in rows {
                u[row as usize] = 0.0;
            }
        }
        for (&g, &u_i) in founder_column.iter().zip(&u) {
            if g >= 0 {
                row_means[g as usize] += u_i;
            }
        }
    }
    Ok(means)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology::structural_depth;

    fn with_ped<T>(mother: &[i32], father: &[i32], f: impl FnOnce(KinshipPedigree<'_>) -> T) -> T {
        let twin = vec![-1; mother.len()];
        let depth = structural_depth(mother, father);
        f(KinshipPedigree::try_new(mother, father, &twin, &depth).unwrap())
    }

    #[test]
    fn equivalent_generations_count_known_generations() {
        // 0..3 found; 4 = (0, 1), 5 = (2, 3), 6 = (4, 5), 7 = (4, -1).
        let (m, f) = ([-1, -1, -1, -1, 0, 2, 4, 4], [-1, -1, -1, -1, 1, 3, 5, -1]);
        let eqg = equivalent_generations(ParentColumns::try_new(&m, &f, None).unwrap()).unwrap();
        assert_eq!(eqg, vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 1.0]);
    }

    #[test]
    fn founder_means_split_each_genome_by_mendelian_halves() {
        // Founders 0, 1, 2 seed genomes 0, 1, 2; 3 = (0, 1); 4 = (3, 2).
        // Cohort 0 is {0, 1, 2}, cohort 1 is {3, 4}.
        let (m, f) = ([-1, -1, -1, 0, 3], [-1, -1, -1, 1, 2]);
        let means = with_ped(&m, &f, |p| {
            founder_contribution_means(p, &[0, 0, 0, 1, 1], 2, &[0, 1, 2, -1, -1], 3).unwrap()
        });
        let third = 1.0 / 3.0;
        assert_eq!(&means[..3], &[third, third, third]);
        // Row 3 is half 0, half 1; row 4 is a quarter each of 0 and 1 and half 2.
        assert_eq!(&means[3..], &[0.375, 0.375, 0.25]);
    }

    #[test]
    fn an_unlabelled_row_joins_no_cohort() {
        let (m, f) = ([-1, -1, 0], [-1, -1, 1]);
        let means = with_ped(&m, &f, |p| {
            founder_contribution_means(p, &[1, 1, 0], 1, &[0, 1, -1], 2).unwrap()
        });
        assert_eq!(means, vec![0.5, 0.5]);
    }

    #[test]
    fn cohort_and_founder_column_ranges_are_checked() {
        let (m, f) = ([-1, -1, 0], [-1, -1, 1]);
        with_ped(&m, &f, |p| {
            assert!(matches!(
                founder_contribution_means(p, &[0, 2, 0], 1, &[0, 1, -1], 2),
                Err(Error::ValueOutOfRange {
                    field: "cohort",
                    position: 1,
                    ..
                })
            ));
            assert!(matches!(
                founder_contribution_means(p, &[0, 0, 0], 1, &[0, 2, -1], 2),
                Err(Error::ValueOutOfRange {
                    field: "founder_column",
                    position: 1,
                    ..
                })
            ));
            assert!(matches!(
                founder_contribution_means(p, &[0, 0], 1, &[0, 1, -1], 2),
                Err(Error::LengthMismatch { .. })
            ));
            assert!(matches!(
                founder_contribution_means(
                    p,
                    &[0, 0, 0],
                    i32::MAX as usize,
                    &[0, 1, -1],
                    usize::MAX / 2
                ),
                Err(Error::ArithmeticOverflow {
                    operation: "founder_means",
                    ..
                })
            ));
        });
    }
}
