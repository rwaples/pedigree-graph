//! The per-generation Ne prerequisites that sweep the parent edges:
//! Maignel's equivalent complete generations, the per-cohort mean
//! founder-genome contributions, and the per-cohort kinship sums.
//!
//! EqG sweeps parents first (the graph rows when they already are, else the
//! stable depth-major order of [`super::depth_order`]); the founder means
//! and the kinship sums walk depth by depth, graph rows ascending within a
//! depth.  EqG and the founder means evaluate each value by the float64
//! expression the 0.9.3 NumPy and Numba code used, in the same order.

use super::depth_order::{DepthOrder, ParentsFirst};
use super::inbreeding::{genome_node, genome_walk};
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

/// Per bucket, the kinship summed over unordered same-bucket pairs of
/// distinct rows that are not MZ co-twins, as float64.
///
/// `labels` is each graph row's bucket in `0..n_buckets`, or `n_buckets`
/// for a row in none.
///
/// No kinship is stored.  Over the genome-node pedigree `A = 2φ` factors as
/// `T D Tᵀ`, with `D` the Mendelian sampling variances the inbreeding walk
/// leaves, so a bucket whose genome counts are `c` has `cᵀ A c = Σ D_i y_i²`
/// for `y = Tᵀ c`.  `y` is one backward sweep: from the deepest member's
/// depth down to 0 every genome node adds `D y²` and sends `y / 2` to each
/// parent node.  Taking away the diagonal `Σ c_x² (1 + F_x)` leaves the
/// ordered pair sum of `A`, a quarter of which is the kinship sum.  Memory
/// is a few float64 per row however related the pedigree is; each bucket
/// costs a pass over the rows at or above its deepest member.
///
/// The value is the exact pedigree kinship to float64 rounding, not a sum
/// of ADR 0009's pinned float32 recurrence.  The two differ by at most that
/// recurrence's rounding, and agree to summation order wherever float32
/// holds every kinship exactly.  A rounding residue below zero is clamped,
/// since no kinship is negative.
///
/// # Errors
///
/// [`Error::LengthMismatch`] when `labels` is not one per row,
/// [`Error::ValueOutOfRange`] on `labels`, on `n_buckets` past int32 or on
/// `depth` as [`inbreeding()`] reports it, and [`Error::AllocationFailed`]
/// for any buffer.
pub fn generation_kinship_sums(
    ped: KinshipPedigree<'_>,
    labels: &[i32],
    n_buckets: usize,
) -> Result<Vec<f64>, Error> {
    const SUMS: Family = Family::KinshipSums;
    let n = ped.len();
    check_column_length("labels", labels.len(), n)?;
    if n_buckets > i32::MAX as usize {
        return Err(Error::ValueOutOfRange {
            field: "n_buckets",
            position: 0,
            value: n_buckets as i64,
            minimum: 0,
            maximum: i64::from(i32::MAX),
        });
    }
    if let Some(position) = labels.iter().position(|&b| b < 0 || b as usize > n_buckets) {
        return Err(Error::ValueOutOfRange {
            field: "labels",
            position,
            value: i64::from(labels[position]),
            minimum: 0,
            maximum: n_buckets as i64,
        });
    }
    let walk = genome_walk(ped)?;
    let (twin, depth) = (ped.twin(), ped.depth());

    // Rows grouped by bucket, ascending within each, so a bucket seeds from
    // its own rows; the walk has put every co-twin at its node's depth.
    let mut starts = alloc::filled(0usize, n_buckets + 2, SUMS, "uint64")?;
    let mut deepest = alloc::filled(0usize, n_buckets + 1, SUMS, "uint64")?;
    for (&b, &d) in labels.iter().zip(depth) {
        starts[b as usize + 1] += 1;
        deepest[b as usize] = deepest[b as usize].max(d as usize);
    }
    for b in 1..starts.len() {
        starts[b] += starts[b - 1];
    }
    let mut members = alloc::filled(0u32, n, SUMS, "uint32")?;
    let mut cursor = alloc::cloned(&starts, SUMS, "uint64")?;
    for (row, &b) in labels.iter().enumerate() {
        members[cursor[b as usize]] = row as u32;
        cursor[b as usize] += 1;
    }
    let mut sums = alloc::filled(0.0f64, n_buckets, SUMS, "float64")?;
    let mut y = alloc::filled(0.0f64, n, SUMS, "float64")?;

    for (b, sum) in sums.iter_mut().enumerate() {
        let rows = &members[starts[b]..starts[b + 1]];
        if rows.is_empty() {
            continue;
        }
        let mut diagonal = 0.0f64;
        for &row in rows {
            let x = genome_node(twin, row as usize);
            // (c + 1)² − c², so the total is Σ c_x² (1 + F_x).
            diagonal += (2.0 * y[x] + 1.0) * (1.0 + walk.f[x]);
            y[x] += 1.0;
        }
        let mut quadratic = 0.0f64;
        for d in (0..=deepest[b]).rev() {
            for &row in walk.sweep.rows_at(d) {
                let i = row as usize;
                let y_i = y[i];
                if y_i == 0.0 {
                    continue;
                }
                quadratic += walk.d_var[i] * y_i * y_i;
                for p in [walk.mother[i], walk.father[i]] {
                    if p >= 0 {
                        y[p as usize] += 0.5 * y_i;
                    }
                }
                y[i] = 0.0;
            }
        }
        *sum = (0.25 * (quadratic - diagonal)).max(0.0);
    }
    Ok(sums)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kinship::inbreeding;
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

    /// The pair sums of the pinned matrix, in float64, per bucket.
    fn matrix_sums(ped: KinshipPedigree<'_>, labels: &[i32], n_buckets: usize) -> Vec<f64> {
        let csc = super::super::kinship_csc(ped).unwrap();
        let mut want = vec![0.0f64; n_buckets];
        for (column, window) in csc.indptr.windows(2).enumerate() {
            let (start, end) = (window[0] as usize, window[1] as usize);
            for (&row, &value) in csc.indices[start..end].iter().zip(&csc.data[start..end]) {
                let (row, b) = (row as usize, labels[column]);
                if row < column
                    && labels[row] == b
                    && (b as usize) < n_buckets
                    && ped.twin()[row] != column as i32
                {
                    want[b as usize] += f64::from(value);
                }
            }
        }
        want
    }

    #[test]
    fn kinship_sums_match_the_pinned_matrix_on_an_inbred_pedigree() {
        let c = crate::relationships::testing::random_pedigree(600, 11);
        let depth = structural_depth(&c.mother, &c.father);
        let ped = KinshipPedigree::try_new(&c.mother, &c.father, &c.twin, &depth).unwrap();
        assert!(c.twin.iter().any(|&t| t >= 0));
        assert!(inbreeding(ped).unwrap().iter().any(|&f| f > 0.0));
        // Buckets by depth, every third row in none.
        let n_buckets = *depth.iter().max().unwrap() as usize + 1;
        let labels: Vec<i32> = depth
            .iter()
            .enumerate()
            .map(|(i, &d)| if i % 3 == 0 { n_buckets as i32 } else { d })
            .collect();
        let got = generation_kinship_sums(ped, &labels, n_buckets).unwrap();
        let want = matrix_sums(ped, &labels, n_buckets);
        for (g, w) in got.iter().zip(&want) {
            assert!((g - w).abs() <= 1e-6 * w.max(1.0), "{got:?} vs {want:?}");
        }
    }

    #[test]
    fn kinship_sums_count_selfing_and_both_co_twins_of_a_bucket() {
        // 0, 1 found; 2 = 0 selfed; 3, 4 MZ of (2, 1); 5 = (3, 4)'s genome
        // mated back to 0.  One bucket holds every row.
        let (m, f) = ([-1, -1, 0, 2, 2, 3], [-1, -1, 0, 1, 1, 0]);
        let twin = [-1, -1, -1, 4, 3, -1];
        let depth = structural_depth(&m, &f);
        let ped = KinshipPedigree::try_new(&m, &f, &twin, &depth).unwrap();
        let labels = [0; 6];
        let got = generation_kinship_sums(ped, &labels, 1).unwrap();
        assert_eq!(got, matrix_sums(ped, &labels, 1));
    }

    #[test]
    fn co_twins_at_different_depths_are_rejected() {
        // 0, 1 found; 2, 3 MZ of (0, 1), with 2 raised a depth above 3.
        let (m, f) = ([-1, -1, 0, 0], [-1, -1, 1, 1]);
        let twin = [-1, -1, 3, 2];
        let depth = [0, 0, 2, 1];
        let ped = KinshipPedigree::try_new(&m, &f, &twin, &depth).unwrap();
        let rejected = |err| {
            matches!(
                err,
                Error::ValueOutOfRange {
                    field: "depth",
                    position: 2,
                    value: 2,
                    minimum: 1,
                    maximum: 1,
                }
            )
        };
        assert!(rejected(inbreeding(ped).unwrap_err()));
        assert!(rejected(
            generation_kinship_sums(ped, &[0, 0, 1, 1], 2).unwrap_err()
        ));
    }

    #[test]
    fn kinship_sums_take_no_bucket_and_check_their_labels() {
        let (m, f) = ([-1, -1, 0], [-1, -1, 1]);
        with_ped(&m, &f, |p| {
            assert_eq!(
                generation_kinship_sums(p, &[0, 0, 0], 0).unwrap(),
                Vec::<f64>::new()
            );
            assert_eq!(
                generation_kinship_sums(p, &[0, 0, 1], 1).unwrap(),
                vec![0.0]
            );
            assert_eq!(
                generation_kinship_sums(p, &[0, 1, 0], 1).unwrap(),
                vec![0.25]
            );
            assert!(matches!(
                generation_kinship_sums(p, &[0, 0], 1),
                Err(Error::LengthMismatch {
                    field: "labels",
                    ..
                })
            ));
            assert!(matches!(
                generation_kinship_sums(p, &[0, 0, 2], 1),
                Err(Error::ValueOutOfRange {
                    field: "labels",
                    position: 2,
                    ..
                })
            ));
            assert!(matches!(
                generation_kinship_sums(p, &[0, -1, 0], 1),
                Err(Error::ValueOutOfRange {
                    field: "labels",
                    position: 1,
                    ..
                })
            ));
        });
    }
}
