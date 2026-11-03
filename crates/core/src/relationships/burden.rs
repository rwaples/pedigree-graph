//! Per-person relationship burden without materialising pair blocks.

use super::check_column_length;
use super::{
    task_ranges, walk_task, Category, CategorySet, Engine, MaxDegree, Pedigree, Progress,
    WorkspacePool, N_CATEGORIES,
};
use crate::alloc::{self, Family};
use crate::error::Error;
use crate::kinship::depth_order::check_structural;
use rayon::prelude::*;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

/// Per-person count columns, one per degree `1..=MaxDegree::MAX`.
pub const DEGREES: usize = MaxDegree::MAX.get() as usize;

/// Counts in registry order, per-person counts for degrees 1..5 in row-major
/// order ([`DEGREES`] per row), and related-pair counts where both endpoints
/// have the same depth.
pub struct Burden {
    pub categories: [u64; N_CATEGORIES],
    pub per_person: Vec<u32>,
    pub same_depth: Vec<u64>,
}

/// Classify each unordered pair once and add its contribution directly to
/// bounded arrays.  MZ pairs count in `categories` and `same_depth` but not
/// `per_person`, matching the degree-1..5 pedsum burden report.
///
/// `depth` buckets the same-depth counts: one per row, each below the row
/// count and strictly above both parents'.
///
/// # Errors
///
/// [`Error::LengthMismatch`] on `depth` when it is not one per row and
/// [`Error::ValueOutOfRange`] at the first depth outside that range;
/// [`Error::AllocationFailed`] from the counters, the engine or a
/// workspace; [`Error::Cancelled`] once `progress` is cancelled.
pub fn relationship_burden(
    ped: &Pedigree<'_>,
    depth: &[i32],
    progress: &Progress,
) -> Result<Burden, Error> {
    let n = ped.len();
    check_column_length("depth", depth.len(), n)?;
    check_structural(ped.mother, ped.father, depth)?;
    let n_depths = depth.iter().copied().max().map_or(0, |d| d as usize + 1);
    let mut cells = alloc::with_capacity(n * DEGREES, Family::RowSet)?;
    cells.extend((0..n * DEGREES).map(|_| AtomicU32::new(0)));
    let mut same_depth = alloc::with_capacity(n_depths, Family::RowSet)?;
    same_depth.extend((0..n_depths).map(|_| AtomicU64::new(0)));
    let categories: [AtomicU64; N_CATEGORIES] = std::array::from_fn(|_| AtomicU64::new(0));

    let engine = Engine::new(ped, MaxDegree::MAX)?;
    let pool = WorkspacePool::for_pairs(n);
    let requested = CategorySet::up_to_degree(MaxDegree::MAX.get());
    let order = engine.family_order();
    progress.walk(n)?;
    task_ranges(n).into_par_iter().try_for_each(|range| {
        walk_task(
            &pool,
            progress,
            range,
            order.as_deref(),
            |_| false,
            |row, ws| {
                engine.emit_row(row, &requested, None, ws, |cat: Category, a, b| {
                    categories[cat.index()].fetch_add(1, Ordering::Relaxed);
                    let degree = cat.degree();
                    if degree > 0 {
                        let column = (degree - 1) as usize;
                        cells[a as usize * DEGREES + column].fetch_add(1, Ordering::Relaxed);
                        cells[b as usize * DEGREES + column].fetch_add(1, Ordering::Relaxed);
                    }
                    let da = depth[a as usize];
                    if da == depth[b as usize] {
                        same_depth[da as usize].fetch_add(1, Ordering::Relaxed);
                    }
                    Ok(())
                })
            },
        )
    })?;
    progress.finish()?;

    Ok(Burden {
        categories: categories.map(AtomicU64::into_inner),
        per_person: cells.into_iter().map(AtomicU32::into_inner).collect(),
        same_depth: same_depth.into_iter().map(AtomicU64::into_inner).collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::relationships::testing::pedigree;

    fn burden_err(depth: &[i32]) -> Error {
        let cols = pedigree(&[(-1, -1), (-1, -1), (0, 1)], &[]);
        let ped = cols.try_borrow().unwrap();
        match relationship_burden(&ped, depth, &Progress::default()) {
            Ok(_) => panic!("depth {depth:?} accepted"),
            Err(err) => err,
        }
    }

    #[test]
    fn a_short_depth_is_a_length_mismatch_not_a_panic() {
        assert!(matches!(
            burden_err(&[0, 0]),
            Error::LengthMismatch {
                field: "depth",
                expected_length: 3,
                actual_length: 2
            }
        ));
    }

    #[test]
    fn a_depth_outside_the_rows_or_at_a_parents_is_refused() {
        for (depth, value, minimum, maximum) in [
            ([0, 0, 3], 3, 1, 2),
            ([0, 0, 0], 0, 1, 2),
            ([-1, 0, 1], -1, 0, 2),
        ] {
            let err = burden_err(&depth);
            assert!(
                matches!(err, Error::ValueOutOfRange { field: "depth", value: v, minimum: lo, maximum: hi, .. }
                    if v == value && lo == minimum && hi == maximum),
                "{err:?}"
            );
        }
    }
}
