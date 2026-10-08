//! Per-person relationship burden without materialising pair blocks.

use super::{
    task_ranges, walk_task, Category, CategorySet, Engine, MaxDegree, Pedigree, Progress,
    WorkspacePool, N_CATEGORIES,
};
use crate::alloc::{self, Family};
use crate::error::Error;
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
/// # Errors
///
/// [`Error::AllocationFailed`] from the counters, the engine or a
/// workspace; [`Error::Cancelled`] once `progress` is cancelled.
pub fn relationship_burden(
    ped: &Pedigree<'_>,
    depth: &[i32],
    progress: &Progress,
) -> Result<Burden, Error> {
    assert_eq!(ped.len(), depth.len());
    let n = ped.len();
    let n_depths = depth.iter().copied().max().map_or(0, |d| d as usize + 1);
    let mut cells = alloc::with_capacity(n * DEGREES, Family::RowSet)?;
    cells.extend((0..n * DEGREES).map(|_| AtomicU32::new(0)));
    let mut same_depth = alloc::with_capacity(n_depths, Family::RowSet)?;
    same_depth.extend((0..n_depths).map(|_| AtomicU64::new(0)));
    let categories: [AtomicU64; N_CATEGORIES] = std::array::from_fn(|_| AtomicU64::new(0));

    let engine = Engine::new(ped, MaxDegree::MAX)?;
    let pool = WorkspacePool::for_pairs(n);
    let requested = CategorySet::up_to_degree(MaxDegree::MAX.get());
    progress.walk(n)?;
    task_ranges(n).into_par_iter().try_for_each(|range| {
        walk_task(
            &pool,
            progress,
            range,
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
