//! Per-person relationship burden without materialising pair blocks.

use super::{
    task_ranges, Category, CategorySet, Engine, MaxDegree, Pedigree, WorkspacePool, N_CATEGORIES,
};
use crate::alloc::{self, Family};
use crate::error::Error;
use rayon::prelude::*;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

/// Counts in registry order, per-person counts for degrees 1..5 in row-major
/// order, and related-pair counts where both endpoints have the same depth.
pub struct Burden {
    pub categories: [u64; N_CATEGORIES],
    pub per_person: Vec<u32>,
    pub same_depth: Vec<u64>,
}

/// Classify each unordered pair once and add its contribution directly to
/// bounded arrays.  MZ pairs count in `categories` and `same_depth` but not
/// `per_person`, matching the degree-1..5 pedsum burden report.
pub fn relationship_burden(ped: &Pedigree<'_>, depth: &[i32]) -> Result<Burden, Error> {
    assert_eq!(ped.len(), depth.len());
    let n = ped.len();
    let n_depths = depth.iter().copied().max().map_or(0, |d| d as usize + 1);
    let mut cells = alloc::with_capacity(n * 5, Family::RowSet, "uint32")?;
    cells.extend((0..n * 5).map(|_| AtomicU32::new(0)));
    let mut same_depth = alloc::with_capacity(n_depths, Family::RowSet, "uint64")?;
    same_depth.extend((0..n_depths).map(|_| AtomicU64::new(0)));
    let categories: [AtomicU64; N_CATEGORIES] = std::array::from_fn(|_| AtomicU64::new(0));

    let engine = Engine::new(ped, MaxDegree::MAX)?;
    let pool = WorkspacePool::new(n, true);
    let requested = CategorySet::up_to_degree(MaxDegree::MAX.get());
    task_ranges(n)
        .into_par_iter()
        .try_for_each(|(start, end)| {
            let mut ws = pool.take()?;
            let mut result = Ok(());
            for row in start..end {
                result = engine.emit_row(row, &requested, None, &mut ws, |cat: Category, a, b| {
                    categories[cat.index()].fetch_add(1, Ordering::Relaxed);
                    let degree = cat.degree();
                    if degree > 0 {
                        let column = (degree - 1) as usize;
                        cells[a as usize * 5 + column].fetch_add(1, Ordering::Relaxed);
                        cells[b as usize * 5 + column].fetch_add(1, Ordering::Relaxed);
                    }
                    let da = depth[a as usize];
                    if da == depth[b as usize] {
                        same_depth[da as usize].fetch_add(1, Ordering::Relaxed);
                    }
                    Ok(())
                });
                if result.is_err() {
                    break;
                }
            }
            pool.give(ws);
            result
        })?;

    Ok(Burden {
        categories: categories.map(AtomicU64::into_inner),
        per_person: cells.into_iter().map(AtomicU32::into_inner).collect(),
        same_depth: same_depth.into_iter().map(AtomicU64::into_inner).collect(),
    })
}
