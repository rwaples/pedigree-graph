//! The stable depth-major sweep order shared by the kinship DP and the
//! inbreeding, lineage and generation sweeps.
//!
//! Every kernel here needs parents visited before children and nothing
//! else from the order, so each walks [`DepthOrder::order`] while indexing
//! graph rows directly: no parent array is remapped and no result is
//! scattered back.  Within a depth the rows stay in ascending graph row,
//! the order [`crate::topology::depth_major_order`] and a NumPy
//! `flatnonzero(depth == d)` both give.

use super::pairwise::KinshipPedigree;
use crate::alloc::{self, Family};
use crate::error::Error;

/// Graph rows in stable depth-major order, bucketed by depth.
pub(crate) struct DepthOrder {
    /// Graph row at each depth-major position.
    pub(crate) order: Vec<u32>,
    /// Rows at depth `d` are `order[starts[d]..starts[d + 1]]`.
    pub(crate) starts: Vec<usize>,
}

impl DepthOrder {
    /// Check `depth` is structural and sort the rows by it, reserving
    /// through `family`.
    ///
    /// # Errors
    ///
    /// [`Error::ValueOutOfRange`] on `depth` when a row's depth is negative
    /// or not above both parents', before anything is allocated, and
    /// [`Error::AllocationFailed`] for either array.
    pub(crate) fn build(ped: &KinshipPedigree<'_>, family: Family) -> Result<DepthOrder, Error> {
        check_structural(ped)?;
        let depth = ped.depth();
        let max_depth = depth.iter().copied().max().unwrap_or(0) as usize;
        let mut starts = alloc::filled(0usize, max_depth + 2, family, "uint64")?;
        for &d in depth {
            starts[d as usize + 1] += 1;
        }
        for d in 1..starts.len() {
            starts[d] += starts[d - 1];
        }
        let mut order = alloc::filled(0u32, depth.len(), family, "uint32")?;
        let mut cursor = alloc::cloned(&starts, family, "uint64")?;
        for (row, &d) in depth.iter().enumerate() {
            order[cursor[d as usize]] = row as u32;
            cursor[d as usize] += 1;
        }
        Ok(DepthOrder { order, starts })
    }

    pub(crate) fn max_depth(&self) -> usize {
        self.starts.len() - 2
    }

    /// The graph rows at depth `d`, ascending.
    pub(crate) fn rows_at(&self, d: usize) -> &[u32] {
        &self.order[self.starts[d]..self.starts[d + 1]]
    }
}

/// Every depth is non-negative and strictly above both parents'.
fn check_structural(ped: &KinshipPedigree<'_>) -> Result<(), Error> {
    let depth = ped.depth();
    for (row, &d) in depth.iter().enumerate() {
        let floor = [ped.mother()[row], ped.father()[row]]
            .into_iter()
            .filter(|&p| p >= 0)
            .map(|p| i64::from(depth[p as usize]) + 1)
            .max()
            .unwrap_or(0);
        if i64::from(d) < floor {
            return Err(Error::ValueOutOfRange {
                field: "depth",
                position: row,
                value: i64::from(d),
                minimum: floor,
                maximum: i64::from(i32::MAX),
            });
        }
    }
    Ok(())
}
