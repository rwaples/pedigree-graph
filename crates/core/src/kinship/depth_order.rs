//! The stable depth-major sweep order shared by the kinship DP and the
//! inbreeding, lineage and generation sweeps.
//!
//! Every kernel here needs parents visited before children and nothing
//! else from the order, so each walks [`DepthOrder::order`] while indexing
//! graph rows directly: no parent array is remapped and no result is
//! scattered back.  Within a depth the rows stay in ascending graph row,
//! the order [`crate::topology::depth_major_order`] and a NumPy
//! `flatnonzero(depth == d)` both give.

use crate::alloc::{self, Family};
use crate::error::Error;
use crate::relationships::{check_column_length, check_row_range};

/// The order a parents-first sweep visits the graph rows in, for the kernels
/// that need nothing else from it.
///
/// Neither order costs a pass of its own: a depth is checked structural,
/// with the parent rows, where it is sorted, and the graph order is checked
/// by the sweep as it goes ([`admit`]).
pub(crate) enum ParentsFirst {
    /// The graph rows as they stand, unchecked.  The sweep skips a parent row
    /// that does not precede its child and says so, and the caller then
    /// discards it for [`ParentsFirst::unordered`].
    Rows {
        /// Why the depth could not be sorted, when there was one.
        depth_error: Option<Error>,
    },
    /// The stable depth-major order, `depth` checked structural and every
    /// parent row `-1` or a row.
    DepthMajor(DepthOrder),
}

impl ParentsFirst {
    /// The depth-major order on `depth` when it sorts, else the graph rows.
    /// Parents-first rows ignore a depth, so a depth that does not sort is
    /// reported only when the rows turn out not to be.
    pub(crate) fn of(
        mother: &[i32],
        father: &[i32],
        depth: Option<&[i32]>,
        family: Family,
    ) -> ParentsFirst {
        let sorted = depth.map(|depth| {
            check_column_length("depth", depth.len(), mother.len())
                .and_then(|()| DepthOrder::build(mother, father, depth, family))
        });
        match sorted {
            Some(Ok(order)) => ParentsFirst::DepthMajor(order),
            Some(Err(error)) => ParentsFirst::Rows {
                depth_error: Some(error),
            },
            None => ParentsFirst::Rows { depth_error: None },
        }
    }

    /// What a sweep in graph order reports when a parent row did not precede
    /// its child: the first `mother`, then `father`, row outside `-1..n`,
    /// then the depth's error, or a missing depth.
    pub(crate) fn unordered(mother: &[i32], father: &[i32], depth_error: Option<Error>) -> Error {
        match check_parent_rows(mother, father) {
            Err(error) => error,
            Ok(()) => depth_error.unwrap_or(Error::LengthMismatch {
                field: "depth",
                expected_length: mother.len(),
                actual_length: 0,
            }),
        }
    }
}

/// The rows a sweep visiting `row` may read: every row in a depth-major
/// sweep (`CHECKED`), whose rows are checked already, and in graph order
/// only those before `row`.
///
/// A sweep looks a parent `p` up as `readable(..).get(p as usize)`, which
/// puts `-1` and every other negative row past the end: one compare, the
/// bounds check the lookup always made, both bounds the row and orders it.
/// A parent it cannot read goes to [`unread`].
#[inline(always)]
pub(crate) fn readable<const CHECKED: bool, T>(values: &[T], row: u32) -> &[T] {
    if CHECKED {
        values
    } else {
        &values[..row as usize]
    }
}

/// [`readable`], mutably.
#[inline(always)]
pub(crate) fn readable_mut<const CHECKED: bool, T>(values: &mut [T], row: u32) -> &mut [T] {
    if CHECKED {
        values
    } else {
        &mut values[..row as usize]
    }
}

/// A parent `p` its sweep could not read ([`readable`]): missing when
/// `p == -1`, else a later row or no row, so the rows are not parents-first
/// and `ordered` clears; the caller then discards the sweep.
#[inline(always)]
pub(crate) fn unread(p: i32, ordered: &mut bool) {
    *ordered &= p == -1;
}

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
    /// [`Error::ValueOutOfRange`] on `mother` or `father` at the first row
    /// outside `-1..n`, then on `depth` when a row's depth is negative,
    /// above `n - 1` or not above both parents', before anything is
    /// allocated, and
    /// [`Error::AllocationFailed`] for either array.
    pub(crate) fn build(
        mother: &[i32],
        father: &[i32],
        depth: &[i32],
        family: Family,
    ) -> Result<DepthOrder, Error> {
        check_structural(mother, father, depth)?;
        let max_depth = depth.iter().copied().max().unwrap_or(0) as usize;
        let Buckets { order, starts } = Buckets::by(depth, max_depth + 1, family)?;
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

/// Rows grouped by an integer key, a stable counting sort: the rows with key
/// `k` are [`Buckets::bucket`]`(k)`, ascending.
pub(crate) struct Buckets {
    /// Row at each key-major position.
    pub(crate) order: Vec<u32>,
    /// Rows with key `k` are `order[starts[k]..starts[k + 1]]`.
    pub(crate) starts: Vec<usize>,
}

impl Buckets {
    /// Sort the rows `0..keys.len()` by `keys`, every key in `0..n_buckets`,
    /// reserving through `family`.
    ///
    /// # Errors
    ///
    /// [`Error::AllocationFailed`] for either array or the cursor.
    pub(crate) fn by(keys: &[i32], n_buckets: usize, family: Family) -> Result<Buckets, Error> {
        let mut starts = alloc::filled(0usize, n_buckets + 1, family)?;
        for &k in keys {
            starts[k as usize + 1] += 1;
        }
        for k in 1..starts.len() {
            starts[k] += starts[k - 1];
        }
        let mut order = alloc::filled(0u32, keys.len(), family)?;
        let mut cursor = alloc::cloned(&starts, family)?;
        for (row, &k) in keys.iter().enumerate() {
            order[cursor[k as usize]] = row as u32;
            cursor[k as usize] += 1;
        }
        Ok(Buckets { order, starts })
    }

    /// The rows with key `k`, ascending.
    pub(crate) fn bucket(&self, k: usize) -> &[u32] {
        &self.order[self.starts[k]..self.starts[k + 1]]
    }
}

/// Every parent row is `-1` or a row of the pedigree, and every depth is
/// non-negative, at most `n - 1`, and strictly above both parents'.  The cap
/// is what any ordering of `n` rows needs, and it keeps a buffer sized by the
/// deepest depth within the pedigree's size.
///
/// The parent rows are checked in the same pass, as the depths are read, so
/// a caller whose rows are already checked pays nothing for it; a range
/// error still wins over a depth error at an earlier row.
///
/// `depth` must have one entry per row.
pub(crate) fn check_structural(mother: &[i32], father: &[i32], depth: &[i32]) -> Result<(), Error> {
    let maximum = depth.len() as i64 - 1;
    for (row, &d) in depth.iter().enumerate() {
        let mut floor = 0i64;
        for (field, p) in [("mother", mother[row]), ("father", father[row])] {
            if p == -1 {
                continue;
            }
            let Some(&parent) = depth.get(p as usize) else {
                // An earlier row's range error, or this one's, is first.
                check_parent_rows(mother, father)?;
                return Err(Error::ValueOutOfRange {
                    field,
                    position: row.into(),
                    value: i64::from(p),
                    minimum: -1,
                    maximum,
                });
            };
            floor = floor.max(i64::from(parent) + 1);
        }
        if i64::from(d) < floor || i64::from(d) > maximum {
            check_parent_rows(mother, father)?;
            return Err(Error::ValueOutOfRange {
                field: "depth",
                position: row.into(),
                value: i64::from(d),
                minimum: floor,
                maximum,
            });
        }
    }
    Ok(())
}

/// The first `mother`, then `father`, row outside `-1..n`.
pub(crate) fn check_parent_rows(mother: &[i32], father: &[i32]) -> Result<(), Error> {
    check_row_range("mother", mother, mother.len())?;
    check_row_range("father", father, mother.len())
}

/// Co-twins are one node, at one depth: each names the other, and they
/// share a depth.  Graph construction refuses a non-reciprocal pair, and
/// structural depth always puts co-twins at one depth, since they share
/// their parents; a raised depth handed to the bindings need not.
///
/// `twin` must already be `-1` or rows of the pedigree.
pub(crate) fn check_twins(twin: &[i32], depth: &[i32]) -> Result<(), Error> {
    for (row, &t) in twin.iter().enumerate() {
        if t < 0 {
            continue;
        }
        let partner = t as usize;
        if twin[partner] != row as i32 {
            return Err(Error::ValueOutOfRange {
                field: "twin",
                position: partner.into(),
                value: i64::from(twin[partner]),
                minimum: row as i64,
                maximum: row as i64,
            });
        }
        if depth[partner] != depth[row] {
            let partner = i64::from(depth[partner]);
            return Err(Error::ValueOutOfRange {
                field: "depth",
                position: row.into(),
                value: i64::from(depth[row]),
                minimum: partner,
                maximum: partner,
            });
        }
    }
    Ok(())
}
