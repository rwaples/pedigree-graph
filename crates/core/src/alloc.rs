//! Fallible allocation for every buffer the input size reaches (ADR 0007,
//! "Safety and allocation").
//!
//! A refused reservation surfaces as [`Error::AllocationFailed`] naming the
//! [`Family`] that asked, the resulting element count, and the element
//! dtype in NumPy spelling, instead of aborting the process.  Because a real
//! allocator refusal cannot be provoked in a test without harming the host,
//! [`fail_next`] plants one failure for a named family; the next reservation
//! of that family then fails as the allocator would have.  The check is one
//! relaxed atomic load on the hot paths.

use crate::error::Error;
use std::sync::atomic::{AtomicU8, Ordering};

/// The allocation families of the relationship engine, each a distinct
/// place a large buffer is sized by the input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Family {
    /// The parent edge list the CSR is built from.
    ParentEdges,
    /// The parent CSR and its transpose.
    Csr,
    /// Sibling groups by original parent id.
    SiblingIndex,
    /// The per-thread sparse accumulator, one marker and one value per row.
    Accumulator,
    /// A per-row relative set, unioned, selected, or collected.
    RowSet,
    /// A task's chunk of emitted pairs.
    TaskChunk,
    /// The per-task tables of chunks or counts the driver gathers.
    TaskTable,
    /// A final pair block.
    PairBlock,
    /// The packed keys a view block is sorted through.
    ViewSortScratch,
}

impl Family {
    pub const ALL: [Family; 9] = [
        Family::ParentEdges,
        Family::Csr,
        Family::SiblingIndex,
        Family::Accumulator,
        Family::RowSet,
        Family::TaskChunk,
        Family::TaskTable,
        Family::PairBlock,
        Family::ViewSortScratch,
    ];

    /// The `operation` field of the error.
    pub fn name(self) -> &'static str {
        match self {
            Family::ParentEdges => "parent_edges",
            Family::Csr => "csr",
            Family::SiblingIndex => "sibling_index",
            Family::Accumulator => "accumulator",
            Family::RowSet => "row_set",
            Family::TaskChunk => "task_chunk",
            Family::TaskTable => "task_table",
            Family::PairBlock => "pair_block",
            Family::ViewSortScratch => "view_sort_scratch",
        }
    }

    /// The family by its `operation` name, for the host's test seam.
    pub fn parse(name: &str) -> Option<Family> {
        Family::ALL.into_iter().find(|f| f.name() == name)
    }

    fn code(self) -> u8 {
        self as u8 + 1
    }
}

/// `0` for none, else `Family::code`.
static FAIL_NEXT: AtomicU8 = AtomicU8::new(0);

/// Make the next reservation of `family` fail, or clear the plant with `None`.
///
/// A test seam, process-wide; the plant is consumed by the first reservation
/// of that family on any thread.
#[doc(hidden)]
pub fn fail_next(family: Option<Family>) {
    FAIL_NEXT.store(family.map_or(0, Family::code), Ordering::SeqCst);
}

#[inline]
fn planted(family: Family) -> bool {
    let code = family.code();
    FAIL_NEXT.load(Ordering::Relaxed) == code
        && FAIL_NEXT
            .compare_exchange(code, 0, Ordering::SeqCst, Ordering::Relaxed)
            .is_ok()
}

fn failed(family: Family, dtype: &'static str, requested_elements: usize) -> Error {
    Error::AllocationFailed {
        operation: family.name(),
        requested_elements,
        dtype,
    }
}

/// Reserve room for `additional` more elements, growing geometrically.
#[inline]
pub(crate) fn reserve<T>(
    vec: &mut Vec<T>,
    additional: usize,
    family: Family,
    dtype: &'static str,
) -> Result<(), Error> {
    let total = vec.len().saturating_add(additional);
    if planted(family) || vec.try_reserve(additional).is_err() {
        return Err(failed(family, dtype, total));
    }
    Ok(())
}

/// Reserve room for exactly `additional` more elements.
#[inline]
pub(crate) fn reserve_exact<T>(
    vec: &mut Vec<T>,
    additional: usize,
    family: Family,
    dtype: &'static str,
) -> Result<(), Error> {
    let total = vec.len().saturating_add(additional);
    if planted(family) || vec.try_reserve_exact(additional).is_err() {
        return Err(failed(family, dtype, total));
    }
    Ok(())
}

/// An empty vector with room for `capacity` elements.
pub(crate) fn with_capacity<T>(
    capacity: usize,
    family: Family,
    dtype: &'static str,
) -> Result<Vec<T>, Error> {
    let mut vec = Vec::new();
    reserve_exact(&mut vec, capacity, family, dtype)?;
    Ok(vec)
}

/// `len` copies of `value`.
pub(crate) fn filled<T: Clone>(
    value: T,
    len: usize,
    family: Family,
    dtype: &'static str,
) -> Result<Vec<T>, Error> {
    let mut vec = with_capacity(len, family, dtype)?;
    vec.resize(len, value);
    Ok(vec)
}

/// Push one element, reserving geometrically when full.
#[inline]
pub(crate) fn push<T>(
    vec: &mut Vec<T>,
    value: T,
    family: Family,
    dtype: &'static str,
) -> Result<(), Error> {
    if vec.len() == vec.capacity() {
        reserve(vec, 1, family, dtype)?;
    }
    vec.push(value);
    Ok(())
}

/// Append every element of `iter`, reserving its lower size bound first.
pub(crate) fn extend<T>(
    vec: &mut Vec<T>,
    iter: impl IntoIterator<Item = T>,
    family: Family,
    dtype: &'static str,
) -> Result<(), Error> {
    let iter = iter.into_iter();
    let (lower, upper) = iter.size_hint();
    reserve(vec, lower, family, dtype)?;
    if upper == Some(lower) {
        // Exact size: the room is reserved, so the bulk path cannot grow.
        vec.extend(iter);
        return Ok(());
    }
    for value in iter {
        push(vec, value, family, dtype)?;
    }
    Ok(())
}

/// Collect `iter` into a new vector.
pub(crate) fn collect<T>(
    iter: impl IntoIterator<Item = T>,
    family: Family,
    dtype: &'static str,
) -> Result<Vec<T>, Error> {
    let mut vec = Vec::new();
    extend(&mut vec, iter, family, dtype)?;
    Ok(vec)
}

/// A copy of `slice`.
pub(crate) fn cloned<T: Copy>(
    slice: &[T],
    family: Family,
    dtype: &'static str,
) -> Result<Vec<T>, Error> {
    let mut vec = with_capacity(slice.len(), family, dtype)?;
    vec.extend_from_slice(slice);
    Ok(vec)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Planting is exercised end to end, per family and in a child process,
    /// by `relationships::pairs::tests`; the seam is process-wide, so a plant
    /// here could be consumed by another test running alongside.
    #[test]
    fn families_round_trip_through_their_names() {
        for family in Family::ALL {
            assert_eq!(Family::parse(family.name()), Some(family));
        }
        assert_eq!(Family::parse("nope"), None);
    }
}
