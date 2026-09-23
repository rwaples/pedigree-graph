//! Row storage for the kinship DP (ADR 0007, "DP row storage chosen by
//! benchmark").
//!
//! The DP keeps one sorted `(column, value)` row per individual, appends to
//! rows in any order, and retires a row once no merge walk will read it
//! again.  A write to a retired row dissolves.  That last rule is a row
//! *state*, not emptiness: an empty live row and a retired row must not look
//! alike, or a descendant's later symmetric write would resurrect the row
//! and the summary would tally it.
//!
//! [`Owned`] gives every row its own two vectors and returns them to the
//! allocator on retirement.  Slice 14's bake-off measured it against a port
//! of the 0.9.1 slab allocator (one flat buffer pair, doubling relocation, a
//! size-bucketed free list) and it won every cell on wall and on peak RSS,
//! by up to 3.2 times on RSS where retirement frees most
//! (`docs/pedigree-graph-0.8-migration/gate/14a/NOTES.md`); the arena was
//! deleted with that record.

use crate::alloc::{self, Family};
use crate::error::Error;

/// Sorted per-row column and value storage with retirement.
pub trait RowStore: Sized {
    /// Empty live rows for `n` individuals.
    fn new(n: usize) -> Result<Self, Error>;

    /// Append `(col, val)` to `row`; a write to a retired row dissolves.
    fn push(&mut self, row: usize, col: u32, val: f32) -> Result<(), Error>;

    /// The row's columns, empty once retired.
    fn cols(&self, row: usize) -> &[u32];

    /// The row's values, aligned with [`RowStore::cols`].
    fn vals(&self, row: usize) -> &[f32];

    /// Both halves of a row, for the MZ overwrite and its insertion.
    fn row_mut(&mut self, row: usize) -> (&mut [u32], &mut [f32]);

    /// Free the row and drop every later write to it.
    fn retire(&mut self, row: usize);

    fn is_retired(&self, row: usize) -> bool;
}

/// One owned vector pair per row.
pub struct Owned {
    rows: Vec<Row>,
}

#[derive(Clone, Default)]
struct Row {
    cols: Vec<u32>,
    vals: Vec<f32>,
    retired: bool,
}

impl RowStore for Owned {
    fn new(n: usize) -> Result<Self, Error> {
        Ok(Owned {
            rows: alloc::filled(Row::default(), n, Family::KinshipScratch, "object")?,
        })
    }

    #[inline]
    fn push(&mut self, row: usize, col: u32, val: f32) -> Result<(), Error> {
        let row = &mut self.rows[row];
        if row.retired {
            return Ok(());
        }
        alloc::push(&mut row.cols, col, Family::KinshipRows, "uint32")?;
        alloc::push(&mut row.vals, val, Family::KinshipRows, "float32")
    }

    #[inline]
    fn cols(&self, row: usize) -> &[u32] {
        &self.rows[row].cols
    }

    #[inline]
    fn vals(&self, row: usize) -> &[f32] {
        &self.rows[row].vals
    }

    fn row_mut(&mut self, row: usize) -> (&mut [u32], &mut [f32]) {
        let row = &mut self.rows[row];
        (&mut row.cols, &mut row.vals)
    }

    fn retire(&mut self, row: usize) {
        let row = &mut self.rows[row];
        row.cols = Vec::new();
        row.vals = Vec::new();
        row.retired = true;
    }

    fn is_retired(&self, row: usize) -> bool {
        self.rows[row].retired
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn exercise<S: RowStore>() {
        let mut store = S::new(4).unwrap();
        for col in 0..40u32 {
            store.push(1, col, col as f32).unwrap();
        }
        assert_eq!(store.cols(1).len(), 40);
        assert_eq!(store.vals(1)[39], 39.0);
        assert!(store.cols(0).is_empty());
        assert!(!store.is_retired(0));

        store.push(2, 7, 0.5).unwrap();
        {
            let (cols, vals) = store.row_mut(2);
            cols[0] = 9;
            vals[0] = 0.25;
        }
        assert_eq!(store.cols(2), &[9]);
        assert_eq!(store.vals(2), &[0.25]);

        store.retire(1);
        assert!(store.is_retired(1));
        assert!(store.cols(1).is_empty());
        store.push(1, 5, 1.0).unwrap();
        assert!(
            store.cols(1).is_empty(),
            "a write to a retired row dissolves"
        );
    }

    #[test]
    fn owned_rows_append_overwrite_and_retire() {
        exercise::<Owned>();
    }
}
