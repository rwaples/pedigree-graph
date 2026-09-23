//! Row storage for the kinship DP: two layouts behind one trait (ADR 0007,
//! "DP row storage chosen by benchmark").
//!
//! The DP keeps one sorted `(column, value)` row per individual, appends to
//! rows in any order, and retires a row once no merge walk will read it
//! again.  A write to a retired row dissolves.  That last rule is a row
//! *state*, not emptiness: an empty live row and a retired row must not look
//! alike, or a descendant's later symmetric write would resurrect the row
//! and the summary would tally it.
//!
//! [`Owned`] gives every row its own two vectors and returns them to the
//! allocator on retirement.  [`Arena`] is the 0.9.1 slab allocator ported as
//! it was: one flat pair of buffers, per-row `(start, count, cap)`, doubling
//! relocation, and a size-bucketed free list.  Slice 14's bake-off selects
//! between them; the loser is deleted with the 14a record.

use crate::alloc::{self, Family};
use crate::error::Error;

/// Which row layout a DP run uses, during the slice 14 bake-off.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layout {
    Owned,
    Arena,
}

impl Layout {
    /// The layout by its bake-off name.
    pub fn parse(name: &str) -> Option<Layout> {
        match name {
            "owned" => Some(Layout::Owned),
            "arena" => Some(Layout::Arena),
            _ => None,
        }
    }
}

/// Sorted per-row column and value storage with retirement.
pub trait RowStore: Sized {
    /// Empty live rows for `n` individuals; `max_depth` sizes the arena's
    /// initial slots as the 0.9.1 heuristic did.
    fn new(n: usize, max_depth: usize) -> Result<Self, Error>;

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
    fn new(n: usize, _max_depth: usize) -> Result<Self, Error> {
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

/// The 0.9.1 flat arena: slots of power-of-two capacity carved from one
/// growing buffer pair, relocated on overflow, recycled through a free list
/// bucketed by capacity.
pub struct Arena {
    cols: Vec<u32>,
    vals: Vec<f32>,
    /// Slot offset per row; [`UNALLOCATED`] before the first write and after
    /// retirement.
    start: Vec<usize>,
    count: Vec<u32>,
    /// Slot capacity per row; `0` marks a retired row.
    cap: Vec<u32>,
    next_alloc: usize,
    /// Bucket `b` holds freed slots of capacity `init_cap << b`.
    free: Vec<Vec<usize>>,
    init_cap: u32,
}

const UNALLOCATED: usize = usize::MAX;

impl Arena {
    /// `2 ** (max_depth + 4)` clamped to `[16, 4096]`, the 0.9.1 heuristic
    /// for a row's first slot.
    fn init_cap(max_depth: usize) -> u32 {
        (1u32 << (max_depth + 4).min(12)).clamp(16, 4096)
    }

    fn bucket(&self, cap: u32) -> usize {
        (cap / self.init_cap).trailing_zeros() as usize
    }

    fn free_push(&mut self, start: usize, cap: u32) -> Result<(), Error> {
        let bucket = self.bucket(cap);
        while self.free.len() <= bucket {
            alloc::push(&mut self.free, Vec::new(), Family::KinshipScratch, "object")?;
        }
        alloc::push(
            &mut self.free[bucket],
            start,
            Family::KinshipScratch,
            "uint64",
        )
    }

    fn acquire(&mut self, cap: u32) -> Result<usize, Error> {
        let bucket = self.bucket(cap);
        if let Some(start) = self.free.get_mut(bucket).and_then(Vec::pop) {
            return Ok(start);
        }
        let needed = self.next_alloc + cap as usize;
        if needed > self.cols.len() {
            let mut len = self.cols.len().max(1) * 2;
            while len < needed {
                len *= 2;
            }
            let extra = len - self.cols.len();
            alloc::reserve_exact(&mut self.cols, extra, Family::KinshipRows, "uint32")?;
            alloc::reserve_exact(&mut self.vals, extra, Family::KinshipRows, "float32")?;
            self.cols.resize(len, 0);
            self.vals.resize(len, 0.0);
        }
        let dest = self.next_alloc;
        self.next_alloc = needed;
        Ok(dest)
    }
}

impl RowStore for Arena {
    fn new(n: usize, max_depth: usize) -> Result<Self, Error> {
        let init_cap = Arena::init_cap(max_depth);
        let initial = (1usize << 16).max(init_cap as usize * 1024);
        Ok(Arena {
            cols: alloc::filled(0u32, initial, Family::KinshipRows, "uint32")?,
            vals: alloc::filled(0.0f32, initial, Family::KinshipRows, "float32")?,
            start: alloc::filled(UNALLOCATED, n, Family::KinshipScratch, "uint64")?,
            count: alloc::filled(0u32, n, Family::KinshipScratch, "uint32")?,
            cap: alloc::filled(init_cap, n, Family::KinshipScratch, "uint32")?,
            next_alloc: 0,
            free: Vec::new(),
            init_cap,
        })
    }

    #[inline]
    fn push(&mut self, row: usize, col: u32, val: f32) -> Result<(), Error> {
        let cap = self.cap[row];
        if cap == 0 {
            return Ok(());
        }
        if self.start[row] == UNALLOCATED {
            self.start[row] = self.acquire(cap)?;
        }
        let count = self.count[row];
        if count == cap {
            let old = self.start[row];
            let new_cap = cap * 2;
            self.free_push(old, cap)?;
            let dest = self.acquire(new_cap)?;
            let n = count as usize;
            self.cols.copy_within(old..old + n, dest);
            self.vals.copy_within(old..old + n, dest);
            self.start[row] = dest;
            self.cap[row] = new_cap;
        }
        let pos = self.start[row] + count as usize;
        self.cols[pos] = col;
        self.vals[pos] = val;
        self.count[row] = count + 1;
        Ok(())
    }

    #[inline]
    fn cols(&self, row: usize) -> &[u32] {
        let start = self.start[row];
        if start == UNALLOCATED {
            return &[];
        }
        &self.cols[start..start + self.count[row] as usize]
    }

    #[inline]
    fn vals(&self, row: usize) -> &[f32] {
        let start = self.start[row];
        if start == UNALLOCATED {
            return &[];
        }
        &self.vals[start..start + self.count[row] as usize]
    }

    fn row_mut(&mut self, row: usize) -> (&mut [u32], &mut [f32]) {
        let start = self.start[row];
        if start == UNALLOCATED {
            return (&mut [], &mut []);
        }
        let end = start + self.count[row] as usize;
        (&mut self.cols[start..end], &mut self.vals[start..end])
    }

    fn retire(&mut self, row: usize) {
        let (start, cap) = (self.start[row], self.cap[row]);
        if start != UNALLOCATED && cap > 0 {
            // Bookkeeping only; a refused free-list growth would lose one
            // slot to reuse, never a value.
            let _ = self.free_push(start, cap);
        }
        self.start[row] = UNALLOCATED;
        self.count[row] = 0;
        self.cap[row] = 0;
    }

    fn is_retired(&self, row: usize) -> bool {
        self.cap[row] == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn exercise<S: RowStore>() {
        let mut store = S::new(4, 3).unwrap();
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

        // A freed slot is recycled by a row growing into its capacity.
        for col in 0..40u32 {
            store.push(3, col, 0.0).unwrap();
        }
        assert_eq!(store.cols(3).len(), 40);
    }

    #[test]
    fn owned_rows_append_overwrite_and_retire() {
        exercise::<Owned>();
    }

    #[test]
    fn arena_rows_append_overwrite_and_retire() {
        exercise::<Arena>();
    }

    #[test]
    fn arena_init_cap_follows_the_0_9_1_heuristic() {
        assert_eq!(Arena::init_cap(0), 16);
        assert_eq!(Arena::init_cap(2), 64);
        assert_eq!(Arena::init_cap(6), 1024);
        assert_eq!(Arena::init_cap(8), 4096);
        assert_eq!(Arena::init_cap(60), 4096);
    }

    #[test]
    fn layout_names_round_trip() {
        assert_eq!(Layout::parse("owned"), Some(Layout::Owned));
        assert_eq!(Layout::parse("arena"), Some(Layout::Arena));
        assert_eq!(Layout::parse("flat"), None);
    }
}
