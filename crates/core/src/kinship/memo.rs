//! The call-local memo of resolved pair values: one small open-addressing
//! table per lower row.
//!
//! Every key is a canonical row pair `lo <= hi`, so the memo is a map from
//! `(u32, u32)` to `f32`, stored as a table of `(hi, value)` slots under `lo`.
//! Each row's table grows on its own, so a rehash copies one row and the
//! peak overhead is one row's table rather than the whole memo (the ADR 0007
//! layout requirement), and the states of one row's ancestor set sit close
//! together while the walk fills them.  Slice 13's bake-off measured this
//! layout against the 0.9.0 flat table ported as is: 8 to 14 times faster on
//! the random fixtures and 0.53 to 0.68 times the peak RSS
//! (`docs/pedigree-graph-0.8-migration/gate/13a/NOTES.md`).
//!
//! The layout never affects a value: the walker computes each key exactly
//! once from its dependencies' stored bits whatever the table shape.
//!
//! The row vector is sized by the pedigree, not the query: 32 bytes a row
//! before any slot is stored, 17 MB on a 536k-row pedigree, paid on every
//! call.  Every consumer issues one bulk query per graph, which is the shape
//! slice 13 measured; a caller looping single pairs on a large pedigree pays
//! that fixed cost per call and should batch its pairs instead.

use crate::alloc::{self, Family};
use crate::error::Error;

/// Grow a row when `entries / capacity` would pass 7/10.
const LOAD_NUM: usize = 7;
const LOAD_DEN: usize = 10;

/// One entry of a row table: the `hi` row and its value.
#[derive(Clone, Copy)]
struct Slot {
    hi: u32,
    value: f32,
}

const EMPTY: u32 = u32::MAX;
const FIRST_CAPACITY: usize = 4;

/// The table of one `lo` row, linear probe, identity hash on `hi`.
#[derive(Clone, Default)]
struct RowMemo {
    slots: Vec<Slot>,
    entries: u32,
}

impl RowMemo {
    fn slot(slots: &[Slot], hi: u32) -> usize {
        let mask = slots.len() - 1;
        let mut i = (hi as usize) & mask;
        while slots[i].hi != EMPTY && slots[i].hi != hi {
            i = (i + 1) & mask;
        }
        i
    }

    #[inline]
    fn get(&self, hi: u32) -> Option<f32> {
        if self.slots.is_empty() {
            return None;
        }
        let i = RowMemo::slot(&self.slots, hi);
        (self.slots[i].hi == hi).then(|| self.slots[i].value)
    }

    fn grow(&mut self) -> Result<(), Error> {
        let capacity = if self.slots.is_empty() {
            FIRST_CAPACITY
        } else {
            self.slots.len() * 2
        };
        let empty = Slot {
            hi: EMPTY,
            value: 0.0,
        };
        let mut slots = alloc::filled(empty, capacity, Family::KinshipMemo, "uint64")?;
        for slot in &self.slots {
            if slot.hi != EMPTY {
                let i = RowMemo::slot(&slots, slot.hi);
                slots[i] = *slot;
            }
        }
        self.slots = slots;
        Ok(())
    }

    #[inline]
    fn insert(&mut self, hi: u32, value: f32) -> Result<(), Error> {
        if (self.entries as usize + 1) * LOAD_DEN >= self.slots.len() * LOAD_NUM {
            self.grow()?;
        }
        let i = RowMemo::slot(&self.slots, hi);
        self.slots[i] = Slot { hi, value };
        self.entries += 1;
        Ok(())
    }
}

/// A map from canonical row pairs to resolved float32 kinship.
pub struct PairMemo {
    rows: Vec<RowMemo>,
    entries: usize,
}

impl PairMemo {
    /// An empty memo for a pedigree of `n_rows` rows.
    pub fn new(n_rows: usize) -> Result<Self, Error> {
        Ok(PairMemo {
            rows: alloc::filled(RowMemo::default(), n_rows, Family::KinshipMemo, "object")?,
            entries: 0,
        })
    }

    /// The stored value of `(lo, hi)`, if resolved.
    #[inline]
    pub fn get(&self, lo: u32, hi: u32) -> Option<f32> {
        self.rows[lo as usize].get(hi)
    }

    /// Store `(lo, hi)`, which the caller knows is absent.
    #[inline]
    pub fn insert(&mut self, lo: u32, hi: u32, value: f32) -> Result<(), Error> {
        self.rows[lo as usize].insert(hi, value)?;
        self.entries += 1;
        Ok(())
    }

    /// Resolved keys.
    pub fn entries(&self) -> usize {
        self.entries
    }

    /// Bytes the tables hold right now.
    pub fn bytes(&self) -> usize {
        self.rows.len() * std::mem::size_of::<RowMemo>()
            + self
                .rows
                .iter()
                .map(|row| row.slots.len() * std::mem::size_of::<Slot>())
                .sum::<usize>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stores_and_grows() {
        let mut memo = PairMemo::new(1000).unwrap();
        assert_eq!(memo.get(3, 7), None);
        for lo in 0..50u32 {
            for hi in lo..50u32 {
                memo.insert(lo, hi, (lo * 100 + hi) as f32).unwrap();
            }
        }
        for lo in 0..50u32 {
            for hi in lo..50u32 {
                assert_eq!(memo.get(lo, hi), Some((lo * 100 + hi) as f32));
            }
            assert_eq!(memo.get(lo, 999), None);
        }
        assert_eq!(memo.entries(), 50 * 51 / 2);
        assert!(memo.bytes() > 0);
    }
}
