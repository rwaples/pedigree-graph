//! The call-local memo of resolved pair values, behind one trait and two
//! layouts under measurement.
//!
//! Every key is a canonical row pair `lo <= hi`, so a memo is a map from
//! `(u32, u32)` to `f32`.  [`Flat`] is the 0.9.0 table ported as is: one
//! open-addressing array that doubles with a full rehash, so growth briefly
//! holds two tables.  [`Rows`] indexes a small table per `lo` row that grows
//! on its own, so a rehash copies one row and peak overhead is one row's
//! table rather than the whole memo (the ADR 0007 layout requirement).
//!
//! Neither layout affects a value: the walker computes each key exactly once
//! from its dependencies' stored bits whatever the table shape.

use crate::alloc::{self, Family};
use crate::error::Error;

/// The memo layout a call runs on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layout {
    /// One open-addressing table over `lo * n + hi` keys.
    Flat,
    /// One small open-addressing table per `lo` row.
    Rows,
}

impl Layout {
    /// The layout by its lower-case name, for a host's bake-off switch.
    pub fn parse(name: &str) -> Option<Layout> {
        match name {
            "flat" => Some(Layout::Flat),
            "rows" => Some(Layout::Rows),
            _ => None,
        }
    }
}

/// A map from canonical row pairs to resolved float32 kinship.
pub trait PairMemo: Sized {
    /// An empty memo for a pedigree of `n_rows` rows, sized for about
    /// `pairs` requested pairs.
    fn new(n_rows: usize, pairs: usize) -> Result<Self, Error>;
    /// The stored value of `(lo, hi)`, if resolved.
    fn get(&self, lo: u32, hi: u32) -> Option<f32>;
    /// Store `(lo, hi)`, which the caller knows is absent.
    fn insert(&mut self, lo: u32, hi: u32, value: f32) -> Result<(), Error>;
    /// Resolved keys.
    fn entries(&self) -> usize;
    /// Bytes the tables hold right now.
    fn bytes(&self) -> usize;
}

/// Grow when `entries / capacity` would pass 7/10.
const LOAD_NUM: usize = 7;
const LOAD_DEN: usize = 10;

fn next_pow2(x: usize) -> usize {
    x.max(1).next_power_of_two()
}

/// One open-addressing table, linear probe, identity hash on `lo * n + hi`.
pub struct Flat {
    keys: Vec<u64>,
    vals: Vec<f32>,
    n: u64,
    entries: usize,
}

const FLAT_EMPTY: u64 = u64::MAX;

impl Flat {
    fn key(&self, lo: u32, hi: u32) -> u64 {
        u64::from(lo) * self.n + u64::from(hi)
    }

    fn slot(keys: &[u64], key: u64) -> usize {
        let mask = keys.len() - 1;
        let mut i = (key as usize) & mask;
        while keys[i] != FLAT_EMPTY && keys[i] != key {
            i = (i + 1) & mask;
        }
        i
    }

    fn grow(&mut self) -> Result<(), Error> {
        let capacity = self.keys.len() * 2;
        let mut keys = alloc::filled(FLAT_EMPTY, capacity, Family::KinshipMemo, "uint64")?;
        let mut vals = alloc::filled(0.0f32, capacity, Family::KinshipMemo, "float32")?;
        for (&key, &val) in self.keys.iter().zip(&self.vals) {
            if key != FLAT_EMPTY {
                let i = Flat::slot(&keys, key);
                keys[i] = key;
                vals[i] = val;
            }
        }
        self.keys = keys;
        self.vals = vals;
        Ok(())
    }
}

impl PairMemo for Flat {
    fn new(n_rows: usize, pairs: usize) -> Result<Self, Error> {
        let capacity = next_pow2(if pairs > 4 { 4 * pairs } else { 16 });
        Ok(Flat {
            keys: alloc::filled(FLAT_EMPTY, capacity, Family::KinshipMemo, "uint64")?,
            vals: alloc::filled(0.0f32, capacity, Family::KinshipMemo, "float32")?,
            n: n_rows as u64,
            entries: 0,
        })
    }

    #[inline]
    fn get(&self, lo: u32, hi: u32) -> Option<f32> {
        let key = self.key(lo, hi);
        let i = Flat::slot(&self.keys, key);
        (self.keys[i] == key).then(|| self.vals[i])
    }

    #[inline]
    fn insert(&mut self, lo: u32, hi: u32, value: f32) -> Result<(), Error> {
        if (self.entries + 1) * LOAD_DEN >= self.keys.len() * LOAD_NUM {
            self.grow()?;
        }
        let key = self.key(lo, hi);
        let i = Flat::slot(&self.keys, key);
        self.keys[i] = key;
        self.vals[i] = value;
        self.entries += 1;
        Ok(())
    }

    fn entries(&self) -> usize {
        self.entries
    }

    fn bytes(&self) -> usize {
        self.keys.len() * 8 + self.vals.len() * 4
    }
}

/// One entry of a row table: the `hi` row and its value.
#[derive(Clone, Copy)]
struct Slot {
    hi: u32,
    value: f32,
}

const ROW_EMPTY: u32 = u32::MAX;
const ROW_FIRST_CAPACITY: usize = 4;

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
        while slots[i].hi != ROW_EMPTY && slots[i].hi != hi {
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
            ROW_FIRST_CAPACITY
        } else {
            self.slots.len() * 2
        };
        let empty = Slot {
            hi: ROW_EMPTY,
            value: 0.0,
        };
        let mut slots = alloc::filled(empty, capacity, Family::KinshipMemo, "uint64")?;
        for slot in &self.slots {
            if slot.hi != ROW_EMPTY {
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

/// One table per `lo` row.
pub struct Rows {
    rows: Vec<RowMemo>,
    entries: usize,
}

impl PairMemo for Rows {
    fn new(n_rows: usize, _pairs: usize) -> Result<Self, Error> {
        Ok(Rows {
            rows: alloc::filled(RowMemo::default(), n_rows, Family::KinshipMemo, "object")?,
            entries: 0,
        })
    }

    #[inline]
    fn get(&self, lo: u32, hi: u32) -> Option<f32> {
        self.rows[lo as usize].get(hi)
    }

    #[inline]
    fn insert(&mut self, lo: u32, hi: u32, value: f32) -> Result<(), Error> {
        self.rows[lo as usize].insert(hi, value)?;
        self.entries += 1;
        Ok(())
    }

    fn entries(&self) -> usize {
        self.entries
    }

    fn bytes(&self) -> usize {
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

    fn exercise<M: PairMemo>() {
        let mut memo = M::new(1000, 3).unwrap();
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

    #[test]
    fn flat_stores_and_grows() {
        exercise::<Flat>();
    }

    #[test]
    fn rows_store_and_grow() {
        exercise::<Rows>();
    }

    #[test]
    fn layouts_parse_by_name() {
        assert_eq!(Layout::parse("flat"), Some(Layout::Flat));
        assert_eq!(Layout::parse("rows"), Some(Layout::Rows));
        assert_eq!(Layout::parse("tree"), None);
    }
}
