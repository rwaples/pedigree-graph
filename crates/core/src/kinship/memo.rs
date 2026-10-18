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
//! A table of up to [`POOLED_MAX`] slots lives in the memo's own pool of
//! [`CHUNK_SLOTS`]-slot chunks.  A table that grows puts its old slots on
//! the free list of their size, where the next row to need that size takes
//! them, so the many rows that hold a few keys cost no allocator call and
//! no allocator header each.  A larger table has its own vector.
//!
//! The row vector is sized by the pedigree, not the query: 12 bytes a row
//! before any slot is stored, 6.4 MB on a 536k-row pedigree, paid by every
//! pool worker of every call, beside the call's 32-byte-a-row ancestor
//! signatures and 16-byte-a-row packed columns.  Every consumer issues one
//! bulk query per graph, which is the shape slice 13 measured; a caller
//! looping single pairs on a large pedigree pays that fixed cost per call
//! and should batch its pairs instead.

use crate::alloc::{self, Family};
use crate::error::Error;
use std::ops::Range;

/// Grow a row when `entries / capacity` would pass 7/10.
const LOAD_NUM: usize = 7;
const LOAD_DEN: usize = 10;

/// One entry of a row table: the `hi` row and its value.
#[derive(Clone, Copy)]
struct Slot {
    hi: u32,
    value: f32,
}

impl alloc::Dtype for Slot {
    const NAME: &'static str = "object";
}

impl alloc::Dtype for Vec<Slot> {
    const NAME: &'static str = "object";
}

const EMPTY: u32 = u32::MAX;
const VACANT: Slot = Slot {
    hi: EMPTY,
    value: 0.0,
};
const FIRST_CAPACITY: usize = 4;

/// The largest table the pool holds, 8 KiB.
const POOLED_MAX: usize = 1024;

/// The pool's size classes, `FIRST_CAPACITY` doubling to `POOLED_MAX`.
const CLASSES: usize = (POOLED_MAX / FIRST_CAPACITY).trailing_zeros() as usize + 1;

/// A pool chunk holds `CHUNK_SLOTS` slots, 512 KiB, and no table spans two.
const CHUNK_SHIFT: u32 = 16;
const CHUNK_SLOTS: usize = 1 << CHUNK_SHIFT;

/// The end of a free list.
const NONE: u32 = u32::MAX;

/// Where one `lo` row's table lives: `at` is its start in the pool, counted
/// in blocks of `FIRST_CAPACITY` slots, when `capacity <= POOLED_MAX`, else
/// its index among the large tables.  A row with no table has capacity 0.
/// Counting blocks lets a `u32` address 2^34 slots, 128 GiB a worker.
#[derive(Clone, Copy, Default)]
struct RowMemo {
    at: u32,
    capacity: u32,
    entries: u32,
}

impl alloc::Dtype for RowMemo {
    const NAME: &'static str = "object";
}

/// The size class of a pooled capacity.
fn class(capacity: usize) -> usize {
    (capacity / FIRST_CAPACITY).trailing_zeros() as usize
}

/// The chunk and slot range of the pooled table of `capacity` slots at
/// block `at`; [`PairMemo::carve`] never lets a table cross a chunk's end.
#[inline(always)]
fn place(at: u32, capacity: usize) -> (usize, Range<usize>) {
    let start = at as usize * FIRST_CAPACITY;
    let offset = start & (CHUNK_SLOTS - 1);
    (start >> CHUNK_SHIFT, offset..offset + capacity)
}

/// The slot holding `hi`, or the empty slot that would: linear probe,
/// identity hash on `hi`.
#[inline(always)]
fn probe(slots: &[Slot], hi: u32) -> usize {
    let mask = slots.len() - 1;
    let mut i = (hi as usize) & mask;
    while slots[i].hi != EMPTY && slots[i].hi != hi {
        i = (i + 1) & mask;
    }
    i
}

/// A map from canonical row pairs to resolved float32 kinship.
pub struct PairMemo {
    rows: Vec<RowMemo>,
    chunks: Vec<Vec<Slot>>,
    /// The first released table of each size class, linked through each
    /// table's first slot's `hi`.
    free: [u32; CLASSES],
    large: Vec<Vec<Slot>>,
    /// A growing pooled table's old entries, copied out while it rehashes.
    spare: Vec<Slot>,
    entries: usize,
}

impl PairMemo {
    /// An empty memo for a pedigree of `n_rows` rows.
    pub fn new(n_rows: usize) -> Result<Self, Error> {
        Ok(PairMemo {
            rows: alloc::filled(RowMemo::default(), n_rows, Family::KinshipMemo)?,
            chunks: Vec::new(),
            free: [NONE; CLASSES],
            large: Vec::new(),
            spare: Vec::new(),
            entries: 0,
        })
    }

    /// The pooled table of `capacity` slots at block `at`.
    #[inline(always)]
    fn pooled(&mut self, at: u32, capacity: usize) -> &mut [Slot] {
        let (chunk, slots) = place(at, capacity);
        &mut self.chunks[chunk][slots]
    }

    #[inline(always)]
    fn table(&self, row: RowMemo) -> &[Slot] {
        let capacity = row.capacity as usize;
        if capacity <= POOLED_MAX {
            let (chunk, slots) = place(row.at, capacity);
            &self.chunks[chunk][slots]
        } else {
            &self.large[row.at as usize]
        }
    }

    #[inline]
    fn table_mut(&mut self, row: RowMemo) -> &mut [Slot] {
        let capacity = row.capacity as usize;
        if capacity <= POOLED_MAX {
            self.pooled(row.at, capacity)
        } else {
            &mut self.large[row.at as usize]
        }
    }

    /// The stored value of `(lo, hi)`, if resolved.
    #[inline(always)]
    pub fn get(&self, lo: u32, hi: u32) -> Option<f32> {
        let row = self.rows[lo as usize];
        if row.capacity == 0 {
            return None;
        }
        let slots = self.table(row);
        let i = probe(slots, hi);
        (slots[i].hi == hi).then(|| slots[i].value)
    }

    /// Store `(lo, hi)`, which the caller knows is absent.
    #[inline]
    pub fn insert(&mut self, lo: u32, hi: u32, value: f32) -> Result<(), Error> {
        let mut row = self.rows[lo as usize];
        if (row.entries as usize + 1) * LOAD_DEN >= row.capacity as usize * LOAD_NUM {
            row = self.grow(row)?;
        }
        let slots = self.table_mut(row);
        let i = probe(slots, hi);
        slots[i] = Slot { hi, value };
        row.entries += 1;
        self.rows[lo as usize] = row;
        self.entries += 1;
        Ok(())
    }

    /// Resolved keys.
    pub fn entries(&self) -> usize {
        self.entries
    }

    /// Move `row`'s table to one of twice the capacity, or give it its first
    /// table, and return where it now lives.  On an error the row is as it
    /// was.
    fn grow(&mut self, row: RowMemo) -> Result<RowMemo, Error> {
        let old = row.capacity as usize;
        let capacity = if old == 0 { FIRST_CAPACITY } else { old * 2 };
        let refused = || alloc::failed::<Slot>(Family::KinshipMemo, capacity);
        let mut grown = RowMemo {
            capacity: u32::try_from(capacity).map_err(|_| refused())?,
            ..row
        };
        if old > POOLED_MAX {
            let mut slots = alloc::filled(VACANT, capacity, Family::KinshipMemo)?;
            for slot in &self.large[row.at as usize] {
                if slot.hi != EMPTY {
                    let i = probe(&slots, slot.hi);
                    slots[i] = *slot;
                }
            }
            self.large[row.at as usize] = slots;
            return Ok(grown);
        }
        // Room for the old entries first, so nothing below can fail after
        // the new table is taken.
        self.spare.clear();
        alloc::reserve(&mut self.spare, old, Family::KinshipMemo)?;
        grown.at = if capacity <= POOLED_MAX {
            self.carve(capacity)?
        } else {
            let at = u32::try_from(self.large.len()).map_err(|_| refused())?;
            let slots = alloc::filled(VACANT, capacity, Family::KinshipMemo)?;
            alloc::push(&mut self.large, slots, Family::KinshipMemo)?;
            at
        };
        if old > 0 {
            let mut spare = std::mem::take(&mut self.spare);
            spare.extend_from_slice(self.table(row));
            let slots = self.table_mut(grown);
            for slot in spare.iter().filter(|slot| slot.hi != EMPTY) {
                let i = probe(slots, slot.hi);
                slots[i] = *slot;
            }
            self.spare = spare;
            self.release(row.at, old);
        }
        Ok(grown)
    }

    /// The first block of a vacant pooled table of `capacity` slots, reused
    /// from its size class's free list when one is there.
    fn carve(&mut self, capacity: usize) -> Result<u32, Error> {
        let class = class(capacity);
        let at = self.free[class];
        if at != NONE {
            let slots = self.pooled(at, capacity);
            let next = slots[0].hi;
            slots.fill(VACANT);
            self.free[class] = next;
            return Ok(at);
        }
        let fits = self
            .chunks
            .last()
            .is_some_and(|chunk| chunk.len() + capacity <= CHUNK_SLOTS);
        if !fits {
            // Every block of the new chunk must sit below `NONE`.  A refusal
            // reports the pool's slots with the chunk, as a refused
            // reservation reports its vector's total.
            let slots = (self.chunks.len() + 1) * CHUNK_SLOTS;
            if slots / FIRST_CAPACITY > NONE as usize {
                return Err(alloc::failed::<Slot>(Family::KinshipMemo, slots));
            }
            let chunk = alloc::with_capacity(CHUNK_SLOTS, Family::KinshipMemo)?;
            alloc::push(&mut self.chunks, chunk, Family::KinshipMemo)?;
        }
        let index = self.chunks.len() - 1;
        let chunk = &mut self.chunks[index];
        let at = (index * CHUNK_SLOTS + chunk.len()) / FIRST_CAPACITY;
        // Within the chunk's reserved capacity, so this cannot reallocate.
        chunk.resize(chunk.len() + capacity, VACANT);
        Ok(at as u32)
    }

    /// Put the pooled table at block `at` on its size class's free list,
    /// linked through its first slot.
    fn release(&mut self, at: u32, capacity: usize) {
        let class = class(capacity);
        self.pooled(at, capacity)[0].hi = self.free[class];
        self.free[class] = at;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

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
    }

    /// Interleaved rows of every size, from one key to past the pool, so
    /// tables cross chunk boundaries, reuse released slots and leave the
    /// pool for their own vectors.
    #[test]
    fn matches_a_map_across_the_pool() {
        const ROWS: u32 = 40_000;
        let mut memo = PairMemo::new(ROWS as usize).unwrap();
        let mut truth = HashMap::new();
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        for step in 0..400_000u32 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            // Every fourth key goes to row 7, which outgrows the pool.
            let lo = if step % 4 == 0 {
                7
            } else {
                (state % ROWS as u64) as u32
            };
            let hi = lo + ((state >> 32) % 1_000_000) as u32;
            if truth.contains_key(&(lo, hi)) {
                continue;
            }
            let value = (state >> 40) as f32;
            memo.insert(lo, hi, value).unwrap();
            truth.insert((lo, hi), value);
        }
        assert!(memo.rows[7].capacity as usize > POOLED_MAX);
        assert!(memo.chunks.len() > 1);
        for (&(lo, hi), &value) in &truth {
            assert_eq!(memo.get(lo, hi), Some(value));
            assert_eq!(memo.get(lo, hi + 1_000_000), None);
        }
        assert_eq!(memo.entries(), truth.len());
    }

    /// Insert keys `hi` into row 0, each valued `hi`.
    fn fill_row(memo: &mut PairMemo, hi: Range<u32>) {
        for hi in hi {
            memo.insert(0, hi, hi as f32).unwrap();
        }
    }

    /// A refused reservation in a grow leaves the row as it was: every key
    /// still reads back, the refused one is absent, and the next insert
    /// succeeds.  The seam is process-wide, so each case runs
    /// [`grow_seam_child`] in a fresh copy of this test binary.
    #[test]
    fn a_refused_grow_leaves_the_row_as_it_was() {
        let exe = std::env::current_exe().unwrap();
        for case in ["spare", "chunk", "large"] {
            let out = std::process::Command::new(&exe)
                .args([
                    "--exact",
                    "kinship::memo::tests::grow_seam_child",
                    "--nocapture",
                ])
                .env("PG_MEMO_SEAM_CASE", case)
                .output()
                .unwrap();
            assert!(
                out.status.success(),
                "{case}:\n{}",
                String::from_utf8_lossy(&out.stderr)
            );
        }
    }

    /// The body of one grow seam case; a no-op unless `PG_MEMO_SEAM_CASE`
    /// is set.
    #[test]
    fn grow_seam_child() {
        use crate::alloc::{fail_next, fail_next_above};
        let Ok(case) = std::env::var("PG_MEMO_SEAM_CASE") else {
            return;
        };
        let mut memo = PairMemo::new(8).unwrap();
        // Keys row 0 holds before the refused insert, and the slots the
        // refused reservation asked for.
        let (kept, requested) = match case.as_str() {
            // 2 keys fill the first table; the 3rd grows it to 8 and first
            // reserves the copy of the 4 old slots.
            "spare" => {
                fill_row(&mut memo, 0..2);
                fail_next(Some(Family::KinshipMemo));
                (2, 4)
            }
            // The first insert carves the pool's first chunk.
            "chunk" => {
                fail_next_above(Some(Family::KinshipMemo), CHUNK_SLOTS);
                (0, CHUNK_SLOTS)
            }
            // 716 keys fill a 1024-slot table; the next leaves the pool.
            "large" => {
                fill_row(&mut memo, 0..716);
                assert_eq!(memo.rows[0].capacity as usize, POOLED_MAX);
                fail_next_above(Some(Family::KinshipMemo), 2 * POOLED_MAX);
                (716, 2 * POOLED_MAX)
            }
            other => panic!("unknown case {other}"),
        };
        let err = memo.insert(0, kept, 0.5).unwrap_err();
        assert_eq!(
            err,
            Error::AllocationFailed {
                operation: "kinship_memo",
                requested_elements: requested,
                dtype: "object",
            }
        );
        assert_eq!(memo.get(0, kept), None);
        assert_eq!(memo.entries(), kept as usize);
        for hi in 0..kept {
            assert_eq!(memo.get(0, hi), Some(hi as f32));
        }
        memo.insert(0, kept, 0.5).unwrap();
        assert_eq!(memo.get(0, kept), Some(0.5));
        fill_row(&mut memo, kept + 1..2000);
        for hi in 0..2000 {
            assert!(memo.get(0, hi).is_some());
        }
    }
}
