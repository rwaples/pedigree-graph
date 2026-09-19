//! Sorted-vector set algebra over row indices, plus the per-thread sparse accumulator.

use super::csr::Csr;
use super::multiplicity::Mult;
use crate::alloc::{self, Family};
use crate::error::Error;

/// A sorted set of rows with a saturated multiplicity each.
pub type Weighted = Vec<(u32, Mult)>;

/// Merge `other` into `set` (both sorted, no duplicates), in place.
///
/// Counts the new members first, grows `set` once, then merges backwards
/// from the end, so a set that is reused across rows keeps its buffer
/// instead of allocating and freeing a replacement on every call.
pub fn union_into(set: &mut Vec<u32>, other: &[u32]) -> Result<(), Error> {
    if other.is_empty() {
        return Ok(());
    }
    if set.is_empty() {
        alloc::reserve(set, other.len(), Family::RowSet, "int32")?;
        set.extend_from_slice(other);
        return Ok(());
    }

    let mut added = other.len();
    let (mut a, mut b) = (0, 0);
    while a < set.len() && b < other.len() {
        match set[a].cmp(&other[b]) {
            std::cmp::Ordering::Less => a += 1,
            std::cmp::Ordering::Greater => b += 1,
            std::cmp::Ordering::Equal => {
                added -= 1;
                a += 1;
                b += 1;
            }
        }
    }
    if added == 0 {
        return Ok(());
    }

    let old_len = set.len();
    alloc::reserve(set, added, Family::RowSet, "int32")?;
    set.resize(old_len + added, 0);
    // Backwards: the write head stays ahead of the read head by the number
    // of `other` members still to place, so nothing is overwritten unread.
    let (mut a, mut b, mut w) = (old_len, other.len(), old_len + added);
    while b > 0 {
        let take_other = a == 0 || set[a - 1] <= other[b - 1];
        if take_other {
            if a > 0 && set[a - 1] == other[b - 1] {
                a -= 1;
            }
            w -= 1;
            b -= 1;
            set[w] = other[b];
        } else {
            w -= 1;
            a -= 1;
            set[w] = set[a];
        }
    }
    Ok(())
}

/// Remove every member of `remove` (sorted) from `set` (sorted) in place.
pub fn subtract(set: &mut Vec<u32>, remove: &[u32]) {
    if set.is_empty() || remove.is_empty() {
        return;
    }
    let mut r = 0;
    set.retain(|&j| {
        while r < remove.len() && remove[r] < j {
            r += 1;
        }
        !(r < remove.len() && remove[r] == j)
    });
}

/// Remove `row` itself from a sorted set.
pub fn drop_self(set: &mut Vec<u32>, row: usize) {
    if let Ok(p) = set.binary_search(&(row as u32)) {
        set.remove(p);
    }
}

/// Number of members strictly greater than `row`: the pairs this row owns.
pub fn count_above(set: &[u32], row: usize) -> u64 {
    (set.len() - set.partition_point(|&j| j <= row as u32)) as u64
}

/// Sparse accumulator over all rows, reused across expansions.
///
/// `marker[j] == stamp` says `j` is live in the current expansion; `value[j]`
/// holds its saturated multiplicity.  One workspace serves one thread.
/// `touched` never exceeds one entry per row, so it is reserved to that bound
/// once, fallibly, with the marker and value arrays.  Its `push` then sits in
/// the innermost loop with nothing to check and no way to reallocate.
pub struct Accumulator {
    stamp: u32,
    marker: Vec<u32>,
    value: Vec<Mult>,
    touched: Vec<u32>,
}

impl Accumulator {
    pub fn new(n: usize) -> Result<Accumulator, Error> {
        Ok(Accumulator {
            stamp: 0,
            marker: alloc::filled(0u32, n, Family::Accumulator, "int32")?,
            value: alloc::filled(Mult::ZERO, n, Family::Accumulator, "uint8")?,
            touched: alloc::with_capacity(n, Family::Accumulator, "int32")?,
        })
    }

    fn begin(&mut self) {
        if self.stamp == u32::MAX {
            self.marker.fill(0);
            self.stamp = 0;
        }
        self.stamp += 1;
        self.touched.clear();
    }

    #[inline]
    fn add(&mut self, j: u32, m: Mult) {
        let idx = j as usize;
        if self.marker[idx] != self.stamp {
            self.marker[idx] = self.stamp;
            self.value[idx] = m;
            // Reserved to one entry per row in `new`, so this cannot grow.
            debug_assert!(self.touched.len() < self.touched.capacity());
            self.touched.push(j);
        } else {
            self.value[idx] = self.value[idx] + m;
        }
    }

    fn drain(&mut self, out: &mut Weighted) -> Result<(), Error> {
        self.touched.sort_unstable();
        out.clear();
        alloc::reserve(out, self.touched.len(), Family::RowSet, "int32")?;
        out.extend(self.touched.iter().map(|&j| (j, self.value[j as usize])));
        Ok(())
    }

    /// One hop of `src` through `adj`, summing path multiplicities.
    pub fn hop(&mut self, adj: &Csr, src: &[(u32, Mult)], out: &mut Weighted) -> Result<(), Error> {
        self.begin();
        for &(s, v) in src {
            let (cols, vals) = adj.row(s as usize);
            for (&c, &w) in cols.iter().zip(vals) {
                self.add(c, v * w);
            }
        }
        self.drain(out)
    }

    /// One hop of an unweighted `src` through `adj`; result support only, as rows.
    pub fn hop_support(&mut self, adj: &Csr, src: &[u32], out: &mut Vec<u32>) -> Result<(), Error> {
        self.begin();
        for &s in src {
            for &c in adj.row(s as usize).0 {
                self.add(c, Mult::ONE);
            }
        }
        self.touched.sort_unstable();
        out.clear();
        alloc::reserve(out, self.touched.len(), Family::RowSet, "int32")?;
        out.extend_from_slice(&self.touched);
        Ok(())
    }

    /// Keep each member of `sets` only in the first set that holds it.
    ///
    /// Visits the sets in order; a member an earlier set claimed is dropped
    /// from every later one.  Uses the marker array, so nothing is allocated.
    pub fn claim_in_order<'a>(&mut self, sets: impl Iterator<Item = &'a mut Vec<u32>>) {
        self.begin();
        let stamp = self.stamp;
        for set in sets {
            set.retain(|&j| self.marker[j as usize] != stamp);
            for &j in set.iter() {
                self.marker[j as usize] = stamp;
            }
        }
    }

    /// Count, per row, how many of the given sorted sets contain it.
    pub fn count_memberships<'a>(
        &mut self,
        sets: impl Iterator<Item = &'a [u32]>,
        out: &mut Weighted,
    ) -> Result<(), Error> {
        self.begin();
        for set in sets {
            for &j in set {
                self.add(j, Mult::ONE);
            }
        }
        self.drain(out)
    }
}

/// The rows of a weighted set, into a buffer the caller reuses.
pub fn support_into(w: &Weighted, out: &mut Vec<u32>) -> Result<(), Error> {
    out.clear();
    alloc::reserve(out, w.len(), Family::RowSet, "int32")?;
    out.extend(w.iter().map(|&(j, _)| j));
    Ok(())
}

/// The rows whose multiplicity `keep` accepts, into a buffer the caller
/// reuses.
pub fn select_into(
    w: &Weighted,
    keep: impl Fn(Mult) -> bool,
    out: &mut Vec<u32>,
) -> Result<(), Error> {
    out.clear();
    alloc::reserve(out, w.len(), Family::RowSet, "int32")?;
    out.extend(w.iter().filter(|&&(_, m)| keep(m)).map(|&(j, _)| j));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The in-place merge against a naive reference, over every overlap
    /// pattern of two small sorted sets, including the empty ones.
    #[test]
    fn union_into_matches_a_naive_merge() {
        fn reference(set: &[u32], other: &[u32]) -> Vec<u32> {
            let mut all: Vec<u32> = set.iter().chain(other).copied().collect();
            all.sort_unstable();
            all.dedup();
            all
        }
        let universe: Vec<u32> = (0..6).collect();
        for a_bits in 0u32..(1 << 6) {
            for b_bits in 0u32..(1 << 6) {
                let a: Vec<u32> = universe
                    .iter()
                    .filter(|&&j| a_bits >> j & 1 == 1)
                    .copied()
                    .collect();
                let b: Vec<u32> = universe
                    .iter()
                    .filter(|&&j| b_bits >> j & 1 == 1)
                    .copied()
                    .collect();
                let mut got = a.clone();
                union_into(&mut got, &b).unwrap();
                assert_eq!(got, reference(&a, &b), "{a:?} u {b:?}");
            }
        }
    }

    /// A reused set keeps its buffer: the capacity never falls, and a merge
    /// that adds nothing does not reallocate.
    #[test]
    fn union_into_reuses_the_buffer() {
        let mut set: Vec<u32> = (0..64).collect();
        let capacity = set.capacity();
        let pointer = set.as_ptr();
        union_into(&mut set, &(0..64).collect::<Vec<u32>>()).unwrap();
        assert_eq!(set.len(), 64);
        assert_eq!(set.capacity(), capacity);
        assert!(std::ptr::eq(set.as_ptr(), pointer));
    }
}
