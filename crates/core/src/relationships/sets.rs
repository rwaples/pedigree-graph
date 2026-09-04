//! Sorted-vector set algebra over row indices, plus the per-thread sparse accumulator.

use super::csr::Csr;
use super::multiplicity::Mult;

/// A sorted set of rows with a saturated multiplicity each.
pub type Weighted = Vec<(u32, Mult)>;

/// Merge `other` into `set` (both sorted, no duplicates).
pub fn union_into(set: &mut Vec<u32>, other: &[u32]) {
    if other.is_empty() {
        return;
    }
    let mut merged = Vec::with_capacity(set.len() + other.len());
    let (mut a, mut b) = (0, 0);
    while a < set.len() && b < other.len() {
        match set[a].cmp(&other[b]) {
            std::cmp::Ordering::Less => {
                merged.push(set[a]);
                a += 1;
            }
            std::cmp::Ordering::Greater => {
                merged.push(other[b]);
                b += 1;
            }
            std::cmp::Ordering::Equal => {
                merged.push(set[a]);
                a += 1;
                b += 1;
            }
        }
    }
    merged.extend_from_slice(&set[a..]);
    merged.extend_from_slice(&other[b..]);
    *set = merged;
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
pub struct Accumulator {
    stamp: u32,
    marker: Vec<u32>,
    value: Vec<Mult>,
    touched: Vec<u32>,
}

impl Accumulator {
    pub fn new(n: usize) -> Accumulator {
        Accumulator {
            stamp: 0,
            marker: vec![0; n],
            value: vec![Mult::ZERO; n],
            touched: Vec::new(),
        }
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
            self.touched.push(j);
        } else {
            self.value[idx] = self.value[idx] + m;
        }
    }

    fn drain(&mut self, out: &mut Weighted) {
        self.touched.sort_unstable();
        out.clear();
        out.extend(self.touched.iter().map(|&j| (j, self.value[j as usize])));
    }

    /// One hop of `src` through `adj`, summing path multiplicities.
    pub fn hop(&mut self, adj: &Csr, src: &[(u32, Mult)], out: &mut Weighted) {
        self.begin();
        for &(s, v) in src {
            let (cols, vals) = adj.row(s as usize);
            for (&c, &w) in cols.iter().zip(vals) {
                self.add(c, v * w);
            }
        }
        self.drain(out);
    }

    /// One hop of an unweighted `src` through `adj`; result support only, as rows.
    pub fn hop_support(&mut self, adj: &Csr, src: &[u32], out: &mut Vec<u32>) {
        self.begin();
        for &s in src {
            for &c in adj.row(s as usize).0 {
                self.add(c, Mult::ONE);
            }
        }
        self.touched.sort_unstable();
        out.clear();
        out.extend_from_slice(&self.touched);
    }

    /// Count, per row, how many of the given sorted sets contain it.
    pub fn count_memberships<'a>(
        &mut self,
        sets: impl Iterator<Item = &'a [u32]>,
        out: &mut Weighted,
    ) {
        self.begin();
        for set in sets {
            for &j in set {
                self.add(j, Mult::ONE);
            }
        }
        self.drain(out);
    }
}

pub fn support(w: &Weighted) -> Vec<u32> {
    w.iter().map(|&(j, _)| j).collect()
}

pub fn select(w: &Weighted, keep: impl Fn(Mult) -> bool) -> Vec<u32> {
    w.iter()
        .filter(|&&(_, m)| keep(m))
        .map(|&(j, _)| j)
        .collect()
}
