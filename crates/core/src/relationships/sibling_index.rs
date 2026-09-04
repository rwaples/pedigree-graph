//! Sibling lookup by original parent id.
//!
//! Siblings are defined on the *original* parent ids, not parent rows, so two
//! rows naming the same unresolved external parent are still siblings
//! (`_core.py::sibling_pairs`).  Only non-twin rows with at least one known
//! parent id take part.  Ids are grouped lexicographically as structured
//! keys, never arithmetic-packed.

/// Rows grouped by a key; `group_of[row] == -1` when the row has no key.
#[derive(Debug)]
struct Groups {
    group_of: Vec<i32>,
    indptr: Vec<usize>,
    members: Vec<u32>,
}

impl Groups {
    fn build<K: Ord + Copy>(n: usize, keyed: impl Iterator<Item = (K, u32)>) -> Groups {
        let mut pairs: Vec<(K, u32)> = keyed.collect();
        pairs.sort_unstable();
        let mut group_of = vec![-1i32; n];
        let mut indptr = vec![0usize];
        let mut members = Vec::with_capacity(pairs.len());
        let mut k = 0;
        while k < pairs.len() {
            let key = pairs[k].0;
            let group = (indptr.len() - 1) as i32;
            while k < pairs.len() && pairs[k].0 == key {
                group_of[pairs[k].1 as usize] = group;
                members.push(pairs[k].1);
                k += 1;
            }
            indptr.push(members.len());
        }
        Groups {
            group_of,
            indptr,
            members,
        }
    }

    #[inline]
    fn members(&self, row: usize) -> &[u32] {
        let g = self.group_of[row];
        if g < 0 {
            return &[];
        }
        &self.members[self.indptr[g as usize]..self.indptr[g as usize + 1]]
    }
}

#[derive(Debug)]
pub struct SiblingIndex {
    family: Groups,
    by_mother: Groups,
    by_father: Groups,
}

impl SiblingIndex {
    pub fn build(twin: &[i32], orig_mother: &[i64], orig_father: &[i64]) -> SiblingIndex {
        let n = twin.len();
        let takes_part = |i: usize| twin[i] < 0 && (orig_mother[i] >= 0 || orig_father[i] >= 0);
        let rows = || (0..n).filter(|&i| takes_part(i));
        SiblingIndex {
            family: Groups::build(
                n,
                rows()
                    .filter(|&i| orig_mother[i] >= 0 && orig_father[i] >= 0)
                    .map(|i| ((orig_mother[i], orig_father[i]), i as u32)),
            ),
            by_mother: Groups::build(
                n,
                rows()
                    .filter(|&i| orig_mother[i] >= 0)
                    .map(|i| (orig_mother[i], i as u32)),
            ),
            by_father: Groups::build(
                n,
                rows()
                    .filter(|&i| orig_father[i] >= 0)
                    .map(|i| (orig_father[i], i as u32)),
            ),
        }
    }

    /// Full sibs of `row` (same known mother and father ids), sorted, excluding `row`.
    pub fn full_sibs(&self, row: usize, out: &mut Vec<u32>) {
        out.clear();
        out.extend(
            self.family
                .members(row)
                .iter()
                .copied()
                .filter(|&j| j as usize != row),
        );
    }

    /// Rows sharing `row`'s known mother id but not both parents, sorted, excluding `row`.
    pub fn maternal_half_sibs(&self, row: usize, out: &mut Vec<u32>) {
        Self::minus(
            self.by_mother.members(row),
            self.family.members(row),
            row,
            out,
        );
    }

    /// Rows sharing `row`'s known father id but not both parents, sorted, excluding `row`.
    pub fn paternal_half_sibs(&self, row: usize, out: &mut Vec<u32>) {
        Self::minus(
            self.by_father.members(row),
            self.family.members(row),
            row,
            out,
        );
    }

    /// Maternal and paternal half sibs together, sorted, excluding `row`.
    pub fn half_sibs(&self, row: usize, scratch: &mut Vec<u32>, out: &mut Vec<u32>) {
        self.maternal_half_sibs(row, out);
        self.paternal_half_sibs(row, scratch);
        super::sets::union_into(out, scratch);
    }

    fn minus(all: &[u32], remove: &[u32], row: usize, out: &mut Vec<u32>) {
        out.clear();
        let mut r = 0;
        for &j in all {
            while r < remove.len() && remove[r] < j {
                r += 1;
            }
            let removed = r < remove.len() && remove[r] == j;
            if !removed && j as usize != row {
                out.push(j);
            }
        }
    }
}
