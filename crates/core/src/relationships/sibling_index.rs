//! Sibling lookup by original parent id.
//!
//! Siblings are defined on the *original* parent ids, not parent rows, so two
//! rows naming the same unresolved external parent are still siblings
//! (`_core.py::sibling_pairs`).  Every row with at least one known parent id
//! takes part, MZ co-twins included: co-twins are full sibs of each other
//! here, and the closest-category fold reports them as MZ (ADR 0010, as
//! amended for issue #29).  Ids are grouped lexicographically as structured
//! keys, never arithmetic-packed.

use crate::alloc::{self, Family};
use crate::error::Error;

/// Rows grouped by a key; `group_of[row] == -1` when the row has no key.
#[derive(Debug)]
struct Groups {
    group_of: Vec<i32>,
    indptr: Vec<usize>,
    members: Vec<u32>,
}

impl Groups {
    fn build<K: Ord + Copy>(
        n: usize,
        keyed: impl Iterator<Item = (K, u32)>,
    ) -> Result<Groups, Error> {
        let mut pairs: Vec<(K, u32)> = alloc::collect(keyed, Family::SiblingIndex)?;
        pairs.sort_unstable();
        let mut group_of = alloc::filled(-1i32, n, Family::SiblingIndex)?;
        let mut indptr = alloc::with_capacity(n + 1, Family::SiblingIndex)?;
        indptr.push(0usize);
        let mut members = alloc::with_capacity(pairs.len(), Family::SiblingIndex)?;
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
        Ok(Groups {
            group_of,
            indptr,
            members,
        })
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
    pub fn build(orig_mother: &[i64], orig_father: &[i64]) -> Result<SiblingIndex, Error> {
        let n = orig_mother.len();
        let rows = || (0..n).filter(|&i| orig_mother[i] >= 0 || orig_father[i] >= 0);
        Ok(SiblingIndex {
            family: Groups::build(
                n,
                rows()
                    .filter(|&i| orig_mother[i] >= 0 && orig_father[i] >= 0)
                    .map(|i| ((orig_mother[i], orig_father[i]), i as u32)),
            )?,
            by_mother: Groups::build(
                n,
                rows()
                    .filter(|&i| orig_mother[i] >= 0)
                    .map(|i| (orig_mother[i], i as u32)),
            )?,
            by_father: Groups::build(
                n,
                rows()
                    .filter(|&i| orig_father[i] >= 0)
                    .map(|i| (orig_father[i], i as u32)),
            )?,
        })
    }

    /// Every row once, each sibling family whole at its first member's
    /// place and every other row where it stands, so full sibs are
    /// classified one after another.  `None` when that is the row order
    /// already, found before anything is allocated, so the walk needs no
    /// order to read.
    pub fn family_order(&self) -> Result<Option<Vec<u32>>, Error> {
        let groups = &self.family;
        let n = groups.group_of.len();
        if self.contiguous() {
            return Ok(None);
        }
        let mut order = alloc::with_capacity(n, Family::SiblingIndex)?;
        let mut placed = alloc::filled(false, groups.indptr.len() - 1, Family::SiblingIndex)?;
        for row in 0..n {
            match usize::try_from(groups.group_of[row]) {
                Err(_) => order.push(row as u32),
                Ok(g) if !std::mem::replace(&mut placed[g], true) => {
                    order.extend_from_slice(groups.members(row));
                }
                Ok(_) => {}
            }
        }
        Ok(Some(order))
    }

    /// Whether every sibling family's members are consecutive rows, so
    /// [`SiblingIndex::family_order`] would be the row order itself.
    fn contiguous(&self) -> bool {
        let groups = &self.family;
        let n = groups.group_of.len();
        let mut row = 0;
        while row < n {
            if groups.group_of[row] < 0 {
                row += 1;
                continue;
            }
            let members = groups.members(row);
            if !members
                .iter()
                .enumerate()
                .all(|(k, &m)| m as usize == row + k)
            {
                return false;
            }
            row += members.len();
        }
        true
    }

    /// The rows sharing `row`'s known mother id, then those sharing its known
    /// father id, each sorted and including `row`; empty where the id is unknown.
    pub fn by_parent_id(&self, row: usize) -> [&[u32]; 2] {
        [self.by_mother.members(row), self.by_father.members(row)]
    }

    /// Full sibs of `row` (same known mother and father ids), sorted, excluding `row`.
    pub fn full_sibs(&self, row: usize, out: &mut Vec<u32>) -> Result<(), Error> {
        out.clear();
        alloc::extend(
            out,
            self.family
                .members(row)
                .iter()
                .copied()
                .filter(|&j| j as usize != row),
            Family::RowSet,
        )
    }

    /// Rows sharing `row`'s known mother id but not both parents, sorted, excluding `row`.
    pub fn maternal_half_sibs(&self, row: usize, out: &mut Vec<u32>) -> Result<(), Error> {
        Self::minus(
            self.by_mother.members(row),
            self.family.members(row),
            row,
            out,
        )
    }

    /// Rows sharing `row`'s known father id but not both parents, sorted, excluding `row`.
    pub fn paternal_half_sibs(&self, row: usize, out: &mut Vec<u32>) -> Result<(), Error> {
        Self::minus(
            self.by_father.members(row),
            self.family.members(row),
            row,
            out,
        )
    }

    /// Maternal and paternal half sibs together, sorted, excluding `row`.
    pub fn half_sibs(
        &self,
        row: usize,
        scratch: &mut Vec<u32>,
        out: &mut Vec<u32>,
    ) -> Result<(), Error> {
        self.maternal_half_sibs(row, out)?;
        self.paternal_half_sibs(row, scratch)?;
        super::sets::union_into(out, scratch)
    }

    fn minus(all: &[u32], remove: &[u32], row: usize, out: &mut Vec<u32>) -> Result<(), Error> {
        out.clear();
        alloc::reserve(out, all.len(), Family::RowSet)?;
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
        Ok(())
    }
}
