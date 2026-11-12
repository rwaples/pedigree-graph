//! External parent ids as founders of the recurrence (ADR 0009, issue #67).
//!
//! A parent id with no row has parent row `-1`, which every kernel reads as
//! a missing parent.  Two rows that name the same external id still share
//! that parent, so kinship runs through it.  [`ExternalFounders`] gives the
//! kernels the pedigree in which it does: the rows, then one parentless,
//! twinless founder row per external id that two or more rows name, in
//! ascending id, with those rows' parent rows pointing at it.  The founders
//! sit at depth 0 and every row one deeper than the host placed it, so the
//! order among the rows, and with it the peel order of every key between
//! two rows, is the host's.
//!
//! An id that one row names gets no founder, nor does one that only the two
//! rows of an MZ pair name, which are one genome.  Its founder would be
//! related to nothing but that genome and its descendants, and the
//! recurrence asks for a founder's kinship only with rows no deeper than its
//! child, so every value is the one a missing parent gives.
//!
//! A kernel runs on [`ExternalFounders::pedigree`] and cuts its output back
//! to the first [`ExternalFounders::rows`] rows.  A pedigree with no shared
//! external id borrows the host's columns and costs one scan of them.

use super::depth_order::{check_structural, check_twins};
use super::pairwise::KinshipPedigree;
use crate::alloc::{self, Family};
use crate::error::Error;
use std::borrow::Cow;

/// One row's use of an external parent id: `slot` is `2 * row`, plus one for
/// the father column.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Named {
    id: i64,
    slot: u32,
}

impl alloc::Dtype for Named {
    const NAME: &'static str = "object";
}

/// A pedigree with its shared external parent ids as founder rows after its
/// own rows.
pub struct ExternalFounders<'a> {
    mother: Cow<'a, [i32]>,
    father: Cow<'a, [i32]>,
    twin: Cow<'a, [i32]>,
    depth: Cow<'a, [i32]>,
    rows: usize,
}

impl<'a> ExternalFounders<'a> {
    /// `ped` with a founder row per external parent id that two or more of
    /// its rows name, the co-twins of an MZ pair counting once; `ped`
    /// itself, borrowed, when there is none or it carries no parent ids.
    ///
    /// # Errors
    ///
    /// When a founder is added: [`Error::ValueOutOfRange`] on `depth` or
    /// `twin` as the kernels report an unordered depth, at the host's row
    /// and value, [`Error::ArithmeticOverflow`] when the rows and founders
    /// together pass int32, and [`Error::AllocationFailed`]
    /// (`kinship_columns`).
    pub fn new(ped: &KinshipPedigree<'a>) -> Result<Self, Error> {
        Self::of(ped, Family::KinshipColumns)
    }

    /// [`ExternalFounders::new`], reserving through `family`.
    pub(crate) fn of(ped: &KinshipPedigree<'a>, family: Family) -> Result<Self, Error> {
        let n = ped.len();
        let (mother, father) = (ped.mother(), ped.father());
        let borrowed = || ExternalFounders {
            mother: Cow::Borrowed(mother),
            father: Cow::Borrowed(father),
            twin: Cow::Borrowed(ped.twin()),
            depth: Cow::Borrowed(ped.depth()),
            rows: n,
        };
        let Some((mother_ids, father_ids)) = ped.parent_ids() else {
            return Ok(borrowed());
        };
        // Counted first, without a branch per row: most pedigrees name no
        // external id, and the kernels behind this are a few hundred
        // instructions a row.
        let columns = [(0, mother, mother_ids), (1, father, father_ids)];
        let external = |parent: i32, id: i64| parent < 0 && id >= 0;
        let uses: usize = columns
            .iter()
            .map(|&(_, rows, ids)| {
                let named = rows.iter().zip(ids).map(|(&p, &id)| external(p, id));
                named.map(usize::from).sum::<usize>()
            })
            .sum();
        if uses < 2 {
            return Ok(borrowed());
        }
        let mut named: Vec<Named> = alloc::with_capacity(uses, family)?;
        for (column, rows, ids) in columns {
            for (row, (&parent, &id)) in rows.iter().zip(ids).enumerate() {
                if external(parent, id) {
                    named.push(Named {
                        id,
                        slot: 2 * row as u32 + column,
                    });
                }
            }
        }
        named.sort_unstable();
        // Two rows or more, not the two co-twins of one genome.
        let twin = ped.twin();
        let shared = |group: &&[Named]| match **group {
            [_] => false,
            [a, b] => twin[a.slot as usize / 2] != (b.slot / 2) as i32,
            _ => true,
        };
        let founders = named.chunk_by(|a, b| a.id == b.id).filter(shared).count();
        if founders == 0 {
            return Ok(borrowed());
        }
        check_structural(mother, father, ped.depth())?;
        check_twins(ped.twin(), ped.depth())?;
        if i32::try_from(n + founders).is_err() {
            return Err(Error::ArithmeticOverflow {
                operation: "kinship_external_founders",
                dtype: "int32",
            });
        }
        let extended = |column: &[i32], shift: i32, founder: i32| {
            let rows = column.iter().map(move |&v| v + shift);
            alloc::collect(rows.chain(std::iter::repeat_n(founder, founders)), family)
        };
        let mut parents = [extended(mother, 0, -1)?, extended(father, 0, -1)?];
        let groups = named.chunk_by(|a, b| a.id == b.id).filter(shared);
        for (founder, group) in groups.enumerate() {
            for name in group {
                parents[name.slot as usize % 2][name.slot as usize / 2] = (n + founder) as i32;
            }
        }
        let [mother, father] = parents;
        Ok(ExternalFounders {
            mother: Cow::Owned(mother),
            father: Cow::Owned(father),
            twin: Cow::Owned(extended(ped.twin(), 0, -1)?),
            depth: Cow::Owned(extended(ped.depth(), 1, 0)?),
            rows: n,
        })
    }

    /// The pedigree the recurrence runs on: the host's rows, then the
    /// founders.  It carries no parent ids, so extending it again borrows it.
    pub fn pedigree(&self) -> KinshipPedigree<'_> {
        KinshipPedigree::from_checked(&self.mother, &self.father, &self.twin, &self.depth)
    }

    /// The host's rows: the first `rows()` rows of [`Self::pedigree`].
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// The founder rows added after the host's.
    pub fn founders(&self) -> usize {
        self.mother.len() - self.rows
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TWIN: [i32; 5] = [-1; 5];

    /// Rows 0 and 1 name mother 900, row 2 names father 800 alone, rows 3
    /// and 4 name father 700; row 4's mother is row 0.
    fn extended(depth: &[i32; 5]) -> Result<[Vec<i32>; 3], Error> {
        let (mother, father) = ([-1, -1, -1, -1, 0], [-1; 5]);
        let ped = KinshipPedigree::try_new(&mother, &father, &TWIN, depth)?
            .with_parent_ids(&[900, 900, -1, -1, 3], &[-1, -1, 800, 700, 700])?;
        let founders = ExternalFounders::new(&ped)?;
        let ped = founders.pedigree();
        assert_eq!((founders.rows(), founders.founders()), (5, 2));
        Ok([ped.mother(), ped.father(), ped.depth()].map(<[i32]>::to_vec))
    }

    #[test]
    fn shared_ids_become_founders_in_ascending_id() {
        let [mother, father, depth] = extended(&[0, 0, 0, 0, 1]).unwrap();
        // 700 is row 5 and 900 row 6; 800, named once, stays missing.
        assert_eq!(mother, [6, 6, -1, -1, 0, -1, -1]);
        assert_eq!(father, [-1, -1, -1, 5, 5, -1, -1]);
        assert_eq!(depth, [1, 1, 1, 1, 2, 0, 0]);
    }

    #[test]
    fn co_twins_and_a_third_row_share_a_founder() {
        let (mother, father) = ([-1; 3], [-1; 3]);
        let (twin, depth) = ([1, 0, -1], [0; 3]);
        let ped = KinshipPedigree::try_new(&mother, &father, &twin, &depth)
            .unwrap()
            .with_parent_ids(&[7, 7, 7], &[-1; 3])
            .unwrap();
        let founders = ExternalFounders::new(&ped).unwrap();
        assert_eq!(founders.pedigree().mother(), [3, 3, 3, -1]);
    }

    #[test]
    fn a_bad_depth_is_reported_as_the_host_gave_it() {
        let err = extended(&[0, 0, 0, 0, 0]).unwrap_err();
        assert!(
            matches!(
                err,
                Error::ValueOutOfRange { field: "depth", position, value: 0, minimum: 1, .. }
                    if position == 4.into()
            ),
            "{err:?}"
        );
    }

    #[test]
    fn a_pedigree_without_a_shared_id_is_borrowed() {
        // Rows 0 and 1 are co-twins.
        let (mother, father) = ([-1, -1, 0], [-1, -1, -1]);
        let (twin, depth) = ([1, 0, -1], [0, 0, 1]);
        let plain = KinshipPedigree::try_new(&mother, &father, &twin, &depth).unwrap();
        let once = plain.with_parent_ids(&[7, 7, 0], &[-1, -1, 8]).unwrap();
        for ped in [plain, once] {
            let founders = ExternalFounders::new(&ped).unwrap();
            assert_eq!((founders.rows(), founders.founders()), (3, 0));
            assert!(std::ptr::eq(founders.pedigree().mother(), ped.mother()));
        }
    }
}
