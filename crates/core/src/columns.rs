//! A built pedigree's columns as one borrow, and the kernel inputs drawn from
//! it.
//!
//! A host that keeps a pedigree's columns in its own memory (NumPy arrays, R
//! vectors) lends them here once per call.  Each projection checks what its
//! kernels need, in the order those checks have always run, so a host never
//! assembles a kernel input itself.

use crate::error::Error;
use crate::kinship::KinshipPedigree;
use crate::lineage::ParentColumns;
use crate::relationships::Pedigree;

/// A built pedigree's columns in graph-space rows, borrowed for one call.
///
/// The columns are unchecked: each projection checks the ones its kernels
/// read.  `mother`, `father` and `twin` are rows, `-1` when absent;
/// `orig_mother` and `orig_father` are the parent ids, `-1` when missing.
/// `depth` orders the rows (non-negative, strictly above both parents') and
/// may be absent when no kernel of the call sorts by it.
/// `rows_topological` is construction's finding that every parent row
/// precedes its child.
#[derive(Clone, Copy, Debug)]
pub struct GraphColumns<'a> {
    pub mother: &'a [i32],
    pub father: &'a [i32],
    pub twin: &'a [i32],
    pub orig_mother: &'a [i64],
    pub orig_father: &'a [i64],
    pub depth: Option<&'a [i32]>,
    pub rows_topological: bool,
}

impl<'a> GraphColumns<'a> {
    /// The relationship engine's input.
    ///
    /// # Errors
    ///
    /// As [`Pedigree::try_new`].
    pub fn relationships(&self) -> Result<Pedigree<'a>, Error> {
        Pedigree::try_new(
            self.mother,
            self.father,
            self.twin,
            self.orig_mother,
            self.orig_father,
        )
    }

    /// The kinship recurrence's input.
    ///
    /// # Errors
    ///
    /// As [`KinshipPedigree::try_new`]; an absent depth is a
    /// [`Error::LengthMismatch`] on `depth`.
    pub fn kinship(&self) -> Result<KinshipPedigree<'a>, Error> {
        KinshipPedigree::try_new(
            self.mother,
            self.father,
            self.twin,
            self.depth.unwrap_or(&[]),
        )
    }

    /// The parents-first sweeps' input, trusting construction for the row
    /// range and `rows_topological`, as [`ParentColumns::validated`] does.
    ///
    /// # Errors
    ///
    /// As [`ParentColumns::validated`].
    pub fn lineage(&self) -> Result<ParentColumns<'a>, Error> {
        ParentColumns::validated(self.mother, self.father, self.depth, self.rows_topological)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kinship::pair_kinship;

    /// Founders 0 and 1, their children 2 and 3.
    const MOTHER: [i32; 4] = [-1, -1, 0, 0];
    const FATHER: [i32; 4] = [-1, -1, 1, 1];
    const TWIN: [i32; 4] = [-1; 4];
    const ORIG_MOTHER: [i64; 4] = [-1, -1, 0, 0];
    const ORIG_FATHER: [i64; 4] = [-1, -1, 1, 1];

    fn columns(depth: Option<&[i32]>) -> GraphColumns<'_> {
        GraphColumns {
            mother: &MOTHER,
            father: &FATHER,
            twin: &TWIN,
            orig_mother: &ORIG_MOTHER,
            orig_father: &ORIG_FATHER,
            depth,
            rows_topological: true,
        }
    }

    #[test]
    fn kinship_accepts_any_depth_ordering() {
        for depth in [[0, 0, 1, 1], [0, 3, 8, 4]] {
            let ped = columns(Some(&depth)).kinship().unwrap();
            assert_eq!(pair_kinship(ped, &[0, 2], &[2, 3]).unwrap(), [0.25, 0.25]);
        }
    }

    #[test]
    fn a_kinship_walk_refuses_a_child_at_its_parents_depth() {
        let depth = [0, 0, 1, 0];
        let ped = columns(Some(&depth)).kinship().unwrap();
        let err = pair_kinship(ped, &[3], &[2]).unwrap_err();
        assert!(matches!(
            err,
            Error::ValueOutOfRange {
                field: "depth",
                value: 0,
                minimum: 1,
                ..
            }
        ));
    }

    #[test]
    fn kinship_without_depth_is_a_length_mismatch() {
        let err = columns(None).kinship().unwrap_err();
        assert!(matches!(
            err,
            Error::LengthMismatch {
                field: "depth",
                expected_length: 4,
                actual_length: 0,
            }
        ));
    }

    #[test]
    fn row_ranges_are_checked_before_depth() {
        let mother = [-1, -1, 7, 0];
        let depth = [0, 0, 0, 0];
        let cols = GraphColumns {
            mother: &mother,
            ..columns(Some(&depth))
        };
        for err in [
            cols.kinship().unwrap_err(),
            cols.relationships().unwrap_err(),
        ] {
            assert!(matches!(
                err,
                Error::ValueOutOfRange {
                    field: "mother",
                    value: 7,
                    ..
                }
            ));
        }
    }

    #[test]
    fn lineage_reads_neither_twin_nor_depth() {
        let short: [i32; 0] = [];
        let cols = GraphColumns {
            twin: &short,
            ..columns(Some(&short))
        };
        assert!(cols.lineage().is_ok());
        assert!(cols.relationships().is_err());
    }
}
