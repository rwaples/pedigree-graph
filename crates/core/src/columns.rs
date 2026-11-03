//! A built pedigree's row columns as one borrow, and the kernel inputs drawn
//! from it.
//!
//! A host that keeps a pedigree's columns in its own memory (NumPy arrays, R
//! vectors) lends its parent and twin rows here once per call.  Each
//! projection takes the further columns its kernels read, by name, and
//! checks them in the order those checks have always run, so a host never
//! assembles a kernel input itself and never lends a column a kernel does
//! not read.

use crate::error::Error;
use crate::kinship::KinshipPedigree;
use crate::lineage::ParentColumns;
use crate::relationships::Pedigree;

/// A built pedigree's parent and twin rows, borrowed for one call.
///
/// Rows are graph-space, `-1` when absent.  They are unchecked here: each
/// projection checks what its kernels read.
#[derive(Clone, Copy, Debug)]
pub struct GraphColumns<'a> {
    pub mother: &'a [i32],
    pub father: &'a [i32],
    pub twin: &'a [i32],
}

impl<'a> GraphColumns<'a> {
    /// The relationship engine's input; `orig_mother` and `orig_father` are
    /// the parent ids, `-1` when missing.
    ///
    /// # Errors
    ///
    /// As [`Pedigree::try_new`].
    pub fn relationships(
        &self,
        orig_mother: &'a [i64],
        orig_father: &'a [i64],
    ) -> Result<Pedigree<'a>, Error> {
        Pedigree::try_new(
            self.mother,
            self.father,
            self.twin,
            orig_mother,
            orig_father,
        )
    }

    /// The kinship recurrence's input.  Whether `depth` orders the rows is
    /// checked where a kernel reads it.
    ///
    /// # Errors
    ///
    /// As [`KinshipPedigree::try_new`].
    pub fn kinship(&self, depth: &'a [i32]) -> Result<KinshipPedigree<'a>, Error> {
        KinshipPedigree::try_new(self.mother, self.father, self.twin, depth)
    }

    /// The parents-first sweeps' input, trusting construction for the row
    /// range and `rows_topological`, as [`ParentColumns::validated`] does.
    /// `depth` may be absent when the rows are topological.
    ///
    /// # Errors
    ///
    /// As [`ParentColumns::validated`].
    pub fn lineage(
        &self,
        depth: Option<&'a [i32]>,
        rows_topological: bool,
    ) -> Result<ParentColumns<'a>, Error> {
        ParentColumns::validated(self.mother, self.father, depth, rows_topological)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kinship::pair_kinship;

    /// Founders 0 and 1, their children 2 and 3.
    const COLUMNS: GraphColumns<'static> = GraphColumns {
        mother: &[-1, -1, 0, 0],
        father: &[-1, -1, 1, 1],
        twin: &[-1; 4],
    };
    const IDS: (&[i64], &[i64]) = (&[-1, -1, 0, 0], &[-1, -1, 1, 1]);

    fn close(got: &[f32], want: &[f32]) -> bool {
        got.len() == want.len() && got.iter().zip(want).all(|(g, w)| (g - w).abs() < 1e-6)
    }

    #[test]
    fn kinship_accepts_any_depth_ordering() {
        for depth in [[0, 0, 1, 1], [0, 1, 3, 2]] {
            let ped = COLUMNS.kinship(&depth).unwrap();
            let got = pair_kinship(ped, &[0, 2], &[2, 3]).unwrap();
            assert!(close(&got, &[0.25, 0.25]), "{got:?}");
        }
    }

    #[test]
    fn a_kinship_walk_refuses_a_child_at_its_parents_depth() {
        let ped = COLUMNS.kinship(&[0, 0, 1, 0]).unwrap();
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
    fn row_ranges_are_checked_before_depth() {
        let cols = GraphColumns {
            mother: &[-1, -1, 7, 0],
            ..COLUMNS
        };
        for err in [
            cols.kinship(&[0, 0, 0, 0]).unwrap_err(),
            cols.relationships(IDS.0, IDS.1).unwrap_err(),
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
        let cols = GraphColumns {
            twin: &[],
            ..COLUMNS
        };
        assert!(cols.lineage(None, true).is_ok());
        assert!(cols.relationships(IDS.0, IDS.1).is_err());
    }
}
