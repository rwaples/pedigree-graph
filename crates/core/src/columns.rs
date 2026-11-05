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
    /// [`Error::LengthMismatch`] when `father`, `twin`, `orig_mother` or
    /// `orig_father` is not as long as `mother`, then
    /// [`Error::ValueOutOfRange`] at the first `mother`, `father` or `twin`
    /// row outside `-1..n`.
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
    /// [`Error::LengthMismatch`] when `father`, `twin` or `depth` is not as
    /// long as `mother`, then [`Error::ValueOutOfRange`] at the first
    /// `mother`, `father` or `twin` row outside `-1..n`.
    pub fn kinship(&self, depth: &'a [i32]) -> Result<KinshipPedigree<'a>, Error> {
        KinshipPedigree::try_new(self.mother, self.father, self.twin, depth)
    }

    /// The parents-first sweeps' input.  The sweeps check the parent rows
    /// themselves, in passes they already make: with `depth`, where it is
    /// checked and sorted; without one, in graph order, each parent row
    /// before its child as it is read.  `depth` may be absent when the rows
    /// are parents-first, and is ignored when they are and it does not sort.
    ///
    /// # Errors
    ///
    /// [`Error::LengthMismatch`] when `father` is not as long as `mother`.
    pub fn lineage(&self, depth: Option<&'a [i32]>) -> Result<ParentColumns<'a>, Error> {
        ParentColumns::try_new(self.mother, self.father, depth)
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
        assert!(cols.lineage(None).is_ok());
        assert!(cols.relationships(IDS.0, IDS.1).is_err());
    }

    #[test]
    fn lineage_checks_the_parent_rows() {
        let cols = GraphColumns {
            father: &[-1, -1, 4, 1],
            ..COLUMNS
        };
        // Out of range before a depth error, absent or not.
        for depth in [None, Some(&[0, 0, 0, 0][..])] {
            let ped = cols.lineage(depth).unwrap();
            assert!(matches!(
                crate::lineage::distinct_ancestor_counts(ped),
                Err(Error::ValueOutOfRange {
                    field: "father",
                    value: 4,
                    ..
                })
            ));
        }
    }

    /// Every parents-first sweep refuses rows it cannot sweep, with or
    /// without a depth, rather than reading out of range, and ignores a
    /// depth that does not sort when the rows are parents-first.
    #[test]
    fn every_sweep_refuses_unreadable_rows_and_ignores_an_unneeded_depth() {
        use crate::kinship::equivalent_generations;
        use crate::lineage::{descendant_path_counts, distinct_ancestor_counts};
        let kinds = |ped: ParentColumns<'_>| {
            [
                distinct_ancestor_counts(ped).err(),
                descendant_path_counts(ped).err(),
                equivalent_generations(ped).err(),
            ]
        };
        let backwards = GraphColumns {
            mother: &[1, -1],
            father: &[-1, -1],
            twin: &[-1, -1],
        };
        let beyond = GraphColumns {
            mother: &[-1, -1, 0, 0],
            father: &[-1, -1, 9, 1],
            ..COLUMNS
        };
        for depth in [None, Some(&[0, 0][..]), Some(&[0][..])] {
            for err in kinds(backwards.lineage(depth).unwrap()) {
                assert!(
                    matches!(
                        err,
                        Some(Error::LengthMismatch { field: "depth", .. })
                            | Some(Error::ValueOutOfRange { field: "depth", .. })
                    ),
                    "{depth:?} {err:?}"
                );
            }
        }
        for depth in [None, Some(&[0, 0, 1, 1][..]), Some(&[0, 0, 0, 0][..])] {
            for err in kinds(beyond.lineage(depth).unwrap()) {
                assert!(
                    matches!(
                        err,
                        Some(Error::ValueOutOfRange {
                            field: "father",
                            value: 9,
                            ..
                        })
                    ),
                    "{depth:?} {err:?}"
                );
            }
        }
        for depth in [
            None,
            Some(&[0, 0, 1, 1][..]),
            Some(&[0, 0, 0, 0][..]),
            Some(&[][..]),
        ] {
            assert_eq!(
                kinds(COLUMNS.lineage(depth).unwrap()),
                [None, None, None],
                "{depth:?}"
            );
        }
    }

    #[test]
    fn lineage_finds_a_child_before_its_parent() {
        // Row 0 is row 1's child: no longer swept as the rows stand, and with
        // no depth to sort by the sweep is refused rather than run backwards.
        let cols = GraphColumns {
            mother: &[1, -1],
            father: &[-1, -1],
            twin: &[-1, -1],
        };
        let err =
            crate::lineage::distinct_ancestor_counts(cols.lineage(None).unwrap()).unwrap_err();
        assert!(matches!(
            err,
            Error::LengthMismatch {
                field: "depth",
                expected_length: 2,
                actual_length: 0,
            }
        ));
        let ok = cols.lineage(Some(&[1, 0])).unwrap();
        assert_eq!(
            crate::lineage::distinct_ancestor_counts(ok).unwrap(),
            vec![1, 0]
        );
    }
}
