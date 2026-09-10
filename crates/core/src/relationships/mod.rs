//! Relationship pairs and counts up to the fifth degree, one row at a time.

mod category;
mod csr;
mod engine;
mod multiplicity;
mod sets;
mod sibling_index;

pub use category::{Category, Counts, N_CATEGORIES};
pub use engine::{Engine, Workspace, EXCLUSIONS};
pub use multiplicity::Mult;

use crate::error::Error;
use rayon::prelude::*;
use std::sync::Mutex;

/// Engine input in graph-space rows, borrowed from the host for one call.
///
/// `mother`, `father`, `twin` are row indices, `-1` when absent (missing or
/// external).  `orig_mother` and `orig_father` are the original parent ids,
/// `-1` when missing; an id with no row is an external parent and still
/// defines siblings.  Rows may be in any acyclic order.
///
/// The fields are private because [`Engine`] indexes by them without a bounds
/// check: only [`Pedigree::try_new`] and [`PedigreeColumns::try_borrow`], which
/// verify that, can build one.
#[derive(Clone, Copy, Debug)]
pub struct Pedigree<'a> {
    mother: &'a [i32],
    father: &'a [i32],
    twin: &'a [i32],
    orig_mother: &'a [i64],
    orig_father: &'a [i64],
}

impl<'a> Pedigree<'a> {
    /// Check the engine's preconditions on five borrowed columns.
    ///
    /// The five columns must have one entry per row, and every row index in
    /// `mother`, `father` and `twin` must be `-1` or a row of the pedigree.
    /// Parent *ids* are unconstrained: an id with no row is an external
    /// parent.
    ///
    /// # Errors
    ///
    /// [`Error::LengthMismatch`] when a column is not as long as `mother`, and
    /// [`Error::ValueOutOfRange`] at the first row index outside `-1..n`.
    pub fn try_new(
        mother: &'a [i32],
        father: &'a [i32],
        twin: &'a [i32],
        orig_mother: &'a [i64],
        orig_father: &'a [i64],
    ) -> Result<Pedigree<'a>, Error> {
        let n = mother.len();
        check_column_length("father", father.len(), n)?;
        check_column_length("twin", twin.len(), n)?;
        check_column_length("orig_mother", orig_mother.len(), n)?;
        check_column_length("orig_father", orig_father.len(), n)?;
        check_row_range("mother", mother, n)?;
        check_row_range("father", father, n)?;
        check_row_range("twin", twin, n)?;
        Ok(Pedigree {
            mother,
            father,
            twin,
            orig_mother,
            orig_father,
        })
    }

    pub fn len(&self) -> usize {
        self.mother.len()
    }

    pub fn is_empty(&self) -> bool {
        self.mother.is_empty()
    }
}

fn check_column_length(field: &'static str, actual_length: usize, n: usize) -> Result<(), Error> {
    if actual_length != n {
        return Err(Error::LengthMismatch {
            field,
            expected_length: n,
            actual_length,
        });
    }
    Ok(())
}

fn check_row_range(field: &'static str, rows: &[i32], n: usize) -> Result<(), Error> {
    let maximum = n as i64 - 1;
    if let Some(position) = rows
        .iter()
        .position(|&row| i64::from(row) < -1 || i64::from(row) > maximum)
    {
        return Err(Error::ValueOutOfRange {
            field,
            position,
            value: i64::from(rows[position]),
            minimum: -1,
            maximum,
        });
    }
    Ok(())
}

/// The same five columns, owned and unchecked; the staging form a reader fills.
///
/// [`PedigreeColumns::try_borrow`] is the gate that turns it into engine input.
#[derive(Clone, Debug, Default)]
pub struct PedigreeColumns {
    pub mother: Vec<i32>,
    pub father: Vec<i32>,
    pub twin: Vec<i32>,
    pub orig_mother: Vec<i64>,
    pub orig_father: Vec<i64>,
}

impl PedigreeColumns {
    /// Borrow the columns as checked engine input.
    ///
    /// # Errors
    ///
    /// Whatever [`Pedigree::try_new`] reports.
    pub fn try_borrow(&self) -> Result<Pedigree<'_>, Error> {
        Pedigree::try_new(
            &self.mother,
            &self.father,
            &self.twin,
            &self.orig_mother,
            &self.orig_father,
        )
    }
}

/// Rows per parallel task.  Small enough to balance load, large enough that
/// the workspace pool is not contended.
const ROWS_PER_TASK: usize = 2048;

/// Exact closest-category pair counts up to `max_degree`, using the current Rayon pool.
///
/// With `selected`, only pairs whose two rows are both selected are counted;
/// classification still runs through every row, so unselected relatives keep
/// connecting the selected ones (the view contract of ADR 0006).
///
/// Rows are independent, so the work is split into row ranges; each task
/// borrows a [`Workspace`] from a pool that never holds more workspaces than
/// there are threads.  Counts are integers summed in any order, so the result
/// is bit-identical for every thread count.
pub fn count_pairs(ped: &Pedigree, max_degree: u8, selected: Option<&[bool]>) -> Counts {
    let engine = Engine::new(ped, max_degree);
    let n = engine.len();
    let pool: Mutex<Vec<Workspace>> = Mutex::new(Vec::new());
    let ranges: Vec<(usize, usize)> = (0..n)
        .step_by(ROWS_PER_TASK)
        .map(|s| (s, (s + ROWS_PER_TASK).min(n)))
        .collect();
    ranges
        .into_par_iter()
        .map(|(start, end)| {
            let mut ws = pool
                .lock()
                .unwrap()
                .pop()
                .unwrap_or_else(|| Workspace::new(n));
            let mut counts = Counts::default();
            for row in start..end {
                if selected.is_some_and(|s| !s[row]) {
                    continue;
                }
                engine.count_row(row, selected, &mut ws, &mut counts);
            }
            pool.lock().unwrap().push(ws);
            counts
        })
        .reduce(Counts::default, Counts::merge)
}

#[cfg(test)]
mod tests {
    use super::{Pedigree, PedigreeColumns};
    use crate::error::Error;

    /// A mutation that breaks one invariant of [`columns`].
    type Corrupt = fn(&mut PedigreeColumns);

    /// Two rows, the second a child of the first; every invariant satisfied.
    fn columns() -> PedigreeColumns {
        PedigreeColumns {
            mother: vec![-1, 0],
            father: vec![-1, -1],
            twin: vec![-1, -1],
            orig_mother: vec![-1, 7],
            orig_father: vec![-1, -1],
        }
    }

    #[test]
    fn well_formed_columns_borrow() {
        let cols = columns();
        let ped = cols.try_borrow().unwrap();
        assert_eq!(ped.len(), 2);
        assert!(!ped.is_empty());
    }

    #[test]
    fn a_short_column_is_a_length_mismatch() {
        let cases: [(&'static str, Corrupt); 4] = [
            ("father", |c| {
                c.father.pop();
            }),
            ("twin", |c| {
                c.twin.pop();
            }),
            ("orig_mother", |c| {
                c.orig_mother.pop();
            }),
            ("orig_father", |c| {
                c.orig_father.pop();
            }),
        ];
        for (field, shorten) in cases {
            let mut cols = columns();
            shorten(&mut cols);
            assert_eq!(
                cols.try_borrow().unwrap_err(),
                Error::LengthMismatch {
                    field,
                    expected_length: 2,
                    actual_length: 1,
                }
            );
        }
    }

    #[test]
    fn a_row_index_outside_the_pedigree_is_out_of_range() {
        let cases: [(&'static str, Corrupt); 3] = [
            ("mother", |c| c.mother[1] = 5),
            ("father", |c| c.father[1] = 5),
            ("twin", |c| c.twin[1] = 5),
        ];
        for (field, corrupt) in cases {
            let mut cols = columns();
            corrupt(&mut cols);
            assert_eq!(
                cols.try_borrow().unwrap_err(),
                Error::ValueOutOfRange {
                    field,
                    position: 1,
                    value: 5,
                    minimum: -1,
                    maximum: 1,
                }
            );
        }
    }

    #[test]
    fn a_row_index_below_minus_one_is_out_of_range() {
        let mut cols = columns();
        cols.mother[0] = -2;
        assert_eq!(
            cols.try_borrow().unwrap_err(),
            Error::ValueOutOfRange {
                field: "mother",
                position: 0,
                value: -2,
                minimum: -1,
                maximum: 1,
            }
        );
    }

    #[test]
    fn parent_ids_are_unconstrained_because_a_parent_may_have_no_row() {
        let mut cols = columns();
        cols.orig_mother[1] = i64::MAX;
        assert!(cols.try_borrow().is_ok());
    }

    #[test]
    fn empty_columns_borrow() {
        let empty = PedigreeColumns::default();
        assert!(empty.try_borrow().unwrap().is_empty());
    }

    /// The entry point the PyO3 host uses, which borrows numpy buffers and so
    /// never has a [`PedigreeColumns`] to hand.
    #[test]
    fn try_new_checks_the_five_slices_directly() {
        assert!(Pedigree::try_new(&[0], &[-1], &[-1], &[-1], &[-1]).is_ok());
        assert_eq!(
            Pedigree::try_new(&[1], &[-1], &[-1], &[-1], &[-1]).unwrap_err(),
            Error::ValueOutOfRange {
                field: "mother",
                position: 0,
                value: 1,
                minimum: -1,
                maximum: 0,
            }
        );
    }
}
