//! A temporary pedigree containing the ancestry needed by a view query.
//!
//! Compact rows retain graph-row order.  Original parent ids are copied
//! unchanged: siblings can share an external parent without either parent
//! having a represented row.  MZ partners are included together because the
//! relationship engine classifies their shared genome explicitly.

use super::{Pedigree, PedigreeColumns};
use crate::alloc::{self, Family};
use crate::error::Error;

pub struct CompactView {
    pub columns: PedigreeColumns,
    /// The caller's view row for each compact row, or -1 for an ancestor.
    pub view_rows: Vec<i32>,
}

impl CompactView {
    /// Retain selected rows, their represented ancestors, and MZ partners.
    /// `view` is a partial permutation of graph rows, already checked by the
    /// caller's public view boundary.
    pub fn build(ped: &Pedigree<'_>, view: &[i32]) -> Result<Self, Error> {
        let n = ped.len();
        let mut retained = alloc::filled(false, n, Family::RowSet, "bool")?;
        let mut stack = alloc::with_capacity(
            view.iter().filter(|&&v| v >= 0).count(),
            Family::RowSet,
            "int32",
        )?;
        for (row, &v) in view.iter().enumerate() {
            if v >= 0 {
                retained[row] = true;
                stack.push(row as i32);
            }
        }
        while let Some(row) = stack.pop() {
            for parent in [
                ped.mother[row as usize],
                ped.father[row as usize],
                ped.twin[row as usize],
            ] {
                if parent >= 0 && !retained[parent as usize] {
                    retained[parent as usize] = true;
                    alloc::reserve(&mut stack, 1, Family::RowSet, "int32")?;
                    stack.push(parent);
                }
            }
        }

        let kept = retained.iter().filter(|&&b| b).count();
        let mut graph_to_compact = alloc::filled(-1i32, n, Family::RowSet, "int32")?;
        let mut compact_row = 0i32;
        for (row, &keep) in retained.iter().enumerate() {
            if keep {
                graph_to_compact[row] = compact_row;
                compact_row += 1;
            }
        }
        let remap = |row: i32| {
            if row < 0 {
                -1
            } else {
                graph_to_compact[row as usize]
            }
        };
        let mut columns = PedigreeColumns {
            mother: alloc::with_capacity(kept, Family::RowSet, "int32")?,
            father: alloc::with_capacity(kept, Family::RowSet, "int32")?,
            twin: alloc::with_capacity(kept, Family::RowSet, "int32")?,
            orig_mother: alloc::with_capacity(kept, Family::RowSet, "int64")?,
            orig_father: alloc::with_capacity(kept, Family::RowSet, "int64")?,
        };
        let mut view_rows = alloc::with_capacity(kept, Family::RowSet, "int32")?;
        for (row, &keep) in retained.iter().enumerate() {
            if !keep {
                continue;
            }
            columns.mother.push(remap(ped.mother[row]));
            columns.father.push(remap(ped.father[row]));
            columns.twin.push(remap(ped.twin[row]));
            columns.orig_mother.push(ped.orig_mother[row]);
            columns.orig_father.push(ped.orig_father[row]);
            view_rows.push(view[row]);
        }
        Ok(CompactView { columns, view_rows })
    }
}
