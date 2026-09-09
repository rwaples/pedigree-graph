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

use rayon::prelude::*;
use std::sync::Mutex;

/// Engine input in graph-space rows, borrowed from the host for one call.
///
/// `mother`, `father`, `twin` are row indices, `-1` when absent (missing or
/// external).  `orig_mother` and `orig_father` are the original parent ids,
/// `-1` when missing; an id with no row is an external parent and still
/// defines siblings.  Rows may be in any acyclic order.
#[derive(Clone, Copy, Debug)]
pub struct Pedigree<'a> {
    pub mother: &'a [i32],
    pub father: &'a [i32],
    pub twin: &'a [i32],
    pub orig_mother: &'a [i64],
    pub orig_father: &'a [i64],
}

impl Pedigree<'_> {
    pub fn len(&self) -> usize {
        self.mother.len()
    }

    pub fn is_empty(&self) -> bool {
        self.mother.is_empty()
    }
}

/// The same five columns, owned; for readers and tests that build a pedigree.
#[derive(Clone, Debug, Default)]
pub struct PedigreeColumns {
    pub mother: Vec<i32>,
    pub father: Vec<i32>,
    pub twin: Vec<i32>,
    pub orig_mother: Vec<i64>,
    pub orig_father: Vec<i64>,
}

impl PedigreeColumns {
    pub fn borrow(&self) -> Pedigree<'_> {
        Pedigree {
            mother: &self.mother,
            father: &self.father,
            twin: &self.twin,
            orig_mother: &self.orig_mother,
            orig_father: &self.orig_father,
        }
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
