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

/// Engine input in graph-space rows.
///
/// `mother`, `father`, `twin` are row indices, `-1` when absent (missing or
/// external).  `orig_mother` and `orig_father` are the original parent ids,
/// `-1` when missing; an id with no row is an external parent and still
/// defines siblings.
#[derive(Clone, Debug, Default)]
pub struct Pedigree {
    pub mother: Vec<i32>,
    pub father: Vec<i32>,
    pub twin: Vec<i32>,
    pub orig_mother: Vec<i64>,
    pub orig_father: Vec<i64>,
}

impl Pedigree {
    pub fn len(&self) -> usize {
        self.mother.len()
    }

    pub fn is_empty(&self) -> bool {
        self.mother.is_empty()
    }
}

/// Rows per parallel task.  Small enough to balance load, large enough that
/// the workspace pool is not contended.
const ROWS_PER_TASK: usize = 2048;

/// Exact per-category pair counts up to `max_degree`, using the current Rayon pool.
///
/// Rows are independent, so the work is split into row ranges; each task
/// borrows a [`Workspace`] from a pool that never holds more workspaces than
/// there are threads.  Counts are integers summed in any order, so the result
/// is bit-identical for every thread count.
pub fn count_pairs(ped: &Pedigree, max_degree: u8) -> Counts {
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
                engine.count_row(row, &mut ws, &mut counts);
            }
            pool.lock().unwrap().push(ws);
            counts
        })
        .reduce(Counts::default, Counts::merge)
}
