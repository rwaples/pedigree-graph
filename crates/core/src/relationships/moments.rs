//! Relationship moments: label-keyed pair counts and fixed-point sums over
//! the row-streaming engine (ADR 0013).
//!
//! [`reduce_pairs`] drives [`Engine::emit_row`] against a [`Reducer`]: every
//! owned pair of every requested category, in semantic orientation, reaches
//! `Reducer::reduce` on one of `W` lanes, and the lanes are merged when the
//! pass ends.  `W` lanes run side by side inside the package pool, each
//! pulling task ranges from a shared cursor, so at most `W` lane
//! accumulators are ever live whatever the pool's thread count.  The lanes
//! are allocated before the pass and merged in lane order; a reducer whose
//! merge is associative and commutative (integer sums) therefore gives the
//! same result for every `W` and every thread count.
//!
//! [`CellReducer`] is the one reducer of 0.11: per requested category and
//! per cell `(first label, second label, equality bits)`, the pair count, the
//! sums and sums of squares of each member's quantized value columns, and
//! the cross sums of the requested products, all as exact `i128`.  A term is
//! at most `2^86` (two quantized values of at most `2^43`), so a cell
//! overflows only past `2^40` pairs, which one check on the counts after the
//! pass rules out.  The merged integers are handed to the host as they are,
//! encoded at their minimal width ([`super::encode_i128`]); every centered
//! moment and every float is derived from them by
//! [`super::MomentsTable`], so the result algebra stays exact (ADR 0015).

use super::category::{Category, CategorySet, N_CATEGORIES};
use super::engine::{Engine, WorkspacePool};
use super::pairs::check_view_map;
use super::progress::{Checkpoint, Progress};
use super::{check_column_length, task_ranges, walk_rows, CompactView, MaxDegree, Pedigree};
use crate::alloc::{self, Family};
use crate::error::Error;
use rayon::prelude::*;
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

/// The largest cell count the `i128` accumulators are proven not to overflow at.
pub const MAX_CELL_PAIRS: i128 = 1 << 40;

/// The largest quantized magnitude a value may carry; the host's scale rule
/// (ADR 0013) guarantees it and the overflow proof above assumes it.
pub const MAX_QUANTIZED: i64 = 1 << 43;

/// Bytes per accumulator beyond the lanes once the pass is over: the
/// encoded table (at most 16, every accumulator being an `i128`) and one
/// host copy of it (R copies it into a raw vector; Python adopts the
/// buffer).  The same figure for both hosts, so a budget refuses the same
/// calls in each (ADR 0015).
pub const HOST_BYTES_PER_ACCUMULATOR: u64 = 16 + 16;

/// The host term of the budget estimate, [`HOST_BYTES_PER_ACCUMULATOR`] per
/// accumulator; `None` when it is not representable.
pub fn host_bytes(accumulators: u64) -> Option<u64> {
    accumulators.checked_mul(HOST_BYTES_PER_ACCUMULATOR)
}

/// A sink for the oriented pairs of one engine pass, accumulated per lane.
///
/// `reduce` runs on the hot path with no `Result`: a reducer sizes its lane
/// in `lane`, where allocation can fail, and reports any post-pass check
/// from whatever consumes the merged lane.
pub trait Reducer: Sync {
    /// One lane's accumulator.
    type Lane: Send;

    /// Allocate an empty lane.
    ///
    /// # Errors
    ///
    /// [`Error::AllocationFailed`] when the lane cannot be built.
    fn lane(&self) -> Result<Self::Lane, Error>;

    /// Add one pair of `cat`, `first` and `second` in the category's
    /// semantic orientation, in the receiver's rows.
    fn reduce(&self, lane: &mut Self::Lane, cat: Category, first: u32, second: u32);

    /// Fold `from` into `into`.
    fn merge(&self, into: &mut Self::Lane, from: Self::Lane);
}

/// How a symmetric pair is oriented before it reaches the reducer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Symmetric {
    /// Once, lower receiver row first (the pair-block rule).
    Canonical,
    /// In both orientations; asymmetric categories are unchanged.
    Both,
}

impl Symmetric {
    /// The public spelling, as the Python keyword accepts it.
    pub fn name(self) -> &'static str {
        match self {
            Symmetric::Canonical => "canonical",
            Symmetric::Both => "both",
        }
    }

    /// The orientation by its public spelling.
    pub fn parse(name: &str) -> Option<Symmetric> {
        match name {
            "canonical" => Some(Symmetric::Canonical),
            "both" => Some(Symmetric::Both),
            _ => None,
        }
    }
}

/// What one pass hands back: the merged lane and how many pairs each lane
/// reduced (in lane order), the latter a diagnostic for the lane tests.
pub struct Reduced<L> {
    pub lane: L,
    pub lane_pairs: Vec<u64>,
}

/// Run `reducer` over every owned pair of the requested categories on
/// `lanes` lanes inside the current Rayon pool, and return the merged lane.
///
/// `view` is the int32 view row of every graph row (`-1` unselected), as for
/// [`super::pair_blocks`], already checked to be a partial permutation by
/// the caller; pairs then arrive in view rows.  With `Symmetric::Both`
/// every symmetric pair is reduced twice, once per orientation.  Once any
/// lane fails, the others stop taking task ranges.
///
/// # Errors
///
/// [`Error::AllocationFailed`] from the engine, a workspace, a row set or a
/// lane; [`Error::Cancelled`] once `progress` is cancelled.
///
/// # Panics
///
/// If `view` does not have one entry per graph row; the host checks that.
#[allow(clippy::too_many_arguments)]
pub fn reduce_pairs<R: Reducer>(
    ped: &Pedigree,
    max_degree: MaxDegree,
    requested: CategorySet,
    view: Option<&[i32]>,
    symmetric: Symmetric,
    lanes: NonZeroUsize,
    reducer: &R,
    progress: &Progress,
) -> Result<Reduced<R::Lane>, Error> {
    let engine = Engine::new(ped, max_degree)?;
    let n = engine.len();
    if let Some(map) = view {
        assert_eq!(map.len(), n, "view map must have one entry per graph row");
    }
    let pool = WorkspacePool::for_pairs(n);
    let ranges = task_ranges(n);
    let cursor = AtomicUsize::new(0);
    let failed = AtomicBool::new(false);
    let both = symmetric == Symmetric::Both;
    let mut slots = Vec::with_capacity(lanes.get());
    for _ in 0..lanes.get() {
        slots.push(reducer.lane()?);
    }
    progress.walk(n)?;
    let finished: Vec<(R::Lane, u64)> = slots
        .into_par_iter()
        .map(|mut lane| {
            let mut ws = pool.take()?;
            // Given back even when a row fails, so the lanes still running
            // reuse it rather than allocate under the pressure that failed
            // this one.
            let mut result = Ok(());
            let mut pairs = 0u64;
            while !failed.load(Ordering::Relaxed) {
                let Some(&(start, end)) = ranges.get(cursor.fetch_add(1, Ordering::Relaxed)) else {
                    break;
                };
                result = walk_rows(
                    (start, end),
                    progress,
                    |row| view.is_some_and(|map| map[row] < 0),
                    |row| {
                        engine.emit_row(row, &requested, view, &mut ws, |cat, a, b| {
                            reducer.reduce(&mut lane, cat, a, b);
                            pairs += 1;
                            if both && cat.symmetric() {
                                reducer.reduce(&mut lane, cat, b, a);
                                pairs += 1;
                            }
                            Ok(())
                        })
                    },
                );
                progress.advance(end - start);
                if result.is_err() {
                    failed.store(true, Ordering::Relaxed);
                }
            }
            pool.give(ws);
            result.map(|()| (lane, pairs))
        })
        .collect::<Result<Vec<_>, Error>>()?;
    progress.finish()?;
    let (accumulators, lane_pairs): (Vec<R::Lane>, Vec<u64>) = finished.into_iter().unzip();
    let mut accumulators = accumulators.into_iter();
    // `lanes` is non-zero, so there is always a first lane; the library does not panic.
    let mut lane = match accumulators.next() {
        Some(first) => first,
        None => reducer.lane()?,
    };
    for from in accumulators {
        progress.checkpoint(Checkpoint::LaneMerge)?;
        reducer.merge(&mut lane, from);
    }
    Ok(Reduced { lane, lane_pairs })
}

/// Which member of a pair a product operand reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Side {
    First,
    Second,
}

/// One cross product: `(side, column) × (side, column)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Product {
    pub a: (Side, usize),
    pub b: (Side, usize),
}

/// The per-individual inputs of one moments call, in receiver rows.
///
/// `values` and `same` are row-major `[n, n_columns]` and `[n, n_same]`.
/// Values are already quantized by the host to integers of magnitude at
/// most [`MAX_QUANTIZED`].
#[derive(Clone, Copy, Debug)]
pub struct MomentsInput<'a> {
    pub labels_first: &'a [i32],
    pub n_labels_first: usize,
    pub labels_second: &'a [i32],
    pub n_labels_second: usize,
    pub values: &'a [i64],
    pub n_columns: usize,
    pub products: &'a [Product],
    pub same: &'a [i64],
    pub n_same: usize,
}

impl MomentsInput<'_> {
    /// The sizes that decide the accumulator layout and the budget.
    pub fn shape(&self, requested: CategorySet) -> MomentsShape {
        MomentsShape {
            n_categories: requested.iter().count(),
            n_labels_first: self.n_labels_first,
            n_labels_second: self.n_labels_second,
            n_columns: self.n_columns,
            n_products: self.products.len(),
            n_same: self.n_same,
        }
    }

    /// Check the inputs against the receiver's row count.
    ///
    /// # Errors
    ///
    /// [`Error::LengthMismatch`] when an array does not have `n` rows,
    /// [`Error::ValueOutOfRange`] at the first label outside its range, the
    /// first quantized value beyond [`MAX_QUANTIZED`], or the first product
    /// operand naming a column at or past `n_columns`.
    pub fn check(&self, n: usize) -> Result<(), Error> {
        check_column_length("labels_first", self.labels_first.len(), n)?;
        check_column_length("labels_second", self.labels_second.len(), n)?;
        check_column_length("values", self.values.len(), n * self.n_columns)?;
        check_column_length("same", self.same.len(), n * self.n_same)?;
        check_label_range("labels_first", self.labels_first, self.n_labels_first)?;
        check_label_range("labels_second", self.labels_second, self.n_labels_second)?;
        if let Some(position) = self
            .values
            .iter()
            .position(|&q| q.unsigned_abs() > MAX_QUANTIZED as u64)
        {
            return Err(Error::ValueOutOfRange {
                field: "values",
                position,
                value: self.values[position],
                minimum: -MAX_QUANTIZED,
                maximum: MAX_QUANTIZED,
            });
        }
        for (position, product) in self.products.iter().enumerate() {
            for column in [product.a.1, product.b.1] {
                if column >= self.n_columns {
                    return Err(Error::ValueOutOfRange {
                        field: "products",
                        position,
                        value: column as i64,
                        minimum: 0,
                        maximum: self.n_columns as i64 - 1,
                    });
                }
            }
        }
        Ok(())
    }
}

fn check_label_range(field: &'static str, labels: &[i32], n_labels: usize) -> Result<(), Error> {
    let maximum = n_labels as i64 - 1;
    if let Some(position) = labels
        .iter()
        .position(|&label| label < 0 || i64::from(label) > maximum)
    {
        return Err(Error::ValueOutOfRange {
            field,
            position,
            value: i64::from(labels[position]),
            minimum: 0,
            maximum,
        });
    }
    Ok(())
}

/// The sizes of one call, enough to plan it without the arrays.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MomentsShape {
    pub n_categories: usize,
    pub n_labels_first: usize,
    pub n_labels_second: usize,
    pub n_columns: usize,
    pub n_products: usize,
    pub n_same: usize,
}

/// The sizes and lane count of one call, fixed before any accumulator exists.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MomentsPlan {
    /// `i128` accumulators per lane: categories × cells × stride.
    accumulators: usize,
    /// Cells per category: `n_labels_first × n_labels_second × 2^n_same`.
    cells: usize,
    /// `i128` accumulators per cell: count, four sums per column, one per product.
    stride: usize,
    /// Lanes the pass runs on.
    pub lanes: NonZeroUsize,
    /// The accumulator peak the pass is planned to reach.
    pub estimated_peak_bytes: u64,
}

const OPERATION: &str = "relationship_moments";

fn over_budget(estimated_bytes: u64, budget_bytes: u64) -> Error {
    Error::MemoryBudgetExceeded {
        operation: OPERATION,
        estimated_bytes,
        budget_bytes,
    }
}

impl MomentsPlan {
    /// Size the accumulators and choose the lane count that fits the budget.
    ///
    /// `W = min(threads, largest W whose estimated peak fits)`, where the
    /// estimate is `W` lanes of 16 bytes per accumulator plus the host term
    /// [`host_bytes`] once; the merge is in place.  Per-lane engine
    /// workspaces and the input arrays are outside the estimate.
    ///
    /// # Errors
    ///
    /// [`Error::MemoryBudgetExceeded`] when one lane plus the host copy does
    /// not fit, or when a size is not representable (reported as
    /// `u64::MAX` estimated bytes).
    pub fn new(
        shape: MomentsShape,
        threads: NonZeroUsize,
        budget_bytes: u64,
    ) -> Result<MomentsPlan, Error> {
        let stride = 1 + 4 * shape.n_columns + shape.n_products;
        let unrepresentable = || over_budget(u64::MAX, budget_bytes);
        let same_cells = u32::try_from(shape.n_same)
            .ok()
            .and_then(|s| 1u64.checked_shl(s))
            .ok_or_else(unrepresentable)?;
        let cells = (shape.n_labels_first as u64)
            .checked_mul(shape.n_labels_second as u64)
            .and_then(|c| c.checked_mul(same_cells))
            .ok_or_else(unrepresentable)?;
        let accumulators = cells
            .checked_mul(shape.n_categories as u64)
            .and_then(|c| c.checked_mul(stride as u64))
            .ok_or_else(unrepresentable)?;
        let lane_bytes = accumulators.checked_mul(16).ok_or_else(unrepresentable)?;
        let host_bytes = host_bytes(accumulators).ok_or_else(unrepresentable)?;
        let one_lane = lane_bytes
            .checked_add(host_bytes)
            .ok_or_else(unrepresentable)?;
        if one_lane > budget_bytes {
            return Err(over_budget(one_lane, budget_bytes));
        }
        let affordable = (budget_bytes - host_bytes)
            .checked_div(lane_bytes)
            .unwrap_or(threads.get() as u64);
        let lanes =
            NonZeroUsize::new(affordable.min(threads.get() as u64) as usize).unwrap_or(threads);
        let cells = usize::try_from(cells).map_err(|_| unrepresentable())?;
        let accumulators = usize::try_from(accumulators).map_err(|_| unrepresentable())?;
        Ok(MomentsPlan {
            accumulators,
            cells,
            stride,
            lanes,
            estimated_peak_bytes: lane_bytes * lanes.get() as u64 + host_bytes,
        })
    }

    /// `i128` accumulators per cell.
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// Cells per category.
    pub fn cells(&self) -> usize {
        self.cells
    }
}

/// The cell reducer: exact `i128` accumulators per category and cell.
pub struct CellReducer<'a> {
    input: MomentsInput<'a>,
    plan: MomentsPlan,
    /// Category index to its slot among the requested categories.
    slot: [usize; N_CATEGORIES],
}

impl<'a> CellReducer<'a> {
    pub fn new(
        input: MomentsInput<'a>,
        requested: CategorySet,
        plan: MomentsPlan,
    ) -> CellReducer<'a> {
        let mut slot = [0usize; N_CATEGORIES];
        for (i, cat) in requested.iter().enumerate() {
            slot[cat.index()] = i;
        }
        CellReducer { input, plan, slot }
    }

    /// Check the merged lane and encode it as the host receives it, `(width,
    /// bytes)`: per requested category in registry order, per cell in
    /// mixed-radix order (first label, second label, then the equality bits
    /// with the first key most significant), `stride` integers: the pair
    /// count; the sums of the first member's columns, of the second's, of
    /// their squares in the same order, and the cross sums per product.
    ///
    /// # Errors
    ///
    /// [`Error::ArithmeticOverflow`] when a cell holds more than
    /// [`MAX_CELL_PAIRS`] pairs, so a partial may have wrapped;
    /// [`Error::AllocationFailed`] for the output.
    pub fn finish(&self, lane: &[i128]) -> Result<(usize, Vec<u8>), Error> {
        if lane
            .chunks_exact(self.plan.stride)
            .any(|acc| acc[0] > MAX_CELL_PAIRS)
        {
            return Err(Error::ArithmeticOverflow {
                operation: OPERATION,
                dtype: "int128",
            });
        }
        super::moments_table::encode_i128(lane, Family::MomentOutput)
    }
}

impl Reducer for CellReducer<'_> {
    type Lane = Vec<i128>;

    fn lane(&self) -> Result<Vec<i128>, Error> {
        alloc::filled(0i128, self.plan.accumulators, Family::MomentLanes)
    }

    #[inline]
    fn reduce(&self, lane: &mut Vec<i128>, cat: Category, first: u32, second: u32) {
        let input = &self.input;
        let (a, b) = (first as usize, second as usize);
        let k = input.n_columns;
        let s = input.n_same;
        let mut cell = input.labels_first[a] as usize * input.n_labels_second
            + input.labels_second[b] as usize;
        for j in 0..s {
            let x = input.same[a * s + j];
            cell = (cell << 1) | usize::from(x >= 0 && x == input.same[b * s + j]);
        }
        let base = (self.slot[cat.index()] * self.plan.cells + cell) * self.plan.stride;
        let acc = &mut lane[base..base + self.plan.stride];
        acc[0] += 1;
        let va = &input.values[a * k..(a + 1) * k];
        let vb = &input.values[b * k..(b + 1) * k];
        let (sum_a, rest) = acc[1..].split_at_mut(k);
        let (sum_b, rest) = rest.split_at_mut(k);
        let (sq_a, rest) = rest.split_at_mut(k);
        let (sq_b, cross) = rest.split_at_mut(k);
        for c in 0..k {
            let (x, y) = (i128::from(va[c]), i128::from(vb[c]));
            sum_a[c] += x;
            sum_b[c] += y;
            sq_a[c] += x * x;
            sq_b[c] += y * y;
        }
        let pick = |(side, col): (Side, usize)| match side {
            Side::First => va[col],
            Side::Second => vb[col],
        };
        for (product, out) in input.products.iter().zip(cross) {
            *out += i128::from(pick(product.a)) * i128::from(pick(product.b));
        }
    }

    fn merge(&self, into: &mut Vec<i128>, from: Vec<i128>) {
        for (x, y) in into.iter_mut().zip(from) {
            *x += y;
        }
    }
}

/// What a moments call hands back.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Moments {
    /// Bytes per accumulator of `table`.
    pub width: usize,
    /// The merged accumulators, laid out as [`CellReducer::finish`]
    /// describes, `width` bytes each, little-endian two's complement.
    pub table: Vec<u8>,
    /// Integers per cell.
    pub stride: usize,
    /// Cells per category.
    pub cells: usize,
    /// The lanes the pass ran on.
    pub lanes: usize,
    /// The pairs each lane reduced, in lane order.
    pub lane_pairs: Vec<u64>,
    /// The planned accumulator peak.
    pub estimated_peak_bytes: u64,
}

/// The receiver's row count: graph rows without a view, view rows with one.
///
/// # Errors
///
/// [`Error::InvalidViewMap`] when `view` is not a partial permutation.
///
/// # Panics
///
/// If `view` does not have one entry per graph row.
pub(super) fn receiver_len(ped: &Pedigree, view: Option<&[i32]>) -> Result<usize, Error> {
    let Some(map) = view else {
        return Ok(ped.len());
    };
    assert_eq!(
        map.len(),
        ped.len(),
        "view map must have one entry per graph row"
    );
    check_view_map(map)?;
    Ok(map
        .iter()
        .copied()
        .filter(|&m| m >= 0)
        .max()
        .map_or(0, |m| m as usize + 1))
}

/// The relationship moments of every requested category, using the current
/// Rayon pool.
///
/// `view` is as for [`super::pair_blocks`]; with `compact` the engine runs
/// on the view's ancestry-compact pedigree, which changes nothing but
/// resource use.  `input` is in receiver rows: graph rows without a view,
/// view rows with one.  `threads` caps the lane count; the budget caps it
/// further.
///
/// # Errors
///
/// [`Error::MemoryBudgetExceeded`] before any accumulator is allocated when
/// one lane does not fit `budget_bytes`; the input errors of
/// [`MomentsInput::check`]; [`Error::InvalidViewMap`] when `view` is not a
/// partial permutation; [`Error::ArithmeticOverflow`] when a cell exceeds
/// [`MAX_CELL_PAIRS`] pairs; [`Error::Cancelled`] once `progress` is
/// cancelled.
///
/// # Panics
///
/// If `view` does not have one entry per graph row.
#[allow(clippy::too_many_arguments)]
pub fn relationship_moments(
    ped: &Pedigree,
    max_degree: MaxDegree,
    requested: CategorySet,
    view: Option<&[i32]>,
    compact: bool,
    input: &MomentsInput<'_>,
    symmetric: Symmetric,
    threads: NonZeroUsize,
    budget_bytes: u64,
    progress: &Progress,
) -> Result<Moments, Error> {
    input.check(receiver_len(ped, view)?)?;
    let plan = MomentsPlan::new(input.shape(requested), threads, budget_bytes)?;
    let reducer = CellReducer::new(*input, requested, plan);
    let reduced = match (view, compact) {
        (Some(map), true) => {
            let compact = CompactView::build(ped, map)?;
            progress.checkpoint(Checkpoint::Compacted)?;
            reduce_pairs(
                &compact.columns.try_borrow()?,
                max_degree,
                requested,
                Some(&compact.view_rows),
                symmetric,
                plan.lanes,
                &reducer,
                progress,
            )?
        }
        _ => reduce_pairs(
            ped, max_degree, requested, view, symmetric, plan.lanes, &reducer, progress,
        )?,
    };
    let (width, table) = reducer.finish(&reduced.lane)?;
    drop(reduced.lane);
    Ok(Moments {
        width,
        table,
        stride: plan.stride,
        cells: plan.cells,
        lanes: plan.lanes.get(),
        lane_pairs: reduced.lane_pairs,
        estimated_peak_bytes: plan.estimated_peak_bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::super::testing::random_pedigree;
    use super::super::{pair_blocks, Execution, PedigreeColumns};
    use super::*;

    /// Brute force from the emitted pair blocks, for the same inputs.
    fn oracle(
        cols: &PedigreeColumns,
        requested: CategorySet,
        view: Option<&[i32]>,
        input: &MomentsInput<'_>,
        symmetric: Symmetric,
    ) -> Vec<i128> {
        let ped = cols.try_borrow().unwrap();
        let blocks = pair_blocks(
            &ped,
            MaxDegree::MAX,
            requested,
            view,
            Execution::Speed,
            &Progress::default(),
        )
        .unwrap();
        let plan = MomentsPlan::new(
            input.shape(requested),
            NonZeroUsize::new(1).unwrap(),
            u64::MAX,
        )
        .unwrap();
        let reducer = CellReducer::new(*input, requested, plan);
        let mut lane = reducer.lane().unwrap();
        for cat in requested.iter() {
            let block = blocks.get(cat);
            for (&a, &b) in block.first.iter().zip(&block.second) {
                reducer.reduce(&mut lane, cat, a as u32, b as u32);
                if symmetric == Symmetric::Both && cat.symmetric() {
                    reducer.reduce(&mut lane, cat, b as u32, a as u32);
                }
            }
        }
        lane
    }

    fn decoded((width, bytes): &(usize, Vec<u8>)) -> Vec<i128> {
        bytes
            .chunks_exact(*width)
            .map(|slot| {
                let fill = if slot[width - 1] & 0x80 == 0 { 0 } else { 0xff };
                let mut wide = [fill; 16];
                wide[..*width].copy_from_slice(slot);
                i128::from_le_bytes(wide)
            })
            .collect()
    }

    fn inputs(n: usize, seed: u64) -> (Vec<i32>, Vec<i32>, Vec<i64>, Vec<i64>) {
        let mut state = seed | 1;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let first: Vec<i32> = (0..n).map(|_| (next() % 3) as i32).collect();
        let second: Vec<i32> = (0..n).map(|_| (next() % 2) as i32).collect();
        let values: Vec<i64> = (0..n * 2).map(|_| (next() % 2001) as i64 - 1000).collect();
        let same: Vec<i64> = (0..n).map(|_| (next() % 5) as i64 - 1).collect();
        (first, second, values, same)
    }

    const PRODUCTS: [Product; 3] = [
        Product {
            a: (Side::First, 0),
            b: (Side::Second, 0),
        },
        Product {
            a: (Side::First, 0),
            b: (Side::First, 1),
        },
        Product {
            a: (Side::Second, 1),
            b: (Side::Second, 1),
        },
    ];

    #[test]
    fn halves_round_trip_every_sign_and_magnitude() {
        let (first, second, values, same) = inputs(2, 1);
        let input = MomentsInput {
            labels_first: &first,
            n_labels_first: 3,
            labels_second: &second,
            n_labels_second: 2,
            values: &values,
            n_columns: 2,
            products: &[],
            same: &same,
            n_same: 0,
        };
        let requested: CategorySet = [Category::FS].into_iter().collect();
        let plan = MomentsPlan::new(
            input.shape(requested),
            NonZeroUsize::new(1).unwrap(),
            u64::MAX,
        )
        .unwrap();
        let reducer = CellReducer::new(input, requested, plan);
        let mut lane = reducer.lane().unwrap();
        let probes = [
            0i128,
            -1,
            1 << 63,
            -(1 << 63),
            i128::MAX,
            i128::MIN,
            1 << 126,
            -(1 << 100) + 7,
        ];
        for (slot, &value) in lane.iter_mut().skip(1).zip(&probes) {
            *slot = value;
        }
        let encoded = reducer.finish(&lane).unwrap();
        assert_eq!(encoded.0, 16);
        assert_eq!(decoded(&encoded), lane);
    }

    #[test]
    fn moments_equal_the_pair_block_oracle_on_graphs_and_views_for_every_lane_count() {
        let cols = random_pedigree(3000, 5);
        let ped = cols.try_borrow().unwrap();
        let n = ped.len();
        let view: Vec<i32> = (0..n as i32)
            .map(|r| {
                if r % 3 == 0 {
                    -1
                } else {
                    r / 3 * 2 + r % 3 - 1
                }
            })
            .collect();
        let n_view = view.iter().filter(|&&v| v >= 0).count();
        let requested = CategorySet::up_to_degree(5);
        for (v, len) in [(None, n), (Some(view.as_slice()), n_view)] {
            let (first, second, values, same) = inputs(len, 9);
            let input = MomentsInput {
                labels_first: &first,
                n_labels_first: 3,
                labels_second: &second,
                n_labels_second: 2,
                values: &values,
                n_columns: 2,
                products: &PRODUCTS,
                same: &same,
                n_same: 1,
            };
            for symmetric in [Symmetric::Canonical, Symmetric::Both] {
                let want = oracle(&cols, requested, v, &input, symmetric);
                let total: u64 = want
                    .chunks_exact(1 + 4 * 2 + 3)
                    .map(|acc| acc[0] as u64)
                    .sum();
                for (threads, compact) in [(1, false), (4, false), (4, true)] {
                    if compact && v.is_none() {
                        continue;
                    }
                    let pool = rayon::ThreadPoolBuilder::new()
                        .num_threads(threads)
                        .build()
                        .unwrap();
                    let got = pool
                        .install(|| {
                            relationship_moments(
                                &ped,
                                MaxDegree::MAX,
                                requested,
                                v,
                                compact,
                                &input,
                                symmetric,
                                NonZeroUsize::new(threads).unwrap(),
                                u64::MAX,
                                &Progress::default(),
                            )
                        })
                        .unwrap();
                    assert_eq!(got.lanes, threads);
                    assert_eq!(got.lane_pairs.len(), threads);
                    assert_eq!(got.lane_pairs.iter().sum::<u64>(), total);
                    assert!(
                        decoded(&(got.width, got.table.clone())) == want,
                        "threads={threads} compact={compact} {symmetric:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn the_budget_bounds_the_lanes_and_refuses_a_single_lane_that_does_not_fit() {
        let shape = MomentsShape {
            n_categories: 2,
            n_labels_first: 3,
            n_labels_second: 2,
            n_columns: 2,
            n_products: 1,
            n_same: 1,
        };
        // 2 categories × 12 cells × (1 + 8 + 1) accumulators.
        let accumulators = 2 * 12 * 10;
        let lane = accumulators * 16;
        let host = host_bytes(accumulators).unwrap();
        let eight = NonZeroUsize::new(8).unwrap();
        let plan = MomentsPlan::new(shape, eight, u64::MAX).unwrap();
        assert_eq!(plan.lanes.get(), 8);
        assert_eq!(plan.estimated_peak_bytes, 8 * lane + host);
        let plan = MomentsPlan::new(shape, eight, 3 * lane + host + 7).unwrap();
        assert_eq!(plan.lanes.get(), 3);
        assert_eq!(plan.estimated_peak_bytes, 3 * lane + host);
        let plan = MomentsPlan::new(shape, eight, lane + host).unwrap();
        assert_eq!(plan.lanes.get(), 1);
        assert_eq!(
            MomentsPlan::new(shape, eight, lane + host - 1).unwrap_err(),
            Error::MemoryBudgetExceeded {
                operation: "relationship_moments",
                estimated_bytes: lane + host,
                budget_bytes: lane + host - 1,
            }
        );
        let huge = MomentsShape {
            n_labels_first: 1 << 40,
            n_labels_second: 1 << 40,
            ..shape
        };
        assert!(matches!(
            MomentsPlan::new(huge, eight, u64::MAX).unwrap_err(),
            Error::MemoryBudgetExceeded {
                estimated_bytes: u64::MAX,
                ..
            }
        ));
    }

    /// `(1 × 1) << n_same` once wrapped to zero cells and the reducer indexed
    /// past its lane; every size is now a checked product.
    #[test]
    fn many_equality_keys_are_refused_not_wrapped() {
        let eight = NonZeroUsize::new(8).unwrap();
        for n_same in [40usize, 62, 63, 64, 100] {
            let shape = MomentsShape {
                n_categories: 1,
                n_labels_first: 1,
                n_labels_second: 1,
                n_columns: 0,
                n_products: 0,
                n_same,
            };
            let err = MomentsPlan::new(shape, eight, 1 << 30).unwrap_err();
            let Error::MemoryBudgetExceeded {
                estimated_bytes, ..
            } = err
            else {
                panic!("n_same={n_same}: {err:?}")
            };
            if n_same == 40 {
                assert_eq!(
                    estimated_bytes,
                    (1 << 40) * 16 + host_bytes(1 << 40).unwrap()
                );
            } else {
                assert_eq!(estimated_bytes, u64::MAX, "n_same={n_same}");
            }
        }
    }

    #[test]
    fn a_cell_past_the_proven_pair_count_is_an_overflow_error() {
        let (first, second, values, same) = inputs(2, 1);
        let input = MomentsInput {
            labels_first: &first,
            n_labels_first: 3,
            labels_second: &second,
            n_labels_second: 2,
            values: &values,
            n_columns: 2,
            products: &[],
            same: &same,
            n_same: 0,
        };
        let requested: CategorySet = [Category::FS].into_iter().collect();
        let plan = MomentsPlan::new(
            input.shape(requested),
            NonZeroUsize::new(1).unwrap(),
            u64::MAX,
        )
        .unwrap();
        let reducer = CellReducer::new(input, requested, plan);
        let mut lane = reducer.lane().unwrap();
        lane[0] = MAX_CELL_PAIRS;
        assert!(reducer.finish(&lane).is_ok());
        lane[0] = MAX_CELL_PAIRS + 1;
        assert_eq!(
            reducer.finish(&lane).unwrap_err(),
            Error::ArithmeticOverflow {
                operation: "relationship_moments",
                dtype: "int128",
            }
        );
    }

    #[test]
    fn labels_values_and_products_are_checked_against_their_ranges() {
        let base = MomentsInput {
            labels_first: &[0, 2],
            n_labels_first: 3,
            labels_second: &[1, 0],
            n_labels_second: 2,
            values: &[1, 2, 3, 4],
            n_columns: 2,
            products: &[],
            same: &[],
            n_same: 0,
        };
        assert!(base.check(2).is_ok());
        let bad_label = MomentsInput {
            labels_second: &[1, 2],
            ..base
        };
        assert_eq!(
            bad_label.check(2).unwrap_err(),
            Error::ValueOutOfRange {
                field: "labels_second",
                position: 1,
                value: 2,
                minimum: 0,
                maximum: 1,
            }
        );
        let too_big = MomentsInput {
            values: &[1, 2, 3, MAX_QUANTIZED + 1],
            ..base
        };
        assert!(matches!(
            too_big.check(2).unwrap_err(),
            Error::ValueOutOfRange {
                field: "values",
                position: 3,
                ..
            }
        ));
        let most_negative = MomentsInput {
            values: &[1, i64::MIN, 3, 4],
            ..base
        };
        assert!(matches!(
            most_negative.check(2).unwrap_err(),
            Error::ValueOutOfRange {
                field: "values",
                position: 1,
                ..
            }
        ));
        let bad_product = MomentsInput {
            products: &[
                Product {
                    a: (Side::First, 0),
                    b: (Side::Second, 1),
                },
                Product {
                    a: (Side::First, 2),
                    b: (Side::Second, 0),
                },
            ],
            ..base
        };
        assert_eq!(
            bad_product.check(2).unwrap_err(),
            Error::ValueOutOfRange {
                field: "products",
                position: 1,
                value: 2,
                minimum: 0,
                maximum: 1,
            }
        );
        let short = MomentsInput {
            labels_first: &[0],
            ..base
        };
        assert!(matches!(
            short.check(2).unwrap_err(),
            Error::LengthMismatch {
                field: "labels_first",
                ..
            }
        ));
    }

    /// Every lane the pass touches is one it allocated up front: the reducer
    /// counts lane builds, and the count is the planned `W`.
    struct CountingReducer {
        built: AtomicUsize,
        pairs: AtomicUsize,
    }

    impl Reducer for CountingReducer {
        type Lane = u64;
        fn lane(&self) -> Result<u64, Error> {
            self.built.fetch_add(1, Ordering::SeqCst);
            Ok(0)
        }
        fn reduce(&self, lane: &mut u64, _: Category, _: u32, _: u32) {
            *lane += 1;
            self.pairs.fetch_add(1, Ordering::SeqCst);
        }
        fn merge(&self, into: &mut u64, from: u64) {
            *into += from;
        }
    }

    #[test]
    fn the_pass_allocates_exactly_the_planned_lanes_and_several_of_them_work() {
        let cols = random_pedigree(3 * super::super::ROWS_PER_TASK + 100, 3);
        let ped = cols.try_borrow().unwrap();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(8)
            .build()
            .unwrap();
        let requested = CategorySet::up_to_degree(3);
        let total = pair_blocks(
            &ped,
            MaxDegree::MAX,
            requested,
            None,
            Execution::Speed,
            &Progress::default(),
        )
        .unwrap()
        .total();
        for lanes in [1usize, 3, 8] {
            let reducer = CountingReducer {
                built: AtomicUsize::new(0),
                pairs: AtomicUsize::new(0),
            };
            let reduced = pool
                .install(|| {
                    reduce_pairs(
                        &ped,
                        MaxDegree::MAX,
                        requested,
                        None,
                        Symmetric::Canonical,
                        NonZeroUsize::new(lanes).unwrap(),
                        &reducer,
                        &Progress::default(),
                    )
                })
                .unwrap();
            assert_eq!(reducer.built.load(Ordering::SeqCst), lanes);
            assert_eq!(reduced.lane as usize, total);
            assert_eq!(reducer.pairs.load(Ordering::SeqCst), total);
            assert_eq!(reduced.lane_pairs.len(), lanes);
            assert_eq!(reduced.lane_pairs.iter().sum::<u64>() as usize, total);
            if lanes > 1 {
                assert!(
                    reduced.lane_pairs.iter().filter(|&&p| p > 0).count() >= 2,
                    "lanes={lanes}: {:?}",
                    reduced.lane_pairs
                );
            }
        }
    }
}
