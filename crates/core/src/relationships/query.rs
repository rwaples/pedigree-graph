//! The relationship queries a host calls: pairs, counts, moments and
//! relatives per person, each over a requested [`CategorySet`] on a
//! [`Receiver`] (issue #52, ADR 0019).
//!
//! The module owns the wiring every host used to repeat around the engine:
//!
//! * the depth: the engine walks to the requested set's top degree, so a
//!   category can never be requested above the walk;
//! * the call that walks nothing, an empty set or a receiver of fewer than
//!   two rows, whose result each entry point returns without running the
//!   engine ([`walks`] says which calls those are, in O(1), so a host can
//!   skip borrowing or copying its columns for them);
//! * whether a view query runs on the view's ancestry-compact pedigree
//!   ([`Compaction::Auto`]);
//! * the view map, built from the view's graph rows when the engine walks.
//!
//! Results are in the receiver's rows whatever path runs.

use std::num::NonZeroUsize;

use super::category::{CategorySet, Counts};
use super::compact::CompactView;
use super::moments::{moments_pass, reduce_pairs_in, Moments, MomentsInput, RowOrder, Symmetric};
use super::pairs::{emit, Execution, PairBlocks};
use super::progress::{Checkpoint, Progress};
use super::relatives::{check_columns, relatives_pass, RelativesPerPerson, ThresholdColumn};
use super::{count_pairs, run_on, MaxDegree, Pedigree};
use crate::alloc::{self, Family};
use crate::error::Error;

/// Whether a view query runs on the view's ancestry-compact pedigree.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Compaction {
    /// The core decides from the view's size ([`compacts`]).  What a host
    /// passes.
    #[default]
    Auto,
    /// Always the full pedigree.  For tests and benchmarks.
    Full,
    /// Always the compact pedigree.  For tests and benchmarks.
    Compact,
}

/// The graph or a view a query runs on; it fixes the result's coordinates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Receiver<'a> {
    Graph,
    /// `graph_rows[v]` is view row `v`'s graph row: distinct rows of the
    /// graph, in any order.
    View {
        graph_rows: &'a [i32],
        compaction: Compaction,
    },
}

impl<'a> Receiver<'a> {
    /// The view `graph_rows` selects, compacted as the core decides.
    pub fn view(graph_rows: &'a [i32]) -> Receiver<'a> {
        Receiver::View {
            graph_rows,
            compaction: Compaction::Auto,
        }
    }

    /// The receiver's row count on a graph of `graph_len` rows.
    pub fn rows(&self, graph_len: usize) -> usize {
        match self {
            Receiver::Graph => graph_len,
            Receiver::View { graph_rows, .. } => graph_rows.len(),
        }
    }
}

/// Whether a query of `requested` on a receiver of `rows` rows has pairs to
/// walk: false for an empty set or fewer than two rows.  Each entry point
/// returns its degenerate result for those calls without running the
/// engine; a host calls this first only to skip borrowing or copying the
/// columns such a call never reads.
pub fn walks(requested: CategorySet, rows: usize) -> bool {
    depth(requested, rows).is_some()
}

/// The degree the engine walks to, or `None` when there is nothing to walk.
fn depth(requested: CategorySet, rows: usize) -> Option<MaxDegree> {
    // Every category's degree is at most `MaxDegree::MAX`, so the top
    // degree of a set is always one the engine counts.
    requested.top_degree().filter(|_| rows >= 2).map(MaxDegree)
}

/// The sparse-view rule: a view of at most a tenth of a graph of 20 000
/// rows or more runs on its compact pedigree.  Measured on `random_30k`:
/// views through 10% were faster with less peak RSS, a 50% view had no
/// reliable RSS advantage, and small graphs do not carry enough engine
/// setup to pay for the extra closure build.
pub fn compacts(graph_rows: usize, view_rows: usize) -> bool {
    graph_rows >= 20_000 && view_rows.saturating_mul(10) <= graph_rows
}

/// The view a walking query runs on: its map from graph rows to view rows,
/// and whether it runs compact.
struct View {
    map: Vec<i32>,
    compact: bool,
}

/// The view of `receiver` on `ped`, or `None` for the graph.
fn view_of(ped: &Pedigree, receiver: Receiver<'_>) -> Result<Option<View>, Error> {
    let Receiver::View {
        graph_rows,
        compaction,
    } = receiver
    else {
        return Ok(None);
    };
    Ok(Some(View {
        map: view_map(graph_rows, ped.len())?,
        compact: compaction.compacts(ped.len(), graph_rows.len()),
    }))
}

impl Compaction {
    /// Whether a view of `view_rows` on a graph of `graph_rows` runs compact.
    fn compacts(self, graph_rows: usize, view_rows: usize) -> bool {
        match self {
            Compaction::Auto => compacts(graph_rows, view_rows),
            Compaction::Full => false,
            Compaction::Compact => true,
        }
    }
}

/// The graph-row → view-row map of the view `graph_rows` selects on a
/// graph of `n` rows, `-1` where unselected.
///
/// # Errors
///
/// [`Error::ValueOutOfRange`] on `graph_rows` at the first entry that is
/// not a graph row; [`Error::InvalidViewMap`] at the graph row selected a
/// second time, with the view row that repeats it;
/// [`Error::AllocationFailed`] for the map.
pub fn view_map(graph_rows: &[i32], n: usize) -> Result<Vec<i32>, Error> {
    let mut map = alloc::filled(-1i32, n, Family::ViewSortScratch)?;
    // Fewer view rows than graph rows, and the graph's rows are int32.
    select(graph_rows, n, |row, view_row| {
        std::mem::replace(&mut map[row], view_row as i32) >= 0
    })?;
    Ok(map)
}

/// The rows `graph_rows` selects on a graph of `n` rows, as the engine's
/// count filter; the errors of [`view_map`] without building the map.
fn selected(graph_rows: &[i32], n: usize) -> Result<Vec<bool>, Error> {
    let mut mask = alloc::filled(false, n, Family::RowSet)?;
    select(graph_rows, n, |row, _| {
        std::mem::replace(&mut mask[row], true)
    })?;
    Ok(mask)
}

/// Mark each graph row of `graph_rows` with `mark(row, view_row)`, which
/// says whether the row was already marked; stop at the first entry that
/// is not a graph row or is marked twice.
fn select(
    graph_rows: &[i32],
    n: usize,
    mut mark: impl FnMut(usize, usize) -> bool,
) -> Result<(), Error> {
    let maximum = n as i64 - 1;
    for (view_row, &row) in graph_rows.iter().enumerate() {
        if row < 0 || i64::from(row) > maximum {
            return Err(Error::ValueOutOfRange {
                field: "graph_rows",
                position: view_row.into(),
                value: i64::from(row),
                minimum: 0,
                maximum,
            });
        }
        if mark(row as usize, view_row) {
            return Err(Error::InvalidViewMap {
                position: (row as usize).into(),
                value: view_row.into(),
                reason: "a graph row is selected twice",
            });
        }
    }
    Ok(())
}

/// The oriented pairs of every requested category, using the current Rayon
/// pool.  On a view they are in view rows, filtered to pairs with both rows
/// selected and sorted by the view-space canonical key; on the graph they
/// are in graph rows and canonical-key order (ADR 0006, ADR 0010).
///
/// # Errors
///
/// The `graph_rows` errors of [`view_map`]; [`Error::AllocationFailed`]
/// from any buffer the pedigree or its pairs size; [`Error::Cancelled`]
/// once `progress` is cancelled.  [`Execution::Memory`] walks every row
/// twice, so its progress total is twice the row count.
pub fn pairs(
    ped: &Pedigree,
    requested: CategorySet,
    receiver: Receiver<'_>,
    execution: Execution,
    progress: &Progress,
) -> Result<PairBlocks, Error> {
    let view = view_of(ped, receiver)?;
    let Some(max_degree) = depth(requested, receiver.rows(ped.len())) else {
        return Ok(PairBlocks::empty());
    };
    let (map, compact) = split(&view);
    run_on(ped, map, compact, progress, |ped, view| {
        emit(ped, max_degree, requested, view, execution, progress)
    })
}

/// Exact closest-category pair counts, using the current Rayon pool.  On a
/// view only pairs with both rows selected count; classification still runs
/// through every row the engine walks, so unselected relatives keep
/// connecting the selected ones (ADR 0006).  Categories above the requested
/// set's top degree are zero; a host reports only the requested ones.
///
/// # Errors
///
/// The `graph_rows` errors of [`view_map`]; [`Error::AllocationFailed`]
/// from the engine, a workspace or a row set; [`Error::Cancelled`] once
/// `progress` is cancelled.
pub fn counts(
    ped: &Pedigree,
    requested: CategorySet,
    receiver: Receiver<'_>,
    progress: &Progress,
) -> Result<Counts, Error> {
    let n = ped.len();
    let Receiver::View {
        graph_rows,
        compaction,
    } = receiver
    else {
        return match depth(requested, n) {
            Some(max_degree) => count_pairs(ped, max_degree, None, progress),
            None => Ok(Counts::default()),
        };
    };
    if !compaction.compacts(n, graph_rows.len()) {
        // The full path filters by a row mask, so the map is never built.
        let selected = selected(graph_rows, n)?;
        return match depth(requested, graph_rows.len()) {
            Some(max_degree) => count_pairs(ped, max_degree, Some(&selected), progress),
            None => Ok(Counts::default()),
        };
    }
    let map = view_map(graph_rows, n)?;
    let Some(max_degree) = depth(requested, graph_rows.len()) else {
        return Ok(Counts::default());
    };
    let compact = CompactView::build(ped, &map)?;
    drop(map);
    progress.checkpoint(Checkpoint::Compacted)?;
    let mut selected = alloc::with_capacity(compact.view_rows.len(), Family::RowSet)?;
    selected.extend(compact.view_rows.iter().map(|&row| row >= 0));
    count_pairs(
        &compact.columns.try_borrow()?,
        max_degree,
        Some(&selected),
        progress,
    )
}

/// The relationship moments of every requested category (ADR 0013), using
/// the current Rayon pool.  `input` is in the receiver's rows; `threads`
/// caps the lane count and the budget caps it further.
///
/// # Errors
///
/// The `graph_rows` errors of [`view_map`]; the input errors of
/// [`MomentsInput::check`]; [`Error::MemoryBudgetExceeded`] before any
/// accumulator is allocated when one lane does not fit `budget_bytes`, a
/// call that walks nothing included; [`Error::ArithmeticOverflow`] when a
/// cell exceeds [`super::MAX_CELL_PAIRS`] pairs; [`Error::Cancelled`] once
/// `progress` is cancelled.
#[allow(clippy::too_many_arguments)]
pub fn moments(
    ped: &Pedigree,
    requested: CategorySet,
    receiver: Receiver<'_>,
    input: &MomentsInput<'_>,
    symmetric: Symmetric,
    threads: NonZeroUsize,
    budget_bytes: u64,
    progress: &Progress,
) -> Result<Moments, Error> {
    let view = view_of(ped, receiver)?;
    let rows = receiver.rows(ped.len());
    input.check(rows)?;
    let Some(max_degree) = depth(requested, rows) else {
        return Moments::degenerate(input.shape(requested), threads, budget_bytes);
    };
    let (map, compact) = split(&view);
    moments_pass(requested, input, threads, budget_bytes, |reducer, lanes| {
        run_on(ped, map, compact, progress, |ped, view| {
            reduce_pairs_in(
                ped,
                max_degree,
                requested,
                view,
                symmetric,
                lanes,
                reducer,
                progress,
                RowOrder::Families,
            )
        })
    })
}

/// Every requested category's relatives of every receiver row (ADR 0014),
/// using the current Rayon pool, on `threads` lanes.  Every column is in
/// the receiver's rows.
///
/// # Errors
///
/// The `graph_rows` errors of [`view_map`]; [`Error::LengthMismatch`] when
/// a column's `relative`, or its `threshold` rows, do not have one entry
/// per receiver row; [`Error::ArithmeticOverflow`] when the counter array's
/// length is not representable; [`Error::AllocationFailed`] for the
/// counters or from the engine; [`Error::Cancelled`] once `progress` is
/// cancelled.
pub fn relatives(
    ped: &Pedigree,
    requested: CategorySet,
    receiver: Receiver<'_>,
    columns: &[ThresholdColumn<'_>],
    threads: NonZeroUsize,
    progress: &Progress,
) -> Result<RelativesPerPerson, Error> {
    let view = view_of(ped, receiver)?;
    let rows = receiver.rows(ped.len());
    check_columns(rows, columns)?;
    let Some(max_degree) = depth(requested, rows) else {
        return RelativesPerPerson::zeros(rows, requested.iter().count(), 1 + columns.len());
    };
    let (map, compact) = split(&view);
    relatives_pass(rows, requested, columns, threads, |reducer| {
        run_on(ped, map, compact, progress, |ped, view| {
            reduce_pairs_in(
                ped,
                max_degree,
                requested,
                view,
                Symmetric::Both,
                threads,
                reducer,
                progress,
                RowOrder::Families,
            )
        })
    })
}

/// The map and compaction flag [`run_on`] takes.
fn split(view: &Option<View>) -> (Option<&[i32]>, bool) {
    match view {
        Some(view) => (Some(&view.map), view.compact),
        None => (None, false),
    }
}

#[cfg(test)]
mod tests {
    use super::super::testing::random_pedigree;
    use super::super::{
        count_view_pairs_compact, pair_blocks, relationship_moments, relatives_per_person,
        Category, MomentsPlan, Snapshot, Threshold, MAX_QUANTIZED,
    };
    use super::*;

    const MODES: [Compaction; 3] = [Compaction::Auto, Compaction::Full, Compaction::Compact];
    const ONE: NonZeroUsize = NonZeroUsize::MIN;

    /// A fixed-seed shuffled selection of `k` of `n` graph rows.
    fn selection(n: usize, k: usize, seed: u64) -> Vec<i32> {
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        let mut order: Vec<i32> = (0..n as i32).collect();
        for i in (1..n).rev() {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            order.swap(i, state as usize % (i + 1));
        }
        order.truncate(k);
        order
    }

    fn view(graph_rows: &[i32], compaction: Compaction) -> Receiver<'_> {
        Receiver::View {
            graph_rows,
            compaction,
        }
    }

    /// The same view as the former entry points' receiver.
    fn legacy(map: &[i32], compaction: Compaction) -> super::super::Receiver<'_> {
        super::super::Receiver::View {
            rows: map,
            compact: compaction == Compaction::Compact,
        }
    }

    fn close() -> CategorySet {
        [Category::MZ, Category::FS, Category::MHS, Category::C1]
            .into_iter()
            .collect()
    }

    /// Moments inputs over `n` receiver rows: two first labels, one second,
    /// one value column, one equality key.
    struct Inputs {
        labels: Vec<i32>,
        zeros: Vec<i32>,
        values: Vec<i64>,
        same: Vec<i64>,
    }

    impl Inputs {
        fn new(n: usize) -> Inputs {
            Inputs {
                labels: (0..n as i32).map(|r| r % 2).collect(),
                zeros: vec![0; n],
                values: (0..n as i64).map(|r| (r % 7) - 3).collect(),
                same: (0..n as i64).map(|r| r % 3).collect(),
            }
        }

        fn input(&self) -> MomentsInput<'_> {
            MomentsInput {
                labels_first: &self.labels,
                n_labels_first: 2,
                labels_second: &self.zeros,
                n_labels_second: 1,
                values: &self.values,
                n_columns: 1,
                products: &[],
                same: &self.same,
                n_same: 1,
            }
        }
    }

    /// `lane_pairs` with its total only: which lane takes which task range
    /// depends on scheduling, so only the total is reproducible.
    fn settled_moments(mut m: Moments) -> Moments {
        m.lane_pairs = vec![m.lane_pairs.iter().sum()];
        m
    }

    /// As [`settled_moments`].
    fn settled_relatives(mut r: RelativesPerPerson) -> RelativesPerPerson {
        r.lane_pairs = vec![r.lane_pairs.iter().sum()];
        r
    }

    #[test]
    fn walks_needs_a_category_and_two_rows() {
        let one: CategorySet = [Category::FS].into_iter().collect();
        for (requested, rows, want) in [
            (CategorySet::EMPTY, 0, false),
            (CategorySet::EMPTY, 2, false),
            (CategorySet::EMPTY, 1_000, false),
            (one, 0, false),
            (one, 1, false),
            (one, 2, true),
            (CategorySet::up_to_degree(0), 2, true),
        ] {
            assert_eq!(walks(requested, rows), want, "{requested:?} on {rows} rows");
        }
    }

    #[test]
    fn the_walk_reaches_the_requested_top_degree() {
        for degree in 0..=MaxDegree::MAX.get() {
            let got = depth(CategorySet::up_to_degree(degree), 2).unwrap();
            assert_eq!(got, MaxDegree::try_new(degree).unwrap());
        }
        let c2_only: CategorySet = [Category::C2].into_iter().collect();
        assert_eq!(depth(c2_only, 2), Some(MaxDegree::MAX));
    }

    /// The rule `_relationship_pairs._should_compact_view` applied before
    /// issue #52, restated, on every boundary it has.
    #[test]
    fn auto_compaction_is_the_rule_the_hosts_applied() {
        let python = |n_graph: usize, n_view: usize| n_graph >= 20_000 && n_view * 10 <= n_graph;
        for n_graph in [0usize, 1, 1_000, 19_999, 20_000, 20_001, 30_300, 303_000] {
            let tenth = n_graph / 10;
            for n_view in [
                0,
                1,
                2,
                tenth.saturating_sub(1),
                tenth,
                tenth + 1,
                n_graph / 2,
                n_graph,
            ] {
                assert_eq!(
                    compacts(n_graph, n_view),
                    python(n_graph, n_view),
                    "{n_view} of {n_graph}"
                );
            }
        }
    }

    #[test]
    fn view_map_inverts_the_selection_and_rejects_bad_rows() {
        assert_eq!(view_map(&[2, 0], 4).unwrap(), vec![1, -1, 0, -1]);
        assert_eq!(view_map(&[], 3).unwrap(), vec![-1, -1, -1]);
        assert_eq!(
            selected(&[2, 0], 4).unwrap(),
            vec![true, false, true, false]
        );
        let duplicate = Error::InvalidViewMap {
            position: 3usize.into(),
            value: 2usize.into(),
            reason: "a graph row is selected twice",
        };
        for (rows, value) in [(&[0, 4][..], 4), (&[-1][..], -1)] {
            let out_of_range = Error::ValueOutOfRange {
                field: "graph_rows",
                position: (rows.len() - 1).into(),
                value,
                minimum: 0,
                maximum: 3,
            };
            assert_eq!(view_map(rows, 4).unwrap_err(), out_of_range);
            assert_eq!(selected(rows, 4).unwrap_err(), out_of_range);
        }
        assert_eq!(view_map(&[3, 1, 3], 4).unwrap_err(), duplicate);
        assert_eq!(selected(&[3, 1, 3], 4).unwrap_err(), duplicate);
    }

    /// Counts check the selection on both paths, a call that walks nothing
    /// included; the full path never builds the map.
    #[test]
    fn counts_reject_a_bad_selection_on_every_path() {
        let cols = random_pedigree(50, 4);
        let ped = cols.try_borrow().unwrap();
        let progress = Progress::default();
        for requested in [close(), CategorySet::EMPTY] {
            for mode in MODES {
                assert!(matches!(
                    counts(&ped, requested, view(&[3, 9, 3], mode), &progress),
                    Err(Error::InvalidViewMap { .. })
                ));
                assert!(matches!(
                    counts(&ped, requested, view(&[50], mode), &progress),
                    Err(Error::ValueOutOfRange {
                        field: "graph_rows",
                        ..
                    })
                ));
            }
        }
    }

    /// The results of a call that walks nothing, pinned to literal values
    /// (the form both hosts built before issue #52), not to the
    /// constructors the entry points return.
    #[test]
    fn calls_that_walk_nothing_return_the_hosts_former_results() {
        let cols = random_pedigree(300, 5);
        let ped = cols.try_borrow().unwrap();
        let progress = Progress::default();
        let threads = NonZeroUsize::new(4).unwrap();
        let budget = 1 << 30;
        let (rows0, rows1) = (Vec::new(), vec![7]);
        let calls: [(CategorySet, Receiver<'_>, usize); 4] = [
            (CategorySet::EMPTY, Receiver::Graph, 300),
            (CategorySet::EMPTY, view(&rows1, Compaction::Auto), 1),
            (close(), view(&rows0, Compaction::Compact), 0),
            (close(), view(&rows1, Compaction::Full), 1),
        ];
        for (requested, receiver, rows) in calls {
            let label = format!("{requested:?} on {receiver:?}");
            let n_categories = requested.iter().count();
            let blocks = pairs(&ped, requested, receiver, Execution::Speed, &progress).unwrap();
            assert_eq!(blocks.0.len(), 23, "{label}");
            assert_eq!(blocks.total(), 0, "{label}");
            assert_eq!(
                counts(&ped, requested, receiver, &progress).unwrap(),
                Counts::default(),
                "{label}"
            );

            let inputs = Inputs::new(rows);
            let input = inputs.input();
            let got = moments(
                &ped,
                requested,
                receiver,
                &input,
                Symmetric::Both,
                threads,
                budget,
                &progress,
            )
            .unwrap();
            // Two first labels, one second, one equality key: 4 cells; one
            // value column: 1 + 4 accumulators per cell.
            let (cells, stride) = (2 * 2, 1 + 4);
            let plan = MomentsPlan::new(input.shape(requested), threads, budget).unwrap();
            assert_eq!(
                got,
                Moments {
                    width: 1,
                    table: vec![0u8; n_categories * cells * stride],
                    stride,
                    cells,
                    lanes: 0,
                    lane_pairs: Vec::new(),
                    estimated_peak_bytes: plan.estimated_peak_bytes,
                },
                "{label}"
            );

            let relative = vec![0.5; rows];
            let columns = [ThresholdColumn {
                relative: &relative,
                threshold: Threshold::Scalar(1.0),
            }];
            let got = relatives(&ped, requested, receiver, &columns, ONE, &progress).unwrap();
            assert_eq!(
                got,
                RelativesPerPerson {
                    counts: vec![0u32; rows * n_categories * 2],
                    rows,
                    n_categories,
                    stride: 2,
                    lanes: 0,
                    lane_pairs: Vec::new(),
                },
                "{label}"
            );
        }
    }

    #[test]
    fn a_call_that_walks_nothing_still_checks_its_inputs_and_budget() {
        let cols = random_pedigree(50, 2);
        let ped = cols.try_borrow().unwrap();
        let progress = Progress::default();
        let rows = vec![3];
        let receiver = view(&rows, Compaction::Auto);
        let bad = [60];
        assert!(matches!(
            pairs(
                &ped,
                CategorySet::EMPTY,
                view(&bad, Compaction::Auto),
                Execution::Speed,
                &progress
            ),
            Err(Error::ValueOutOfRange {
                field: "graph_rows",
                ..
            })
        ));

        let inputs = Inputs::new(1);
        let call = |input: &MomentsInput<'_>, requested, budget| {
            moments(
                &ped,
                requested,
                receiver,
                input,
                Symmetric::Both,
                ONE,
                budget,
                &progress,
            )
        };
        assert!(matches!(
            call(&inputs.input(), close(), 0),
            Err(Error::MemoryBudgetExceeded { .. })
        ));
        let out_of_range = [5i32];
        let input = MomentsInput {
            labels_first: &out_of_range,
            ..inputs.input()
        };
        assert!(matches!(
            call(&input, close(), 1 << 30),
            Err(Error::ValueOutOfRange {
                field: "labels_first",
                ..
            })
        ));
        let big = [MAX_QUANTIZED + 1];
        let input = MomentsInput {
            values: &big,
            ..inputs.input()
        };
        assert!(matches!(
            call(&input, CategorySet::EMPTY, 1 << 30),
            Err(Error::ValueOutOfRange {
                field: "values",
                ..
            })
        ));

        let short: Vec<f64> = Vec::new();
        let columns = [ThresholdColumn {
            relative: &short,
            threshold: Threshold::Scalar(0.0),
        }];
        assert!(matches!(
            relatives(&ped, CategorySet::EMPTY, receiver, &columns, ONE, &progress),
            Err(Error::LengthMismatch {
                field: "relative",
                ..
            })
        ));
    }

    /// Every entry point on every receiver and compaction equals the
    /// former entry point on the same view map: identical blocks, counts,
    /// moments tables and per-person counts.
    #[test]
    fn query_results_match_the_former_entry_points_on_every_path() {
        let progress = Progress::default();
        let threads = NonZeroUsize::new(3).unwrap();
        for (n, seed) in [(400usize, 1u64), (2_500, 9)] {
            let cols = random_pedigree(n, seed);
            let ped = cols.try_borrow().unwrap();
            for requested in [close(), CategorySet::up_to_degree(MaxDegree::MAX.get())] {
                let max_degree = depth(requested, n).unwrap();
                for k in [2, n / 10, n / 2, n] {
                    let rows = selection(n, k, seed + k as u64);
                    let map = view_map(&rows, n).unwrap();
                    let mut selected = vec![false; n];
                    for &row in &rows {
                        selected[row as usize] = true;
                    }
                    let want_counts =
                        count_pairs(&ped, max_degree, Some(&selected), &progress).unwrap();
                    assert_eq!(
                        count_view_pairs_compact(&ped, max_degree, &map, &progress).unwrap(),
                        want_counts
                    );
                    let inputs = Inputs::new(k);
                    let input = inputs.input();
                    let relative: Vec<f64> = (0..k).map(|r| (r % 5) as f64).collect();
                    let columns = [ThresholdColumn {
                        relative: &relative,
                        threshold: Threshold::Scalar(2.0),
                    }];
                    for mode in MODES {
                        let label = format!("n={n} k={k} {mode:?} {requested:?}");
                        let (new, old) = (view(&rows, mode), legacy(&map, mode));
                        for execution in [Execution::Speed, Execution::Memory] {
                            assert_eq!(
                                pairs(&ped, requested, new, execution, &progress).unwrap(),
                                pair_blocks(&ped, max_degree, requested, old, execution, &progress)
                                    .unwrap(),
                                "{label} {execution:?}"
                            );
                        }
                        assert_eq!(
                            counts(&ped, requested, new, &progress).unwrap(),
                            want_counts,
                            "{label}"
                        );
                        assert_eq!(
                            settled_moments(
                                moments(
                                    &ped,
                                    requested,
                                    new,
                                    &input,
                                    Symmetric::Canonical,
                                    threads,
                                    1 << 30,
                                    &progress
                                )
                                .unwrap()
                            ),
                            settled_moments(
                                relationship_moments(
                                    &ped,
                                    max_degree,
                                    requested,
                                    old,
                                    &input,
                                    Symmetric::Canonical,
                                    threads,
                                    1 << 30,
                                    &progress,
                                )
                                .unwrap()
                            ),
                            "{label}"
                        );
                        assert_eq!(
                            settled_relatives(
                                relatives(&ped, requested, new, &columns, threads, &progress)
                                    .unwrap()
                            ),
                            settled_relatives(
                                relatives_per_person(
                                    &ped, max_degree, requested, old, &columns, threads, &progress
                                )
                                .unwrap()
                            ),
                            "{label}"
                        );
                    }
                }
                assert_eq!(
                    pairs(
                        &ped,
                        requested,
                        Receiver::Graph,
                        Execution::Speed,
                        &progress
                    )
                    .unwrap(),
                    pair_blocks(
                        &ped,
                        max_degree,
                        requested,
                        super::super::Receiver::Graph,
                        Execution::Speed,
                        &progress
                    )
                    .unwrap()
                );
                assert_eq!(
                    counts(&ped, requested, Receiver::Graph, &progress).unwrap(),
                    count_pairs(&ped, max_degree, None, &progress).unwrap()
                );
            }
        }
    }

    /// The path a view takes shows in progress: the walk's total is the
    /// rows of the pedigree the engine walks.  `Auto` must walk what the
    /// hosts' rule chose before issue #52.
    #[test]
    fn auto_walks_the_compact_pedigree_exactly_when_the_rule_says() {
        let n = 20_000;
        let cols = random_pedigree(n, 3);
        let ped = cols.try_borrow().unwrap();
        let requested: CategorySet = [Category::FS].into_iter().collect();
        let total = |receiver: Receiver<'_>, execution| {
            let progress = Progress::default();
            pairs(&ped, requested, receiver, execution, &progress).unwrap();
            match progress.snapshot() {
                Snapshot::Finishing { total } => total,
                other => panic!("call ended in {other:?}"),
            }
        };
        for k in [2, n / 10, n / 10 + 1, n / 2] {
            let rows = selection(n, k, k as u64);
            let compact = total(view(&rows, Compaction::Compact), Execution::Speed);
            assert_eq!(total(view(&rows, Compaction::Full), Execution::Speed), n);
            assert!(compact < n, "{k}: {compact}");
            let want = if compacts(n, k) { compact } else { n };
            assert_eq!(
                total(view(&rows, Compaction::Auto), Execution::Speed),
                want,
                "{k}"
            );
            assert_eq!(
                total(view(&rows, Compaction::Auto), Execution::Memory),
                2 * want,
                "{k}"
            );
        }
        assert_eq!(total(Receiver::Graph, Execution::Speed), n);
    }
}
