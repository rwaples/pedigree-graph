//! Per-person relative counts: for each receiver row and requested category,
//! how many relatives the row has, and how many of them pass each threshold
//! column.
//!
//! One pass of [`reduce_pairs`] with [`Symmetric::Both`] credits the `first`
//! member of every oriented pair, so a symmetric category credits both
//! members and a directional one only its first (junior) member.  Every lane
//! adds into one shared `[rows][categories][1 + K]` array of relaxed
//! `AtomicU32` counters, so the lanes carry nothing and there is nothing to
//! merge; integer adds commute, so the counts are the same for every lane
//! and thread count.  A row's relatives in one category are distinct rows,
//! fewer than `2^32`, so no counter can wrap.

use super::category::{Category, CategorySet, N_CATEGORIES};
use super::moments::{receiver_len, reduce_pairs, Reducer, Symmetric};
use super::progress::Progress;
use super::{check_column_length, on_receiver, MaxDegree, Pedigree, Receiver};
use crate::alloc::{self, Family};
use crate::error::Error;
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicU32, Ordering};

/// The credited member's side of a threshold column.
#[derive(Clone, Copy, Debug)]
pub enum Threshold<'a> {
    /// One threshold for every row, never broadcast into an array.
    Scalar(f64),
    /// One threshold per receiver row.
    Rows(&'a [f64]),
}

impl Threshold<'_> {
    #[inline]
    fn at(&self, row: usize) -> f64 {
        match *self {
            Threshold::Scalar(t) => t,
            Threshold::Rows(rows) => rows[row],
        }
    }
}

/// One count column: a relative counts when `relative[relative row] <=
/// threshold[credited row]`, compared as IEEE floats, so NaN on either side
/// never counts.  Both sides are borrowed in receiver rows, and two columns
/// may borrow the same slice.
#[derive(Clone, Copy, Debug)]
pub struct ThresholdColumn<'a> {
    pub relative: &'a [f64],
    pub threshold: Threshold<'a>,
}

/// Credits the first member of each pair in a shared counter array.
struct PersonReducer<'a> {
    /// `[receiver row][requested category slot][1 + K]`.
    counts: &'a [AtomicU32],
    columns: &'a [ThresholdColumn<'a>],
    n_categories: usize,
    /// Category index to its slot among the requested categories.
    slot: [usize; N_CATEGORIES],
}

impl Reducer for PersonReducer<'_> {
    type Lane = ();

    fn lane(&self) -> Result<(), Error> {
        Ok(())
    }

    #[inline]
    fn reduce(&self, _: &mut (), cat: Category, first: u32, second: u32) {
        let (first, second) = (first as usize, second as usize);
        let base = (first * self.n_categories + self.slot[cat.index()]) * (1 + self.columns.len());
        self.counts[base].fetch_add(1, Ordering::Relaxed);
        for (k, column) in self.columns.iter().enumerate() {
            if column.relative[second] <= column.threshold.at(first) {
                self.counts[base + 1 + k].fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    fn merge(&self, _: &mut (), _: ()) {}
}

/// What a per-person call hands back.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RelativesPerPerson {
    /// Row-major `[rows][n_categories][stride]`: per receiver row and
    /// requested category in registry order, the relative count, then the
    /// count passing each threshold column in the caller's order.
    pub counts: Vec<u32>,
    /// Receiver rows.
    pub rows: usize,
    /// Requested categories.
    pub n_categories: usize,
    /// Counts per row and category: one plus the threshold columns.
    pub stride: usize,
    /// The lanes the pass ran on.
    pub lanes: usize,
    /// The oriented pairs each lane credited, in lane order.
    pub lane_pairs: Vec<u64>,
}

const OPERATION: &str = "relative_counts";

/// Every requested category's relatives of every receiver row, using the
/// current Rayon pool.
///
/// `receiver` is as for [`super::pair_blocks`], and every column is in its
/// rows.  The pass runs on `threads` lanes.
///
/// # Errors
///
/// [`Error::InvalidViewMap`] when the view map is not a partial permutation;
/// [`Error::LengthMismatch`] when a column's `relative`, or its
/// `threshold` rows, do not have one entry per receiver row;
/// [`Error::ArithmeticOverflow`] when the counter array's length is not
/// representable; [`Error::AllocationFailed`] for the counters or from the
/// engine; [`Error::Cancelled`] once `progress` is cancelled.
///
/// # Panics
///
/// If the view map does not have one entry per graph row.
pub fn relatives_per_person(
    ped: &Pedigree,
    max_degree: MaxDegree,
    requested: CategorySet,
    receiver: Receiver<'_>,
    columns: &[ThresholdColumn<'_>],
    threads: NonZeroUsize,
    progress: &Progress,
) -> Result<RelativesPerPerson, Error> {
    let rows = receiver_len(ped, receiver)?;
    for column in columns {
        check_column_length("relative", column.relative.len(), rows)?;
        if let Threshold::Rows(threshold) = column.threshold {
            check_column_length("threshold", threshold.len(), rows)?;
        }
    }
    let n_categories = requested.iter().count();
    let stride = 1 + columns.len();
    let len = rows
        .checked_mul(n_categories)
        .and_then(|c| c.checked_mul(stride))
        .ok_or(Error::ArithmeticOverflow {
            operation: OPERATION,
            dtype: "uint32",
        })?;
    let mut counts = alloc::with_capacity(len, Family::RelativeCounts)?;
    counts.extend((0..len).map(|_| AtomicU32::new(0)));
    let mut slot = [0usize; N_CATEGORIES];
    for (i, cat) in requested.iter().enumerate() {
        slot[cat.index()] = i;
    }
    let reducer = PersonReducer {
        counts: &counts,
        columns,
        n_categories,
        slot,
    };
    let reduced = on_receiver(ped, receiver, progress, |ped, view| {
        reduce_pairs(
            ped,
            max_degree,
            requested,
            view,
            Symmetric::Both,
            threads,
            &reducer,
            progress,
        )
    })?;
    Ok(RelativesPerPerson {
        counts: into_counts(counts),
        rows,
        n_categories,
        stride,
        lanes: threads.get(),
        lane_pairs: reduced.lane_pairs,
    })
}

/// The settled counters as plain integers.  `AtomicU32` and `u32` share
/// size and alignment, so std's in-place collect reuses the allocation
/// rather than doubling the peak; a unit test holds it to that.
fn into_counts(counts: Vec<AtomicU32>) -> Vec<u32> {
    counts.into_iter().map(AtomicU32::into_inner).collect()
}

#[cfg(test)]
mod tests {
    use super::super::testing::{pedigree, random_pedigree};
    use super::super::{pair_blocks, Execution, PedigreeColumns, ROWS_PER_TASK};
    use super::*;

    fn xorshift(seed: u64) -> impl FnMut() -> u64 {
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        }
    }

    /// Values in `[0, 1)` with about one in sixteen each NaN, `+inf`, `-inf`.
    fn floats(n: usize, seed: u64) -> Vec<f64> {
        let mut next = xorshift(seed);
        (0..n)
            .map(|_| match next() % 16 {
                0 => f64::NAN,
                1 => f64::INFINITY,
                2 => f64::NEG_INFINITY,
                _ => (next() % 1000) as f64 / 1000.0,
            })
            .collect()
    }

    /// Half the graph rows, in a shuffled view order.
    fn partial_view(n: usize, seed: u64) -> Vec<i32> {
        let mut next = xorshift(seed);
        let mut order: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            order.swap(i, next() as usize % (i + 1));
        }
        let mut view = vec![-1i32; n];
        for (v, &row) in order.iter().take(n / 2).enumerate() {
            view[row] = v as i32;
        }
        view
    }

    /// K = 0, 1, 3, 3: scalar thresholds of `+inf`, `-inf`, NaN and a finite
    /// value, and two columns sharing one threshold slice.
    fn column_sets<'a>(a: &'a [f64], b: &'a [f64], t: &'a [f64]) -> Vec<Vec<ThresholdColumn<'a>>> {
        let col = |relative, threshold| ThresholdColumn {
            relative,
            threshold,
        };
        vec![
            vec![],
            vec![col(a, Threshold::Scalar(f64::INFINITY))],
            vec![
                col(a, Threshold::Rows(t)),
                col(b, Threshold::Rows(t)),
                col(b, Threshold::Scalar(0.5)),
            ],
            vec![
                col(a, Threshold::Scalar(f64::NEG_INFINITY)),
                col(b, Threshold::Scalar(f64::NAN)),
                col(a, Threshold::Rows(t)),
            ],
        ]
    }

    /// Brute force from the emitted pair blocks: a directional pair credits
    /// its first member, a symmetric one both, each with the other as the
    /// relative.
    fn oracle(
        cols: &PedigreeColumns,
        requested: CategorySet,
        view: Option<&[i32]>,
        columns: &[ThresholdColumn<'_>],
        rows: usize,
    ) -> Vec<u32> {
        let ped = cols.try_borrow().unwrap();
        let blocks = pair_blocks(
            &ped,
            MaxDegree::MAX,
            requested,
            Receiver::from(view),
            Execution::Speed,
            &Progress::default(),
        )
        .unwrap();
        let n_cat = requested.iter().count();
        let stride = 1 + columns.len();
        let mut counts = vec![0u32; rows * n_cat * stride];
        let mut credit = |slot: usize, person: usize, relative: usize| {
            let base = (person * n_cat + slot) * stride;
            counts[base] += 1;
            for (k, column) in columns.iter().enumerate() {
                let threshold = match column.threshold {
                    Threshold::Scalar(t) => t,
                    Threshold::Rows(t) => t[person],
                };
                if column.relative[relative] <= threshold {
                    counts[base + 1 + k] += 1;
                }
            }
        };
        for (slot, cat) in requested.iter().enumerate() {
            let block = blocks.get(cat);
            for (&a, &b) in block.first.iter().zip(&block.second) {
                credit(slot, a as usize, b as usize);
                if cat.symmetric() {
                    credit(slot, b as usize, a as usize);
                }
            }
        }
        counts
    }

    fn run(
        cols: &PedigreeColumns,
        requested: CategorySet,
        view: Option<&[i32]>,
        compact: bool,
        columns: &[ThresholdColumn<'_>],
        threads: usize,
    ) -> RelativesPerPerson {
        let ped = cols.try_borrow().unwrap();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        pool.install(|| {
            relatives_per_person(
                &ped,
                MaxDegree::MAX,
                requested,
                view.map_or(Receiver::Graph, |rows| Receiver::View { rows, compact }),
                columns,
                NonZeroUsize::new(threads).unwrap(),
                &Progress::default(),
            )
        })
        .unwrap()
    }

    fn subset() -> CategorySet {
        [
            Category::MO,
            Category::FS,
            Category::PHS,
            Category::Av,
            Category::C1,
            Category::C1R1,
        ]
        .into_iter()
        .collect()
    }

    #[test]
    fn counts_equal_the_pair_block_oracle_for_every_column_set() {
        for seed in [1, 2, 3] {
            let cols = random_pedigree(2000, seed);
            let n = cols.mother.len();
            let (a, b, t) = (floats(n, seed), floats(n, seed + 10), floats(n, seed + 20));
            for requested in [CategorySet::up_to_degree(MaxDegree::MAX.get()), subset()] {
                let n_cat = requested.iter().count();
                for columns in column_sets(&a, &b, &t) {
                    let got = run(&cols, requested, None, false, &columns, 1);
                    let want = oracle(&cols, requested, None, &columns, n);
                    assert_eq!(
                        (got.rows, got.n_categories, got.stride),
                        (n, n_cat, 1 + columns.len())
                    );
                    assert!(got.counts == want, "seed={seed} K={}", columns.len());
                    let column_total = |k: usize| -> u64 {
                        got.counts
                            .iter()
                            .skip(k)
                            .step_by(got.stride)
                            .map(|&c| u64::from(c))
                            .sum()
                    };
                    let relatives = column_total(0);
                    assert!(relatives > 0);
                    assert_eq!(got.lane_pairs.iter().sum::<u64>(), relatives);
                    for (k, column) in columns.iter().enumerate() {
                        let passing = column_total(1 + k);
                        match column.threshold {
                            Threshold::Scalar(s) if s.is_nan() => assert_eq!(passing, 0),
                            _ => assert!(0 < passing && passing < relatives, "K={k}"),
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn a_reordered_partial_view_matches_the_oracle_with_and_without_compaction() {
        let cols = random_pedigree(3000, 7);
        let n = cols.mother.len();
        let view = partial_view(n, 7);
        let rows = n / 2;
        let (a, b, t) = (floats(rows, 4), floats(rows, 5), floats(rows, 6));
        for requested in [CategorySet::up_to_degree(MaxDegree::MAX.get()), subset()] {
            for columns in column_sets(&a, &b, &t) {
                let want = oracle(&cols, requested, Some(&view), &columns, rows);
                let full = run(&cols, requested, Some(&view), false, &columns, 2);
                let compact = run(&cols, requested, Some(&view), true, &columns, 2);
                assert_eq!(full.rows, rows);
                assert!(full.counts == want, "K={}", columns.len());
                assert!(full.counts == compact.counts, "K={}", columns.len());
                assert_eq!(
                    full.lane_pairs.iter().sum::<u64>(),
                    compact.lane_pairs.iter().sum::<u64>()
                );
            }
        }
    }

    #[test]
    fn counts_are_identical_for_every_lane_count() {
        let cols = random_pedigree(3 * ROWS_PER_TASK + 100, 3);
        let n = cols.mother.len();
        let (a, b, t) = (floats(n, 1), floats(n, 2), floats(n, 3));
        let columns = &column_sets(&a, &b, &t)[2];
        let requested = CategorySet::up_to_degree(MaxDegree::MAX.get());
        let serial = run(&cols, requested, None, false, columns, 1);
        for lanes in [2, 4] {
            let got = run(&cols, requested, None, false, columns, lanes);
            assert!(got.counts == serial.counts, "lanes={lanes}");
            assert_eq!(got.lanes, lanes);
            assert_eq!(got.lane_pairs.len(), lanes);
            assert_eq!(
                got.lane_pairs.iter().sum::<u64>(),
                serial.lane_pairs[0],
                "lanes={lanes}"
            );
            assert!(
                got.lane_pairs.iter().filter(|&&p| p > 0).count() >= 2,
                "lanes={lanes}: {:?}",
                got.lane_pairs
            );
        }
    }

    /// Rows 0 and 1 are the mother and father of full sibs 2 and 3; each row
    /// holds its MO, FO and FS counts.
    #[test]
    fn a_directional_pair_credits_the_child_and_a_symmetric_pair_both_sibs() {
        let cols = pedigree(&[(-1, -1), (-1, -1), (0, 1), (0, 1)], &[]);
        let requested: CategorySet = [Category::MO, Category::FO, Category::FS]
            .into_iter()
            .collect();
        let got = run(&cols, requested, None, false, &[], 1);
        assert_eq!(got.counts, [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]);
    }

    #[test]
    fn the_settled_counts_reuse_the_atomic_allocation() {
        let atomics: Vec<AtomicU32> = (0..1000).map(AtomicU32::new).collect();
        let before = atomics.as_ptr() as usize;
        let counts = into_counts(atomics);
        assert_eq!(counts.as_ptr() as usize, before);
        assert!(counts.iter().copied().eq(0..1000));
    }

    #[test]
    fn columns_and_the_view_are_checked_against_the_receiver() {
        let cols = random_pedigree(100, 1);
        let ped = cols.try_borrow().unwrap();
        let one = NonZeroUsize::new(1).unwrap();
        let requested = subset();
        let call = |view: Option<&[i32]>, columns: &[ThresholdColumn<'_>]| {
            relatives_per_person(
                &ped,
                MaxDegree::MAX,
                requested,
                Receiver::from(view),
                columns,
                one,
                &Progress::default(),
            )
        };
        let (fit, short) = (vec![0.0; 100], vec![0.0; 99]);
        assert!(call(
            None,
            &[ThresholdColumn {
                relative: &fit,
                threshold: Threshold::Rows(&fit),
            }]
        )
        .is_ok());
        assert_eq!(
            call(
                None,
                &[ThresholdColumn {
                    relative: &short,
                    threshold: Threshold::Scalar(0.0),
                }]
            )
            .unwrap_err(),
            Error::LengthMismatch {
                field: "relative",
                expected_length: 100,
                actual_length: 99,
            }
        );
        assert_eq!(
            call(
                None,
                &[ThresholdColumn {
                    relative: &fit,
                    threshold: Threshold::Rows(&short),
                }]
            )
            .unwrap_err(),
            Error::LengthMismatch {
                field: "threshold",
                expected_length: 100,
                actual_length: 99,
            }
        );
        let mut view = vec![-1i32; 100];
        view[3] = 0;
        view[8] = 0;
        assert!(matches!(
            call(Some(&view), &[]).unwrap_err(),
            Error::InvalidViewMap { position: 8, .. }
        ));
    }
}
