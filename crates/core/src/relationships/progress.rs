//! How far a relationship call has got, and whether its host has cancelled it
//! (ADR 0017).
//!
//! The engine only stores atomics here and checks the cancel flag; it never
//! calls the host.  A host thread polls [`Progress::snapshot`] and sets
//! [`Progress::cancel`], so progress and interrupts work from the thread that
//! owns the host runtime (ADR 0007).
//!
//! A call moves through three phases: preparing (compaction and engine
//! setup, total unknown), walking (row visits), and finishing (whatever runs
//! after the last row: lane merges, block copies, the view sort).  Phase
//! changes are `Release` stores and the observer reads the phase with
//! `Acquire` before the counters, so a walking snapshot always carries the
//! published total and a finishing one comes after every row advance.

use super::Category;
use crate::error::Error;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicUsize, Ordering};

/// Rows between two cancel checks in a row loop.  Task ranges start at
/// multiples of it, so each task checks on its first row and then every
/// `CHECK_EVERY` rows; a cancel waits for at most that many rows per worker.
pub(crate) const CHECK_EVERY: usize = 64;

const PREPARING: u8 = 0;
const WALKING: u8 = 1;
const FINISHING: u8 = 2;

/// What an observer sees; each phase has one shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Snapshot {
    /// The total is not known yet.
    Preparing,
    /// `done <= total` row visits.
    Walking { done: usize, total: usize },
    /// Every row visit is done; the result is being assembled.
    Finishing { total: usize },
}

/// A named place outside the row loops where the engine checks for
/// cancellation.  The work between two of them cannot be interrupted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Checkpoint {
    /// The ancestry-compact pedigree of a view is built.
    Compacted,
    /// The walk is about to start.
    Walk,
    /// Pass one of a two-pass emission has returned.
    BetweenPasses,
    /// Before one category's exact block is allocated and zero-filled.
    CategoryAlloc(Category),
    /// The walk has returned.
    Finish,
    /// Before one category's task chunks are copied into its block.
    CategoryCopy(Category),
    /// Before one category's block is sorted into view order.
    CategorySort(Category),
    /// Before one lane is merged into the total.
    LaneMerge,
}

/// The shared state of one call: the engine writes, the host reads and cancels.
#[derive(Debug, Default)]
pub struct Progress {
    phase: AtomicU8,
    done: AtomicUsize,
    total: AtomicUsize,
    cancelled: AtomicBool,
    /// Trips [`Progress::cancel`] when this checkpoint is reached.
    #[cfg(test)]
    cancel_at: Option<Checkpoint>,
    /// Every checkpoint reached, in order.
    #[cfg(test)]
    reached: std::sync::Mutex<Vec<Checkpoint>>,
}

impl Progress {
    /// The phase and counts as of now.
    pub fn snapshot(&self) -> Snapshot {
        match self.phase.load(Ordering::Acquire) {
            PREPARING => Snapshot::Preparing,
            WALKING => {
                let total = self.total.load(Ordering::Relaxed);
                let done = self.done.load(Ordering::Relaxed).min(total);
                Snapshot::Walking { done, total }
            }
            _ => Snapshot::Finishing {
                total: self.total.load(Ordering::Relaxed),
            },
        }
    }

    /// Ask the call to stop; it returns [`Error::Cancelled`] at the next row
    /// check or checkpoint any worker reaches.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Relaxed);
    }

    /// Publish the number of row visits and start walking.
    pub(crate) fn walk(&self, total: usize) -> Result<(), Error> {
        self.total.store(total, Ordering::Relaxed);
        self.phase.store(WALKING, Ordering::Release);
        self.checkpoint(Checkpoint::Walk)
    }

    /// Count `rows` visited, once per task range.
    pub(crate) fn advance(&self, rows: usize) {
        self.done.fetch_add(rows, Ordering::Relaxed);
    }

    /// Every row is visited; what follows assembles the result.
    pub(crate) fn finish(&self) -> Result<(), Error> {
        self.phase.store(FINISHING, Ordering::Release);
        self.checkpoint(Checkpoint::Finish)
    }

    pub(crate) fn checkpoint(&self, at: Checkpoint) -> Result<(), Error> {
        #[cfg(test)]
        {
            self.reached.lock().unwrap().push(at);
            if self.cancel_at == Some(at) {
                self.cancel();
            }
        }
        #[cfg(not(test))]
        let _ = at;
        self.check()
    }

    /// The row loops' cancel check: reads the flag on every
    /// [`CHECK_EVERY`]th row only.
    #[inline]
    pub(crate) fn check_row(&self, row: usize) -> Result<(), Error> {
        if row % CHECK_EVERY == 0 {
            self.check()
        } else {
            Ok(())
        }
    }

    fn check(&self) -> Result<(), Error> {
        if self.cancelled.load(Ordering::Relaxed) {
            Err(Error::Cancelled)
        } else {
            Ok(())
        }
    }
}

#[cfg(test)]
impl Progress {
    /// A progress that cancels itself on reaching `at`.
    pub(crate) fn cancelling_at(at: Checkpoint) -> Progress {
        Progress {
            cancel_at: Some(at),
            ..Progress::default()
        }
    }

    /// A progress cancelled before the call starts.
    pub(crate) fn cancelled() -> Progress {
        let progress = Progress::default();
        progress.cancel();
        progress
    }

    pub(crate) fn reached(&self) -> Vec<Checkpoint> {
        self.reached.lock().unwrap().clone()
    }

    pub(crate) fn done(&self) -> usize {
        self.done.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::super::testing::random_pedigree;
    use super::super::{count_pairs, MaxDegree, Pedigree};
    use super::*;

    #[test]
    fn snapshots_normalize_each_phase() {
        let progress = Progress::default();
        assert_eq!(progress.snapshot(), Snapshot::Preparing);
        progress.walk(10).unwrap();
        assert_eq!(
            progress.snapshot(),
            Snapshot::Walking { done: 0, total: 10 }
        );
        progress.advance(4);
        progress.advance(6);
        assert_eq!(
            progress.snapshot(),
            Snapshot::Walking {
                done: 10,
                total: 10
            }
        );
        progress.finish().unwrap();
        assert_eq!(progress.snapshot(), Snapshot::Finishing { total: 10 });
        assert_eq!(progress.reached(), [Checkpoint::Walk, Checkpoint::Finish]);
    }

    #[test]
    fn a_cancelled_progress_fails_every_check() {
        let progress = Progress::cancelled();
        assert_eq!(progress.check_row(0), Err(Error::Cancelled));
        assert_eq!(progress.check_row(CHECK_EVERY + 1), Ok(()));
        assert_eq!(progress.walk(1), Err(Error::Cancelled));
        assert_eq!(progress.finish(), Err(Error::Cancelled));
    }

    #[test]
    fn cancel_at_trips_on_that_checkpoint_only() {
        let progress = Progress::cancelling_at(Checkpoint::Finish);
        progress.walk(3).unwrap();
        assert_eq!(progress.check_row(0), Ok(()));
        assert_eq!(progress.finish(), Err(Error::Cancelled));
    }

    /// An observer polling during a real walk sees the phases in order, a
    /// published total while walking, and `done` never decreasing.
    #[test]
    fn a_concurrent_observer_sees_a_consistent_walk() {
        let cols = random_pedigree(20_000, 3);
        let ped = cols.try_borrow().unwrap();
        let progress = Progress::default();
        let reference = count_pairs(&ped, MaxDegree::MAX, None, &Progress::default()).unwrap();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        let (counts, seen) = std::thread::scope(|s| {
            let worker =
                s.spawn(|| pool.install(|| count_pairs(&ped, MaxDegree::MAX, None, &progress)));
            let mut seen = Vec::new();
            while !worker.is_finished() {
                seen.push(progress.snapshot());
            }
            seen.push(progress.snapshot());
            (worker.join().unwrap().unwrap(), seen)
        });
        assert_eq!(counts, reference);
        let rank = |s: &Snapshot| match s {
            Snapshot::Preparing => 0,
            Snapshot::Walking { .. } => 1,
            Snapshot::Finishing { .. } => 2,
        };
        assert!(seen.windows(2).all(|w| rank(&w[0]) <= rank(&w[1])));
        let mut last_done = 0;
        for snapshot in &seen {
            if let Snapshot::Walking { done, total } = *snapshot {
                assert_eq!(total, 20_000);
                assert!(done >= last_done && done <= total);
                last_done = done;
            }
        }
        assert_eq!(*seen.last().unwrap(), Snapshot::Finishing { total: 20_000 });
        assert_eq!(progress.done(), 20_000);
    }

    type Run<'a> = Box<dyn Fn(&Progress) -> Result<(), Error> + 'a>;

    /// One entry point on one input: what it runs, the checkpoints it
    /// reaches without a cancel, and its row-visit total.
    struct Path<'a> {
        name: &'static str,
        run: Run<'a>,
        expected: Vec<Checkpoint>,
        total: usize,
    }

    const N: usize = 1000;
    const LANES: usize = 3;

    /// Every entry point, on the graph and through a compact view where it
    /// has one, with the checkpoints it must reach in order.
    fn paths<'a>(ped: &'a Pedigree<'a>, view: &'a [i32], zeros: &'a [i32]) -> Vec<Path<'a>> {
        use super::super::{
            count_view_pairs_compact, pair_blocks, relationship_burden, relationship_moments,
            relatives_per_person, CategorySet, CompactView, Execution, MomentsInput, Receiver,
            Symmetric,
        };
        use std::num::NonZeroUsize;
        use Checkpoint::*;
        let degree = MaxDegree::try_new(3).unwrap();
        let all = CategorySet::up_to_degree(degree.get());
        let lanes = NonZeroUsize::new(LANES).unwrap();
        let compact_n = CompactView::build(ped, view).unwrap().columns.mother.len();
        assert!(compact_n < N, "the view must compact the pedigree");
        let graph_input = MomentsInput {
            labels_first: zeros,
            n_labels_first: 1,
            labels_second: zeros,
            n_labels_second: 1,
            values: &[],
            n_columns: 0,
            products: &[],
            same: &[],
            n_same: 0,
        };
        let n_view = view.iter().filter(|&&v| v >= 0).count();
        let view_input = MomentsInput {
            labels_first: &zeros[..n_view],
            labels_second: &zeros[..n_view],
            ..graph_input
        };
        let copies = |v: Option<&[i32]>| -> Vec<Checkpoint> {
            let blocks = match v {
                None => pair_blocks(
                    ped,
                    degree,
                    all,
                    Receiver::Graph,
                    Execution::Speed,
                    &Progress::default(),
                ),
                Some(v) => pair_blocks(
                    ped,
                    degree,
                    all,
                    Receiver::View {
                        rows: v,
                        compact: true,
                    },
                    Execution::Speed,
                    &Progress::default(),
                ),
            }
            .unwrap();
            Category::ALL
                .into_iter()
                .filter(|&c| !blocks.get(c).is_empty())
                .map(CategoryCopy)
                .collect()
        };
        let per_cat = |f: fn(Category) -> Checkpoint| all.iter().map(f).collect::<Vec<_>>();
        let merges = vec![LaneMerge; LANES - 1];
        let cat = |parts: &[&[Checkpoint]]| parts.concat();
        vec![
            Path {
                name: "count_pairs",
                run: Box::new(move |p| count_pairs(ped, degree, None, p).map(drop)),
                expected: vec![Walk, Finish],
                total: N,
            },
            Path {
                name: "count_view_pairs_compact",
                run: Box::new(move |p| count_view_pairs_compact(ped, degree, view, p).map(drop)),
                expected: vec![Compacted, Walk, Finish],
                total: compact_n,
            },
            Path {
                name: "relationship_burden",
                run: Box::new(move |p| relationship_burden(ped, zeros, p).map(drop)),
                expected: vec![Walk, Finish],
                total: N,
            },
            Path {
                name: "pair_blocks speed",
                run: Box::new(move |p| {
                    pair_blocks(ped, degree, all, Receiver::Graph, Execution::Speed, p).map(drop)
                }),
                expected: cat(&[&[Walk, Finish], &copies(None)]),
                total: N,
            },
            Path {
                name: "pair_blocks memory",
                run: Box::new(move |p| {
                    pair_blocks(ped, degree, all, Receiver::Graph, Execution::Memory, p).map(drop)
                }),
                expected: cat(&[&[Walk, BetweenPasses], &per_cat(CategoryAlloc), &[Finish]]),
                total: 2 * N,
            },
            Path {
                name: "pair_blocks compact view speed",
                run: Box::new(move |p| {
                    pair_blocks(
                        ped,
                        degree,
                        all,
                        Receiver::View {
                            rows: view,
                            compact: true,
                        },
                        Execution::Speed,
                        p,
                    )
                    .map(drop)
                }),
                expected: cat(&[
                    &[Compacted, Walk, Finish],
                    &copies(Some(view)),
                    &per_cat(CategorySort),
                ]),
                total: compact_n,
            },
            Path {
                name: "pair_blocks compact view memory",
                run: Box::new(move |p| {
                    pair_blocks(
                        ped,
                        degree,
                        all,
                        Receiver::View {
                            rows: view,
                            compact: true,
                        },
                        Execution::Memory,
                        p,
                    )
                    .map(drop)
                }),
                expected: cat(&[
                    &[Compacted, Walk, BetweenPasses],
                    &per_cat(CategoryAlloc),
                    &[Finish],
                    &per_cat(CategorySort),
                ]),
                total: 2 * compact_n,
            },
            Path {
                name: "relationship_moments",
                run: Box::new(move |p| {
                    let input = graph_input;
                    relationship_moments(
                        ped,
                        degree,
                        all,
                        Receiver::Graph,
                        &input,
                        Symmetric::Canonical,
                        lanes,
                        u64::MAX,
                        p,
                    )
                    .map(drop)
                }),
                expected: cat(&[&[Walk, Finish], &merges]),
                total: N,
            },
            Path {
                name: "relationship_moments compact",
                run: Box::new(move |p| {
                    let input = view_input;
                    relationship_moments(
                        ped,
                        degree,
                        all,
                        Receiver::View {
                            rows: view,
                            compact: true,
                        },
                        &input,
                        Symmetric::Canonical,
                        lanes,
                        u64::MAX,
                        p,
                    )
                    .map(drop)
                }),
                expected: cat(&[&[Compacted, Walk, Finish], &merges]),
                total: compact_n,
            },
            Path {
                name: "relatives_per_person",
                run: Box::new(move |p| {
                    relatives_per_person(ped, degree, all, Receiver::Graph, &[], lanes, p).map(drop)
                }),
                expected: cat(&[&[Walk, Finish], &merges]),
                total: N,
            },
            Path {
                name: "relatives_per_person compact",
                run: Box::new(move |p| {
                    relatives_per_person(
                        ped,
                        degree,
                        all,
                        Receiver::View {
                            rows: view,
                            compact: true,
                        },
                        &[],
                        lanes,
                        p,
                    )
                    .map(drop)
                }),
                expected: cat(&[&[Compacted, Walk, Finish], &merges]),
                total: compact_n,
            },
        ]
    }

    /// Run `body` on every path, inside a one-thread pool so that the
    /// parallel copy tasks reach their checkpoints in a fixed order.
    fn each_path(body: impl Fn(&Path) + Sync) {
        let cols = random_pedigree(N, 9);
        let ped = cols.try_borrow().unwrap();
        // Rows below 500 only, so no ancestor lies past them.
        let mut next = 0;
        let view: Vec<i32> = (0..N)
            .map(|r| {
                if r % 7 == 0 && r < 500 {
                    next += 1;
                    next - 1
                } else {
                    -1
                }
            })
            .collect();
        let zeros = vec![0i32; N];
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        pool.install(|| {
            for path in paths(&ped, &view, &zeros) {
                body(&path);
            }
        });
    }

    #[test]
    fn every_path_reaches_its_checkpoints_and_ends_with_every_row_visited() {
        each_path(|path| {
            let progress = Progress::default();
            (path.run)(&progress).unwrap();
            assert_eq!(progress.reached(), path.expected, "{}", path.name);
            assert_eq!(
                progress.snapshot(),
                Snapshot::Finishing { total: path.total },
                "{}",
                path.name
            );
            assert_eq!(progress.done(), path.total, "{}", path.name);
        });
    }

    #[test]
    fn a_progress_cancelled_before_the_call_cancels_every_path() {
        each_path(|path| {
            assert_eq!(
                (path.run)(&Progress::cancelled()),
                Err(Error::Cancelled),
                "{}",
                path.name
            );
        });
    }

    /// Cancelling at a checkpoint stops the call there: nothing after it
    /// starts, so `reached` ends at it.
    #[test]
    fn a_cancel_at_any_checkpoint_stops_every_path_there() {
        each_path(|path| {
            for (i, &at) in path.expected.iter().enumerate() {
                if path.expected[..i].contains(&at) {
                    continue;
                }
                let progress = Progress::cancelling_at(at);
                let label = format!("{} at {at:?}", path.name);
                assert_eq!((path.run)(&progress), Err(Error::Cancelled), "{label}");
                let reached = progress.reached();
                if let Checkpoint::CategoryCopy(_) = at {
                    // The copy tasks run in parallel and each checks for
                    // itself, so only what precedes the copies is fixed.
                    let finish = path.expected.iter().position(|&c| c == Checkpoint::Finish);
                    let before = &path.expected[..=finish.unwrap()];
                    assert_eq!(&reached[..before.len()], before, "{label}");
                    assert!(reached.contains(&at), "{label}");
                    assert!(
                        !reached
                            .iter()
                            .any(|c| matches!(c, Checkpoint::CategorySort(_))),
                        "{label}"
                    );
                    continue;
                }
                assert_eq!(reached, path.expected[..=i], "{label}");
                match at {
                    Checkpoint::Walk => assert_eq!(progress.done(), 0, "{label}"),
                    Checkpoint::BetweenPasses | Checkpoint::CategoryAlloc(_) => {
                        assert_eq!(progress.done(), path.total / 2, "{label}")
                    }
                    _ => {}
                }
            }
        });
    }
}
