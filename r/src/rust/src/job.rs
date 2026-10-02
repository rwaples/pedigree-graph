//! A relationship call running on the package pool while R watches it
//! (ADR 0017; pedigree-graph #39).
//!
//! A start entry point spawns the engine call as a [`Task`] and hands R a
//! [`Job`] handle.  R's loop (`.pg_watch()`) waits on the handle a tick at a
//! time, services its own events between ticks (Ctrl-C, time limits), and
//! then collects the result or cancels the call.  The task never touches R:
//! it owns copies of its inputs and returns plain Rust values, which the
//! job's `finish` turns into R objects on the R thread.

use crate::errors::{HostError, HostResult};
use extendr_api::prelude::*;
use pedigree_graph_core::error::Error;
use pedigree_graph_core::relationships::{Progress, Snapshot};
use std::any::Any;
use std::cell::RefCell;
use std::panic::{self, AssertUnwindSafe};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::time::Duration;

type Output = Box<dyn Any + Send>;
type Outcome = std::thread::Result<Result<Output, Error>>;

/// Where a task's outcome is.  The worker moves `Running` to `Ready` once;
/// `take` and `cancel` move `Ready` to `Collected`.
enum Slot {
    Running,
    Ready(Outcome),
    Collected,
}

struct Shared {
    progress: Progress,
    slot: Mutex<Slot>,
    ready: Condvar,
    /// When set, `wait` reports this in place of the engine's progress.
    #[cfg(feature = "test-hooks")]
    scripted: Mutex<Option<Snapshot>>,
}

impl Shared {
    fn slot(&self) -> MutexGuard<'_, Slot> {
        self.slot.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn until_ready(&self) -> MutexGuard<'_, Slot> {
        self.ready
            .wait_while(self.slot(), |slot| matches!(slot, Slot::Running))
            .unwrap_or_else(PoisonError::into_inner)
    }
}

/// `take` on a task whose outcome was already taken or dropped.
#[derive(Debug, PartialEq, Eq)]
pub struct AlreadyCollected;

/// One engine call running on a pool, free of R.
pub struct Task {
    shared: Arc<Shared>,
}

impl Task {
    pub fn spawn<T: Send + 'static>(
        pool: &rayon::ThreadPool,
        work: impl FnOnce(&Progress) -> Result<T, Error> + Send + 'static,
    ) -> Task {
        let shared = Arc::new(Shared {
            progress: Progress::default(),
            slot: Mutex::new(Slot::Running),
            ready: Condvar::new(),
            #[cfg(feature = "test-hooks")]
            scripted: Mutex::new(None),
        });
        let worker = Arc::clone(&shared);
        pool.spawn(move || {
            // A panic that leaves a rayon job aborts the process.
            let outcome = panic::catch_unwind(AssertUnwindSafe(|| {
                work(&worker.progress).map(|t| Box::new(t) as Output)
            }));
            *worker.slot() = Slot::Ready(outcome);
            worker.ready.notify_all();
        });
        Task { shared }
    }

    /// Block until the task is no longer running or `tick` passes: the
    /// progress then, or `None` once it is not running.
    pub fn wait(&self, tick: Duration) -> Option<Snapshot> {
        let (slot, _) = self
            .shared
            .ready
            .wait_timeout_while(self.shared.slot(), tick, |slot| {
                matches!(slot, Slot::Running)
            })
            .unwrap_or_else(PoisonError::into_inner);
        if !matches!(*slot, Slot::Running) {
            return None;
        }
        drop(slot);
        #[cfg(feature = "test-hooks")]
        if let Some(scripted) = *self
            .shared
            .scripted
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
        {
            return Some(scripted);
        }
        Some(self.shared.progress.snapshot())
    }

    /// Wait for the outcome and take it, leaving the task collected
    /// whatever the outcome is.  A panic in the work resumes here.
    pub fn take(&self) -> Result<Result<Output, Error>, AlreadyCollected> {
        let taken = std::mem::replace(&mut *self.shared.until_ready(), Slot::Collected);
        match taken {
            Slot::Ready(Ok(result)) => Ok(result),
            Slot::Ready(Err(payload)) => panic::resume_unwind(payload),
            Slot::Collected => Err(AlreadyCollected),
            Slot::Running => unreachable!("until_ready waits while running"),
        }
    }

    /// Stop the work, wait for it and drop its outcome.  A task no longer
    /// running returns at once.
    pub fn cancel(&self) {
        self.shared.progress.cancel();
        let outcome = std::mem::replace(&mut *self.shared.until_ready(), Slot::Collected);
        // Freed outside the lock: a result can be large.
        drop(outcome);
    }
}

type Finish = Box<dyn FnOnce(Output) -> HostResult<Robj>>;

/// What R's handle holds: the task and, until it is collected, how to turn
/// its output into the R result.  It lives on the R thread only.
pub struct Job {
    task: Task,
    finish: RefCell<Option<Finish>>,
}

/// Spawn `work` on `pool` and return R's handle to it; collecting the handle
/// passes the work's value to `finish`.
pub fn spawn<T: Send + 'static>(
    pool: &rayon::ThreadPool,
    work: impl FnOnce(&Progress) -> Result<T, Error> + Send + 'static,
    finish: impl FnOnce(T) -> HostResult<Robj> + 'static,
) -> Robj {
    let finish: Finish = Box::new(move |out: Output| {
        finish(
            *out.downcast::<T>()
                .expect("a task's output has its work's type"),
        )
    });
    ExternalPtr::new(Job {
        task: Task::spawn(pool, work),
        finish: RefCell::new(Some(finish)),
    })
    .into()
}

impl Job {
    /// The R result, or the work's error; the job is collected either way.
    pub fn collect(&self) -> HostResult<Robj> {
        let finish = self.finish.take();
        match self.task.take() {
            Err(AlreadyCollected) => Err(HostError::usage(
                "this relationship call was already collected or cancelled".to_string(),
            )),
            Ok(Err(err)) => Err(err.into()),
            Ok(Ok(out)) => finish.expect("an uncollected job keeps its finish")(out),
        }
    }

    pub fn cancel(&self) {
        self.task.cancel();
        self.finish.take();
    }

    /// Report `snapshot` from now on in place of the engine's progress.
    #[cfg(feature = "test-hooks")]
    pub(crate) fn script(&self, snapshot: Snapshot) {
        *self
            .task
            .shared
            .scripted
            .lock()
            .unwrap_or_else(PoisonError::into_inner) = Some(snapshot);
    }
}

/// R's garbage collector drops an uncollected handle: stop its work.
impl Drop for Job {
    fn drop(&mut self) {
        self.task.cancel();
    }
}

/// The job an R handle holds.
pub(crate) fn of(handle: &Robj) -> HostResult<&Job> {
    <&ExternalPtr<Job>>::try_from(handle)
        .and_then(|ptr| ptr.try_addr())
        .map_err(|_| HostError::usage("expected a relationship call handle".to_string()))
}

/// `list(finished, phase, rows_done, rows_total)`: `finished` once the work
/// is no longer running, each other field `NA` where the phase has no value.
pub(crate) fn snapshot_robj(state: Option<Snapshot>) -> Robj {
    let (phase, done, total) = match state {
        None => (None, None, None),
        Some(Snapshot::Preparing) => (Some("preparing"), None, None),
        Some(Snapshot::Walking { done, total }) => (Some("walking"), Some(done), Some(total)),
        Some(Snapshot::Finishing { total }) => (Some("finishing"), None, Some(total)),
    };
    // Doubles: a "memory" walk is 2n visits, which can pass an int32.
    let count = |c: Option<usize>| {
        Doubles::from_values([c.map_or_else(Rfloat::na, |c| Rfloat::from(c as f64))]).into_robj()
    };
    List::from_names_and_values(
        ["finished", "phase", "rows_done", "rows_total"],
        [
            state.is_none().into(),
            Strings::from_values([phase.map_or_else(Rstr::na, Rstr::from)]).into_robj(),
            count(done),
            count(total),
        ],
    )
    .expect("four names and four values")
    .into_robj()
}

/// Wait up to `tick` seconds; see [`Task::wait`].
pub fn wait(handle: &Robj, tick: f64) -> HostResult<Robj> {
    let tick = Duration::try_from_secs_f64(tick)
        .ok()
        .filter(|t| !t.is_zero())
        .ok_or_else(|| {
            HostError::usage(format!(
                "tick must be a positive number of seconds, got {tick}"
            ))
        })?;
    Ok(snapshot_robj(of(handle)?.task.wait(tick)))
}

pub fn collect(handle: &Robj) -> HostResult<Robj> {
    of(handle)?.collect()
}

pub fn cancel(handle: &Robj) -> HostResult<Robj> {
    of(handle)?.cancel();
    Ok(().into())
}

/// Whether `progress` was cancelled.  `Progress` has no getter, so this asks
/// the engine itself: a walk over two rows fails at once once cancelled.
#[cfg(any(test, feature = "test-hooks"))]
pub(crate) fn cancelled(progress: &Progress) -> bool {
    use pedigree_graph_core::relationships::{count_pairs, MaxDegree, Pedigree};
    let none = [-1i32; 2];
    let ids = [0i64; 2];
    let ped = Pedigree::try_new(&none, &none, &none, &ids, &ids).expect("a two-row pedigree");
    let degree = MaxDegree::try_new(1).expect("degree 1");
    matches!(
        count_pairs(&ped, degree, None, progress),
        Err(Error::Cancelled)
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use pedigree_graph_core::relationships::MaxDegree;
    use std::sync::mpsc;
    use std::time::Instant;

    fn pool(threads: usize) -> rayon::ThreadPool {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
    }

    fn until_cancelled(progress: &Progress) -> Result<i32, Error> {
        while !cancelled(progress) {
            std::thread::sleep(Duration::from_millis(1));
        }
        Err(Error::Cancelled)
    }

    fn value(out: Output) -> i32 {
        *out.downcast::<i32>().unwrap()
    }

    fn panic_message(task: &Task) -> String {
        let payload = panic::catch_unwind(AssertUnwindSafe(|| task.take())).unwrap_err();
        payload.downcast::<&str>().unwrap().to_string()
    }

    #[test]
    fn a_finished_task_waits_none_and_takes_its_value_once() {
        let pool = pool(1);
        let task = Task::spawn(&pool, |_| Ok(7i32));
        while task.wait(Duration::from_millis(10)).is_some() {}
        assert_eq!(value(task.take().unwrap().unwrap()), 7);
        assert_eq!(task.take().err(), Some(AlreadyCollected));
        assert_eq!(task.wait(Duration::from_millis(10)), None);
    }

    #[test]
    fn a_running_task_reports_progress_and_wakes_its_waiter_when_ready() {
        let pool = pool(2);
        let (release, released) = mpsc::channel::<()>();
        let task = Task::spawn(&pool, move |_| {
            released.recv().unwrap();
            Ok(1i32)
        });
        assert_eq!(
            task.wait(Duration::from_millis(5)),
            Some(Snapshot::Preparing)
        );
        release.send(()).unwrap();
        let start = Instant::now();
        assert_eq!(task.wait(Duration::from_secs(60)), None);
        assert!(start.elapsed() < Duration::from_secs(10));
        assert_eq!(value(task.take().unwrap().unwrap()), 1);
    }

    #[test]
    fn cancel_stops_a_running_task_and_collects_it() {
        let pool = pool(1);
        let task = Task::spawn(&pool, until_cancelled);
        assert!(task.wait(Duration::from_millis(5)).is_some());
        task.cancel();
        assert_eq!(task.take().err(), Some(AlreadyCollected));
        task.cancel();
        assert_eq!(task.wait(Duration::from_millis(5)), None);
    }

    #[test]
    fn cancel_after_take_returns_without_waiting() {
        let pool = pool(1);
        let task = Task::spawn(&pool, |_| Ok(3i32));
        assert_eq!(value(task.take().unwrap().unwrap()), 3);
        task.cancel();
        task.cancel();
        assert_eq!(task.take().err(), Some(AlreadyCollected));
    }

    #[test]
    fn a_core_error_is_taken_once_and_cancel_after_it_is_a_no_op() {
        let pool = pool(1);
        let task = Task::spawn(&pool, |_| MaxDegree::try_new(99).map(|_| 0i32));
        assert!(matches!(
            task.take(),
            Ok(Err(Error::MaxDegreeOutOfRange { .. }))
        ));
        task.cancel();
        assert_eq!(task.take().err(), Some(AlreadyCollected));
    }

    #[test]
    fn a_panic_lands_ready_and_resumes_in_take_once() {
        let pool = pool(1);
        let task = Task::spawn(&pool, |_| -> Result<i32, Error> { panic!("deliberate") });
        while task.wait(Duration::from_millis(10)).is_some() {}
        assert_eq!(panic_message(&task), "deliberate");
        assert_eq!(task.take().err(), Some(AlreadyCollected));
        task.cancel();
    }

    #[test]
    fn cancel_drops_a_ready_outcome_and_a_panic() {
        let pool = pool(1);
        let done = Task::spawn(&pool, |_| Ok(5i32));
        while done.wait(Duration::from_millis(10)).is_some() {}
        done.cancel();
        assert_eq!(done.take().err(), Some(AlreadyCollected));
        let panicked = Task::spawn(&pool, |_| -> Result<i32, Error> { panic!("deliberate") });
        while panicked.wait(Duration::from_millis(10)).is_some() {}
        panicked.cancel();
        assert_eq!(panicked.take().err(), Some(AlreadyCollected));
    }
}
