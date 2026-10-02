//! Test-only entry points, compiled with the `test-hooks` feature.
//!
//! `pixi run -e r r-install` builds with it (`PG_CARGO_FEATURES`); the source
//! tarball never does.  Without the feature the module registers nothing.

use extendr_api::prelude::*;

#[cfg(feature = "test-hooks")]
use crate::{
    errors::{finish, HostError, HostResult},
    job::{self, cancelled},
    kernels::package_pool,
};
#[cfg(feature = "test-hooks")]
use pedigree_graph_core::{
    error::Error,
    relationships::{Progress, Snapshot},
};
#[cfg(feature = "test-hooks")]
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Condvar, Mutex, PoisonError,
};
#[cfg(feature = "test-hooks")]
use std::time::Duration;

/// Panic inside a native call, so the package tests can show a Rust panic
/// reaches R as an error and leaves the session usable.
#[cfg(feature = "test-hooks")]
#[extendr]
fn panic_for_test() {
    panic!("pedigree-graph test hook: deliberate panic");
}

/// A pair total of `count` (a decimal string) as `relationship_counts`
/// hands it to R, so the tests can reach the 2^53 refusal.
#[cfg(feature = "test-hooks")]
#[extendr]
fn count_as_double_for_test(count: &str) -> Robj {
    let count: u64 = count.parse().expect("a decimal u64");
    crate::errors::finish(crate::kernels::exact_double("FS", count).map(Robj::from))
}

/// Whether `release_for_test()` has let the scripted job return.
#[cfg(feature = "test-hooks")]
static RELEASED: Mutex<bool> = Mutex::new(false);
#[cfg(feature = "test-hooks")]
static RELEASE: Condvar = Condvar::new();

/// Set when the cancellable job saw its cancel.
#[cfg(feature = "test-hooks")]
static CANCELLED: AtomicBool = AtomicBool::new(false);

#[cfg(feature = "test-hooks")]
fn spawn_i32(
    work: impl FnOnce(&Progress) -> Result<i32, Error> + Send + 'static,
) -> HostResult<Robj> {
    Ok(job::spawn(package_pool()?, work, |value| Ok(value.into())))
}

/// A job that reports what `script_for_test()` sets, starting at
/// `preparing`, and returns `42L` once `release_for_test()` is called.
#[cfg(feature = "test-hooks")]
#[extendr]
fn scripted_job_for_test() -> Robj {
    *RELEASED.lock().unwrap_or_else(PoisonError::into_inner) = false;
    finish(
        spawn_i32(|progress| loop {
            let released = RELEASE
                .wait_timeout(
                    RELEASED.lock().unwrap_or_else(PoisonError::into_inner),
                    Duration::from_millis(1),
                )
                .unwrap_or_else(PoisonError::into_inner)
                .0;
            if *released {
                return Ok(42);
            }
            drop(released);
            if cancelled(progress) {
                return Err(Error::Cancelled);
            }
        })
        .and_then(|handle| {
            job::of(&handle)?.script(Snapshot::Preparing);
            Ok(handle)
        }),
    )
}

/// Make the scripted job report `phase` with `done` and `total` row visits.
#[cfg(feature = "test-hooks")]
#[extendr]
fn script_for_test(handle: Robj, phase: &str, done: f64, total: f64) -> Robj {
    let snapshot = match phase {
        "preparing" => Snapshot::Preparing,
        "walking" => Snapshot::Walking {
            done: done as usize,
            total: total as usize,
        },
        "finishing" => Snapshot::Finishing {
            total: total as usize,
        },
        _ => return HostError::usage(format!("unknown phase {phase:?}")).into_robj(),
    };
    finish(job::of(&handle).map(|job| {
        job.script(snapshot);
        ().into()
    }))
}

/// Let the scripted job return.
#[cfg(feature = "test-hooks")]
#[extendr]
fn release_for_test() {
    *RELEASED.lock().unwrap_or_else(PoisonError::into_inner) = true;
    RELEASE.notify_all();
}

/// A job whose work panics.
#[cfg(feature = "test-hooks")]
#[extendr]
fn panicking_job_for_test() -> Robj {
    finish(spawn_i32(|_| {
        panic!("pedigree-graph test hook: deliberate job panic")
    }))
}

/// A job that runs until it is cancelled; `was_cancelled_for_test()` then
/// turns `TRUE`.
#[cfg(feature = "test-hooks")]
#[extendr]
fn cancellable_job_for_test() -> Robj {
    CANCELLED.store(false, Ordering::SeqCst);
    finish(spawn_i32(|progress| {
        while !cancelled(progress) {
            std::thread::sleep(Duration::from_millis(1));
        }
        CANCELLED.store(true, Ordering::SeqCst);
        Err(Error::Cancelled)
    }))
}

#[cfg(feature = "test-hooks")]
#[extendr]
fn was_cancelled_for_test() -> bool {
    CANCELLED.load(Ordering::SeqCst)
}

#[cfg(feature = "test-hooks")]
extendr_module! {
    mod test_hooks;
    fn panic_for_test;
    fn count_as_double_for_test;
    fn scripted_job_for_test;
    fn script_for_test;
    fn release_for_test;
    fn panicking_job_for_test;
    fn cancellable_job_for_test;
    fn was_cancelled_for_test;
}

#[cfg(not(feature = "test-hooks"))]
extendr_module! {
    mod test_hooks;
}
