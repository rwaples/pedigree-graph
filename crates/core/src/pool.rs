//! The one package-owned Rayon pool (ADR 0007, "Threads and determinism").
//!
//! Every host configures it once, from its committed thread budget, before
//! the first parallel call; counts and pairs then run inside it.  Configuring
//! it again with the same value is a no-op, and a different value is an
//! error, so work already dispatched cannot have its budget changed behind
//! its back.  Integer results are the same for every thread count.
//!
//! The pool is owned per process, not per address space.  `fork` copies the
//! pool's memory but none of its worker threads, so a child that inherited a
//! parent's pool would push jobs onto a queue nobody drains and block for
//! ever.  The slot therefore records the process that built it and rebuilds
//! when it is first used under a different one.  The inherited pool is leaked
//! rather than dropped: dropping it would join worker threads that do not
//! exist in the child.

use crate::error::Error;
use std::num::NonZeroUsize;
use std::sync::Mutex;

/// The pool of one process, and the size and process it was built for.
struct Slot {
    pid: u32,
    threads: NonZeroUsize,
    pool: &'static rayon::ThreadPool,
}

static POOL: Mutex<Option<Slot>> = Mutex::new(None);

/// The package pool, built on first use in this process with `threads` workers.
///
/// # Errors
///
/// [`Error::ThreadPoolConflict`] when this process's pool already has a
/// different size, and [`Error::ThreadPoolUnavailable`] when the operating
/// system refuses the worker threads.
pub fn configure(threads: NonZeroUsize) -> Result<&'static rayon::ThreadPool, Error> {
    let pid = std::process::id();
    // A poisoned lock leaves the slot readable; a panic under it cannot have
    // left a half-built pool, since the slot is written in one assignment.
    let mut slot = POOL.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(current) = slot.as_ref() {
        if current.pid == pid {
            if current.threads != threads {
                return Err(Error::ThreadPoolConflict {
                    configured: current.threads.get(),
                    requested: threads.get(),
                });
            }
            return Ok(current.pool);
        }
        // Inherited across a fork: its workers did not survive.  Leave it
        // leaked and build this process its own.
    }
    let built = rayon::ThreadPoolBuilder::new()
        .num_threads(threads.get())
        .build()
        .map_err(|e| Error::ThreadPoolUnavailable {
            reason: e.to_string(),
        })?;
    let pool: &'static rayon::ThreadPool = Box::leak(Box::new(built));
    *slot = Some(Slot { pid, threads, pool });
    Ok(pool)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_size_is_a_no_op_and_a_different_size_conflicts() {
        let one = NonZeroUsize::new(1).unwrap();
        let first = configure(one).expect("first configure");
        let again = configure(one).expect("same size again");
        assert!(std::ptr::eq(first, again));
        let err = configure(NonZeroUsize::new(2).unwrap()).expect_err("different size");
        assert!(matches!(err, Error::ThreadPoolConflict { .. }));
    }
}
