//! The one package-owned Rayon pool (ADR 0007, "Threads and determinism").
//!
//! Every host configures it once, from its committed thread budget, before
//! the first parallel call; counts and pairs then run inside it.  Configuring
//! it again with the same value is a no-op, and a different value is an
//! error, so work already dispatched cannot have its budget changed behind
//! its back.  Integer results are the same for every thread count.

use crate::error::Error;
use std::num::NonZeroUsize;
use std::sync::OnceLock;

static POOL: OnceLock<(NonZeroUsize, rayon::ThreadPool)> = OnceLock::new();

/// The package pool, built on first use with `threads` workers.
///
/// # Errors
///
/// [`Error::ThreadPoolConflict`] when the pool already has a different size,
/// and [`Error::ThreadPoolUnavailable`] when the operating system refuses
/// the worker threads.
pub fn configure(threads: NonZeroUsize) -> Result<&'static rayon::ThreadPool, Error> {
    if POOL.get().is_none() {
        let built = rayon::ThreadPoolBuilder::new()
            .num_threads(threads.get())
            .build()
            .map_err(|e| Error::ThreadPoolUnavailable {
                reason: e.to_string(),
            })?;
        // Another thread may have won the race; its pool is then checked below.
        let _ = POOL.set((threads, built));
    }
    let (configured, pool) = POOL.get().expect("set above");
    if *configured != threads {
        return Err(Error::ThreadPoolConflict {
            configured: configured.get(),
            requested: threads.get(),
        });
    }
    Ok(pool)
}

/// The pool's size once configured.
pub fn configured_threads() -> Option<NonZeroUsize> {
    POOL.get().map(|(threads, _)| *threads)
}
