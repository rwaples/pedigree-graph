//! The package thread budget (ADR 0007; slice 16, decision 7).
//!
//! Python's rules (`pedigree_graph/_threads.py`): `configure_threads(n)`
//! records `n`; the first `thread_budget()` commits it, else
//! `PEDIGREE_GRAPH_THREADS`, else 1; after that only the committed value may
//! be configured again.  The state lives here, not in R, so there is one
//! source of truth per process.  A forked child inherits it; the core pool
//! rebuilds itself under the child's pid (`pool.rs`).

use crate::errors::{HostError, HostResult};
use std::num::NonZeroUsize;
use std::sync::Mutex;

const ENV_VAR: &str = "PEDIGREE_GRAPH_THREADS";

struct Budget {
    configured: Option<NonZeroUsize>,
    committed: Option<NonZeroUsize>,
}

static BUDGET: Mutex<Budget> = Mutex::new(Budget {
    configured: None,
    committed: None,
});

fn conflict(committed: NonZeroUsize, requested: NonZeroUsize) -> HostError {
    HostError::from(pedigree_graph_core::error::Error::ThreadPoolConflict {
        configured: committed.get(),
        requested: requested.get(),
    })
}

/// The largest budget: `thread_budget()` returns it as an R integer.
pub const MAX_THREADS: usize = i32::MAX as usize;

/// `n` as a budget, if it is one: from 1 to [`MAX_THREADS`].
fn in_range(n: usize) -> Option<NonZeroUsize> {
    NonZeroUsize::new(n).filter(|n| n.get() <= MAX_THREADS)
}

pub fn configure(n: usize) -> HostResult<()> {
    let n = in_range(n).ok_or_else(|| {
        HostError::usage(format!(
            "configure_threads(n) requires a whole number from 1 to {MAX_THREADS}, got {n}"
        ))
    })?;
    let mut budget = BUDGET.lock().unwrap_or_else(|e| e.into_inner());
    match budget.committed {
        Some(committed) if committed != n => Err(conflict(committed, n)),
        Some(_) => Ok(()),
        None => {
            budget.configured = Some(n);
            Ok(())
        }
    }
}

fn from_env() -> HostResult<NonZeroUsize> {
    let raw = match std::env::var(ENV_VAR) {
        Ok(raw) => raw,
        Err(std::env::VarError::NotPresent) => return Ok(NonZeroUsize::MIN),
        Err(std::env::VarError::NotUnicode(raw)) => {
            return Err(HostError::usage(format!(
                "{ENV_VAR} must be a decimal integer >= 1, got {raw:?}"
            )))
        }
    };
    match raw.parse::<usize>().ok().and_then(in_range) {
        Some(n) if raw.bytes().all(|b| b.is_ascii_digit()) => Ok(n),
        _ => Err(HostError::usage(format!(
            "{ENV_VAR} must be a decimal integer from 1 to {MAX_THREADS}, got {raw:?}"
        ))),
    }
}

/// The committed budget, committing it on the first call.
pub fn budget() -> HostResult<NonZeroUsize> {
    let mut budget = BUDGET.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(committed) = budget.committed {
        return Ok(committed);
    }
    let committed = match budget.configured {
        Some(n) => n,
        None => from_env()?,
    };
    budget.committed = Some(committed);
    Ok(committed)
}
