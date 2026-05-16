# ADR 0001: Splitting `_dp_kinship` into production + bench entry points — rejected

**Status:** rejected
**Date:** 2026-05-16
**Context branch:** `refactor/dp-config-bundle` (Phase C, Gate 2)

## Context

`_dp_kinship` in `pedigree_graph/_kinship_kernel.py` is the kinship-DP
hot kernel.  Its signature carries two arguments that exist
exclusively for benchmark instrumentation:

* `grow_stats: np.ndarray` (int64[3]) — captures
  `[grow_call_count, total_entries_copied, peak_allocated]` for
  `_grow_global` and `_acquire_slot`.  Production callers don't read
  it; they pass a fresh zeroed array.
* `initial_buffer_override: np.int64` — lets bench callers swap the
  `max(1<<16, init_cap_per_row*1024)` heuristic.  Production callers
  pass `0` (use heuristic).

The Phase C plan proposed splitting `_dp_kinship` into:

* `_dp_kinship` — production entry, drops the two bench args
* `_dp_kinship_bench` — bench entry, keeps them

The rationale offered in the plan was:

> Microbenchmark again: expect neutral or faster (fewer branches in
> the production-dispatched numba specialization).

## Decision

**Do not split.**  The performance premise is incorrect.

Numba specializes `@numba.njit` functions on **argument types**, not
argument values.  The current `_dp_kinship` produces exactly one
specialization (verified via `_dp_kinship.signatures` count = 1
after warming both `retire` branches).  Splitting into two entry
points either:

1. Creates **two** specializations (one per kernel), which is *more*
   dispatch fragmentation, not less; or
2. Yields one specialization per shared kernel body, in which case
   the split is just Python-layer renaming with no perf
   consequence.

A second probe (`/tmp/numba_defaults_probe.py`, kept for one cycle as
a reference; not committed) confirmed that adding default values to
an `@njit` signature creates a separate `omitted(default=…)`
specialization per call shape — exactly the fragmentation the gate
was supposed to avoid.

## Consequences

* The two bench args (`grow_stats`, `initial_buffer_override`) remain
  on `_dp_kinship`'s production signature.  `_run_dp_core` (the
  production Python wrapper introduced in Phase B) continues to
  supply them as `np.zeros(3, dtype=np.int64)` and `np.int64(0)`
  defaults — a single allocation per `per_gen_mean_kinship` call,
  amortized over the entire DP traversal.  Negligible cost.
* The test in `tests/test_kinship_kernel.py:180` continues to call
  `_dp_kinship` directly with positional `grow_stats` /
  `initial_buffer_override`.  Acceptable: it's the only
  test-direct caller, and the `_run_dp_core` Python layer (Phase B)
  already hides the bench knobs from real production paths.
* If a future refactor genuinely needs an instrumentation split, the
  right path is a *separate compilation unit* (different module-level
  `@njit` function with its own body), not a default-arg overload
  on the same function name.

## What we did keep from the Phase C gate

* **Gate 1** (the `_FreelistBuffers` NamedTuple): shipped as commit
  `3ff9729`.  Bundles 4 hot-loop args at 6 call sites; numba's
  `inline="always"` flattens it cleanly.  `_dp_kinship.signatures`
  count unchanged at 1; perf delta within run-to-run noise (-1.85%
  small, -0.18% medium).
* **Gate 3** (the `KinshipDPConfig` Python NamedTuple for the three
  booleans): planned as a follow-on commit on this branch.  It
  operates entirely at the Python layer — the `@njit` kernel still
  receives three plain bools, so dispatch is unaffected.

## References

* The benchmark probe artifact (`bench_phase_c.py`) lives in the
  branch root but is `.gitignore`-able and not committed; rerun
  after any Phase-C-adjacent change with
  `python bench_phase_c.py --scales small,medium --iters 3
  --compare /tmp/phase_c_baseline.json`.
