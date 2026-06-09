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

## Empirical follow-up (2026-06-09): profiling corroborates the rejection

A separate investigation asked whether any pedigree-graph hot path
should be rewritten in C++ for speed.  A profiling harness
(`benchmarks/profile_pedigree_graph.py`) was built over the consumed
`PedigreeGraph` surface, run on simACE-simulated pedigrees at two
scales (16k and 80k individuals).  Two findings bear directly on this
ADR:

1. **The DP kinship kernel *is* the dominant cost — a real, tempting
   optimization target.**  In a representative 16k-individual cProfile
   sequence, `_run_dp_core` (the `@njit` DP) was the single largest
   self-time frame: **44.5s of 67.9s** wall.  This is exactly the kind
   of hot kernel whose call boundary the rejected split proposed to
   shave.

2. **But the cost is super-linear and output-dominated, not dispatch-
   or language-bound.**  `per_gen_mean_kinship` went **38.5s → 320s**
   from 16k → 80k individuals (8.3× for 5× rows, ≈ O(n^1.4)).
   `compute_pair_kinship` builds a **53.3M-nonzero** kinship matrix at
   16k (≈21% dense) and OOMs at 80k.  The DP materialises a near-dense
   matrix whose size grows super-linearly with population, because
   pedigree relatedness density rises with N.

A dispatch-level tweak — the rejected production/bench split, and by
the same reasoning a like-for-like C++ port of the kernel — is noise
against an output that scales like O(n^1.4).  This empirically
confirms the original decision's premise: optimising the kernel's
**call boundary** (or its host language) cannot move a cost that is
set by the **size of what it produces**.  For contrast, the scipy
sparse matmuls in `extract_pairs` were *not* a bottleneck (~1.6s at
80k); the cost centre is the DP's algorithmic shape, not the sparse
algebra around it.

The genuine lever is algorithmic: avoid materialising the full kinship
matrix when only specific pairs are needed (the twin/inbreeding-gated
path in `_core.py:compute_pair_kinship`).  That work is scoped
separately as a direct pairwise-kinship routine and, if pursued,
warrants its own ADR.

Harness + saved reports live under `benchmarks/` (gitignore-able, not
committed); rerun with
`python benchmarks/profile_pedigree_graph.py --scale {small,medium}`.

## References

* The benchmark probe artifact (`bench_phase_c.py`) lives in the
  branch root but is `.gitignore`-able and not committed; rerun
  after any Phase-C-adjacent change with
  `python bench_phase_c.py --scales small,medium --iters 3
  --compare /tmp/phase_c_baseline.json`.
