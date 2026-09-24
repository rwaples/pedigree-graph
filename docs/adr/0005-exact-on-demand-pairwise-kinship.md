# ADR 0005: `pair_kinship` is always exact, via an on-demand pairwise recurrence

**Status:** accepted; the value definition (a float32 recurrence with a pinned peel rule) is ADR 0009's
**Date:** 2026-06-09
**Context:** profiling follow-up to ADR 0001 (the algorithmic lever it pointed at)

Revised 2026-09-24 to match 0.10.0; earlier wording in git history.

## Context

`PedigreeGraph.pair_kinship` returns the kinship coefficient of each
requested pair. Its predecessor, `compute_pair_kinship(pairs)`, had two paths:

1. **Nominal fast path.** When inbreeding was all zero and there were no MZ
   twins, return the nominal kinship constant of each pair's relationship
   code.
2. **Matrix slow path.** Otherwise build the full kinship matrix and read off
   the requested cells.

Both were wrong or unscalable:

* The nominal fast path is **not exact** whenever a pair is related through
  *multiple* lineages, even with no inbreeding and no twins. Double first
  cousins (each parent couple a full-sib pair) have `phi = 0.125`, but the
  fast path returned the single-path `1C` constant `0.0625`.
* The matrix slow path was the package's dominant super-linear cost. ADR 0001
  measured a 53.3M-nonzero matrix at 16k individuals and an OOM at 80k, and
  named the lever as "avoid materialising the full kinship matrix when only
  specific pairs are needed".

An earlier plan proposed **pruning** the matrix DP by a kinship threshold
derived from the requested codes. That was proven incorrect and abandoned:
threshold pruning is lossy for *cross-generation* propagation. A
sub-threshold kinship between two mates feeds their descendants'
above-threshold kinship, and pruning deletes it at the parents' generation.
Disproof: half-first-cousin parents (`phi = 1/32`) and their child. The
child's exact parent-offspring kinship is `0.265625`, but any threshold that
drops `1/32` collapses it to `0.25`. No global magnitude threshold can be
exact, because `phi(i, j)` needs the kinship sub-matrix over
`ancestors(i) ∪ ancestors(j)`.

## Decision

**`pair_kinship` is always exact and has no nominal fast path.** It computes
kinship for **only the requested pairs** with a memoised Karigl recurrence
and never builds the `n × n` matrix:

* `phi(a, a) = (1 + phi(mother_a, father_a)) / 2`, a missing parent
  contributing 0, and an MZ co-twin taking the self formula;
* otherwise peel the endpoint `c` of greater structural depth, ties to the
  greater row, with `o` the other endpoint:
  `phi(a, b) = (phi(mother_c, o) + phi(father_c, o)) / 2`.

ADR 0009 pins this recurrence to float32, one correctly rounded half-sum per
step, and that is the value the package returns. `kinship_matrix()` and
`relationship_kinship_matrix(...)` entries are bit-identical to
`pair_kinship` for the same pair within one graph.

The recurrence runs in the Rust core (`kinship::pair_kinship`,
`crates/core/src/kinship/pairwise.rs`), reached through
`_native.pair_kinship` from `pedigree_graph/_kinship_pairwise.py`. One memo
is built per call, shared across every pair of the query, and freed before
the call returns (ADR 0007). `pair_kinship` never reads a cached matrix, so
its result does not depend on call history. The readable recurrence is the
test oracle `tests/oracle/pair_kinship.py`, which the package never imports
and which the native kernel must match bit for bit.

## Consequences

* **Behaviour change (the point):** multi-path pairs (double cousins and
  similar) return their true kinship instead of the nominal code value.
  Inbreeding and MZ genome identity are likewise exact. Single-path,
  non-inbred pairs keep the nominal value, since the recurrence yields the
  same dyadic number.
* **Output dtype is float32** (ADR 0009). This ADR first chose float64; ADR
  0009 replaced that when it pinned the recurrence to float32.
* **Scaling:** the work follows the ancestor pairs the requested pairs reach
  through the memo, not `n²`. Deeply inbred or high-overlap pedigrees can
  still grow the memo; pathologically deep pedigrees are out of scope.
* `kinship_matrix()` is a separate path with its own cache.
  `mean_kinship_by_generation()` streams kinship from the DP, or walks the
  complete matrix when it is already cached.
* The in-workspace production consumer,
  `fitACE/fitace/exports/tables.py::export_pairwise_relatedness`, calls
  `pair_kinship`.

## Alternatives considered

* **Threshold-prune the DP matrix.** Rejected; cannot be made exact (the
  half-first-cousin disproof above).
* **Restrict the existing DP to the ancestor sub-pedigree and reuse it.** No
  help; the relationship pairs span most of the population, so the ancestor
  closure is about the whole graph, and the matrix blowup comes from row
  *density*, not row count. Only a per-cell recurrence avoids the density.
* **numba `typed.Dict` for the memo.** Rejected for the original Numba kernel
  in favour of a hand-rolled open-addressing table keyed on the canonical
  pair, which cached cleanly under `@njit(cache=True)`. The Rust kernel
  replaced the Numba one; its memo is `crates/core/src/kinship/memo.rs`.
