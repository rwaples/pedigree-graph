# ADR 0005: `compute_pair_kinship` is always exact, via an on-demand pairwise recurrence

**Status:** accepted; superseded in part by ADR 0009 (output dtype is float32, the recurrence is float32 with a pinned peel rule, and the cached-matrix sampling branch is removed)
**Date:** 2026-06-09
**Context:** profiling follow-up to ADR 0001 (the algorithmic lever it pointed at)

## Context

`PedigreeGraph.compute_pair_kinship(pairs)` returns the kinship coefficient for
each requested relationship pair. It previously had two paths:

1. **Nominal fast path** — when `compute_inbreeding()` was all-zero and there
   were no MZ twins, return the `PAIR_KINSHIP[code]` constant per pair.
2. **Matrix slow path** — otherwise build the full `kinship_matrix(0.0)` and
   read off the requested cells.

Both were wrong or unscalable:

* The nominal fast path is **not exact** whenever a pair is related through
  *multiple* lineages, even with no inbreeding and no twins. Double first
  cousins (each parent-couple a full-sib pair) have `phi = 0.125`, but the
  fast path returned the single-path `1C` constant `0.0625` — contradicting the
  method's own "exact kinship per pair" contract.
* The matrix slow path is the package's dominant super-linear cost (ADR 0001's
  empirical follow-up): a 16k-individual pedigree builds a ~53M-nonzero matrix
  and ~47s of DP; simACE-scale pedigrees OOM. ADR 0001 named the genuine lever
  as "avoid materialising the full kinship matrix when only specific pairs are
  needed" and deferred it to its own ADR — this one.

An earlier plan proposed **pruning** the DP matrix (deriving `min_kinship` from
the requested codes). That was proven incorrect and abandoned: DP
threshold-pruning is lossy for *cross-generation* propagation. A sub-threshold
kinship between two mates feeds their descendants' above-threshold kinship, and
pruning deletes it at the parents' generation. Disproof — half-first-cousin
parents (`phi = 1/32`) → child: the child's exact parent-offspring kinship is
`0.265625`, but any threshold that drops `1/32` collapses it to `0.25`. No
global magnitude threshold can be exact, because `phi(i, j)` needs the full
kinship sub-matrix over `ancestors(i) ∪ ancestors(j)`.

## Decision

**`compute_pair_kinship` is always exact, with no nominal fast path.** It no
longer calls `compute_inbreeding()` for branch selection. For non-empty input
it either:

* samples the exact `kinship_matrix(0.0)` if it is already cached (symmetric,
  so input orientation is irrelevant; no `.tocsr()` duplication); or
* computes exact kinship for **only the requested pairs** via a direct memoized
  Karigl recurrence (`pedigree_graph/_kinship_pairwise.py`), never materialising
  the `n × n` matrix:
  `phi(a,a) = (1+F_a)/2`, `F_a = phi(mother_a, father_a)`, MZ pair → self-kinship,
  else `phi(a,b) = ½·(phi(mother_c, o) + phi(father_c, o))` for `c = max(a,b)`,
  `o = min(a,b)`.

The recurrence is **exact-by-construction** against the matrix DP: that path
already derives the diagonal `F` as `phi(mother, father)` inside the kernel
(not from the MZ-naive ML `compute_inbreeding`), its merge walk is
`½·(K[m,k] + K[f,k])`, and its MZ pass writes the inbred self-kinship to the
twin off-diagonal — the recurrence reproduces each rule.

The code ships **two implementations**: a pure-Python `functools.cache`
reference (`_pairwise_kinship_py`, the readable bit-oracle) and a
`@numba.njit(cache=True)` production kernel (`pairwise_kinship`, an iterative
work-stack with a hand-rolled open-addressing `int64 → float64` memo keyed on
canonical `lo·n + hi`). They are validated bit-for-bit against each other and
to `atol=1e-6` against `kinship_matrix(0.0)`.

## Consequences

* **Behavior change (the point):** multi-path pairs (double cousins, etc.) now
  return their true kinship instead of the nominal code value. Inbreeding and
  MZ co-coalescence are likewise exact. Plain single-path, non-inbred pairs are
  unchanged (the recurrence yields the same dyadic value the constant did).
* **Output dtype is float64** (was float32 from the matrix). Every kinship value
  is a dyadic rational, exact in float32 up to depth 24 and float64 up to depth
  53, so for realistic pedigrees (`G_ped ≈ 8`) the values are byte-identical —
  the downstream `pairwise_relatedness.tsv` export does not change. float64 only
  diverges, more accurately, at inbreeding depth > 24.
* **Scales** to the simACE-relevant range: cost is `O(P + ancestor-pairs)` for
  `P` requested pairs, worst case `O(P · A²)` in the max distinct-ancestor count
  `A`. The full matrix is never built unless already cached. Deeply inbred /
  high-overlap pedigrees can still inflate the shared memo (a `benchmarks`
  stress pedigree guards this); pathologically deep pedigrees are out of scope.
* `kinship_matrix()` itself is untouched — other consumers
  (`per_gen_mean_kinship`, GRM export) are unaffected.
* The one in-workspace production consumer,
  `fitACE/fitace/exports/tables.py::export_pairwise_relatedness`, speeds up with
  no code change.

## Alternatives considered

* **Threshold-prune the DP matrix** — rejected; cannot be made exact (the
  half-first-cousin disproof above).
* **Restrict the existing DP to the ancestors sub-pedigree and reuse it** —
  no help; `extract_pairs` spans most of the population, so the ancestor closure
  is ≈ the whole graph, and the matrix blowup is from row *density*, not row
  count. Only a per-cell recurrence avoids the density.
* **numba `typed.Dict` for the memo** — rejected in favour of a hand-rolled
  open-addressing table, matching the package's existing kernel style
  (`_kinship_dp` freelist, `_inbreeding_kernel` scratch), which caches cleanly
  under `@njit(cache=True)` and avoids per-lookup container overhead.
