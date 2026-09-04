# ADR 0009: kinship is one pinned float32 recurrence, returned recurrence-only

**Status:** accepted
**Date:** 2026-09-04
**Context:** resolves issue #6 and the `pair_kinship` dtype question deferred by ADR 0006; supersedes ADR 0005's float64 and cached-matrix clauses

## Context

ADR 0005 made `compute_pair_kinship` return float64 from a float64 Karigl
recurrence while `kinship_matrix` stores float32 per generation. ADR 0006 kept
float64 for the redesigned `pair_kinship`, said the CSC contract "permits only
this final float32 rounding", and deferred to issue #6 whether `pair_kinship`
may sample an already cached complete matrix, on the condition that "both paths
must agree".

A decision study (`simACE/plans/pair-kinship-float32-study.md`) measured the
candidates on a focused corpus (all 23 registry motifs, the ADR 0008 MZ
fixtures, selfing, backcross, double cousins, 50 to 60 generation inbred
lineages, 120 random pedigrees) and on four simACE pedigrees from 20k to 536k
rows, and ran every fitACE and pedsum consumer under a float32 result.

* Every kinship value is a dyadic rational. On every simACE pedigree and every
  shallow fixture the float32 and float64 values are bit-identical; no consumer
  output, threshold decision, or inbred flag changed. Values that float32
  cannot hold need more than 24 significant bits, which takes about 24
  generations of sustained inbreeding loops.
* The two paths did **not** agree. The matrix DP rounds to float32 every
  generation and peels the deeper endpoint (depth-major order, ties by row),
  while the pairwise kernel peels the larger row. On deep inbred pedigrees
  they differ by up to 2 float32 ulps, and a float32 kernel that keeps
  row-order peeling differs from both.
* Making the pairwise kernel peel by depth then row, in float32, made it
  bit-identical to the matrix on every corpus, including row-permuted
  pedigrees, and independent of endpoint order (0 differences over 490k pairs
  where row-order peeling gave 5,807).
* The memo, not the output, dominates memory. A float32 memo value cuts each
  slot from 16 to 12 bytes and peak RSS by 20 to 24 percent at 536k rows; the
  output cast alone saves nothing measurable.

## Decision

The package defines the kinship coefficient it returns as the value of **one
float32 Karigl recurrence with a pinned peel rule**, and both kernels implement
it:

* `phi(a, a) = (1 + phi(mother_a, father_a)) / 2`; MZ co-twins take the
  self-kinship of the higher row; a missing parent contributes 0.
* Otherwise peel the endpoint with the greater structural depth, ties broken
  by the larger row index: `phi(a, b) = (phi(mother_c, o) + phi(father_c, o)) / 2`
  where `c` is the peeled endpoint. Each step is the correctly rounded float32
  of the exact half-sum of two float32 operands.

Consequences of that definition:

* `pair_kinship(first_rows, second_rows)`, `pair_kinship(block)`, and
  `pair_kinship(pairs)` return read-only **float32**. The memo stores float32.
  The pairwise kernel takes structural depth as an input.
* `kinship_matrix()` and `relationship_kinship_matrix(...)` entries are
  **bit-identical** to `pair_kinship` for the same pair. That parity is a
  property test on every fixture, including deep inbred and row-permuted ones,
  and in view coordinates. ADR 0006's "only this final float32 rounding" is
  replaced by this per-step, pinned rounding.
* `pair_kinship` is **recurrence-only**. It never reads a cached matrix, so its
  result does not depend on call history. Issue #6 closes on this; the
  second-graph workaround in fitACE (`fitace/kinship/kinship.py`) can be
  deleted once 0.8.0 ships.
* Zero is exact: a returned 0 means the exact kinship is 0, and reversed
  endpoint order gives identical bits.
* Bit parity is a **within-graph** property. For one constructed graph, pair
  and matrix values agree bit-for-bit, and so do the two endpoint orders.
  Two graphs built from the same pedigree in different input-row orders are
  a different case: the peel rule breaks depth ties by row, so a permutation
  can change the evaluation order of a deep inbred pair and hence its
  rounding path. Such graphs must agree within the recurrence envelope
  `abs(a - b) <= 2 * (depth_a + depth_b + 1) * 2**-25`, the sum of one
  half-ulp per rounding step along both peel paths. Relationship
  categories, pair sets, and every integer output remain exactly invariant
  under permutation. On the review corpus (571,305 random deep-inbred pairs)
  7,735 pairs differed, by at most 4 ulps; that number is diagnostic, not a
  contract, and tests report ULP distance alongside the envelope check.
* Canonical `inbreeding()` stays the float64 Meuwissen-Luo walk of ADR 0008
  (1 s and 19 MB at 536k rows, against 6 s and 1.2 GB through self pairs).
  `F_i = 2·phi(i,i) − 1` remains a tested invariant with tolerance `2^-22`
  on every fixture and exact equality on shallow ones.
* Callers that threshold `pair_kinship` values against a non-dyadic cutoff
  must widen to float64 first; under NumPy's NEP 50 a float32 array compared
  with a Python float compares in float32, which can admit a pair one float32
  ulp below the cutoff. fitACE's pair export widens at its comparison when it
  migrates; today its degree gate makes the comparison inert.

## Considered options

* **float64 output, float64 internal** (ADR 0005 and 0006 as written). Exact
  rounded once, but the matrix can never match it on deep pedigrees without a
  float64 matrix, and every consumer narrows anyway. Rejected for the parity
  gap and the foregone memo saving.
* **float32 output, float64 internal.** Correctly rounded; same parity gap
  with the matrix; no memory benefit beyond the output array. Rejected once
  the pinned rule showed bit parity was reachable.
* **float32 memo without changing the peel rule.** Takes the memory win but
  leaves pairwise and matrix disagreeing by up to 2 ulps. Rejected.
* **Allow `pair_kinship` to sample a cached complete matrix.** Now
  semantics-preserving, but the complete matrix exists only on pedigrees small
  enough that the recurrence is already cheap (53k rows: 6.8 s to build the
  matrix, 5.7 s for the degree-3 recurrence) and a second path must be kept in
  parity forever. Rejected in favour of one path.
* **Derive `inbreeding()` from the recurrence** for an exact identity.
  Rejected on the measured cost above.

## Consequences

* The error of a returned value against the exact rational is bounded by the
  recurrence depth times `2^-25`; measured worst case `1.5e-7` on a
  50-generation closed herd and 0 on every simACE pedigree.
* The peel rule is part of the value's definition. The Rust core (ADR 0007)
  implements the same rule, and cross-implementation parity tests compare
  bits, not tolerances.
* The rehash copy in the pairwise memo (old and new tables coexist) costs
  about 30 percent of peak RSS at 536k rows independent of dtype. That is a
  memo-layout requirement for the Rust core, not a 0.8.0 Python change.
* ADR 0005's "output dtype is float64" and "samples the exact
  `kinship_matrix(0.0)` if already cached" clauses are superseded by this ADR.
