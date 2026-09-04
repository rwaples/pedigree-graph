# ADR 0008: `inbreeding()` is the Meuwissen–Luo walk over the genome-node pedigree

**Status:** accepted
**Date:** 2026-09-04
**Context:** resolves issue #8, a blocker named by ADR 0006 for the 0.8.0 kinship slice

## Context

ADR 0006 fixes the contract `F_i = 2·phi(i, i) − 1` for canonical `inbreeding()`
and leaves open how it is computed and whether the 0.7.1 MZ-naive
Meuwissen–Luo kernel survives. The 0.7.1 kernel treats MZ co-twins as two
individuals, so an inbreeding path that runs through both members of an MZ
pair is under-weighted (twins with parents count as full sibs, kinship 1/4
instead of 1/2) or missed entirely (founder twins count as two unrelated
founders).

The obvious reading of the contract, derive F from the pairwise recurrence's
self pairs, was benchmarked and fails at production scale. Its memo holds the
ancestor-pair closure of every mating, and on a 1M-row, nine-generation
simACE pedigree it passed 26 GB of RSS after 48 minutes without finishing.
The current kernel does the same pedigree in 18 s and 78 MB.

## Decision

`inbreeding()` runs the Meuwissen–Luo ancestor walk over the **genome-node
pedigree**: every row maps to a genome node, MZ co-twins share the node with
the lower row index, the walk follows the parent's genome node instead of the
parent row, and a non-canonical twin row copies its node's F and Mendelian
sampling variance. Because ADR 0006 requires represented MZ references to be
reciprocal, two-member, and parent-identical, this one-line canonicalisation
is exact on any graph that passed construction.

The MZ-naive kernel is deleted in the same slice. `compute_inbreeding()`
becomes an adapter over `inbreeding()` until slice 7 removes it, so it returns
MZ-aware values from slice 5 onward rather than preserving 0.7.1 output.

`F_i = 2·phi(i, i) − 1` is a tested invariant, not the implementation:
parity tests hold `inbreeding()` equal to `pair_kinship` self pairs and to
`2·diag(kinship_matrix()) − 1` on fixtures covering MZ ancestry with and
without loops, founder twins, and inbred twins.

## Consequences

* No slower than the 0.7.1 kernel: on the 1M-row, nine-generation pedigree
  16.0 s against 16.35 s, on a 600k-row pedigree without twins 1.02 s
  against 1.01 s (medians of interleaved fresh-process runs). Peak RSS rises
  by about 2% on pedigrees with twins, from two canonicalised parent arrays
  allocated only when twin rows exist.
* F never decreases. On the deepest simACE pedigree benchmarked, 0.12% of
  individuals change, every one with an MZ ancestor, by at most 1/128; mean F
  moves from 2.276e-5 to 2.285e-5. Effective-size estimators, pedsum, and
  fitACE's inbreeding export see shifts under 1%, so 0.7.1 golden values on
  pedigrees with MZ twins need a tolerance, not bit-exact parity.
* The Rust port (ADR 0007) ports this walk, not a diagonal extraction.

## Alternatives considered

* **`2·phi(i, i) − 1` via the pairwise recurrence.** Exact, but out of memory
  at 1M rows for the reason above.
* **Uncapped kinship-matrix diagonal.** Exact, but materialises the complete
  matrix, the cost ADR 0005 exists to avoid.
* **Keep the MZ-naive kernel under an explicit name.** No caller wants
  MZ-naive F, the correction is small and one-directional, and a second
  kernel through the Rust port is maintenance without a consumer.

Benchmark record: pedigree-graph issue #8, comment of 2026-09-04.
