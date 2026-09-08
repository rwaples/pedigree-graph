# ADR 0011: the scalar estimate labels only six codes exact and has no lineal opt-in

**Status:** accepted
**Date:** 2026-09-05
**Context:** refines the `relationship_counts` / `estimate_relationship_counts` contract of ADR 0006 (slice 4c of the 0.8.0 plan)

## Context

ADR 0006 replaces `count_pairs_streaming` with `estimate_relationship_counts`,
which returns a `RelationshipCountResult` whose `exact`, `approximate`, and
`clamped` sets say per requested code how the value was obtained. The
registry's `streaming_exact` flag marks ten codes (MZ, MO, FO, FS, MHS, PHS,
GP, GGP, GGGP, G3GP) as bit-identical between the scalar counter and the
matrix engine, and the first cut of slice 4c put exactly those ten in
`exact`.

That flag was defined against the 0.7.1 matrix engine, which counts every
pair in every category it belongs to. The 0.8 `relationship_counts` applies
the closest-category precedence fold (ADR 0006 pair contract 4): a pair is
filed once, under the lowest-degree category and then registry order. The
scalar formulas count raw structure. The two disagree wherever a pair is
related in two ways:

| fixture | fold removals of the six affected codes, by claiming category |
|---|---|
| backcross_and_selfing_like | MHS by FO 2, PHS by MO 2, GP by MO/FO 4, GGP by GP 6, GGGP by GP 2 and GGP 3, G3GP by GGP 2 |
| one_parent_known | GGP by GP 1 |
| random_1k | GGP by GP 12, GGGP by GGP 71, G3GP by GGP 4 and GGGP 223 |

A backcross child's father is also its maternal grandfather, so the fold
files the pair under FO and the GP count drops by one. Only MZ, MO, FO, and FS
can never be claimed by a closer category; everything else in the ten can.

The plan's done-criterion for the slice is that the result metadata
truthfully distinguishes exact from approximate. `exact` meaning "exact under
the semantics of the method this one replaces" fails that.

## Options considered

1. **Fold-aware correction for all six affected codes.** Subtract, per code,
   the pairs a closer category claims. For MHS and PHS the only closer claims
   are parent-offspring, an O(N) predicate on the parent arrays. For the four
   lineal codes the closer lineal and half-sib claims are an elementwise
   product of each adjacency power against the union of the shorter ones
   plus a gather over the power's entries.
2. **Shrink `exact`** to the codes whose correction is free: MZ, MO, FO, FS,
   MHS, PHS. The four lineal codes become `approximate`.
3. **Option 1 behind an opt-in keyword** `exact_lineal=True`, with option 2
   as the default.
4. **Keep the ten** and document `exact` as "exact path count before
   precedence".

## What was measured

The lineal correction was implemented and benchmarked, interleaved against
`count_pairs_streaming` at the previous commit, five runs per block, two
stash cycles, medians of ten, peak RSS from the child's `ru_maxrss`:

| pedigree | wall ratio | RSS ratio |
|---|---|---|
| random_30k | 1.24 | 1.02 |
| random_300k | 1.73 | 1.14 |

A copy-free rewrite of the overlap step measured within noise of that, and
key-membership on int64 `(row, col)` keys was ten times slower, so the cost
is the power products themselves, not the implementation. The counter's own
work is a handful of `nnz` reads and `bincount` sums, so any per-entry pass
over the fifth power is a large fraction of it. That fails the project's
5 % regression gate by a wide margin, which ruled out option 1 as the
default and led to option 3.

Option 3 was then implemented, its default path measured within 1 % of
baseline wall time with identical peak RSS, and its opt-in path at the
ratios above. A code
review of that tree fuzzed the opt-in against `relationship_counts` with the
parity generator's skip-generation setting. A lineal pair that is also a
lower-degree *collateral* pair (a great-grandmother who is also an aunt, a
great-great-grandparent who is also a first cousin once removed) is filed
under the collateral code by the fold and left in the lineal count by the
correction, because counting it needs the avuncular and cousin membership
rules on the lineal nonzeros, which is the machinery the memory-bounded
path exists to avoid:

| generator | seeds where a code in `exact` disagreed with `relationship_counts` |
|---|---|
| `random_pedigree(p_skip_generation=0.5)`, 60 seeds | 24 (G3GP 19, GGGP 13) |
| same, reviewer's run, 100 seeds | 39 |
| reviewer's close-mating generator, 100 seeds | 98 |

The residual had been described as narrow. It is routine under inbreeding.
The opt-in therefore bought a better approximation and labelled it `exact`,
the same untruth the slice set out to remove.

## Decision

* `estimate_relationship_counts(*, max_degree)` has no `exact_lineal`
  keyword and no lineal correction. `exact` is `requested ∩ {MZ, MO, FO, FS,
  MHS, PHS}`. GP, GGP, GGGP, and G3GP are `approximate`: raw ancestor-path
  counts that over-count a pair also related at a shorter depth, as a
  half-sib, or as a closer collateral. `exact` carries no footnote.
* The MHS and PHS parent-offspring correction stays: it is one O(N) pass,
  measured at no cost, and it is complete. The only categories closer than
  the half-sib codes are MZ, MO, FO, and FS; MZ and FS pairs share both
  parents and are excluded from the half-sib blocks by definition, so the
  only possible closer claim on a half-sib pair is parent-offspring, which
  the correction counts exactly.
* The registry is the single source of truth. `EngineSupport.estimate_exact`
  names the six codes and `estimate_exact_codes()` derives the set. The
  0.7.1 `streaming_exact` flag kept its original meaning (the unfolded
  `count_pairs_streaming` equals the unfolded `count_pairs`) until slice 7
  deleted it with the adapter and tests that read it.
* Exact lineal counts are what `relationship_counts` is for. A caller who
  needs them on a pedigree too large for the matrix engine waits for the
  row-streaming engine of ADR 0010, which classifies each pair once by
  construction and does not have this problem.

## Consequences

* The default estimate costs the same as the 0.7.1 streaming path. The
  final interleaved benchmark (`benchmarks/bench_estimate_counts.py`, five
  runs per block, two stash cycles, medians of ten) measured wall ratios of
  1.005 on 30k rows and 1.007 on 300k rows with peak RSS ratios of 1.000 on
  both. Its `exact` set is exact by construction, and a 120-pedigree fuzz
  (skip-generation and missing-parent settings of the parity generator)
  found no disagreement with `relationship_counts` on any exact code.
* Four codes that are exact on every pedigree without inbreeding loops are
  labelled approximate. That is the price of the label meaning one thing.
* `benchmarks/bench_estimate_counts.py` remains as the rerunnable
  default-versus-baseline gate for future changes to the counter.
* The lineal correction code is deleted, not kept behind a flag or in a
  branch. If the row-streaming engine lands, the question is moot; if a
  cheaper exact formulation appears, it starts from this ADR's measurements.

## Rejected

* **Option 4, documenting the ten as "exact before precedence".** It keeps a
  public result whose `exact` set disagrees with the package's own exact
  method on inbred data.
* **Option 3, the opt-in.** Implemented, measured, fuzzed, removed: the
  fuzz numbers above are the reason.
* **A tri-state registry flag** (`always / with_exact_lineal / never`). No
  longer needed once the opt-in is gone; two booleans with two lifetimes
  (`estimate_exact` permanent, `streaming_exact` deleted with 0.7.1) say it
  more plainly.
