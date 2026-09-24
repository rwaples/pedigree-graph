# ADR 0011: the scalar count covers only six exact codes and has no lineal opt-in

**Status:** accepted
**Date:** 2026-09-05
**Context:** refines the `relationship_counts` / scalar-count contract of ADR 0006 (the 0.8.0 counts work); narrowed by issue #17 in 0.9.0

Revised 2026-09-24 to match 0.10.0; earlier wording in git history.

## Context

ADR 0006 replaced `count_pairs_streaming` with `estimate_relationship_counts`,
which returned a `RelationshipCountResult` whose `exact`, `approximate`, and
`clamped` sets said per requested code how the value was obtained. The
registry's `streaming_exact` flag marked ten codes (MZ, MO, FO, FS, MHS, PHS,
GP, GGP, GGGP, G3GP) as bit-identical between the scalar counter and the
matrix engine, and the first 0.8.0 cut put exactly those ten in
`exact`.

That flag was defined against the 0.7.1 matrix engine, which counted every
pair in every category it belonged to. The 0.8 `relationship_counts` applies
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

The plan's done-criterion for the slice was that the result metadata
truthfully distinguish exact from approximate. `exact` meaning "exact under
the semantics of the method this one replaces" failed that.

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

Issue #17 later removed the approximate codes altogether; the decision
below is the result of both steps.

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
was the power products themselves, not the implementation. The counter's own
work was a handful of `nnz` reads and `bincount` sums, so any per-entry pass
over the fifth power was a large fraction of it. That failed the project's
5 % regression gate by a wide margin, which ruled out option 1 as the
default and led to option 3.

Option 3 was then implemented, its default path measured within 1 % of
baseline wall time with identical peak RSS, and its opt-in path at the
ratios above. A code
review of that tree fuzzed the opt-in against `relationship_counts` with the
parity generator's skip-generation setting. A lineal pair that is also a
lower-degree *collateral* pair (a great-grandmother who is also an aunt, a
great-great-grandparent who is also a first cousin once removed) is filed
under the collateral code by the fold and was left in the lineal count by
the correction, because counting it needs the avuncular and cousin
membership rules on the lineal nonzeros, which is the machinery the
memory-bounded path existed to avoid:

| generator | seeds where a code in `exact` disagreed with `relationship_counts` |
|---|---|
| `random_pedigree(p_skip_generation=0.5)`, 60 seeds | 24 (G3GP 19, GGGP 13) |
| same, reviewer's run, 100 seeds | 39 |
| reviewer's close-mating generator, 100 seeds | 98 |

The residual had been described as narrow. It is routine under inbreeding.
The opt-in therefore bought a better approximation and labelled it `exact`,
the same untruth the slice set out to remove.

## Decision

* The scalar API is `close_relative_counts()`. It replaced
  `estimate_relationship_counts(*, max_degree)` in 0.9.0 (issue #17) with
  no alias. It always computes MZ, MO, FO, FS, MHS and PHS, exactly under
  closest-category precedence. Its mapping has all 23 registry keys, with
  `None` for every other category, and `requested == exact` is the six-code
  set. It takes no degree or category selector, is full-graph only, needs
  only the parent and twin arrays and sibling-group sums, and caches one
  immutable result per graph.
* There is no `exact_lineal` keyword and no lineal correction. GP, GGP,
  GGGP and G3GP are not computed by the scalar path. In 0.8 they were
  reported as `approximate`: raw ancestor-path counts that over-count a
  pair also related at a shorter depth, as a half-sib, or as a closer
  collateral. Issue #17 removed those and the other approximate formulas,
  clamps, the warning and the adjacency-power release.
* `RelationshipCountResult.approximate` and `.clamped` are removed, from
  exact graph and view results too. Keeping permanently empty fields would
  misrepresent an API that no longer computes approximations. The breaking
  change is in the 0.9.0 changelog.
* The MHS and PHS parent-offspring correction stays: it is one O(N) pass,
  measured at no cost, and it is complete. The only categories closer than
  the half-sib codes are MZ, MO, FO, and FS; MZ and FS pairs share both
  parents and are excluded from the half-sib blocks by definition, so the
  only possible closer claim on a half-sib pair is parent-offspring, which
  the correction counts exactly.
* The registry is the single source of truth. `EngineSupport.estimate_exact`
  in `REL_PLAN` names the six codes and `estimate_exact_codes()` derives the
  set; both keep their internal names. The 0.7.1 `streaming_exact` flag kept
  its original meaning (the unfolded `count_pairs_streaming` equals the
  unfolded `count_pairs`) until 0.8.0 deleted it with the adapter and
  tests that read it.
* Exact counts for every category are what `relationship_counts` is for. It
  runs the row-streaming Rust engine of ADR 0010, which classifies each pair
  once by construction, in linear memory. The scalar method remains for
  callers who need only close relatives and want to skip classifying more
  distant pairs. pedsum made that move: its CLI calls
  `relationship_counts(max_degree=5)` (pedsum commit 4869144,
  `pedsum/cli.py`).

## Consequences

* When the six-code set was first adopted, the estimate cost the same as
  the 0.7.1 streaming path. The final interleaved benchmark (five runs per
  block, two stash cycles, medians of ten) measured wall ratios of 1.005 on
  30k rows and 1.007 on 300k rows with peak RSS ratios of 1.000 on both. A
  120-pedigree fuzz (skip-generation and missing-parent settings of the
  parity generator) found no disagreement with `relationship_counts` on any
  exact code.
* `benchmarks/bench_estimate_counts.py` now records a one-arm baseline of
  `close_relative_counts` on `random_30k` and `random_300k`. It has no gate:
  the `count_pairs_streaming` arm it compared against was deleted with the
  0.7.1 adapters.
* The lineal correction code is deleted, not kept behind a flag or in a
  branch. The row-streaming engine has landed and gives exact lineal counts,
  so the question is moot for callers who can use `relationship_counts`. If
  a cheaper exact scalar formulation is ever wanted, it starts from this
  ADR's measurements.

## Rejected

* **Option 4, documenting the ten as "exact before precedence".** It keeps a
  public result whose `exact` set disagrees with the package's own exact
  method on inbred data.
* **Option 3, the opt-in.** Implemented, measured, fuzzed, removed: the
  fuzz numbers above are the reason.
* **A tri-state registry flag** (`always / with_exact_lineal / never`). Not
  needed once the opt-in was gone; two booleans with two lifetimes
  (`estimate_exact` permanent, `streaming_exact` deleted with 0.7.1) said it
  more plainly.
