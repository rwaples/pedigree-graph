# ADR 0010: the relationship engine streams rows and saturates multiplicity at two

**Status:** accepted
**Date:** 2026-09-04
**Context:** resolves issues #11 (memory-bounded exact counts) and #9 (multiplicity overflow); refines the relationship-engine invariants of ADR 0007

Revised 2026-09-24 to match 0.10.0; earlier wording in git history.

## Context

When this was decided, neither pair engine gave an exact count on a 20M-row
pedigree with 30 GiB of RAM. The SciPy matrix engine (`_pair_extractor.py`)
was exact but materialised every pair list up to degree 5 at about 75 bytes
per pair, 150 to 560 GiB extrapolated. The scalar counter
(`count_pairs_streaming`) was O(N) but its cousin and collateral categories
were inclusion–exclusion residuals with fixed coefficients that diverged and
clamped to zero on real pedigrees.

ADR 0007 committed the production relationship engine to Rust and left two
questions open: how its intermediates are bounded, and how path
multiplicities avoid silent wrap-around (issue #9). The first Rust spike
carried unchecked `i32` multiplicity and materialised the same global
products as the Python engine.

## Decision

### Rows are the unit of work

Every relationship category in the reference engine is a sparse product whose
row `i` depends only on row `i` of its left operand, followed by a
multiplicity predicate and a subtraction of closer categories. Row `i`
therefore sees every path that decides pair `(i, j)`. The engine classifies
all pairs one row at a time and never holds a pair list. Its global state is
the parent CSR, its transpose, the sibling index by original parent id, and
the parent arrays, all linear in N. Per-thread scratch is one sparse
accumulator (a 32-bit stamp marker and a one-byte multiplicity per row) plus
the current row's relative sets.

A pair is unordered, and the reference engine takes the nonzeros of a possibly
asymmetric product in both orientations before canonicalising. So each row
evaluates `M[i, :]` and `M^T[i, :]`; the transposed row of `X @ Y^T` is
`Y[i, :] @ X^T`, and every operand chain reduces to expansions upward through
the parent CSR and downward through its transpose. Each unordered pair is
counted once, at its lower row. Total product work is about twice the matrix
engine's; memory is what the issue asked to bound.

Rows are independent, so the work is split into row ranges over the Rayon
pool with a workspace pool that never exceeds the thread count. Counts are
integers summed in any order and are bit-identical across thread counts.

### Categories are the 0.7.1 definitions, with exclusions as a table

The engine's category definitions reproduce `count_pairs(max_degree=5)` of
pedigree-graph 0.7.1 bit for bit, including two idiosyncrasies that must
not be "fixed" silently: first cousins count *distinct* shared grandparents
while the removed cousins and second cousins count *paths*, and the first
cousin sibling exclusion is "shares a known parent id", which is wider than
the twin-filtered sibling lists the collateral categories subtract. The
per-category subtraction lists are one constant table (`EXCLUSIONS`).

### Closest-category reporting is a separate fold

The ADR 0006 closest-category rule ("lowest degree, then registry
precedence") is **not** a change to `EXCLUSIONS`. It is a per-row
precedence fold applied to the final category sets after classification:
each category loses the members every earlier registry category claims,
the operation the Python `_fold_precedence` performed on whole blocks.
Definitions decide membership; the fold decides reporting (`CONTEXT.md`,
"Closest category"). The fold is always on, so `count_pairs` returns the
counts `PedigreeGraph.relationship_counts` publishes, and the parity
fixtures under `crates/core/tests/fixtures` hold those folded counts. The
engine has no mode that returns the 0.7.1 raw counts.

### Multiplicity is saturated at two

The engine asks three questions of a path count: zero, exactly one, at least
two. The map `s(n) = min(n, 2)` respects both semiring operations under
saturation: `s(a + b) = min(s(a) + s(b), 2)` and
`s(a * b) = min(s(a) * s(b), 2)`. So every product is evaluated in one byte
per entry, nothing can overflow, and every decision is exactly the decision
unbounded integers would make. This is issue #9's third option, with the
proof in the module doc of `multiplicity.rs` and an exhaustive unit test.
Checked or wide integer arithmetic is not used in the pair engine.

### Pair sinks and orientation provenance

The classifier has more than one consumer. Counting adds the members above
the owner row after the precedence fold. Pair emission instead hands each
owned pair to a sink in the category's semantic orientation. To do so the
engine records, only in workspaces built for emission, a per-category
*first arm*: the members for whom the owner row carries `first_role`,
captured after multiplicity filtering and before the two arms of an
asymmetric product are unioned.

* Lineal codes: the up arm (the row is the descendant).
* `MO` and `FO`: the row's own parent (the row is the offspring).
* The avuncular family: the sibs-of-ancestors arm (the row is the niece or
  nephew).
* Removed cousins: the backward arm, in which the row sits the greater
  number of meioses from the shared ancestor and is the junior cousin; the
  Python matrix engine read these products with `row_is_first=False`.

A pair valid in both orientations is in both raw arms, hence in the first
arm, so lower-row ownership reproduces ADR 0006's lower-row tie break.
Exclusions and the precedence fold remove pairs; they never change a
survivor's orientation. Because the owner row rises across ordered task
ranges and every set is sorted, graph-receiver blocks arrive in canonical
key order and are never sorted; view receivers are relabelled through the
graph-to-view map and sorted by the view-space key. Emission and counting
share one pool of workspaces, and emission is byte-identical across thread
counts and across the output assemblies ADR 0007's amendment selects. The
counting path never allocates or fills the first arms. The evidence is in
`benchmarks/bench_pair_emitters.md`.

### Where it lives

`crates/core` is `pedigree-graph-core` under the ADR 0007 layout,
`#![forbid(unsafe_code)]`, unpublished, with the engine in
`src/relationships/`. The Rust toolchain comes from the pedigree-graph pixi
manifest. The PyO3 binding in `crates/python` exposes it as
`pedigree_graph._native`, which `relationship_counts`,
`relationship_pairs` and `relationship_burden` call. `pgr-count` is a
benchmark and parity CLI over the array dump written by
`tests/parity/dump_relationship_inputs.py`.

## Evidence

When the engine landed, its unfolded counts were bit-identical to the
0.7.1 Python matrix engine on all 26 dumped fixtures (the
registry motifs, `random_1k`, `deep_inbred_60g`, `random_30k`,
`small_pedigree`, five random inbred pedigrees with twins and missing
parents, three simACE pedigrees at depth 2, 3 and 6), and on simACE
pedigrees of 120k and 300k rows, at one and at several threads. Those raw
counts remain historical evidence of the per-category sets; the fixtures
now hold the folded counts.

Measured with `/usr/bin/time -v` on the 12-core, 30 GiB workstation, the
`pedsum_2M` and `pedsum_20M` simACE pedigrees (2M and 20M rows, 8 recorded
generations), release build, all 23 categories:

| rows | threads | wall | peak RSS |
|---|---|---|---|
| 2M | 1 | 41 s | 209 MiB |
| 2M | 12 | 6.7 s | 302 MiB |
| 20M | 1 | 498 s | 2.00 GiB |
| 20M | 12 | 83 s | 2.86 GiB |

The 20M result is the issue's acceptance case: 2.1 billion pairs classified
exactly in under 3 GiB where the matrix engine needed an estimated 150 GiB
and the scalar counter reached 12.3 GiB for approximate cousin counts.

## Consequences

* The Rust pair engine of ADR 0007 is this engine with a pair sink in place
  of the counter; it is not a separate implementation. `relationship_burden`
  is a third sink over the same classifier
  (`crates/core/src/relationships/burden.rs`).
* `count_pairs_streaming` and its `streaming_exact` metadata are deleted.
  The only scalar counting path left is `close_relative_counts`, which
  counts six close codes exactly (ADR 0011).
* The SciPy matrix engine is no longer in the package. It is kept verbatim
  as the differential test oracle in `tests/oracle/relationship_pairs.py`.
* Contributors building from source need the Rust toolchain from the pixi
  manifest.

## Alternatives considered

* **Row-block chunking of the SciPy engine** — same locality, but the Python
  engine was scheduled for deletion under ADR 0007, so it would have been
  throwaway.
* **Count-only mode on global products** — the product matrices are the pair
  lists, one entry per candidate pair, so memory is not bounded.
* **Anchoring on shared ancestors** — needs a global merge to apply the
  at-least-two rule across anchors; the row anchor gives the same locality
  with no global state.
* **Checked `u32`/`u64` multiplicity** — correct but adds an error path and
  four to eight bytes per entry for information the predicates never use.

