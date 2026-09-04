# ADR 0010: the relationship engine streams rows and saturates multiplicity at two

**Status:** accepted
**Date:** 2026-09-04
**Context:** resolves issues #11 (memory-bounded exact counts) and #9 (multiplicity overflow); refines the relationship-engine invariants of ADR 0007

## Context

Neither pair engine gives an exact count on a 20M-row pedigree with 30 GiB of
RAM. The matrix engine (`_pair_extractor.py`) is exact but materialises every
pair list up to degree 5 at about 75 bytes per pair, 150 to 560 GiB
extrapolated. The scalar counter (`_streaming_counter.py`) is O(N) but its
cousin and collateral categories are inclusion–exclusion residuals with fixed
coefficients that diverge and clamp to zero on real pedigrees.

ADR 0007 committed the production relationship engine to Rust and left two
questions open: how its intermediates are bounded, and how path
multiplicities avoid silent wrap-around (issue #9). The Rust spike carried
unchecked `i32` multiplicity and materialised the same global products as the
Python engine.

## Decision

### Rows are the unit of work

Every relationship category in the reference engine is a sparse product whose
row `i` depends only on row `i` of its left operand, followed by a
multiplicity predicate and a subtraction of closer categories. Row `i`
therefore sees every path that decides pair `(i, j)`. The engine classifies
all pairs one row at a time and never holds a pair list. Its global state is
the parent CSR, its transpose, the sibling index by original parent id, and
the parent arrays, all linear in N. Per-thread scratch is one sparse
accumulator (an int32 stamp and one byte per row) plus the current row's
relative sets.

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

The engine reproduces `count_pairs(max_degree=5)` of pedigree-graph 0.7.1
bit for bit on every parity fixture, including two idiosyncrasies that must
not be "fixed" silently: first cousins count *distinct* shared grandparents
while the removed cousins and second cousins count *paths*, and the first
cousin sibling exclusion is "shares a known parent id", which is wider than
the twin-filtered sibling lists the collateral categories subtract. The
per-category subtraction lists are one constant table (`EXCLUSIONS`). The
ADR 0006 rule "lowest degree, then registry precedence" is a change to that
table when slice 4c adopts the engine, not a new engine.

### Multiplicity is saturated at two

The engine asks three questions of a path count: zero, exactly one, at least
two. The map `s(n) = min(n, 2)` respects both semiring operations under
saturation: `s(a + b) = min(s(a) + s(b), 2)` and
`s(a * b) = min(s(a) * s(b), 2)`. So every product is evaluated in one byte
per entry, nothing can overflow, and every decision is exactly the decision
unbounded integers would make. This is issue #9's third option, with the
proof in the module doc of `multiplicity.rs` and an exhaustive unit test.
Checked or wide integer arithmetic is not used in the pair engine.

### Where it lives

`crates/core` is `pedigree-graph-core` under the ADR 0007 layout,
`#![forbid(unsafe_code)]`, unpublished, with the engine in
`src/relationships/`. The Rust toolchain comes from the pedigree-graph pixi
manifest. `pgr-count` is a benchmark and parity CLI over the array dump
written by `tests/parity/dump_relationship_inputs.py`; the PyO3 binding and
the `relationship_counts` wiring wait for the native-scaffold slice so this
work does not pre-empt the packaging gates.

## Evidence

Bit-identical to the Python matrix engine on all 26 dumped fixtures (the
registry motifs, `random_1k`, `deep_inbred_60g`, `random_30k`,
`small_pedigree`, five random inbred pedigrees with twins and missing
parents, three simACE pedigrees at depth 2, 3 and 6), and on simACE
pedigrees of 120k and 300k rows, at one and at several threads.

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

* The Rust pair engine of ADR 0007 slice 5 is this engine with a pair sink in
  place of the counter; it is not a separate implementation.
* `count_pairs_streaming` and its `REL_PLAN.streaming_exact` metadata are
  unchanged until slice 4c wires `relationship_counts` to the native engine;
  pedsum keeps reporting approximate cousin counts until then.
* Contributors building from source need the Rust toolchain from the pixi
  manifest; there is no `cargo` on the ambient path.

## Alternatives considered

* **Row-block chunking of the SciPy engine** — same locality, but the Python
  engine is scheduled for deletion under ADR 0007, so it would be throwaway.
* **Count-only mode on global products** — the product matrices are the pair
  lists, one entry per candidate pair, so memory is not bounded.
* **Anchoring on shared ancestors** — needs a global merge to apply the
  at-least-two rule across anchors; the row anchor gives the same locality
  with no global state.
* **Checked `u32`/`u64` multiplicity** — correct but adds an error path and
  four to eight bytes per entry for information the predicates never use.
