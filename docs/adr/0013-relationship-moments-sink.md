# ADR 0013: relationship moments are an exact fixed-point sink on the row-streaming engine

**Status:** accepted
**Date:** 2026-09-30
**Context:** issue #28 (simACE #25); builds on the pair sinks of ADR 0010 and
the host boundary, thread and allocation rules of ADR 0007, which this ADR
qualifies for one operation. Plan: simACE
`plans/relationship-moments-v10.md`.

## Context

Every consumer that reduces relationship pairs to a small table did so from
a pair list: `relationship_pairs`, then NumPy sums per category and stratum.
At 20M rows the degree-5 pair list is 15.8 GiB (ADR 0010 evidence), and
simACE's validate step capped the pairs it read at 5,000 per category to
stay inside a job. The statistics wanted are sufficient statistics: pair
counts, sums, sums of squares and cross sums over caller-supplied
per-individual values, keyed by a few per-member strata and flags. They can
be accumulated in the engine's row pass with no pair list at all.

Three families of consumers were checked (plan D10): per-category
label-keyed sums (simACE stats and validate, fitACE Falconer, PA-FGRS,
the fitACE_epimight sandwich), per-person sums over relatives (pedsum,
fitACE_epimight `create_input`), and pair lists as the product (tetraher,
pedsum `--pairs`). This ADR serves the first in 0.11.0 and leaves the hooks
the second needs.

## Decision

### A reducer trait over `emit_row`

`crates/core/src/relationships/moments.rs` adds `reduce_pairs`, which drives
`Engine::emit_row` (ADR 0010) against a `Reducer`: per lane a `Lane`
accumulator, `reduce(lane, category, first, second)` on the hot path with no
`Result`, and `merge`. Pairs reach the reducer in the category's semantic
orientation and the receiver's rows, exactly as pair blocks do, so a
reducer sees what a pair-list consumer would have seen. The only 0.11
implementation is the `CellReducer`. A per-person reducer (counts and sums
of the other member's values per person, category and cell) is planned for
0.12 with pedsum#3 and the fitACE_epimight `create_input` migration; it
plugs into the same pass, with a lane that is a shared array of integer
atomics and an empty merge.

### Named factors, packed by the host into per-role pair labels

The engine is keyed by one int32 pair label per individual per role
(`labels_first`, `labels_second`) and a label count per role. The Python
boundary (`pedigree_graph/_relationship_moments.py`) takes named factors
(`first={"generation": g, "sex": s}`, `second={"sex": s}`), maps each
factor's distinct values to contiguous ascending levels with `np.unique`,
packs a role's factors by mixed radix (last factor fastest) into its label,
and keeps the levels as the axes of the result. `second=None` reuses the
first member's packing. Per-role labels are what every stratified consumer
needs, and they keep the cell count at `L1 × L2` rather than the square of
one shared label space: at 36 generations × sex × two flags, 288 × 8 cells
against 288² (plan D1 arithmetic).

### Cells: product sides, symmetric orientation, equality keys

A cell is `(first label, second label, equality bits)` per requested
category, in mixed-radix order with the first equality key most
significant. Per cell the reducer keeps the pair count, per value column
the sum and sum of squares over each member, and per requested product the
sum of the two operands' product. A product names each operand's side,
`("first.x", "second.y")`, so first × first and second × second products
serve partial correlations and same-member covariates; the default is the
first × second diagonal of every column.

`symmetric="canonical"` reduces a symmetric pair once, lower receiver row
first (the pair-block rule). `symmetric="both"` reduces it in both
orientations and leaves asymmetric categories unchanged, for intraclass
correlations and first-member stratification of symmetric codes.

`same={"household": hh}` adds one cell bit per key: whether the two
members' key values are equal, where a negative value never equals anything
so unknowns are "not same". It costs one integer compare per pair per key
and doubles the cell count. A difference-bucket key (generation gap) would
be a new keyword whose keys add axes with more than two levels, an
additive change: existing `same=` calls, cell layouts and result axes stay
as they are.

### Fixed-point `i128` accumulation with integer-exponent scales

Values are quantized once per individual and column, before the compact or
full execution path is chosen, so both paths see the same integers. For a
column with `M = max|x|` over the receiver's rows the exponent `e` is the
largest integer with `M · 2^e <= 2^43`, read from the binary exponent
(`frexp`) rather than computed in floating point, so a subnormal `M` gets
`e > 1023` and a near-maximal `M` a negative `e` without overflow; an
all-zero column gets `e = 0`. `q = rint(ldexp(x, e))` rounds half to even,
and `ldexp` scales in one exact step. The absolute quantization error is
below `2^-43 · M` per value.

The engine sums the integers in `i128`. A term is at most `2^86`, so a cell
overflows only past `2^40` pairs; one check on the cell counts after the
pass proves no partial overflowed, and there is no per-pair check. Merges
are integer sums, so the result is the same for every thread count, lane
count and view execution path. The merged integers cross the boundary as
they are, split into int64 halves; the engine derives no float. Non-finite
inputs are rejected; masking is by a factor level, and because the scale is
taken over every receiver row a masked row should still carry a finite
value of ordinary magnitude.

### Exact accumulators in the host, floats derived on access

`RelationshipMoments` (`pedigree_graph/moments.py`) keeps the accumulators
as NumPy object arrays of Python `int` (`counts`, `q_sum_first`,
`q_sum_second`, `q_sumsq_first`, `q_sumsq_second`, `q_cross`) with the
per-column exponents beside them. It is a labelled table over the axes
`category`, `first_<factor>`, `second_<factor>` and `same_<key>`;
`select` narrows an axis, `sum` folds one away and `merge` combines two
results over the same axes, and all three are integer additions, exact and
order-independent. `merge` aligns columns whose exponents differ by
multiplying the coarser operand's integers up to the finer exponent, so
the merged result carries the larger exponent and no rounding. Grouping
categories (`PO` from `MO` and `FO`, `HS` from `MHS` and `PHS`) is a fold,
not an engine argument.

Every float view (`sum_first`, `sumsq_first`, `cross`, `m2_first`,
`comoment`, `mean`, ...) is derived on access by one path: an exact
integer numerator (`n·Σq² − (Σq)²` and `n·Σq_a q_b − Σq_a Σq_b` for the
centered moments), and the exact rational `numerator / (n · 2^e)` (with
`n = 1` for raw sums) rounded once to the nearest `f64`, ties to even, by
Python's integer true division. The integers can far exceed `f64` after a
merge aligns very different exponents; only the scaled value has to fit,
and one that does not raises `ValueError` naming the column (finite in,
finite out; underflow follows `f64`). This is one rounding, not the plan's
two (convert, then divide by `n`). `pearson` is
`sign(N_ab) · sqrt(N_ab² / (N_aa · N_bb))`: the ratio is exact integer
arithmetic rounded once and at most 1 by Cauchy-Schwarz, so column scales
cancel, nothing overflows or underflows, and `|r| <= 1`; a constant
operand gives NaN. Axis levels are copied and frozen like the arrays.
Consequences: a constant
column has `m2 = 0` exactly, before and after any fold or merge; swapping
the two members swaps `m2_first` and `m2_second` and leaves a diagonal
co-moment unchanged, so its `r` is bit-identical under the swap; a column
whose mean dwarfs its spread loses nothing to cancellation; and the
centered moments of a fold equal those of a direct engine call bit for
bit. Tetrachoric correlation stays with the consumer, since it is a
statistic rather than pedigree structure.

The plan (D10.2) had the centered co-moments computed in the engine and
folded in the host with the pairwise update
`M = M_a + M_b + δ_x δ_y n_a n_b / n`. That fold runs on float64-rounded,
already-scaled sums, so after a fold a constant non-dyadic column showed
`m2 ≈ 7e-31` and a finite, fabricated `r`. Keeping the integers and
deriving the floats last is what makes the algebra exact, and it removed
the engine's 256-bit arithmetic.

### A memory budget and pooled lanes

`relationship_moments` takes `memory_budget_bytes` (default 1 GiB) as a
ceiling on its accumulators. The plan sizes one lane
(`categories × cells × (1 + 4k + P)` accumulators of 16 bytes) and the
host's copy of the merged accumulators once (72 bytes per accumulator: the
two int64 halves the binding hands over, 16, plus the object-array slot, 8,
and a CPython 3.13 `int` of up to `2^126`, 48) plus the scratch of the
conversion, which runs 4096 accumulators at a time at up to 128 bytes each
(about 100 measured with `tracemalloc`; converting the whole table at once
peaked near 153 bytes per accumulator), with every size a checked product, refuses with the structured `ResourceError("memory_budget_exceeded")`
before anything is allocated when one lane plus the host copy does not fit
or a size is not representable, and otherwise runs on
`W = min(thread budget, largest W whose W lanes plus the host copy fit)`
lanes. The lanes are allocated up front and run side by side inside the one
package pool, each pulling 2048-row task ranges from a shared cursor until
one lane fails, so at most `W` accumulators are live whatever the pool's
thread count; the merge is in place into the first lane. Rayon
`fold`/`try_reduce` identities were not used because they allocate per
split and their live count is not bounded by `W`. The effective `W`, the
planned peak and the pairs each lane reduced are returned and logged. A
degenerate call (no categories, fewer than two receiver rows) plans the
same sizes and is refused the same way without running the engine.

Outside the estimate, as part of the job's RSS: the graph, the input
arrays and the quantized values, and one engine workspace per lane (about
9 bytes per graph row each, as for `relationship_pairs`). Each equality
key doubles the cells; at most 16 keys are accepted.

This qualifies ADR 0007's "no default memory budget is set": this one
operation has a default budget, and its lane limit restricts concurrency
within the shared pool rather than sizing a pool.

## Rejected

* **Sampling pairs** (simACE's 5,000-pair cap): biased `n_pairs`, and every
  statistic downstream inherits the sampling error. Exact sums over every
  pair cost one engine pass.
* **Task-ordered `f64` partials**: about 10k partials at 20M rows, and the
  compact and full view paths would differ in their last bits. Integer
  accumulation makes the result independent of partition.
* **A shared label array for both members**: squares the stratum count and
  would cut the lane count under the default budget at 36 generations.
* **One small table per requested breakdown** (plan option B): grows
  linearly in the number of traits where the joint table grows as
  `144 · 4^T`, but simACE has two traits and the joint table is small;
  revisit if a scenario needs `T >= 3`.

## Consequences

* One engine pass replaces the pair list for pattern-A consumers; on
  `pedsum_2M` the numbers are in `benchmarks/bench_relationship_moments.md`.
* Pearson correlations against a NumPy float oracle move to a tolerance
  (about `1e-12`); pair-swap symmetry, row-permutation invariance and the
  result algebra stay exact.
* Two allocation families (`moment_lanes`, `moment_output`) and one error
  code (`memory_budget_exceeded`) join the tables; `relationships/moments.rs`
  joins the parallel-module map with its cross-budget test.
* The R package does not expose moments in 0.11.0 (issue #30).
