# ADR 0015: the moments boundary arithmetic is one core implementation both hosts share

**Status:** accepted
**Date:** 2026-10-01
**Context:** issue #30 (the R binding of `relationship_moments`,
`relationship_counts` and `relationship_burden`). Amends ADR 0013's "Exact
accumulators in the host, floats derived on access" and the host term of
its memory budget. Plan: simACE `plans/pedigree-graph-issue-30-r-moments-v3.md` (session-local draft),
with D1 revised as recorded below.

## Context

Through 0.11 the Python host did all of the arithmetic around the moments
engine pass: it packed named factors into labels with `np.unique`,
quantized each value column with `frexp`, `ldexp` and `rint`, built Python
ints from the engine's int64 halves, and derived every float by Python
integer true division (ADR 0013). The core only accumulated. An R binding
would have needed its own copy of all of it, held to Python by goldens: two
implementations of correctly rounded big-rational division, subnormal
quantization and exponent alignment, kept equal by test data rather than by
construction.

Two defects were in the Python copy. `merge` and `sum` added the int64
`counts` with NumPy, which wraps silently past 2^63 − 1. `__repr__` totalled
the counts the same way.

## Decision

### Core owns the arithmetic

`crates/core/src/relationships/moments_table.rs` holds everything on either
side of the engine pass, and both bindings call it:

- `pack_labels`: mixed-radix labels over each factor's sorted distinct
  values (the occurring values only).
- `quantize_column`: the exponent from the binary exponent of `max|x|`, and
  `q = x · 2^e` rounded half to even in integer steps from the float's bits;
  a value that is not finite is refused (`NonFiniteValue`).
- `MomentsTable`: `select`, `sum` and `merge` (exponent alignment by shifts)
  as exact integer operations, `counts`, `total_pairs`, and `derive`, every
  float statistic per cell (`sum_*`, `sumsq_*`, `cross`, `m2_*`,
  `comoment`, `mean_*`, `pearson`).
- `ratio`: `numerator / (denominator · 2^shift)` rounded once to the nearest
  `f64`, ties to even, subnormals exact, `None` past the range. It is
  CPython's integer true division, held to it on 106 recorded vectors and by
  property tests.

Arithmetic is `i128` where every operand fits and `num-bigint` where one does
not; a test stores one table at its own width and padded past 16 bytes and
requires identical results from the two paths. The 0.11 Python code is kept
unchanged as `tests/oracle/moments_arithmetic.py`, and
`tests/test_moments_arithmetic.py` holds core to it bit for bit: labels,
quantized integers, exact tables after folds and merges across exponent
gaps of 2,000 bits, and every derived float.

### One encoding, held by both hosts

A table is its axis sizes, column count, product operands, per-column
exponents and its accumulators: per cell the count, the sums, the sums of
squares and the cross sums, each a little-endian two's complement integer of
one width per table. Core borrows a host's bytes at any width and returns
every table at the smallest width that holds its values, so one set of
values has one encoding. The engine encodes its merged lane directly at that
width; no int64 halves cross the boundary any more.

Both hosts keep this encoding as the result's exact state. Python's
`RelationshipMoments` holds `width` and `encoded` (uint8); `counts` is
decoded once, and `q_sum_first` and the other `q_*` arrays are properties
that decode to object arrays of Python ints on access. R's
`relationship_moments` object holds the bytes as a raw matrix, one
accumulator per column, so it saves with `saveRDS`. Every view, fold, merge
and selection hands the bytes to core and keeps the bytes it returns.

The plan (D1) kept Python ints as Python's state and encoded them for each
core call. Measured on the 16,128-cell, 177,408-accumulator table of
`bench_relationship_moments.py`, encoding took about 41 ms per call and
decoding a result about 100 ms, against 16.6 ms for `m2_first`, 4.6 ms for
`sum` and 20 ms for `merge` in 0.11. Sums of squares pass 2^63, so no int64
fast path applies. That design could not meet the 5% gate, and the plan was
revised to the encoding above.

### Counts are exact and checked

A cell's count is an accumulator like any other, at most 2^63 − 1. `sum` and
`merge` refuse a result that would pass it with `ResourceError` /
`pedigree_graph_resource_error` (`arithmetic_overflow`, operation
`moments_sum` or `moments_merge`); nothing wraps. Python keeps a public int64
`counts`, which every valid cell fits. A total over cells is exact: Python's
`repr` adds Python ints and R's `print` shows core's `u128` total. R refuses
an `n` above 2^53 in `as.data.frame` (`count_exceeds_double`), and every
other statistic is still derived from the exact count. `relationship_counts`
and `relationship_burden` refuse any total above 2^53 the same way.

### Errors

An output past the float64 range raises `NotRepresentable`, a usage error
whose message names the statistic and the column or product
(`sumsq_first of 'x' is not representable in float64 (an output
overflowed)`); R appends how to leave it out with `stats =`. An inconsistent
table handed to core is `InvalidMomentsTable`.

### R surface

`relationship_moments(pg, max_degree, categories, first, second, values,
products, same, symmetric, memory_budget_bytes)` takes Python's keywords;
`products` is a list of length-2 character vectors. `first` and `second`
take factors, integer, logical and whole-number double vectors. A factor
enters core as its codes and its axis keeps the labels that occur. Character
vectors and `NA` are refused. `NA` in a `same` key is unknown, never equal,
like a negative key. `moments_select` takes labels on a factor axis and
values elsewhere, `moments_sum` takes axis names, and `moments_merge`
refuses results whose factor labels differ, naming the axis.
`as.data.frame(m, stats =)` gives one row per cell. `relationship_counts`
and `relationship_burden` return named doubles and an integer `n × 5`
matrix.

### Memory

The engine's host term is 32 bytes per accumulator for both hosts: the
encoded table (at most 16) and one host copy of it (R copies it into a raw
vector; Python adopts the buffer without a copy). It was 72 bytes plus a
conversion chunk (ADR 0013). Lanes, refusals and the planned peak are
otherwise unchanged and the same in both hosts.

A later operation is outside the budget, as before. It allocates its output
at an upper-bound width (the input width plus the bytes the fold's length or
the merge's shift needs), narrows it in place, and holds a few big integers
of scratch, because each accumulator is decoded when it is read.
`crates/core/tests/moments_table_scratch.rs` asserts that bound with a
counting allocator at two table sizes, at widths below and above 16 bytes.
On the Python side a view holds its float output and nothing else, and a
fold, merge or selection holds nothing that grows with the table
(`tracemalloc` in `test_relationship_moments.py`).

## Rejected

- **Each host keeps its own copy, held together by goldens.** Two copies of
  correctly rounded division and subnormal quantization drift, and goldens
  only sample them.
- **Fixed-width `i256` in house.** A merge across exponents 2,000 bits apart
  needs integers past any fixed width; `num-bigint` covers them, and the
  `i128` fast path covers the common case.
- **Python ints as Python's state, encoded per call.** It is too slow, as
  measured above.
- **Python ints plus cached bytes.** Views would be fast, but every fold and
  merge would still decode its whole output into ints, and memory would grow
  to 88 bytes per accumulator.

## Consequences

- Python and R give the same floats by construction. The R goldens
  (`moments*.tsv` per fixture, both `symmetric` modes, folded and merged) and
  `tools/r_parity.py` compare them byte for byte.
- `bench_relationship_moments.md` records the gate: every derived region is
  faster (0.13 to 0.62 of 0.11's wall), the engine call is within its
  spread, and every checksum is unchanged.
- Python API: `RelationshipMoments(...)` takes `width` and `encoded` in
  place of `counts` and the five `q_*` arrays, which are now read-only
  properties; `RelationshipMoments.from_exact(...)` takes the 0.11 fields
  and encodes them, for tables built without a pedigree. `CONVERSION_CHUNK` and `CONVERSION_BYTES_PER_ACCUMULATOR` are
  gone, `HOST_BYTES_PER_ACCUMULATOR` is 32, and `sum` and `merge` raise
  instead of wrapping.
- New allocation families: `moment_input` (packed labels) and
  `moment_table` (table outputs); `moment_output` now covers the engine's
  encoded table and each view's floats.
- The core depends on `num-bigint`, `num-integer` and `num-traits` (MSRV
  1.60, under the R package's `rustc >= 1.85`); the R source tarball vendors
  them.
