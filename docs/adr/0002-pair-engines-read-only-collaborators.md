# ADR 0002: Pair engines are read-only collaborators over PedigreeGraph

**Status:** accepted
**Date:** 2026-05-29
**Context:** PGQ-003 (decompose `PedigreeGraph`; move relationship engines out of `_core.py`)

Revised 2026-09-24 to match 0.10.0; earlier wording in git history.

## Context

`_core.py` had grown past 2,000 lines because `PedigreeGraph` owned both the
graph data (parent matrices, adjacency powers, sibling matrices, caches) and
the two relationship engines of the time: an exact sparse-matrix pair
extractor and a scalar streaming pair counter. Small changes kept accreting
ad-hoc branches onto the central class.

## Decision

Relationship engines live outside `PedigreeGraph` and follow one contract:
**an engine reads graph data and returns its result; the public wrapper owns
whatever is cached.** An engine never writes the graph's result state.

The engines today:

* The Rust row-streaming engine (`crates/core/src/relationships/`, ADR 0010)
  serves `relationship_pairs` and `relationship_counts`. The wrappers in
  `_relationship_pairs.py` and `_relationship_counts.py` hand it the graph's
  built columns (`graph._built`) and wrap what it returns. Nothing is stored
  on the graph.
* `_count_close_relatives` (`_streaming_counter.py`) counts the six close
  categories from the public row and id columns (`twin_rows`, `mother_rows`,
  `father_rows`, `mother_ids`, `father_ids`) and returns a dict. Only the
  public wrapper, `close_relative_counts`, writes
  `pg._close_relative_counts_cache`.
* The relationship codes come from one registry, `_registry.py`.

When this ADR was written the engines were Python classes that held a `pg`
reference and read its private matrices (`pg._A`, `pg._A2`,
`pg._full_sib_matrix`, …) directly. That read-coupling was accepted because
the lazy `cached_property` triggers and the degree-gated cache ordering (a
half-1C set found at degree 3 and consumed at degree 4) were too fragile to
reproduce outside the graph. Slice 12 moved `relationship_pairs` to the Rust
engine and deleted the matrix extractor, its helpers, and the graph's matrix
properties, so there are no private matrices left to read. The matrix engine
survives only as the differential oracle `tests/oracle/relationship_pairs.py`,
where `_Matrices(graph)` builds the matrices from the graph's public columns.

## Considered options

- **Pass matrices as explicit constructor args (full decoupling).** Rejected
  at the time: the wrapper would have had to reproduce the lazy-trigger and
  degree-gated cache-population ordering. The Rust engine made the question
  moot; it takes the built columns and holds no graph matrices.
- **Engines write back to `pg` directly.** Rejected: it splits cache-mutation
  logic across files and makes the engines impossible to test without
  asserting on graph side effects. The read-only contract is what makes
  `tests/test_pair_engines.py::TestEngineReadOnlyContract` meaningful.

## Consequences

- `_core.py` shrank from about 2,071 to about 1,159 lines at the split; it is
  615 lines today.
- The engine classes and functions are not exported. The public surface is
  the `PedigreeGraph` and `PedigreeView` methods and the names in
  `pedigree_graph.__all__`.
- `TestEngineReadOnlyContract` checks that `_count_close_relatives` leaves
  `_close_relative_counts_cache` unset and that the matrix oracle returns
  every registry code with only the requested ones populated.
- A new engine follows the same contract: read graph data, return results,
  let the public wrapper persist them.
