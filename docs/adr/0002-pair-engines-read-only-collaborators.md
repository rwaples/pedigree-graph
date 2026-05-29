# ADR 0002: Pair engines are read-only collaborators over PedigreeGraph

**Status:** accepted
**Date:** 2026-05-29
**Context:** PGQ-003 (decompose `PedigreeGraph`; move relationship engines out of `_core.py`)

## Context

`_core.py` had grown past 2,000 lines because `PedigreeGraph` owned both the
graph data (parent matrices, adjacency powers, sibling matrices, caches) and
the two relationship engines: the exact matrix pair extractor (~350-line
`extract_pairs` plus a dozen `_*_pairs` helpers) and the scalar streaming pair
counter (~260 lines). Small changes kept accreting ad-hoc branches onto the
central class.

## Decision

Split the engines into separate collaborators — `MatrixPairExtractor`
(`_pair_extractor.py`) and `StreamingPairCounter` (`_streaming_counter.py`) —
with stateless shared helpers in `_pair_utils.py` and the relationship-code
registry in `_registry.py`. `PedigreeGraph` keeps the graph data, the matrix
`cached_property`s, the shared graph-data accessors (`_get_Ak`,
`sibling_pairs`, `_mz_twin_pairs`, `_parent_offspring_pairs`), and thin public
wrappers (`extract_pairs`, `count_pairs_streaming`).

Two boundary rules make the split work:

1. **Engines hold a `pg` reference and read its private matrices directly**
   (`pg._A`, `pg._A2`, `pg._full_sib_matrix`, …) rather than receiving them as
   constructor arguments. The matrices are genuinely graph data; lazy
   `cached_property` triggering and the degree-gated cache-population ordering
   (a half-1C set found at degree 3 and consumed at degree 4) are subtle enough
   that reproducing them outside the graph would be more fragile than the
   coupling it removes. The `~8` underscore attributes the engines read are the
   documented engine-facing surface.

2. **Engines are read-only with respect to the graph's *result* state.**
   `extract()` / `count()` return their results; the thin wrappers persist them
   to `pg._pair_count_cache` and call `pg._release_pair_matrices()`. Engines may
   still trigger and free the *lazy matrix caches* they themselves drive (e.g.
   the mid-extraction `del pg._Am, pg._Af` memory optimisation), but never the
   result cache. Degree-gated run-state (the half-1C pairs) became
   `MatrixPairExtractor` instance state — fresh per `extract()` call — which
   also eliminated a latent staleness risk that existed when it was stashed on
   the long-lived graph.

## Considered options

- **Pass matrices as explicit constructor args (full decoupling).** Rejected:
  the wrapper would have to reproduce the lazy-trigger and degree-gated
  cache-population ordering, trading a clean read-coupling for fragile
  duplication.
- **Engines write back to `pg` directly.** Rejected: it splits cache-mutation
  logic across files and makes the engines impossible to test without asserting
  on graph side effects. The read-only contract is what makes the
  side-effect-freeness test in `tests/test_pair_engines.py` meaningful.

## Consequences

- `_core.py` shrank from ~2,071 to ~1,159 lines.
- Public API is unchanged: `PedigreeGraph.extract_pairs` / `.count_pairs` /
  `.count_pairs_streaming` and the `pedigree_graph` package exports are
  identical. The engine classes are intentionally *not* exported.
- New engines (e.g. the experimental BFS counter reassessed in PGQ-009) should
  follow the same contract: read graph data, return results, let a wrapper
  persist them.
