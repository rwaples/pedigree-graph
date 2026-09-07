# Architecture & contracts

Orientation for contributors. The day-to-day vocabulary lives in
[`CONTEXT.md`](../CONTEXT.md) (a glossary — graph-space vs caller-space,
relationship pair/category, degree); the *decisions* live in
[`docs/adr/`](adr/). This file maps the module layout and the hidden
contracts that aren't obvious from any single file, and points at the
source-of-truth and the regression test for each.

## Module map

The package is decomposed into focused modules behind a few thin facades.
Prefer adding a **new** focused module over extending an oversized one (see
*Guardrails* below).

| Module | Responsibility |
|---|---|
| `_core.py` | The `PedigreeGraph` class: graph data, parent CSR, kinship/inbreeding, the canonical `from_frame` / `from_arrays` constructors, and thin receiver methods for views, relationship operations, and pair kinship. Matrix receiver methods live with their implementation in `_kinship_matrix.py`; compatibility pair/count bodies live in `_compat.py`; input validation lives in `_input.py`. `_from_input` is the one builder every entry point funnels through and where the 0.7.1 defaults are switched. |
| `_properties.py` | `PedigreeProperties`, the ADR [0006](adr/0006-public-api-and-coordinate-semantics.md) read-only property surface mixed into `PedigreeGraph`: the owned input arrays handed back unchanged, plus lazily computed structural `depth`. |
| `_view.py` | `PedigreeView` and `CoordinateToken`: the `PedigreeGraph.view(ids=...)` / `view(rows=...)` selection boundary (reusing the `_input.py` coercion helpers), the owned read-only `ids` / `graph_rows` arrays, the opaque per-receiver coordinate token, and the view receiver's `relationship_pairs` / `relationship_counts` / `pair_kinship`. The view owns the graph-row → view-row projection table (`_graph_to_view`, `-1` where unselected) that both the view pair path and the legacy `from_subsample` adapter project through (ADR [0006](adr/0006-public-api-and-coordinate-semantics.md)). |
| `_compat.py` | The bodies of the 0.7.1 `from_subsample` (a full graph carrying a private legacy `PedigreeView`), `extract_pairs` (`legacy_extract_pairs`: 0.7.1 code selection, `(min, max)` collateral orientation, projection through the legacy view plus the 0.7.1 all-codes `(min, max)` fold, count cache), and the two count adapters (`legacy_count_pairs`: the unfolded matrix-engine dict with the subsample scope; `legacy_count_pairs_streaming`: `estimate_relationship_counts` with `None` written as `0`, the 0.7.1 scope rules and count-cache write), kept apart so 0.8.0 removes them by deleting the module. |
| `_input.py` | `PedigreeInput` and `parse_pedigree_input` / `parse_pedigree_arrays`: the **one** input boundary. Lossless integer coercion, range and uniqueness checks, id→row mapping (`_map_ids_to_rows`), cycle detection, MZ pair validation, optional-metadata normalization, and owned read-only storage. |
| `_errors.py` | `PedigreeValidationError`, `MissingMetadataError`, `ResourceError` and the three code registries. **Single source of truth** for the structured-error codes and their required `.fields` (ADR [0006](adr/0006-public-api-and-coordinate-semantics.md)). |
| `_threads.py` | Package-wide thread budget: `configure_threads(n)` > `PEDIGREE_GRAPH_THREADS` > `1`, committed on first use (ADR [0007](adr/0007-rust-core-host-boundary-and-release.md)). |
| `_registry.py` | `RelationshipCategory` and the immutable ordered `RELATIONSHIPS` mapping, the internal selectors, `REL_PLAN` + helpers (per-code engine semantics), and the detached 0.7.1 `RelType` / `REL_REGISTRY` / `PAIR_KINSHIP` snapshots. **Single source of truth** for codes, kinship, degree range, and engine divergence. |
| `relationships.py` | The public relationship vocabulary and result types: re-exports `RELATIONSHIPS` / `RelationshipCategory` from `_registry.py` and defines `RelationshipPairBlock` (owned read-only int32 `first_rows` / `second_rows`, roles, `requested`, private receiver token), the immutable 23-key `RelationshipPairs` mapping, and the frozen 23-key `RelationshipCountResult` mapping with its `requested` / `exact` / `approximate` / `clamped` code sets (ADR [0006](adr/0006-public-api-and-coordinate-semantics.md)). |
| `_pair_utils.py` | Free functions shared by the pair engines: `canonical_keys` / `sort_by_canonical_key` (the unordered `min * n + max` key), `subtract_pairs` (the one canonical-key membership subtraction), `oriented_pairs_from_sparse` (oriented, dual-valid-deduplicated read of an asymmetric product), `pairs_from_groups`, `dedup_pairs` (BFS engine), and `project_pairs` (keep both-endpoints-selected pairs, relabelled through a graph-to-view table). |
| `_pair_extractor.py` | `MatrixPairExtractor` — exact, path-counting matrix pair extraction returning oriented graph-space blocks for a dependency-closed code set — plus the assembly steps shared by both receivers (`_requested_codes` selector validation, `_classify` = `dependency_closure` + thread budget + registry-order precedence fold, `_build_result` block construction), `relationship_pairs` (graph rows) and `view_relationship_pairs` (projection, symmetric re-canonicalisation, view-key re-sort), and the `check_exclusive` invariant checker (tests, or `PEDIGREE_GRAPH_DEBUG_EXCLUSIVITY=1`). |
| `_streaming_counter.py` | `StreamingPairCounter` — memory-bounded scalar counter returning `ScalarCounts(raw, overlaps, clamped)`: the 0.7.1 unfolded counts, the MHS / PHS pairs a parent-offspring category claims under the fold, and the residual codes floored at zero — and `estimate_relationship_counts` / `_estimate`: validation, the thread budget (public path only), the per-cutoff `_estimate_cache` of `CachedEstimate(result, raw)` on the graph, the typed `RelationshipCountResult` (`raw - overlaps`, `exact` from `estimate_exact_codes()`), the one `RuntimeWarning` per clamped computation raised before the cache write, and matrix release. |
| `_bfs_engine.py` / `experimental.py` | Experimental BFS counter (`count_pairs_bfs`); `experimental.py` is the thin public-experimental surface. |
| `_kinship_kernel.py` | Facade re-exporting the numba kinship kernel, split into `_kinship_depth`, `_kinship_allocator`, `_kinship_csc`, `_kinship_dp` (DP orchestration + driver + theta), `_kinship_dp_depth` (one-depth recurrence, MZ fill, and candidate capture), and `_inbreeding_kernel`. |
| `_kinship_matrix.py` | `PedigreeMatrixMethods`, mixed into `PedigreeGraph`, and the three graph-space matrix families with their operation/selector caches: complete DP support; closest-category support from `relationship_pairs`; and the old propagation-pruned candidate support. Sparse relationship support streams retained coordinates through the ADR 0009 pair recurrence in deterministic fixed-size chunks, starting each chunk from the graph's retained pair memo (`_kinship_pairwise.memoised_kinship`) and leaving the closure behind for the next `pair_kinship`. Dense approximate support maps candidates into stable topology space and captures only those values during one complete retiring DP pass. Both paths write symmetric CSC positions and freeze sorted float32/int32 arrays. |
| `effective_size.py` | Public final Ne surface: the eight estimators and their observed-cohort result records, implemented in `_cohorts`, `_ne_common`, `_ne_results`, `_ne_family_size`, `_ne_founders`, `_ne_caballero_toro`, `_ne_hill`, `_ne_rates`. |
| `_effective_size.py` | `# 0.8.0-DELETE` facade: the 0.7.1 root estimators, `_ne_legacy` dense records, and `compute_all_ne`. |
| `_topology.py` | `build_topology` and the `Topology` value: structural depth plus the private stable depth-major order and the graph ↔ topological row maps every order-dependent kernel routes through. |
| `_lineage.py`, `_lineage_kernel.py`, `_cohort_utils.py` | Lineage surfaces (`distinct_ancestor_counts`, `descendant_path_counts`, `connected_component_ids` bodies: coordinate mapping, memo, min-ID component labels) over the numba/scipy count kernels; cohort-eligibility windows. |
| `crates/core/src/relationships/` | Rust `pedigree-graph-core`: the row-streaming exact relationship engine and `pgr-count` CLI (ADR [0010](adr/0010-row-streaming-relationship-engine.md)). Parity fixtures under `crates/core/tests/fixtures/` come from `tests/parity/dump_relationship_inputs.py`. |

The pair engines are **read-only collaborators** of `PedigreeGraph`: they
hold a reference, read private matrices/accessors, and return results; the
graph owns caches and matrix lifetimes (ADR
[0002](adr/0002-pair-engines-read-only-collaborators.md)).

## Hidden contracts

These invariants are easy to break in a refactor and not visible from any
one call site. Each has a documented source of truth and a regression test.

| Contract | What it means | Source of truth | Regression test |
|---|---|---|---|
| **Coordinate space** | The kinship matrix is indexed in *graph-space* (full-pedigree rows); `extract_pairs` on a `from_subsample` graph returns *caller-space* (subsample rows), and `view.relationship_pairs` returns *view rows*. Mixing them silently returns wrong kinship. | `CONTEXT.md` glossary; `PedigreeGraph._legacy_view` (`compute_pair_kinship` maps back through its `graph_rows`); `PedigreeView._graph_to_view`; `project_pairs` | `tests/test_pedigree_graph.py::…::test_reversed_subsample_pair_kinship_uses_graph_coords`; `tests/test_view_relationship_pairs.py::TestOracleEquality` |
| **Exact vs approximate counts** | The matrix engine is exact (counts paths) for every code. `estimate_relationship_counts` equals `relationship_counts` for MZ, MO, FO, FS, MHS, PHS only (the half-sib pairs a parent-offspring category claims under the fold are subtracted); every other code is `approximate`, the lineal codes because raw ancestor-path counts over-count a pair also related at a shorter depth, as a half-sib, or as a closer collateral (ADR [0011](adr/0011-scalar-estimate-exact-set-excludes-lineal-codes.md)). `RelationshipCountResult.exact / approximate / clamped` say which per requested code (`clamped` marks a residual floored at `0` on this pedigree). `count_pairs_streaming` returns the unfolded raw counts, exact for the ten `streaming_exact` codes against the unfolded `count_pairs`. | `REL_PLAN[...].estimate_exact` / `estimate_exact_codes()` in `_registry.py`; `_estimate` in `_streaming_counter.py` | `tests/test_estimate_relationship_counts.py::TestValues`; `tests/test_count_pairs_streaming.py::test_streaming_exact_codes_match_matrix`; `tests/test_relationship_plan.py` |
| **Three matrix supports, one value definition** | `kinship_matrix()` has complete nonzero support; `relationship_kinship_matrix` has selected closest-category support; `approximate_kinship_matrix` has propagation-pruned candidate support, which is not a final-value cutoff. Every retained value in all three is the ADR 0009 pair recurrence bit, and every diagonal is present. Do not return propagated values from the approximate family or substitute degree support for fitACE's `0.001` candidate support. | ADR [0006](adr/0006-public-api-and-coordinate-semantics.md); `_kinship_matrix.py` | `tests/test_kinship_matrices.py` |
| **Path-count vs distinct-ancestor** | Under inbreeding the BFS engine counts *distinct* shared ancestors while the matrix engine counts *paths*; they diverge on exactly 4 cousin codes. | `REL_PLAN[...].bfs_diverges_under_inbreeding` / `bfs_divergent_codes()` | `tests/test_experimental.py::test_inbred_with_cousins_{non_cousin_codes_match,cousin_codes_diverge}`; `tests/test_relationship_plan.py::test_bfs_divergent_codes_are_the_four_cousin_codes` |
| **Dense vs sparse IDs** | IDs may be sparse/high-valued; construction must remap to a dense row space, never allocate a dense `max(id)`-sized table. | `_input.validate_id_field` / `_input._map_ids_to_rows` | `tests/test_pedigree_graph.py::TestInputValidation::test_sparse_high_ids_do_not_allocate_dense_table`, `…::test_unsorted_ids_remap_correctly` |
| **Default all-zero sex** | `sex=` defaults to all-female; the sex-dependent Ne estimators warn (not error) so a forgotten `sex=` is diagnosable. | `_warn_if_uniform_sex` in `_ne_family_size.py` | `tests/test_from_arrays_sex.py::test_ne_{sex_ratio,variance_family_size}_warns_when_sex_defaulted`, `…::test_no_warning_when_sex_is_supplied` |
| **Relationship code set** | All three engines (matrix, streaming, BFS) return exactly the `RELATIONSHIPS` key set. | `RELATIONSHIPS` in `_registry.py` | `tests/test_relationship_plan.py::TestAllEnginesReturnRegistryKeySet` |
| **Private topological order** | Input rows may arrive in any acyclic order. Kernels that need parents before children (inbreeding, descendant counts, pairwise kinship, the kinship DP, Caballero–Toro) run on parent arrays remapped into one stable depth-major order and map their per-row outputs back; for pairwise kinship that order is also the ADR 0009 peel rule. Skipping the remap silently zeroes inbreeding terms; skipping the map-back silently misaligns every result. Supplied `generation` labels never enter it. | `build_topology` / `Topology` in `_topology.py`; `PedigreeGraph._topology` | `tests/test_topology.py`; `tests/test_row_order.py` |

Statistical-correctness gotchas (booleanise-after-multiplicity, ≥2 shared
ancestors for full/half, `_get_Ak(0)` = identity, pair-key int64 overflow,
degree-gating cache side effects) are catalogued in the umbrella
`CLAUDE.md`; touch the relevant module's tests when changing that code.

## Guardrails

* **Line budget.** `tests/test_architecture_guardrails.py` fails if any
  production module exceeds the budget (default 1000 lines), so large-file
  growth is visible in review. Reviewed exceptions live in that file's
  `ALLOWLIST` with a per-file cap; an allowlisted file that drops back under
  the default budget is flagged so the exception can be removed. Prefer a new
  focused module over pushing an existing one past the budget.
* **Single source of truth for relationship semantics.** Add a new code in
  `RELATIONSHIPS` + `REL_PLAN` (a test asserts the two stay in lockstep) and
  implement it in each engine — do not re-document kinship or divergence in
  engine docstrings (ADR
  [0003](adr/0003-relationship-plan-documents-not-drives-engines.md)).
