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
| `_core.py` | The `PedigreeGraph` class: graph data, parent CSR, kinship/inbreeding, the canonical `from_frame` / `from_arrays` constructors, and thin receiver methods for views, relationship operations, and pair kinship. Matrix receiver methods live with their implementation in `_kinship_matrix.py`; host coercion lives in `_input.py` and every pedigree-semantic check in the Rust core. `_from_built` is the one builder every entry point funnels through. |
| `_properties.py` | `PedigreeProperties`, the ADR [0006](adr/0006-public-api-and-coordinate-semantics.md) read-only property surface mixed into `PedigreeGraph`: the owned input arrays handed back unchanged, plus lazily computed structural `depth`. |
| `_view.py` | `PedigreeView` and `CoordinateToken`: the `PedigreeGraph.view(ids=...)` / `view(rows=...)` selection boundary (reusing the `_input.py` coercion helpers), the owned read-only `ids` / `graph_rows` arrays, the opaque per-receiver coordinate token, and the view receiver's `relationship_pairs` / `relationship_counts` / `pair_kinship`. The view owns the graph-row → view-row projection table (`_graph_to_view`, `-1` where unselected) that the view pair path projects through (ADR [0006](adr/0006-public-api-and-coordinate-semantics.md)). |
| `_input.py` | `HostColumns` and `host_columns` / `host_columns_from_arrays`: the numpy half of the **one** input boundary. Field presence, shape, length, and lossless int64 coercion of numpy dtypes, pandas nullable columns, and host nulls (the `-1` sentinel), plus the selection coercers the view and pair endpoints share. Everything pedigree-semantic (range, sex encoding, duplicate and shared-parent ids, id→row mapping, cycle detection, MZ pair validation, optional-metadata normalization, birth-year order) runs in `pedigree_graph._native.build_pedigree` (`crates/core/src/graph.rs`), which hands back owned numpy columns. |
| `_errors.py` | `PedigreeValidationError`, `MissingMetadataError`, `ResourceError` and the three code registries. **Single source of truth** for the structured-error codes and their required `.fields` (ADR [0006](adr/0006-public-api-and-coordinate-semantics.md)). |
| `_threads.py` | Package-wide thread budget: `configure_threads(n)` > `PEDIGREE_GRAPH_THREADS` > `1`, committed on first use (ADR [0007](adr/0007-rust-core-host-boundary-and-release.md)). |
| `_registry.py` | `RelationshipCategory` and the immutable ordered `RELATIONSHIPS` mapping, the internal selectors, and `REL_PLAN` + helpers (per-code engine semantics). **Single source of truth** for codes, kinship, degree range, and scalar count coverage. |
| `relationships.py` | The public relationship vocabulary and result types: re-exports `RELATIONSHIPS` / `RelationshipCategory` from `_registry.py` and defines `RelationshipPairBlock` (owned read-only int32 `first_rows` / `second_rows`, roles, `requested`, private receiver token), the immutable 23-key `RelationshipPairs` mapping, and the frozen 23-key `RelationshipCountResult` mapping with its `requested` / `exact` code sets (ADR [0006](adr/0006-public-api-and-coordinate-semantics.md)). |
| `_selection.py` | `RelationshipSelection`: the resolved form of the one-of-two `max_degree=` / `categories=` selector the general relationship endpoints take; `close_relative_counts()` has no selector. The selection is parsed once at the public boundary by `RelationshipSelection.parse` and passed to the pair, count, and matrix engines. Carries the requested `codes`, their registry-`ordered` tuple (the canonical cache key, so a cutoff and the code list it names are one selection), and `top_degree` (`None` exactly when empty — the cutoff the row-streaming counter runs at). Parsing once also means a one-shot `categories` iterable is consumed exactly once, so no engine has to materialise it defensively. Private: no consumer constructs one. |
| `_relationship_pairs.py` | `relationship_pairs` (graph rows) and `view_relationship_pairs` (view rows, as the int32 view row of every graph row): the `_selection.py` selection parsed at the boundary, one call into `_native.relationship_pairs` on the package pool at the selection's `top_degree` with the requested codes and the `execution` mode, and the owned read-only blocks wrapped without a copy by `_input._own_native`. Classification, orientation, closest-category precedence, view projection and ordering all happen in the Rust engine (ADR 0010 as amended); the retired SciPy matrix engine lives on as the differential oracle in `tests/oracle/relationship_pairs.py`. |
| `_relationship_counts.py` | `relationship_counts` (graph) and `view_relationship_counts` (view, as a boolean row mask): the `_selection.py` selection parsed at the boundary, the thread budget, one call into `_native.relationship_counts` passing the graph's own `BuiltPedigree` at the selection's `top_degree`, and the typed exact `RelationshipCountResult` keyed by the codes the binding returns. No pair list, no native state retained (ADR [0010](adr/0010-row-streaming-relationship-engine.md) as amended). |
| `_streaming_counter.py` | `close_relative_counts` commits the thread budget and caches one immutable `RelationshipCountResult` per graph. `_count_close_relatives` reads parent/twin arrays and sums sibling-group combinations, subtracting MHS/PHS pairs claimed by parent-offspring categories. The six computed codes come from `estimate_exact_codes()`; every other registry key is `None`. No approximation, clamp, warning, pair list or adjacency-power access/release. |
| `_kinship_kernel.py` | Facade re-exporting the numba kinship kernel, split into `_kinship_depth` (EqG and retirement depth only), `_kinship_allocator`, `_kinship_csc`, `_kinship_dp` (DP orchestration + driver + theta), `_kinship_dp_depth` (one-depth recurrence, MZ fill, and candidate capture), and `_inbreeding_kernel`. |
| `_kinship_matrix.py` | `PedigreeMatrixMethods`, mixed into `PedigreeGraph`, and the three graph-space matrix families with their operation/selector caches: complete DP support; closest-category support from `relationship_pairs`, cached under the selection's canonical code order so equivalent selectors share one entry; and the old propagation-pruned candidate support. Sparse relationship support streams retained coordinates through the ADR 0009 pair recurrence in deterministic fixed-size chunks, starting each chunk from the graph's retained pair memo (`_kinship_pairwise.memoised_kinship`) and leaving the closure behind for the next `pair_kinship`. Dense approximate support maps candidates into stable topology space and captures only those values during one complete retiring DP pass. Both paths write symmetric CSC positions and freeze sorted float32/int32 arrays. |
| `effective_size.py` | Public final Ne surface: the eight estimators and their observed-cohort result records, implemented in `_cohorts`, `_ne_common`, `_ne_results`, `_ne_family_size`, `_ne_founders`, `_ne_group_coancestry`, `_ne_hill`, `_ne_rates`. |
| `_topology.py` | `build_topology` and the `Topology` value: structural depth plus the private stable depth-major order and the graph ↔ topological row maps every order-dependent kernel routes through. Depth, the order, the topological check, and the cycle witness are computed in Rust (`_native`). |
| `_native` (`crates/python/`) | The PyO3 extension module over `pedigree-graph-core`; `_native.pyi` is its typed surface. Core `Error` variants cross as the `_errors.py` classes by code. |
| `_lineage.py`, `_lineage_kernel.py`, `_cohort_utils.py` | Lineage surfaces and kernels: `distinct_ancestor_counts` merges sorted closed ancestor sets in stable topological order and reuses power-of-two slots after each row's last direct child; `descendant_path_counts` is the reverse topological scalar sweep; `connected_component_ids` owns the SciPy component labelling and minimum-ID policy; `_cohort_utils` owns cohort-eligibility windows. |
| `crates/core/src/` | Rust `pedigree-graph-core`: `topology` (depth, depth-major order, cycle witness), `error` (structured codes), and `relationships`, the row-streaming exact relationship engine (category definitions, then the per-row closest-category fold, then the count or the row mask) with the `pgr-count` CLI. `Pedigree`'s slices are private: `Pedigree::try_new` and `PedigreeColumns::try_borrow` are the only ways in, and they check the column lengths and row ranges the engine indexes by without bounds checks, so the PyO3 binding no longer carries its own copy of that validation (ADR [0010](adr/0010-row-streaming-relationship-engine.md), as amended). Parity fixture inputs under `crates/core/tests/fixtures/` come from the frozen `tests/parity/dump_relationship_inputs.py`; their `.counts.json` oracles from `tests/parity/dump_relationship_counts.py` (`relationship_counts(max_degree=5)` on a graph rebuilt from each TSV). |

The relationship engines are **read-only collaborators** of `PedigreeGraph`:
they receive the graph's `BuiltPedigree` (or, for the pure-Python counter,
its parent and twin arrays), compute, and return results; the graph owns
its caches (ADR [0002](adr/0002-pair-engines-read-only-collaborators.md)).
No production module builds adjacency powers or sibling matrices any more;
the one place that still does is the test oracle.

## Hidden contracts

These invariants are easy to break in a refactor and not visible from any
one call site. Each has a documented source of truth and a regression test.

| Contract | What it means | Source of truth | Regression test |
|---|---|---|---|
| **Coordinate space** | The kinship matrix and every graph method are indexed in *graph-space* (full-pedigree rows); `view.relationship_pairs` and `view.pair_kinship` work in *view rows*, and each receiver's coordinate token refuses rows from the other. Mixing them silently returns wrong kinship. | `CONTEXT.md` glossary; `CoordinateToken` and `PedigreeView._graph_to_view` in `_view.py`; the relabel and view-key sort in `crates/core/src/relationships/pairs.rs` | `tests/test_pedigree_graph.py::TestPairKinship::test_reversed_view_pair_kinship_resolves_through_the_graph`; `tests/test_view_relationship_pairs.py::TestOracleEquality` |
| **Exact count coverage** | All public counts use closest-category precedence. `close_relative_counts()` computes exactly MZ, MO, FO, FS, MHS and PHS, matching `relationship_counts` even under inbreeding. It has no selector. Other registry keys are `None`, never approximate values or zero placeholders. `requested` and `exact` contain the six codes. `RelationshipCountResult.approximate` and `.clamped` are removed from every result type use. See amended ADR [0011](adr/0011-scalar-estimate-exact-set-excludes-lineal-codes.md). | `REL_PLAN[...].estimate_exact` / `estimate_exact_codes()` in `_registry.py`; `_count_close_relatives` in `_streaming_counter.py` | `tests/test_close_relative_counts.py`; `tests/test_relationship_plan.py` |
| **Three matrix supports, one value definition** | `kinship_matrix()` has complete nonzero support; `relationship_kinship_matrix` has selected closest-category support; `approximate_kinship_matrix` has propagation-pruned candidate support, which is not a final-value cutoff. Every retained value in all three is the ADR 0009 pair recurrence bit, and every diagonal is present. Do not return propagated values from the approximate family or substitute degree support for fitACE's `0.001` candidate support. | ADR [0006](adr/0006-public-api-and-coordinate-semantics.md); `_kinship_matrix.py` | `tests/test_kinship_matrices.py` |
| **Dense vs sparse IDs** | IDs may be sparse/high-valued; construction must remap to a dense row space, never allocate a dense `max(id)`-sized table. | `graph::IdIndex` in `crates/core/src/graph.rs` (sorted-id binary search, exposed as `_native.IdIndex`) | `tests/test_pedigree_graph.py::TestInputValidation::test_sparse_high_ids_do_not_allocate_dense_table`, `…::test_unsorted_ids_remap_correctly` |
| **No sex default** | An absent or partly unknown sex column is refused by the sex-dependent Ne estimators with `MissingMetadataError("missing_sex")`; a fully known but uniform column is valid and warns, returning `ne=None`. Nothing invents a sex. | `_require_complete_sex` in `_ne_metadata.py`; `_warn_if_uniform_sex` in `_ne_family_size.py` | `tests/test_effective_size_metadata.py::test_sex_dependent_estimators_require_complete_sex`, `…::test_uniform_fully_known_sex_is_valid_and_warns`; `tests/test_from_arrays_sex.py::test_no_warning_when_both_sexes_are_present` |
| **Relationship code set** | Pair lists, general counts and close-relative counts return exactly the `RELATIONSHIPS` key set. | `RELATIONSHIPS` in `_registry.py` | `tests/test_relationship_plan.py::TestAllEnginesReturnRegistryKeySet` |
| **Private topological order** | Input rows may arrive in any acyclic order. Kernels that need parents before children (inbreeding, descendant counts, pairwise kinship, the kinship DP) run on parent arrays remapped into one stable depth-major order and map their per-row outputs back; for pairwise kinship that order is also the ADR 0009 peel rule. Skipping the remap silently zeroes inbreeding terms; skipping the map-back silently misaligns every result. Supplied `generation` labels never enter it. | `build_topology` / `Topology` in `_topology.py`; `PedigreeGraph._topology` | `tests/test_topology.py`; `tests/test_row_order.py` |

Statistical-correctness gotchas (booleanise-after-multiplicity, ≥2 shared
ancestors for full/half, `_get_Ak(0)` = identity, pair-key int64 overflow,
degree-gating cache side effects) are catalogued in the umbrella
`CLAUDE.md`; since slice 12 the matrix code they describe is the test oracle
in `tests/oracle/relationship_pairs.py`, and the production engine's own
invariants (saturated multiplicity, the `EXCLUSIONS` table, the per-row
precedence fold, first-arm orientation) are stated in ADR 0010.

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
