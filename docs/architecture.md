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
| `_core.py` | The `PedigreeGraph` class: graph data, parent CSR, kinship/inbreeding, constructors, and thin `extract_pairs` / `count_pairs` / `count_pairs_streaming` wrappers. |
| `_registry.py` | `RelType`, `REL_REGISTRY`, `PAIR_KINSHIP` (kinship coefficients) and `REL_PLAN` + helpers (per-code engine semantics). **Single source of truth** for codes, kinship, degree range, and engine divergence. |
| `_pair_utils.py` | Free functions shared by the pair engines: `dedup_pairs`, `extract_from_sparse`, `pairs_from_groups`, `remap_pairs_to_caller`. |
| `_pair_extractor.py` | `MatrixPairExtractor` — exact, path-counting matrix pair extraction. |
| `_streaming_counter.py` | `StreamingPairCounter` — memory-bounded scalar counter (exact for 10 codes, approximate for the rest). |
| `_bfs_engine.py` / `experimental.py` | Experimental BFS counter (`count_pairs_bfs`); `experimental.py` is the thin public-experimental surface. |
| `_kinship_kernel.py` | Facade re-exporting the numba kinship kernel, split into `_kinship_depth`, `_kinship_allocator`, `_kinship_csc`, `_kinship_dp` (DP + driver + theta), and `_inbreeding_kernel`. |
| `_effective_size.py` | Facade for the Ne estimators, split into `_ne_common`, `_ne_results`, `_ne_family_size`, `_ne_founders`, `_ne_caballero_toro`, `_ne_hill`, `_ne_rates`. |
| `_lineage_kernel.py`, `_cohort_utils.py` | Ancestor/descendant counts; cohort-eligibility windows. |

The pair engines are **read-only collaborators** of `PedigreeGraph`: they
hold a reference, read private matrices/accessors, and return results; the
graph owns caches and matrix lifetimes (ADR
[0002](adr/0002-pair-engines-read-only-collaborators.md)).

## Hidden contracts

These invariants are easy to break in a refactor and not visible from any
one call site. Each has a documented source of truth and a regression test.

| Contract | What it means | Source of truth | Regression test |
|---|---|---|---|
| **Coordinate space** | The kinship matrix is indexed in *graph-space* (full-pedigree rows); `extract_pairs` returns *caller-space* (subsample rows). Mixing them silently returns wrong kinship. | `CONTEXT.md` glossary; `PedigreeGraph._subsample_inverse`; `remap_pairs_to_caller` | `tests/test_pedigree_graph.py::…::test_reversed_subsample_pair_kinship_uses_graph_coords` |
| **Exact vs approximate counts** | The matrix engine is exact (counts paths) for every code; the streaming counter is exact for 10 codes and approximate for the rest. | `REL_PLAN[...].streaming_exact` / `streaming_exact_codes()` in `_registry.py` | `tests/test_count_pairs_streaming.py::test_streaming_exact_codes_match_matrix`, `…::test_streaming_cousin_codes_approximate`; `tests/test_relationship_plan.py` |
| **Path-count vs distinct-ancestor** | Under inbreeding the BFS engine counts *distinct* shared ancestors while the matrix engine counts *paths*; they diverge on exactly 4 cousin codes. | `REL_PLAN[...].bfs_diverges_under_inbreeding` / `bfs_divergent_codes()` | `tests/test_experimental.py::test_inbred_with_cousins_{non_cousin_codes_match,cousin_codes_diverge}`; `tests/test_relationship_plan.py::test_bfs_divergent_codes_are_the_four_cousin_codes` |
| **Dense vs sparse IDs** | IDs may be sparse/high-valued; construction must remap to a dense row space, never allocate a dense `max(id)`-sized table. | `PedigreeGraph._validate_id_column` / `_map_ids_to_rows` | `tests/test_pedigree_graph.py::TestInputValidation::test_sparse_high_ids_do_not_allocate_dense_table`, `…::test_unsorted_ids_remap_correctly` |
| **Default all-zero sex** | `sex=` defaults to all-female; the sex-dependent Ne estimators warn (not error) so a forgotten `sex=` is diagnosable. | `_warn_if_uniform_sex` in `_ne_family_size.py` | `tests/test_from_arrays_sex.py::test_ne_{sex_ratio,variance_family_size}_warns_when_sex_defaulted`, `…::test_no_warning_when_sex_is_supplied` |
| **Relationship code set** | All three engines (matrix, streaming, BFS) return exactly the `REL_REGISTRY` key set. | `REL_REGISTRY` in `_registry.py` | `tests/test_relationship_plan.py::TestAllEnginesReturnRegistryKeySet` |

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
  `REL_REGISTRY` + `REL_PLAN` (a test asserts the two stay in lockstep) and
  implement it in each engine — do not re-document kinship or divergence in
  engine docstrings (ADR
  [0003](adr/0003-relationship-plan-documents-not-drives-engines.md)).
