# ADR 0006: Public API, coordinate spaces, and relationship semantics for 0.8.0

**Status:** accepted
**Date:** 2026-09-04
**Context:** the pedigree-graph Rust core redesign
(simACE `docs/plans/pedigree-graph-rust-core.md`); precedes any native code

## Context

The 0.7.x public surface grew by accretion around the internal matrix engine,
and several of its contracts are implicit or misleading:

* `from_subsample` stores the full graph but returns relationship pairs in the
  caller's coordinates, while kinship matrices stay in graph coordinates
  (`_core.py:880-943`). The hidden split caused PGQ-001.
* Public `mother` / `father` arrays hold remapped row coordinates although the
  names read as original IDs (`_core.py:244-248`).
* A supplied `generation` column drives kinship-DP traversal depth
  (`_kinship_dp.py:104-123`), so cohort metadata can change a structural
  computation. Structural depth is derived separately elsewhere.
* Pair orientation for lineal categories is canonicalised during
  view/subsample remap (`_pair_utils.py:124-153`), so which endpoint is the
  ancestor depends on the construction path.
* `min_kinship` on the DP is propagation pruning, not a final-value filter
  (ADR 0005 records the inbred counterexample).
* `count_pairs_streaming` mixes exact and approximate categories under an
  implementation-oriented name.
* Descendant counts are path counts while ancestor counts are distinct counts.
* fitACE reaches into `_Am` / `_Af` for connected components, and PA-FGRS
  imports a private `_compute_depth`.
* `REL_REGISTRY` and `PAIR_KINSHIP` are two registries for one concept.

Porting this surface to Rust as-is would freeze every one of these into a
host boundary. The API is therefore redesigned first, in pure Python, and
0.8.0 becomes the frozen differential baseline for the native migration
(ADR 0007).

## Decision

**Break the public API once, in 0.8.0, with no deprecation shims.** The
canonical vocabulary is `CONTEXT.md` (graph-space, view-space, structural
depth, generation label, nominal vs pedigree-specific kinship, role-ordered
relationship pair).

### Construction

* `PedigreeGraph.from_frame(frame)` and `PedigreeGraph.from_arrays(ids=,
  mother_ids=, father_ids=, twin_ids=, sex=, generation=, birth_year=)`
  replace the loose `PedigreeGraph(frame_or_dict)` constructor.
  `from_subsample` is removed.
* Required fields are `id`, `mother`, `father`. Frames accept host-native
  nulls; arrays additionally accept `-1`. IDs are non-negative int64; rows are
  int32.
* Any acyclic input row order is accepted. A stable private topological order
  is built internally; every public graph-space row stays aligned with the
  input. Cycles are a structured validation error.
* Unresolved parent/twin IDs are valid partial-pedigree references, distinct
  from missing. Unresolved parent IDs still support sibling classification. A
  child cannot name the same known individual in both parent roles.
* Represented MZ references must be non-self, reciprocal, two-member,
  parent-identical, and sex-identical when both sexes are known. An external
  co-twin does not establish an internal MZ pair.
* Sex is `Female` / `Male` / `Unknown`, transported as `0` / `1` / `-1`.
  Parent-role/sex conflicts do not block construction, so validation tools can
  represent imperfect data.

### Coordinate spaces

* `PedigreeGraph` operations are graph-space. `full.view(ids=...)` or
  `full.view(rows=...)` (exactly one keyword; order preserved; duplicates,
  missing IDs, and out-of-range rows are structured errors) returns a
  `PedigreeView` whose operations are view-space.
* `PedigreeView` initially exposes only `relationship_pairs`,
  `relationship_counts`, `pair_kinship`, read-only `ids` and `graph_rows`,
  and `len`. Matrix, inbreeding, lineage, connectivity, and effective-size
  operations stay on the full graph until a view contract for them is
  scientifically clear.
* Relationship results carry an opaque coordinate-space token so they cannot
  be used against the wrong receiver.

### Read-only properties

`ids`, `mother_ids` / `father_ids` / `twin_ids` (int64, `-1` missing),
`mother_rows` / `father_rows` / `twin_rows` (int32, `-1` absent/external),
`sex` (int8 or `None`), `depth` (int32, always present), `generation_labels`
(int32 or `None`), `birth_year` (int32 or `None`), `n_individuals` / `len`.
The ambiguous `generation`, `mother`, `father`, `twin`, and `n` names are
gone. Exposed arrays are read-only; mutation fails rather than appearing to
change the graph.

### Relationship registry

One immutable `RELATIONSHIPS` mapping replaces `REL_REGISTRY` and
`PAIR_KINSHIP`. Each `RelationshipCategory` carries `code`, `label`,
`degree`, `nominal_kinship`, `up`, `down`, `ancestor_count`, `first_role`,
`second_role`. Registry order is the documented same-degree precedence for
closest-category classification. The ordered 23-variant Rust enum becomes the
eventual source.

### Relationship pairs and counts

* `relationship_pairs(max_degree=)` or `relationship_pairs(categories=)`,
  exactly one selector. The same selector contract applies to
  `relationship_counts` and `relationship_kinship_matrix`. Selection is an
  output filter: closer-category dependencies are always resolved internally.
* `RelationshipPairs` is an immutable mapping over all 23 codes. Each
  `RelationshipPairBlock` has `first_rows`, `second_rows`, `first_role`,
  `second_role`, `requested`, and `len`, and unpacks as
  `(first_rows, second_rows)`. Unrequested blocks are empty with
  `requested=False`; counts report them as unrequested, not zero.
* Pair contracts:
  1. asymmetric categories have fixed semantic roles (offspring→mother,
     offspring→father, descendant→ancestor, and the collateral analogues);
  2. symmetric categories have no role distinction;
  3. internal subtraction keys canonicalise independently of public role
     order;
  4. every unordered pair appears in at most one category: lowest degree,
     then registry precedence;
  5. when both orientations of an asymmetric category are valid via different
     paths, the pair is returned once, orientation chosen deterministically
     by input-row order;
  6. blocks are sorted by canonical unordered row key, so scheduling cannot
     affect output order;
  7. rows are int32 in Python and 1-based integers in R;
  8. results own their arrays and never retain the graph or view.
* `sibling_pairs()` is removed; callers request the sibling categories.
  Per-category exclusion lists stay beside each category implementation.
* `relationship_counts` is exact. `estimate_relationship_counts` replaces
  `count_pairs_streaming` with a typed result (values, requested, exact,
  approximate, clamped categories) and one `RuntimeWarning` per clamped call.
  It is full-graph-only initially.

### Kinship and inbreeding

* `pair_kinship(first_rows, second_rows)`, `pair_kinship(block)`, and
  `pair_kinship(pairs)` return read-only float32 and accept arbitrary and self
  pairs. The value is the pinned float32 recurrence of ADR 0009, bit-identical
  to the matrix entry for the same pair. A collection query is one core call
  sharing one memo.
* `kinship_matrix()` is complete (every nonzero pedigree kinship).
  `relationship_kinship_matrix(...)` is structurally limited to the selected
  closest categories, but every retained coefficient is the full-pedigree
  value, never a propagation-pruned one. Both include the diagonal.
* CSC contract: SciPy `csc_matrix`, float32 data, int32 indices/indptr, sorted
  rows per column, read-only cached arrays. "Pedigree-specific" permits only
  the per-step float32 rounding of the pinned recurrence (ADR 0009); every
  entry equals the `pair_kinship` value for that pair.
* Canonical `inbreeding()` is MZ-aware and satisfies `F_i = 2·phi(i,i) − 1`.
  Issue #8 decides whether the MZ-naive Meuwissen–Luo implementation is
  deleted or retained under an explicit non-canonical name.
* `pair_kinship` is recurrence-only and never reads a cached matrix; its
  result does not depend on call history (ADR 0009, resolving issue #6).
  Callers thresholding against a non-dyadic cutoff widen to float64 first.

### Generation summaries, lineage, connectivity

* `mean_kinship_by_generation()` returns `generations`, `mean_kinship`,
  `pair_counts`, `unlabelled_individual_count`. Wholly absent labels fall back
  to structural depth; partial labels exclude and report the unlabelled, never
  silently assign depth. Only observed labels are returned.
* Effective-size estimators needing generations reject partial labels;
  sex-dependent estimators reject unknown sex; sex-independent ones continue.
* `distinct_ancestor_counts()` and `descendant_path_counts()` name their
  differing semantics. `connected_component_ids()` returns int64 rows aligned
  to input, each the smallest original ID in its represented parent-edge
  component, matching fitACE's deterministic FID contract without importing
  its founder-family policy.

### Namespace

Root exports: `PedigreeGraph`, `PedigreeView`, `RelationshipCategory`,
`RelationshipPairs`, `RelationshipPairBlock`, `RELATIONSHIPS`,
`PedigreeValidationError`, `MissingMetadataError`, `ResourceError`,
`configure_threads`. `FrameLike` moves to `pedigree_graph.typing`.
Effective-size functions, cohort utilities, and result classes move to public
`pedigree_graph.effective_size`; `compute_all_ne` becomes
`estimate_effective_sizes`. Estimator formulas do not change.

### Errors

Construction failures raise `PedigreeValidationError` (a `ValueError`) with a
stable `.code`; missing analysis metadata raises `MissingMetadataError`;
allocation/capacity failures raise `ResourceError`. Tests assert codes and
fields, not prose. Existing regex-matched messages are not compatibility
requirements.

## Consequences

* Every consumer (simACE, fitACE core and method packages, fitACE_epimight,
  PA-FGRS, pedsum) migrates in one coordinated change with 0.8.0. fitACE's
  private connected-components reach and PA-FGRS's `_compute_depth` import are
  replaced by public calls.
* Lineal pair orientation becomes a property of the category, not of how the
  graph was built. Consumers that relied on canonical `(lo, hi)` ordering of
  lineal pairs must read roles from the block.
* Supplying `generation` no longer changes any relationship or kinship value.
* `kinship_matrix(min_kinship=...)` semantics disappear; the relationship-
  limited matrix is the supported sparse form and is compared, in the
  performance gate, against fitACE's current exact construction rather than
  against pruned output.
* The experimental BFS engine is kept only as far as the pure-Python break
  requires and is deleted before the Rust relationship migration (issue #7).
* 0.8.0 may be the last setuptools-scm release; it is the frozen baseline the
  native slices are differentially tested against.

## Alternatives considered

* **Keep the 0.7 API and add deprecation shims** — rejected. The problems are
  in the semantics (coordinate spaces, orientation, generation-as-depth), not
  the names, and shims would carry them into the Rust boundary.
* **Redesign during the Rust port** — rejected. Two moving targets at once
  removes the pure-Python baseline that makes native slices differentially
  testable.
* **Expose the full graph API on views** — deferred. Matrix, inbreeding, and
  lineage semantics over a reordered subset are not yet scientifically pinned;
  exposing them would repeat the `from_subsample` mistake.
* **One generic exclusion table for closest-category subtraction** —
  rejected in favour of explicit per-category lists plus a global
  uniqueness/precedence check, so a missing exclusion is caught rather than
  hidden in table lookup.
