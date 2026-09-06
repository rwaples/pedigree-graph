# Changelog

This file tracks public-API changes per release.  For per-commit
history, see `git log`.  Historical release notes prior to v0.5.0
live on the corresponding GitHub release pages.

## Unreleased

- **Added: three explicit kinship-matrix families** (ADR 0006, ADR 0009,
  slice 5b). `kinship_matrix()` is complete;
  `relationship_kinship_matrix(max_degree=...)` or `(categories=...)` contains
  exactly the selected closest-category pairs plus the diagonal; and
  `approximate_kinship_matrix(min_propagated_kinship=0.001)` preserves the
  0.7.1 propagation-pruned candidate support plus the diagonal. The approximate
  threshold applies to intermediate propagation values, is not a final-value
  cutoff, and can admit or omit pairs relative to thresholding
  pedigree-expected coefficients. Every retained value in every family is now
  the pinned float32 recurrence and is bit-identical to `pair_kinship`; the
  approximate family discards the old propagated values after selecting their
  support and captures exact candidate coefficients during one complete
  retiring-DP pass; sparse relationship support uses deterministic bounded pair
  chunks. All matrices
  are cached by operation and selector and return CSC with float32 data, int32
  indices/indptr, sorted rows, and read-only arrays. A zero approximate
  threshold delegates to the complete matrix; non-finite or out-of-range
  thresholds are `ValueError`. The 0.7.1 overloaded
  `kinship_matrix(min_kinship=..., max_degree=...)` remains temporarily, routing
  positive resolved thresholds to the approximate-support family rather than
  describing propagation pruning as exact relationship support. Profiling and
  the runtime/memory decision are recorded in `benchmarks/matrix_exactification.md`.

- **Added: `PedigreeGraph.pair_kinship` and `PedigreeView.pair_kinship`**
  (ADR 0006, ADR 0009, slice 5a).  Three call forms:
  `pair_kinship(first_rows, second_rows)` for any pairs, self pairs included;
  `pair_kinship(block)` for one `RelationshipPairBlock`; and
  `pair_kinship(pairs)` for a whole `RelationshipPairs`, which runs one
  recurrence with one shared memo and returns an immutable mapping over all
  23 codes.  Values are read-only float32, positionally aligned to the input.
  Each value is the pinned float32 recurrence: peel the endpoint with the
  greater structural depth, ties by the greater row, every half-sum rounded
  once to float32.  Within one receiver the value is bit-identical to the
  `kinship_matrix` entry for the same pair, to the reversed endpoint order,
  and to itself before and after a matrix is cached, because the call never
  reads a cached matrix (issue #6).  A returned `0` is an exact `0`.  Two
  graphs built from one pedigree in different row orders agree within
  `2 * (depth_a + depth_b + 1) * 2**-25` on deep inbred pairs (measured: at
  most 2 ulp on the 60-generation closed herd, none on any shallow fixture).
  Widen to float64 before comparing against a non-dyadic cutoff.  A block or
  collection from another receiver fails with `coordinate_space_mismatch`;
  row arguments fail with `invalid_shape`, `invalid_integer_value`,
  `pair_row_out_of_range`, or `pair_length_mismatch`; a memo past its cap
  is `ResourceError("memo_capacity_exceeded")`.  The call commits the package
  thread budget like every 0.8 operation and runs on one thread.
- **Changed: `compute_pair_kinship` returns float32** (0.8.0-DELETE adapter).
  The dict form and the caller-space rows of a `from_subsample` graph are
  kept, but the values are those of `pair_kinship`, so on pedigrees deep
  enough for a kinship to need more than 24 significant bits (about 24
  generations of sustained inbreeding loops) they differ from the 0.7.1
  float64 recurrence within the envelope above.  The arrays are also read-only
  now, as every 0.8 result is, so a consumer that mutated them in place must
  copy first.  The cached-matrix sampling branch is gone; results no longer
  depend on call history.
- **Changed: the pairwise kernel is float32 and runs in the stable
  depth-major order**, where peeling the greater row is the ADR 0009 rule.
  On the `random_30k` fixture (interleaved fresh processes, medians of three)
  the degree-3 batch of 566,720 pairs and the 30,300 self pairs run at the
  0.7.1 kernel's wall time with 25 percent less kernel-attributable RSS
  (560 MB against 747 MB, 289 MB against 384 MB).

- **Fixed: the scalar counter no longer allocates a table the size of the
  largest parent id.**  `_per_sex_anchor_sums` binned per-parent sums by
  original parent id, so a pedigree with ten-digit ids allocated a dense
  table of that size per sex side and per degree (480 MB peak on a 20-row
  pedigree with ids near 10^7).  Parents are now grouped by dense index.
  Counts are unchanged; on the parity fixtures, whose ids start at 10^7,
  `estimate_relationship_counts(max_degree=5)` drops from 0.45 s to 0.04 s
  on 30k rows and from 0.88 s to 0.46 s on 300k rows, with peak RSS at 0.28x
  and 0.42x of before (`benchmarks/bench_estimate_counts.py`, interleaved,
  medians of ten).  This predates 0.8 (`count_pairs_streaming` had it too).

- **Added: `PedigreeGraph.estimate_relationship_counts(max_degree=...)`**
  (ADR 0006, ADR 0011, slice 4c).  The memory-bounded scalar estimate that
  `count_pairs_streaming` computed, returned as a `RelationshipCountResult`:
  `None` above the cutoff, and per code whether the value is `exact`,
  `approximate`, or `clamped` (an inclusion-exclusion residual that
  underflowed and was floored at `0`; that `0` is not a true absence).
  MZ, MO, FO, FS, MHS, and PHS are exact and equal `relationship_counts`
  (the half-sib pairs a parent-offspring category claims under the
  precedence fold are subtracted).  Every other code is approximate: GP,
  GGP, GGGP, and G3GP are raw ancestor-path counts that over-count a pair
  also related at a shorter depth, as a half-sib, or as a closer collateral,
  and the cousin / collateral formulas assume a full complement of known
  ancestors.  The exact set is `REL_PLAN.estimate_exact` /
  `estimate_exact_codes()`; why it excludes the lineal codes is ADR 0011
  (`docs/adr/0011-scalar-estimate-exact-set-excludes-lineal-codes.md`).
  The result for each `max_degree` is computed once per graph and the same
  frozen object is returned afterwards.  The computation, and only the
  computation, emits one `RuntimeWarning` naming the clamped codes in
  registry order when there are any, before the result is cached; a cached
  retrieval is silent, and a different cutoff computes and warns on its
  own.  The call commits the package thread budget like every 0.8
  operation; the counter itself is single-threaded and its integer results
  do not depend on the budget.  Full-graph only.  `count_pairs_streaming`
  is now an adapter over the same computation (the unfolded raw counts, `0`
  above the cutoff, the 0.7.1 dict and scope rules kept, no thread-budget
  commit, no longer written to the matrix count cache), so it can emit that
  `RuntimeWarning` where 0.7.1 wrote a `logging` warning, and it shares the
  per-cutoff cache with the new method.

- **Added: `PedigreeView.relationship_pairs`, `relationship_counts` on both
  receivers, and `RelationshipCountResult`** (ADR 0006, slice 4b).
  `view.relationship_pairs(max_degree=...)` / `(categories=...)` classifies
  through the full graph, so a relationship whose connecting ancestors are
  unselected is still found, then reports only the pairs with both endpoints
  in the view, as view rows (`0 <= row < len(view)`).  Asymmetric blocks keep
  their role orientation; symmetric blocks store `first < second` in view
  rows; every block is sorted by the canonical unordered view-row key and
  carries the view's own coordinate token.  A view of fewer than two rows
  returns all-empty blocks after selector validation.
  `graph.relationship_counts(...)` and `view.relationship_counts(...)` take
  the same selectors and return `RelationshipCountResult`, a frozen mapping
  over all 23 codes to the exact block length (`None` where unrequested),
  with `requested` / `exact` / `approximate` / `clamped` code sets (in this
  release `exact == requested` and the other two are empty).  Exported from
  `pedigree_graph.relationships` and the package root.  The 0.7.1
  `from_subsample` now builds the full graph first and resolves the
  subsample ids through `view(ids=...)`, so a full pedigree that is both
  invalid and missing subsample ids reports the pedigree fault before
  `unknown_view_id`; its `extract_pairs` output is unchanged.

- **Added: `PedigreeGraph.relationship_pairs`, `RelationshipPairs`, and
  `RelationshipPairBlock`** (ADR 0006, slice 4a).  `relationship_pairs(
  max_degree=...)` or `relationship_pairs(categories=...)` (exactly one
  selector) returns an immutable mapping over all 23 registry codes.  Each
  block owns read-only int32 graph rows: for an asymmetric category
  `first_rows` carries `first_role` (offspring, descendant, niece_nephew,
  junior_cousin) and `second_rows` the counterpart; a symmetric category
  stores `first < second`.  Every unordered pair appears in exactly one
  category (lowest degree, then registry order), a pair valid in both
  orientations of an asymmetric category appears once with the lower graph
  row first, and blocks are sorted by the canonical unordered row key, so the
  output is bit-identical across thread counts.  Selection is an output
  filter: the closer categories a selected one depends on are always
  resolved.  Unselected blocks are empty with `requested=False`.  The engine
  honours `configure_threads`.  Both types are exported from
  `pedigree_graph.relationships` and the package root.  `extract_pairs` and
  `sibling_pairs` keep their 0.7.1 orientation and membership until 0.8.0
  removes them, but within a code they may now emit pairs in a different
  order.

- **Added: a Rust row-streaming relationship engine with exact,
  memory-bounded pair counts** (ADR 0010; issues #11 and #9).  `crates/core`
  (`pedigree-graph-core`) classifies every relationship pair up to degree 5
  one individual at a time, so peak memory is linear in the pedigree size:
  the 20M-row simACE pedigree counts all 23 categories exactly in 83 s on 12
  threads within 2.9 GiB, where the matrix engine needed an estimated
  150 GiB.  Counts are bit-identical to `count_pairs(max_degree=5)` on every
  parity fixture and bit-identical across thread counts.  Path multiplicity
  is saturated at two, which is provably exact for the engine's predicates
  and removes the overflow question of issue #9.  Not yet reachable from
  Python: `pgr-count` is a benchmark and parity CLI over the array dump from
  `tests/parity/dump_relationship_inputs.py`; the binding lands with the
  native scaffold.  The pixi manifest now provides the Rust toolchain.

- **Added: canonical construction** (ADR 0006).  `PedigreeGraph.from_frame(frame,
  *, sex_encoding="simace")` takes a dict of columns or any FrameLike table, and
  `PedigreeGraph.from_arrays(*, ids=, mother_ids=, father_ids=, twin_ids=None,
  sex=None, generation=None, birth_year=None, sex_encoding="simace")` takes the
  columns separately.  Neither applies a default: an omitted or wholly unknown
  `sex`, `generation`, or `birth_year` reads as absent rather than as a
  fabricated column.  The 0.7.1 entry points — `PedigreeGraph(data)`,
  `from_dataframe`, `from_subsample`, and the positional
  `from_arrays(ids, mothers, fathers, ...)` — keep their names, their
  depth-derived `generation` fallback, and their all-female `sex` default until
  0.8.0 removes them.  `from_arrays` serves both call forms: the canonical one
  is keyword-only and the 0.7.1 one is selected by any positional argument or by
  `mothers=`/`fathers=`/`twins=`.  Mixing the two, or naming neither, is a
  `TypeError`, as is passing `sex_encoding=` to the 0.7.1 form.

- **Added: read-only properties** (ADR 0006).  A graph now exposes `ids`,
  `mother_ids` / `father_ids` / `twin_ids` (int64, `-1` missing), `mother_rows` /
  `father_rows` / `twin_rows` (int32, `-1` absent or external), `sex` (int8 or
  `None`), `depth` (int32, always present), `generation_labels` (int32 or
  `None`), `birth_year` (int32 or `None`), `n_individuals`, and `len(pg)`.  Each
  array property hands back the graph's own storage — the same object on every
  access, read-only, so writing into what you read raises instead of silently
  changing the graph.  `depth` is structural and is computed on first access,
  never at construction.  The 0.7.1 names `mother`, `father`, `twin`,
  `generation`, and `n` still work and still mean rows, depth-fallback labels,
  and the row count; 0.8.0 removes them.

- **Added: canonical relationship registry** (ADR 0006, slice 2).  The root
  now exports `RELATIONSHIPS`, an immutable ordered mapping of all 23
  relationship codes to frozen `RelationshipCategory` records, also importable
  from `pedigree_graph.relationships`.  Each record carries `code`, `label`,
  `degree`, `nominal_kinship`, `up`, `down`, `ancestor_count`, `first_role`,
  and `second_role` (roles drawn from the closed `RelationshipRole` literal set
  exported by `pedigree_graph.relationships`), replacing the two parallel `REL_REGISTRY` / `PAIR_KINSHIP`
  lookups with one.  Iteration order is the documented same-degree precedence
  for closest-category classification.  `first` is the pair member with at
  least as many meioses to the shared ancestor(s), so `up` counts meioses from
  `first` up to the ancestor(s), `down` counts them from the ancestor(s) down
  to `second`, and `up >= down` holds for every category.  This flips the
  stored collateral orientation: `Av` is now `up=2, down=1`, where 0.7.1's
  `RelType` stored `up=1, down=2`.  Asymmetric categories name their two
  positions (`offspring`/`mother`, `offspring`/`father`,
  `descendant`/`ancestor`, `niece_nephew`/`aunt_uncle`,
  `junior_cousin`/`senior_cousin`, where junior means generationally further
  from the shared ancestors, not younger by birth year); the seven symmetric
  categories carry `None` for both and report `symmetric` as `True`.
  `REL_REGISTRY`, `PAIR_KINSHIP`, and `RelType` keep their 0.7.1 names, values,
  and orientation, but are now detached snapshots built once from
  `RELATIONSHIPS`: mutating them no longer changes the registry or any engine
  output.  0.8.0 removes all three.

- **Added: pedigree views** (ADR 0006, slice 3).  `graph.view(ids=[...])` and
  `graph.view(rows=[...])` return a `PedigreeView` over exactly that selection,
  in exactly the order given; naming both keywords, or neither, is a
  `TypeError`.  An empty selection of any dtype is a valid empty view.  A view
  exposes read-only `ids` (int64) and `graph_rows` (int32), each its own
  contiguous storage handed back unchanged on every access, plus
  `n_individuals` and `len(view)`; mutating the selection array afterwards
  cannot change the view.  A bad selection raises one of four structured
  errors: `duplicate_view_id`, `unknown_view_id` (which is also how a negative
  id reads), `duplicate_view_row`, and `view_row_out_of_range` (there is no
  negative indexing).  Each selector checks single entries before pairs:
  membership or range first, duplicates last, so a value too large for
  int64 reads as unknown or out of range rather than as a fifth code.  The
  graph memoises a sorted-id index on the first `ids=` view, so later id
  views cost the selection's size, not a sort of the whole pedigree.
  Shape and lossless-integer failures report
  `invalid_shape` and `invalid_integer_value` naming the `ids` or `rows`
  argument.  Each graph and each separately built view owns a distinct opaque
  coordinate token, so equivalent selections from two `view(...)` calls are not
  interchangeable receivers.  Relationship methods on views arrive in a later
  slice, and `from_subsample` is unchanged for now.

- **Changed: MZ pairs are validated at construction.**  Every constructor now
  rejects a represented MZ reference that is self-directed
  (`mz_self_reference`), not reciprocated (`mz_nonreciprocal`, which is also how
  a third row pointing into a pair is reported), names different parents
  (`mz_parent_mismatch`, with the offending roles in `parent_roles`), or pairs
  two individuals of different known sex (`mz_sex_mismatch`).  Parents are
  compared by id, so co-twins sharing one unrepresented parent agree.  A co-twin
  outside the represented rows forms no pair and is not checked.
  `compute_inbreeding()` no longer performs this check; it raised
  `mz_nonreciprocal` and `mz_parent_mismatch` lazily in 0.7.1, and a pedigree
  that used to construct and fail later now fails at construction.

- **Changed: partly known generation labels are rejected.**  A supplied
  `generation` column containing `-1` now raises
  `MissingMetadataError("missing_generation_labels", status="partial")`, with
  `missing_count`, from `per_gen_mean_kinship()`, every generation-indexed Ne
  estimator, and `compute_all_ne`.  Previously the `-1` rows wrapped into the
  last cohort bucket of the kinship DP theta sums and the Caballero-Toro
  founder sweep, silently biasing `ne_caballero_toro` and `ne_inbreeding`,
  while `ne_coancestry` and `per_gen_mean_kinship()` failed with an
  unstructured `ValueError` from `np.bincount`.  A wholly absent column is
  unchanged: the 0.7.1 estimators still fall back to structural depth.

- **Added: `configure_threads` is exported from the package root.**  One
  package-wide budget resolving `configure_threads(n)` >
  `PEDIGREE_GRAPH_THREADS` > `1`, committed the first time it is read.
  Repeating the committed value is accepted; changing it afterwards raises
  `RuntimeError`.  There is no per-call thread argument (ADR 0007).

- **Changed: any acyclic input row order is accepted.**  A parent no longer
  has to precede its child; construction rejects only genuine cycles.  Public
  outputs stay aligned to the input rows.  Internally the graph derives one
  private stable depth-major order (`_topology.build_topology`) and runs the
  kernels that need parents first — the Meuwissen-Luo inbreeding walk, the
  descendant path-count sweep, the pairwise kinship recurrence, the kinship
  DP, and the Caballero-Toro founder sweep — in that order, mapping their
  results back to graph rows.  Ties inside a depth keep input row order and no
  original id ever enters the ordering.

- **Changed: a supplied `generation` label no longer affects any relationship
  or kinship output.**  Structural depth drives the kinship DP and every other
  order-dependent kernel; labels are metadata.  `per_gen_mean_kinship()` still
  groups its cohorts by the supplied label, and so do the label-indexed
  effective-size results.  Callers who passed labels that disagreed with
  structural depth will see kinship and relationship results change to the
  structurally correct values.

  Two floating consequences of accepting arbitrary order, both governed by
  ADR 0009: two graphs built from the same pedigree in different row orders
  agree exactly on every integer, category and pair result, and agree on
  kinship within the recurrence envelope
  `abs(a - b) <= 2 * (depth_a + depth_b + 1) * 2**-25`.  The propagated
  `kinship_matrix(min_kinship > 0)` support is approximate by construction
  (ADR 0005) and is the one output whose *support* a permutation can move,
  by a fraction of a percent on the test corpus.

- **Added: structured errors** (ADR 0006).  `PedigreeValidationError` and
  `MissingMetadataError` (both `ValueError`) and `ResourceError` (a
  `RuntimeError`) are exported from the package root.  Each carries a stable
  `.code` string and an immutable `.fields` mapping naming the offending
  field, row, value, or limit; messages are prose and are not a contract.
  Construction failures, `max_degree` range failures, MZ-invariant failures,
  and the `compute_n_descendants()` int32 overflow (previously an
  `OverflowError`) now raise these instead of bare `ValueError` /
  `OverflowError`.  Tests should assert `.code`, not the message text.

- **Changed: only `id`, `mother`, and `father` are required.**  `twin`,
  `sex`, `generation`, and `birth_year` are optional in every dict and frame
  input; a frame column that is wholly missing (all `-1` or all host nulls)
  now reads exactly like an omitted one.  The 0.7.1 attributes keep their
  0.7.1 defaults for now — `pg.sex` is all-female, `pg.generation` falls back
  to structural depth — but `pg.birth_year` is `None` for a wholly unknown
  birth-year column, where 0.7.1 returned an all-`-1` array.

- **Changed: numeric input is coerced losslessly and range-checked.**
  Integer, integral-float, and object columns (pandas nullable, mixed
  lists) are accepted; polars nulls and pandas `pd.NA` become the `-1`
  missing sentinel everywhere except `id`, which has no missing value.
  Non-integral floats, infinities, `bool` columns, strings, `uint64` values
  above the int64 maximum, and out-of-range values are rejected with
  `invalid_integer_value` or `value_out_of_range` naming the position.
  `sex_encoding="plink"` maps `1 -> 1` male, `2 -> 0` female, `0 -> -1`
  unknown; the default `"simace"` encoding stores `0` / `1` / `-1` as given.

- **Added: cyclic parent references are rejected** with a `cycle` error
  carrying one deterministic witness — the tuple of ids around the cycle,
  the same for a given graph whatever order its rows arrive in.  A child
  naming one id in both parent roles is `same_parent_id`, external ids
  included.

- **Changed: construction owns its arrays.**  Every column is copied into
  contiguous, read-only storage, so mutating the caller's arrays after
  construction cannot change the graph.

- **Changed: `compute_inbreeding()` is MZ-aware** (#8, ADR 0008).  The
  Meuwissen–Luo walk now runs over the genome-node pedigree, in which MZ
  co-twins share one node, so `F` equals `2 * phi(i, i) - 1` from
  `compute_pair_kinship()` and from the `kinship_matrix()` diagonal on every
  pedigree.  Previously co-twins were walked as two individuals: twins with
  parents counted as full sibs and founder co-twins as unrelated founders,
  so an inbreeding path through both members of an MZ pair was
  under-weighted or missed.

  **Numeric change for existing callers** on pedigrees containing MZ twins:
  `F` never decreases; on a 1M-row simACE pedigree 0.12% of individuals
  change, by at most 1/128, and mean `F` moves under 1%.  Everything
  derived from `compute_inbreeding()` (`ne_inbreeding`,
  `ne_individual_delta_f`, the Caballero–Toro estimators, pedsum's
  inbreeding section) shifts accordingly.  Wall time is unchanged; peak
  memory rises by about 2% on pedigrees that contain twins.

  `compute_inbreeding()` now raises `PedigreeValidationError`
  (`mz_nonreciprocal` or `mz_parent_mismatch`, both `ValueError`s) when a
  represented MZ reference is not reciprocal or the co-twins do not share
  both parent rows.  An absent co-twin (`twin == -1`) is not an MZ pair.

## v0.7.1

- **Fixed: founder MZ co-twins were dropped by the kinship DP** (#5).  The MZ
  twin pass ran only inside the depth ≥ 1 loop, so a twin pair sitting at
  depth 0 — both co-twins founders — never had its off-diagonal written.
  `kinship_matrix()` returned `0.0` for the pair on the capped and uncapped
  paths alike, disagreeing with `compute_pair_kinship()`, which was correct.

  The missing edge was not the whole cost: because it was absent at depth 0,
  every merge walk below the pair propagated the zero, so **descendants of
  founder co-twins were unrelated to each other** in the returned matrix
  (children of co-twins: `0.0` against an exact φ = 0.125).  Twins with
  parents were never affected.

  The pass is now a shared `_mz_twin_pass()` run once per depth, depth 0
  included, before that depth's retirement.

  **Numeric change for existing callers** on pedigrees containing founder
  co-twins: `kinship_matrix()` gains those entries and everything derived from
  them, and `per_gen_mean_kinship()` rises accordingly for the generations
  below such a pair.  Pedigrees with no founder co-twin pair are unaffected.

## v0.7.0

- **Structural frame protocol (`FrameLike`), exported.**  Every constructor
  (`PedigreeGraph(...)`, `from_dataframe`, `from_subsample`) now accepts any
  column-addressable table exposing `.columns`, string `__getitem__`, and
  column `.to_numpy()` — pandas *and* polars DataFrames both qualify — while
  the package continues to import neither frame library at runtime.
  `dict[str, np.ndarray]` input remains accepted everywhere.  Columns are
  extracted via `.to_numpy()` (previously pandas-only `.values`); NA-free
  pandas nullable-integer columns are now accepted.  A frame missing a
  required column now reports the uniform `ValueError` instead of a raw
  `KeyError`.  `from_dataframe` is kept as a compatibility name.
  Coercion lives in the new focused `pedigree_graph/_frames.py` module.
- Test fixtures serve polars frames (the family's primary library);
  focused pandas compatibility coverage — including nullable integers —
  lives in `tests/test_frame_inputs.py`.  `polars` joins pandas in the
  `test` extra only; runtime dependencies are unchanged.

## v0.6.0

- **First release published to PyPI.**  `pip install pedigree-graph` now
  works, retiring the `git+https://...@vX.Y.Z` install form that consumers
  had to carry because the project was unavailable on the index.
  Distributions are built and uploaded by a tag-triggered GitHub Actions
  workflow using PyPI trusted publishing (OIDC), so no API token is
  stored in the repository or in CI secrets.  Downstream packages pinning
  a git URL can move to a version range such as
  `pedigree-graph>=0.6,<0.7`; the bound is worth keeping tight because
  `PAIR_KINSHIP` and `extract_pairs` are consumed directly by simace,
  fitace, and pedsum.
- **Packaging metadata completed for the index listing.**  `readme`,
  `license` (SPDX `MIT`), `license-files`, `authors`, `classifiers`, and
  `[project.urls]` are now declared, so the PyPI page renders the README
  and links back to the repository and this changelog.  The build backend
  floor moved from `setuptools>=64` to `setuptools>=77`, which is where
  PEP 639 SPDX license support lands.
- **`py.typed` marker added.**  The package now advertises inline type
  information under PEP 561, so type checkers read its annotations when
  it is installed as a wheel rather than as an editable source checkout.
  Runtime code is unchanged.

## v0.5.4

- **Self-kinship diagonal fixed for inbred individuals when rows are not
  in generation-monotonic order.**  The matrix-DP kinship kernel assumed
  every relative discovered during a row's merge walk had a smaller row
  index, but `from_arrays` only requires topological order (parents
  before children).  When a relative at an earlier generation had a
  higher row index, the diagonal append broke the row's sorted order and
  the binary search reading `phi(mother, father)` silently returned 0 —
  so the self-kinship diagonal read `(1+0)/2` instead of `(1+F)/2`, and
  the GRM diagonal consumed downstream was wrong for inbred individuals.
  Off-diagonals and the pairwise `compute_pair_kinship` path were
  unaffected, as was any pedigree loaded in generation order (the common
  case, which now hits a zero-overhead fast path).

- **Pair-set subtraction and id validation no longer go through
  `np.unique`/`np.isin`.**  Three internal hot spots were rewritten with
  sort/searchsorted or diff-based equivalents, verified output-identical
  on real data and under differential fuzzing (5,000 randomized trials
  per site, including empties, duplicates, and ids near the int64
  pair-key bound):
  - `PedigreeGraph._subtract_pairs` now sorts the remove keys and
    binary-searches candidates instead of `np.isin` (~6.9x on a 300k-N
    pedigree; halves `sibling_pairs()` wall time).  Same pattern —
    and rationale — as `extract_from_sparse` already used.
  - The duplicate-id check in `_validate_id_column` counts equal
    adjacent elements after a sort instead of `len(np.unique)` (~37x).
  - `pairs_from_groups` detects group boundaries by diffing its
    already-sorted key array instead of re-running `np.unique` on it.

  Returned pairs, counts, orderings, and error messages are unchanged —
  this change is performance-only.

## v0.5.3

- **`count_pairs_streaming` warns when a cousin/collateral residual
  underflows.**  The scalar engine derives `H1C`, `1C1R`, `1C2R`, and
  `H1C1R` by inclusion–exclusion — subtracting closer-relationship
  contributions with fixed coefficients that are exact only on non-inbred,
  single-mating pedigrees.  On inbred or structurally complex real
  pedigrees those corrections can over-count, driving the raw residual
  negative; it was then silently clamped to `0`, indistinguishable from a
  true absence (e.g. millions of `1C` but `H1C == 0`).  The clamp now logs a
  `WARNING` naming the code and the underflow magnitude and points to the
  matrix engine (`extract_pairs`) for an exact count.  Returned counts are
  unchanged — only the diagnostic is new.

## v0.5.2

- **`count_pairs_streaming()` releases its transient matrices on exit.**
  The scalar streaming counter builds the adjacency powers `_A`…`_A5` and
  now drops them via `_release_pair_matrices()` before returning, exactly
  as `extract_pairs()` already did.  Previously they stayed resident for
  the graph's lifetime, inflating peak memory of any later inbreeding / Ne
  / lineage work on the same graph (~400–520 MiB on a 1M-row pedigree).
  The counts remain cached and the matrices rebuild lazily via
  `_ensure_parent_csr()` if pair work runs again, so callers that reached
  into the private `_release_pair_matrices()` after a streaming call (e.g.
  pedsum `summarize`) can drop that workaround.  Fixes #4.

## v0.5.0

- **Registry-aligned `max_degree` semantics.**  `extract_pairs`,
  `count_pairs`, and `count_pairs_streaming` now include exactly the
  relationship categories whose `REL_REGISTRY[code].degree` is less
  than or equal to the cutoff.  `max_degree=0` is MZ-only,
  `max_degree=2` stops before 1st cousins, and `max_degree=3`
  includes 1st cousins plus the other degree-3 categories.  The public
  defaults changed from `2` to `3` to preserve the old default behavior
  of including 1st cousins.

- **`PedigreeGraph.compute_n_ancestors()`** — new cached method.
  Returns the per-individual count of *distinct* strict ancestors
  (`int32`, length `n`).  Backed by a sparse boolean transitive
  closure of the parent graph; memory scales with the total closure
  size.  Suitable for pedigrees up to ~1M rows on commodity hardware;
  deeper / wider pedigrees may need a future retirement-style DP
  variant.

- **`PedigreeGraph.compute_n_descendants()`** — new cached method.
  Returns the per-individual descendant *path count* (`int32`, length
  `n`).  In non-inbred pedigrees this equals the unique-descendant
  count; in inbred pedigrees it over-counts a descendant reachable via
  multiple ancestor paths.  Matches the convention used historically
  by `pedsum` (`compute_descendants`) and by the matrix engine's GP /
  Av / 1C pair counts.  Raises `OverflowError` if any per-individual
  path count exceeds `int32` max (the kernel accumulates in `int64`
  and the cast happens after a bounds check, so deeply inbred
  pedigrees cannot silently wrap).

- **`PedigreeGraph.from_arrays(...)`** — accepts a new optional `sex`
  kwarg (`np.ndarray | None`).  When omitted, behaviour is unchanged
  (sex defaults to zeros).  Existing callers do not need updates.

- **Defensive warning for the `sex`-default foot-gun.**
  ``ne_sex_ratio`` and ``ne_variance_family_size`` now emit a
  ``RuntimeWarning`` when ``pg.sex`` is uniformly 0 or 1 — almost
  always a sign that the caller forgot to pass ``sex=`` to
  ``from_arrays`` and is consuming silently-degenerate (single-sex)
  Ne results.  The estimator return values are unchanged (``ne=None``);
  the warning is the new diagnostic.  Kinship-only callers
  (relationship-pair extraction, GRMs, PA-FGRS) are not affected
  because they don't invoke the sex-aware estimators.

- New private kernel module `pedigree_graph/_lineage_kernel.py` houses
  the descendant (numba-JIT) and ancestor (scipy sparse) primitives.

- **`PedigreeGraph.count_pairs_streaming(max_degree=2, scope="full")`**
  — new method.  Memory-bounded relationship pair counts via pure
  scalar arithmetic; no pair-key arrays are ever materialized.  Peak
  memory is O(N) regardless of pedigree density.  Returns all 23
  codes from `REL_REGISTRY`.  Bit-identical to `count_pairs` for
  the 10 simple codes (`MZ`, `MO`, `FO`, `FS`, `MHS`, `PHS`, `GP`,
  `GGP`, `GGGP`, `G3GP`); approximate (~1% on deep low-inbreeding
  pedigrees) for the 13 cousin / collateral codes (`Av`, `1C`,
  `H1C`, `HAv`, `GAv`, `GGAv`, `G3Av`, `HGAv`, `HGGAv`, `1C1R`,
  `H1C1R`, `1C2R`, `2C`).  The scalar path is **full-graph only**:
  `scope='subsample'` raises `NotImplementedError` on graphs built
  via `from_subsample` (use `count_pairs` for subsample-restricted
  counts).  See `LIMITATIONS.md` for the full precision contract.
  Benchmark: 5 seconds on a 783K-row stallion-heavy livestock
  pedigree where both matrix and BFS engines OOM at 30 GB.

- **`max_degree` validation** — `extract_pairs`, `count_pairs`, and
  `count_pairs_streaming` now reject `max_degree` outside `[0, 5]`
  with `ValueError`.  Degree 0 is accepted (cheap codes MZ / MO /
  FO / FS are computed regardless; the cap controls the expensive
  matrix products at degree 2 and above).
