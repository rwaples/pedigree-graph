# Changelog

This file tracks public-API changes per release.  For per-commit
history, see `git log`.  Historical release notes prior to v0.5.0
live on the corresponding GitHub release pages.

## Unreleased

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
