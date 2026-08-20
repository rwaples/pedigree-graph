# Changelog

This file tracks public-API changes per release.  For per-commit
history, see `git log`.  Historical release notes prior to v0.5.0
live on the corresponding GitHub release pages.

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
