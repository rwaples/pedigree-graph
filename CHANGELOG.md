# Changelog

This file tracks public-API changes per release.  For per-commit
history, see `git log`.  Historical release notes prior to v0.5.0
live on the corresponding GitHub release pages.

## v0.5.0 (unreleased)

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
