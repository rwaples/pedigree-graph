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
  Av / 1C pair counts.

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
