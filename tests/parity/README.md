# 0.7.1 parity baseline

`tests/data/parity_v0.7.1/` holds pedigree-graph 0.7.1 outputs frozen at tag
`v0.7.1` (`9469a3c`). The 0.8 slices compare against it (plan:
`simACE/plans/pedigree-graph-0.8.0-slices-v2.md`, preflight item 4).

## Files

- `pedigrees.py` builds every input deterministically and imports nothing from
  `pedigree_graph`, so it runs unchanged under 0.7.1 and 0.8.
- `generate_baseline.py` runs the 0.7.1 API against a package root you name and
  refuses to run if `pedigree_graph` resolves anywhere else.
- `manifest.json` records the generator version, package commit, fixture
  parameters, SHA-256 of every input, and SHA-256 of every output.
- One `<fixture>.npz` per small fixture with the full arrays. Large fixtures
  have hashes and counts only.

## Regenerate

```bash
git worktree add ../pedigree-graph-v0.7.1 v0.7.1
pixi run python tests/parity/generate_baseline.py --package-root ../pedigree-graph-v0.7.1
```

Regeneration must reproduce every hash in `manifest.json`. Bump
`GENERATOR_VERSION` in `pedigrees.py` when an input changes.

## What is frozen, per fixture

- `extract_pairs(max_degree=5)` in 0.7.1 orientation, sorted by `(first, second)`.
- `compute_pair_kinship` (float64) aligned to those pairs.
- `count_pairs_streaming(max_degree=5)`.
- `compute_inbreeding`, `compute_n_ancestors`, `compute_n_descendants`,
  `per_gen_mean_kinship`, and the derived `generation` (structural depth).
- Upper triangle of `kinship_matrix(min_kinship=0.001)`: the propagated
  support slice 5b must reproduce. Small fixtures also freeze
  `kinship_matrix(0.0)`.
- `from_subsample` pairs over a seeded half-size shuffled selection, in
  caller coordinates.

## Baseline facts worth knowing

- `deep_inbred_60g` overflows int32 in `compute_n_descendants` under 0.7.1;
  the manifest records `n_descendants_overflow: true` and no array.
- `count_pairs_streaming` clamps negative residuals on inbred fixtures and
  prints a warning; the clamped values are what is frozen.
- `random_30k` takes about four minutes to freeze, nearly all of it in the
  propagated `0.001` matrix (26.9M upper-triangle entries).
