# 11a gate for pedigree-graph 0.8.3 (2026-09-09)

Slice 11 (`plans/pedigree-graph-slice-11-relationship-counts.md`): `relationship_counts`
on the Rust row-streaming engine, with the closest-category fold in the engine
(ADR 0010 as amended).

## Semantics check before any wiring

`pgr-count` with the fold, on the 18 fixtures the 0.8 golden lock covers
(`tests/data/relationship_pairs_v0.8/manifest.json`, block lengths of
`relationship_pairs(max_degree=5)`): all 23 codes equal on every fixture,
`random_30k` included. The pre-fold engine differed on 13 of the 26 dumped fixtures
(for example `random_30k` 1C1R 149,392 raw against 21,618 folded).

## Fold cost

The first fold (a running sorted union per category) cost 27 percent of engine wall
on `random_30k` (3.21 s to 4.07 s single-threaded, `pgr-count`). Replaced by a
stamp-based claim over the accumulator's marker array (no allocation, no extra
memory): 3.36 s, within 4 percent of the pre-fold engine; `sim_d6_n3000` 1.21 s to
1.24 s.

## Tests

- `cargo test --release`: core `47 passed` (five new fold and mask tests), parity
  `2 passed` against the regenerated `.counts.json` (26 fixtures, one and four threads).
- `cargo fmt --check`, `cargo clippy --all-targets -D warnings`, `ruff check`,
  `ruff format --check`, `ty check`: clean.
- `pytest -m "not slow"`: `2284 passed, 7 skipped, 9 deselected in 526.84s`, exit 0
  (198 of them the new `tests/test_native_relationship_counts.py`).
- `pytest -m slow`: `9 passed, 2291 deselected in 931.32s`, exit 0.

## Perf gate (behavior-equivalent, 5 percent blocker)

`bench_counts.py` (this directory): `relationship_counts(max_degree=5)`, one fresh
process per measurement, five interleaved runs per version, PyPI 0.8.2 wheel against
the 0.8.3 wheel built from this tree, two venvs on the pixi Python
(`counts-bench.tsv`). Medians, CPU at its 2.6 GHz base clock:

| pedigree | threads | 0.8.2 wall | 0.8.3 wall | ratio | 0.8.2 peak RSS | 0.8.3 peak RSS | ratio |
|---|---|---|---|---|---|---|---|
| `random_30k` (30k rows, 3.07M pairs) | 1 | 1.629 s | 0.977 s | 0.60 | 426 MiB | 185 MiB | 0.43 |
| `random_30k` | 6 | 1.050 s | 0.204 s | 0.19 | 536 MiB | 186 MiB | 0.35 |
| `synthetic_300k` (300k rows, 33.9M pairs) | 1 | 21.700 s | 17.200 s | 0.79 | 2820 MiB | 238 MiB | 0.08 |
| `synthetic_300k` | 6 | 12.710 s | 3.670 s | 0.29 | 3753 MiB | 241 MiB | 0.06 |

Total pair counts and the 1C1R count agree between versions in every row. No
regression.

An earlier pass of the same runs (`counts-bench-throttled-800mhz.tsv`,
`capability-bench-throttled-800mhz.tsv`) was taken while the host was clamped to
800 MHz by an external PROCHOT assertion; the ratios were the same (0.65 / 0.24 /
0.81 / 0.28), the absolute walls three times higher. Kept for the record, not used.

## New capability (no baseline that fits)

`capability-bench.tsv`, the 0.8.3 wheel on the simACE `bench_pedsum` pedigrees, graph
already built, whole-process peak RSS:

| rows | threads | wall | RSS before | peak RSS | pairs |
|---|---|---|---|---|---|
| 2M | 1 | 65.6 s | 632 MiB | 677 MiB | 212,626,359 |
| 2M | 12 | 9.6 s | 651 MiB | 738 MiB | 212,626,359 |
| 20M | 1 | 714 s | 2.90 GiB | 3.35 GiB | 2,124,650,324 |
| 20M | 12 | 102 s | 2.90 GiB | 4.22 GiB | 2,124,650,324 |

`estimate_relationship_counts(max_degree=5)` on the 2M graph: 2.0 s, 1.28 GiB,
1C1R 24 percent above exact (decision 8 input). The 20M estimate needs about 10 GiB
and was stopped by the host's memory guard on the first pass; not re-run.

## Wheel gate (`tools/pg08_wheel_gate.sh`)

Built from a clean worktree of `v0.8` at `ce698ec` (version 0.8.3): the sdist and
the `cp313-abi3` wheel install into fresh venvs, the import resolves from
site-packages, and the package suite against the installed wheel from a cwd outside
the tree passes: `2279 passed, 3 skipped, 9 deselected in 203.11s`, exit 0
(`wheel-gate.json`, `wheel-pytest.log`).

## Consumer gate

`EPIMIGHT_CONDA_ENV=epimight-master pixi run --frozen python tools/pg08_release_gate.py
run --stage 11a --routing <wheel-site> --unit <unit>`, one unit per invocation (a
single all-units run was stopped by the host's memory guard during the simACE unit),
locks still pinning 0.8.2, every consumer's `routing` step resolving `pedigree_graph`
under the 0.8.3 wheel site. All eleven green, `failed units: none`:

| unit | suite | wall |
|---|---|---:|
| pedigree-graph | 2285 passed, 7 skipped, 9 deselected (+ ruff, format, ty) | 175.2 s |
| simACE | 1470 passed, 3 skipped (+ smoke `--forceall`, atlas) | 119.7 s |
| fitACE | 383 passed | 21.0 s |
| fitACE_pcgc | 211 passed, 4 skipped | 8.8 s |
| fitACE_iter_reml | 110 passed, 4 skipped | 48.2 s |
| ace_iter_reml | six binary test units, all exit 0 | 18.3 s |
| fitACE_tetraher | 33 passed | 6.4 s |
| fitACE_pafgrs | 119 passed | 13.5 s |
| fitACE_frailty | 7 passed | 5.0 s |
| fitACE_epimight | 256 passed, 19 deselected | 53.9 s |
| pedsum | 325 passed (+ TSV export, CLI smoke) | 189.5 s |

The first pass of this gate, at `23823bb`, failed the simACE unit (6 failed, 19
errors in `tests/analysis`, the `analyze` rule down in the smoke run):
`simace/analysis/stats/runner.py:203` calls `RelationshipCountResult.from_pairs`,
which slice 11 had deleted on a "no caller" check that covered only pedigree-graph's
tree. `ce698ec` restores it; the records here are from the rerun at that head.

## Next

Tag `v0.8.3` at `ce698ec` and publish (the maintainer's push), relock simACE, fitACE,
and pedsum, then `--stage 11b --routing locked`.
