# 10c post-publish lock gate for pedigree-graph 0.8.1 (2026-09-09)

`v0.8.1` (`19b6b20`) was tagged and pushed; `publish.yml` run 34352521001
succeeded and PyPI serves five `cp313-abi3` wheels (manylinux x86-64 and
AArch64, macOS x86-64 and arm64, Windows x86-64) plus the sdist. The published
sdist sha256 `d80a67d9…` is byte-identical to the one the 10a wheel gate built
locally; the linux x86-64 wheel is `e3299955…`. The first `ci.yml` run on the
branch push failed (wheel job import mode, and last-bit float equality in the
Ne golden test on the runner's BLAS); both fixed in `6aa081d`, pushed, rerun
pending at the time of writing.

## Locks

pixi's PyPI index cache did not yet list 0.8.1, so `pixi lock` and
`pixi update pedigree-graph` first reported the locks up to date; after
`pixi clean cache --pypi`, `pixi update pedigree-graph` in simACE, fitACE, and
pedsum each reported exactly one change, `~ (pypi) pedigree-graph 0.8.0 -> 0.8.1`,
to the same x86-64 wheel. fitACE's diff also reorders an unchanged `coverage`
7.15.4 block. `pixi install --locked` then succeeded in all three, and
`pedigree_graph._native` imports from each env's site-packages with
`core_version() == 0.8.1`.

## Gate

`EPIMIGHT_CONDA_ENV=epimight-master pixi run python tools/pg08_release_gate.py
run --stage 10c --routing locked --unit <the nine consumer units>`.

First pass: eight units green, `fitACE_pcgc` red on one test,
`test_continuous_perf.py::test_tiny_under_budget[reference]`, a wall-clock
budget (2x a stored baseline) that the 10a run and three isolated reruns of the
module (11.3 s, 11.5 s, 11.4 s, all green) do not reproduce; the record below
is the unit rerun, which replaced the failing record. Every `routing` step
resolved `pedigree_graph` under the unit's own `.pixi/envs/default/…/site-packages/`.

| unit | suite | wall |
|---|---|---:|
| simACE | 1470 passed, 3 skipped (+ smoke `--forceall`, atlas) | 363.1 s |
| fitACE | 383 passed | 92.0 s |
| fitACE_pcgc | 211 passed, 4 skipped (rerun) | 24.7 s |
| fitACE_iter_reml | 110 passed, 4 skipped | 97.9 s |
| fitACE_tetraher | 33 passed | 17.5 s |
| fitACE_pafgrs | 119 passed | 40.0 s |
| fitACE_frailty | 7 passed | 13.2 s |
| fitACE_epimight | 256 passed, 19 deselected | 169.7 s |
| pedsum | 325 passed (+ TSV export, CLI smoke) | 551.6 s |

Not run: the slow suites and the `ace_iter_reml` unit, as in 9c and 10a.
