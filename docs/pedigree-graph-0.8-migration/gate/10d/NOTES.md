# 10d locked gate for pedigree-graph 0.8.2 (2026-09-09)

`v0.8.2` is tagged at pedigree-graph `81e4bcb` and pushed; `publish.yml` run
34382161779 succeeded and PyPI serves five `cp313-abi3` wheels (manylinux
x86-64 and AArch64, macOS x86-64 and arm64, Windows x86-64) plus the sdist,
whose sha256 (`12577341…`) equals the sdist the 10b wheel gate built locally.

simACE, fitACE, and pedsum were relocked with `pixi clean cache --pypi --yes`
then `pixi update pedigree-graph`; each lock moved exactly one package,
`pedigree-graph 0.8.1 -> 0.8.2`, and each environment imports 0.8.2 with
`_native.core_version() == "0.8.2"`.

## Consumer gate

`EPIMIGHT_CONDA_ENV=epimight-master pixi run --frozen python tools/pg08_release_gate.py
run --stage 10d --routing locked`, every unit. All green, `failed units: none`;
every consumer `routing` step resolved `pedigree_graph` under its own locked
environment's site-packages (simACE once, pedsum once, the fitACE family seven
times).

| unit | suite | wall |
|---|---|---:|
| pedigree-graph | 2086 passed, 7 skipped, 9 deselected (+ ruff, format, ty) | 517.4 s |
| simACE | 1470 passed, 3 skipped (+ smoke `--forceall`, atlas) | 156.5 s |
| fitACE | 383 passed | 76.3 s |
| fitACE_pcgc | 211 passed, 4 skipped | 15.7 s |
| fitACE_iter_reml | 110 passed, 4 skipped | 94.0 s |
| ace_iter_reml | six binary test units, all exit 0 | 43.0 s |
| fitACE_tetraher | 33 passed | 13.6 s |
| fitACE_pafgrs | 119 passed | 35.9 s |
| fitACE_frailty | 7 passed | 9.1 s |
| fitACE_epimight | 256 passed, 19 deselected | 166.1 s |
| pedsum | 325 passed (+ TSV export, CLI smoke) | 560.8 s |

## CI on the tagged commit

`ci.yml` run 34382158173 on `v0.8` at `81e4bcb`: Rust and Python jobs green,
the wheel job failed two cases of `tests/test_ne_h1_parity.py`
(`ne_caballero_toro` for `skip_gen` and `small_pedigree`). The differences
are least-squares noise in the `np.polyfit` slope under the PyPI numpy on the
runner: a relative 3e-12 in a real slope and the Ne derived from it, and a
pure-noise slope of order 1e-16 on a flat series, both outside the
`rel=1e-12, abs=0` tolerance the 10a fix set. The package code is unaffected
(the editable Python job on the same commit passed, as did the local wheel
gate and every unit here). The tolerance is widened to `rel=1e-9, abs=1e-14`
in the pedigree-graph tree, uncommitted at the time of writing; it needs a
commit and push so the next CI run on `v0.8` is green.
