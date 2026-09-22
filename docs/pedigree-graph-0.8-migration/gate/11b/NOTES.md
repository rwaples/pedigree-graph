# 11b locked gate for pedigree-graph 0.8.3 (2026-09-09)

`v0.8.3` is tagged (annotated) at pedigree-graph `ce698ec` and pushed;
`publish.yml` run 34407893755 succeeded and PyPI serves five `cp313-abi3` wheels
(manylinux x86-64 and AArch64, macOS x86-64 and arm64, Windows x86-64) plus the
sdist, whose sha256 (`ed47e54c…`) equals the sdist the 11a wheel gate built
locally from the same commit (`11a/wheel-gate.json`). `ci.yml` run 34407891413
on `v0.8` at `ce698ec` is green (Rust, Python, and wheel jobs), the Ne parity
tolerance from the 10d record having been committed in between.

simACE, fitACE, and pedsum were relocked with `pixi clean cache --pypi --yes`
then `pixi update pedigree-graph`. The first attempt reported every lock
"already up-to-date": `pixi clean cache --pypi` drops the cached wheels but not
uv's cached simple index (`uv-cache/simple-v21/pypi/pedigree-graph.rkyv`), which
still listed 0.8.2 as the newest release. Deleting that one file and rerunning
moved exactly one package in each lock, `pedigree-graph 0.8.2 -> 0.8.3` (the
fitACE diff also reorders the unchanged `execnet 2.1.2` entry). After
`pixi install --locked`, each environment imports 0.8.3 from its own
site-packages with `_native.core_version() == "0.8.3"`.

## Consumer gate

`EPIMIGHT_CONDA_ENV=epimight-master pixi run python tools/pg08_release_gate.py
run --stage 11b --routing locked --unit <unit>`, one unit per invocation as in
11a. All eleven green, `failed units: none`; every consumer `routing` step
resolved `pedigree_graph` under its own locked environment's site-packages
(simACE once, pedsum once, the fitACE family seven times).

| unit | suite | wall |
|---|---|---:|
| pedigree-graph | 2285 passed, 7 skipped, 9 deselected (+ ruff, format, ty) | 191.7 s |
| simACE | 1470 passed, 3 skipped (+ smoke `--forceall`, atlas) | 117.6 s |
| fitACE | 383 passed | 27.6 s |
| fitACE_pcgc | 211 passed, 4 skipped | 8.7 s |
| fitACE_iter_reml | 110 passed, 4 skipped | 48.1 s |
| ace_iter_reml | six binary test units, all exit 0 | 18.5 s |
| fitACE_tetraher | 33 passed | 6.3 s |
| fitACE_pafgrs | 119 passed | 13.2 s |
| fitACE_frailty | 7 passed | 4.9 s |
| fitACE_epimight | 256 passed, 19 deselected | 52.8 s |
| pedsum | 325 passed (+ TSV export, CLI smoke) | 182.8 s |

Walls match the 11a wheel-site run within noise (pedigree-graph 175.2 s there,
pedsum 189.5 s), as expected for the same wheel on the same host.
