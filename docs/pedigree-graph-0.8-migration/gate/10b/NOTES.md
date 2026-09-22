# 10b pre-release gate for pedigree-graph 0.8.2 (2026-09-09)

Slice 10b of ADR 0007 (native construction: `crates/core/src/graph.rs`,
`_native.build_pedigree`, the Python parser deleted) is committed on
pedigree-graph `v0.8` as `e0a0e06..81e4bcb`, with
`[workspace.package].version = "0.8.2"`. Not yet tagged or pushed at the time of
writing. Plan and execution record: `plans/pedigree-graph-slice-10b-native-construction.md`.

## Wheel gate (`tools/pg08_wheel_gate.sh`)

Built from a clean worktree at `81e4bcb`: `pedigree_graph-0.8.2-cp313-abi3-linux_x86_64.whl`
(sha256 `609eac9b…`) and `pedigree_graph-0.8.2.tar.gz` (sha256 `12577341…`), see
`wheel-gate.json`. Both install into fresh venvs, import from site-packages with
`_native.abi3.so`, `_native.pyi`, and `py.typed` beside the sources, and
`_native.core_version()` equals the distribution version. The package suite run
against the installed wheel from a cwd outside the tree (`wheel-pytest.log`):
`2080 passed, 3 skipped, 9 deselected in 643.33s`, exit 0.

Also run in the pedigree-graph checkout before commit: the not-slow suite
(`2086 passed, 7 skipped, 9 deselected`), `pytest -m slow` (`9 passed in 1772.84s`),
`cargo test --release` (core `42 passed`, parity `2 passed`), clippy, fmt, ruff,
and ty, all clean.

## Construction perf gate

`bench_build.py` (this directory) builds a synthetic 12-generation pedigree in a
random row and id order with sex and birth years through `PedigreeGraph.from_frame`,
one measurement per fresh process. Five interleaved runs per version, the PyPI
0.8.1 wheel against the 0.8.2 wheel built from `81e4bcb`, in two venvs on the
pixi Python with numpy 2.5.3 (`construction-bench.tsv`; the recorded runs used
the pre-lint version of the script, which imported the package after generating
the frame, so `rss_before_mib` there excludes the import):

| rows | 0.8.1 median wall | 10b median wall | ratio | 0.8.1 peak RSS | 10b peak RSS | ratio |
|---|---|---|---|---|---|---|
| 300,000 | 0.779 s | 0.413 s | 0.53 | 253.9 MiB | 247.2 MiB | 0.97 |
| 20,000,000 | 88.96 s | 81.47 s | 0.92 | 3300.5 MiB | 2845.9 MiB | 0.86 |

`mother_rows` checksums agree between versions at both sizes. No regression.

## Consumer gate

`EPIMIGHT_CONDA_ENV=epimight-master pixi run --frozen python tools/pg08_release_gate.py
run --stage 10b --routing <wheel-site>`, every unit, locks still pinning 0.8.1.
All green; every consumer `routing` step resolved `pedigree_graph` under the
wheel site (nine identical `routed` lines).

| unit | suite | wall |
|---|---|---:|
| pedigree-graph | 2086 passed, 7 skipped, 9 deselected (+ ruff, format, ty) | 503.2 s |
| simACE | 1470 passed, 3 skipped (+ smoke `--forceall`, atlas) | 103.9 s |
| fitACE | 383 passed | 55.6 s |
| fitACE_pcgc | 211 passed, 4 skipped | 20.7 s |
| fitACE_iter_reml | 110 passed, 4 skipped | 88.6 s |
| ace_iter_reml | six binary test units, all exit 0 | 40.8 s |
| fitACE_tetraher | 33 passed | 14.7 s |
| fitACE_pafgrs | 119 passed | 36.7 s |
| fitACE_frailty | 7 passed | 10.5 s |
| fitACE_epimight | 256 passed, 19 deselected | 165.4 s |
| pedsum | 325 passed (+ TSV export, CLI smoke) | 538.1 s |

A first attempt of this stage was aborted after the simACE `ruff` and `format`
steps failed on the benchmark script newly placed in this directory; the script
was made lint-clean and the stage rerun from the start. Not run: the consumer
slow suites (as in 9c, 10a, 10c).

## Next

The maintainer tags `v0.8.2` at `81e4bcb` and pushes; `publish.yml` builds five
abi3 wheels plus the sdist and publishes them, and `ci.yml` runs. After PyPI
serves 0.8.2, relock simACE, fitACE, and pedsum (`pixi clean cache --pypi --yes`
then `pixi update pedigree-graph`, expect one change each) and rerun this gate
with `--routing locked` as stage 10d.
