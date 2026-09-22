# 10a pre-release gate for pedigree-graph 0.8.1 (2026-09-09)

Slice 10a of ADR 0007 (maturin build, PyO3 `_native`, topology kernels in Rust)
is committed on pedigree-graph `v0.8` as `f81f0dc..19b6b20`, with
`[workspace.package].version = "0.8.1"`. Not yet tagged or pushed at the time of
writing; `publish.yml` and `ci.yml` are unverified until then.

## Wheel gate (`tools/pg08_wheel_gate.sh`, adapted to maturin)

Built from a clean worktree at `19b6b20`: `pedigree_graph-0.8.1-cp313-abi3-linux_x86_64.whl`
(sha256 `63ceff7b…`) and `pedigree_graph-0.8.1.tar.gz` (sha256 `d80a67d9…`), see
`wheel-gate.json`. Both install into fresh venvs, import from site-packages with
`_native.abi3.so`, `_native.pyi`, and `py.typed` beside the sources, and
`_native.core_version()` equals the distribution version. The package suite run
against the installed wheel from a cwd outside the tree (`wheel-pytest.log`):
`2074 passed, 3 skipped, 9 deselected in 602.08s`.

Two script changes were needed and are the reason the first two runs failed:
the version now comes from `Cargo.toml` (no `SETUPTOOLS_SCM_PRETEND_VERSION`),
and pytest runs from the work dir rather than the clean tree, because the fresh
child processes the thread tests spawn (`sys.executable -c`, no `-P`) imported
the source package from cwd, which has no extension module. One test fix landed
in pedigree-graph (`19b6b20`): `tests/conftest.py` puts the repository root on
`sys.path` so `tests.oracle` imports from a wheel install too.

Also run in the pedigree-graph checkout: `pytest -m slow` (`9 passed in 1746.70s`,
slowed by the concurrent wheel build), and before commit the not-slow suite,
`cargo test --release`, clippy, fmt, ruff, and ty (recorded in
`plans/pedigree-graph-slice-10-native-scaffold.md`).

## Consumer gate

`EPIMIGHT_CONDA_ENV=epimight-master pixi run --frozen python tools/pg08_release_gate.py
run --stage 10a --routing <wheel-site> --unit <the nine consumer units>`, locks
still pinning 0.8.0. All nine green; every `routing` step resolved
`pedigree_graph` under the wheel site (nine identical `routed` lines).

| unit | suite | wall |
|---|---|---:|
| simACE | 1470 passed, 3 skipped (+ smoke `--forceall`, atlas) | 362.3 s |
| fitACE | 383 passed | 58.2 s |
| fitACE_pcgc | 211 passed, 4 skipped | 22.9 s |
| fitACE_iter_reml | 110 passed, 4 skipped | 89.0 s |
| fitACE_tetraher | 33 passed | 17.2 s |
| fitACE_pafgrs | 119 passed | 39.0 s |
| fitACE_frailty | 7 passed | 13.1 s |
| fitACE_epimight | 256 passed, 19 deselected | 167.6 s |
| pedsum | 325 passed (+ TSV export, CLI smoke) | 543.4 s |

Not run: the consumer slow suites (as in 9c), and the `ace_iter_reml` unit,
which never imports pedigree-graph.

## Next

The maintainer tags `v0.8.1` at `19b6b20` and pushes; `publish.yml` must build
five abi3 wheels plus the sdist and publish them. After PyPI serves 0.8.1,
relock simACE, fitACE, and pedsum (expect one change each) and rerun this gate
with `--routing locked` as stage 10c. Then slice 10b (native construction).
