# 9b built-artifact gate (2026-09-09)

Built by `tools/pg08_wheel_gate.sh` from a clean `git worktree` of `v0.8` at
`c8c963a` with `SETUPTOOLS_SCM_PRETEND_VERSION=0.8.0` (an untagged branch
would build `0.8.0.devN`, which fails the consumers' `>=0.8` pin).
`wheel-gate.json` carries the wheel and sdist sha256.

- Wheel and sdist each install into a fresh venv: import resolves under
  `site-packages`, `py.typed` present, every name in the frozen root
  `__all__` and the four public modules import.
- The package suite against the installed wheel: 2065 passed, 3 skipped
  (`-m "not slow"`). The source run collects 7 more only because
  `test_benchmark_contract` parametrizes over gitignored local
  `benchmarks/reports/*.json` files absent from the clean worktree.
- Consumer units against the wheel installed with `pip --target` and routed
  by `PYTHONPATH` (records in this directory, `routing.log` per unit):
  simACE (test, smoke, atlas), fitACE core, pcgc, iter_reml, tetraher,
  pafgrs, frailty, epimight (default suite), pedsum (tests + CLI smoke).
  All green; slow suites were run in 9a only.
- Working-tree sweep after the gate: simACE carries the gate evidence and
  tool edits, pedigree-graph carries the changelog heading and the 300k
  benchmark suite, fitACE / fitACE_epimight / pedsum are clean. No manifest
  or lock has a `path =` or find-links routing to pedigree-graph.
- The build's isolated env printed "listing git files failed" from
  setuptools-scm; the sdist nevertheless contains all 42 package modules.
