# 0.11.1 consumer gate (2026-09-30)

Release gate for 0.11.1: `relatives_per_person` (#33, ADR 0014). Plan:
simACE `plans/pedigree-graph-issue-33-relatives-per-person.md` (session-local draft).

## Build stage

`pixi run --frozen python external/pedigree-graph/tools/consumer_gate.py run --stage 0.11.1 --wheel-ref 8dff17d`
from the umbrella root of a clean `pg-0.11.1` worktree (`build.json`):

| artifact | sha256 |
|---|---|
| `pedigree_graph-0.11.1-cp313-abi3-linux_x86_64.whl` | `a3fe7a629758f939cc6e4928d7da1fdfc2d5709f0471e87934756eb26dcca20a` |
| `pedigree_graph-0.11.1.tar.gz` | `2d9e39201422561bf3e1ce7d69698f0c35a2b63b3bb7821845c6158cb541d77c` |

Both installs pass the import check (15 root names; `RelativesPerPerson`
is the new one). The package's own fast suite against the installed wheel:
`3322 passed, 2 skipped, 14 deselected in 306.14s`, exit 0.

## Consumer units

All 13 units ok, every routed unit resolving `pedigree_graph` under
`target/consumer-gate/0.11.1/wheel-site`. The worktree held simACE `dev`
at `3649411` and fitACE at `26b5013`, both clean. fitACE_epimight
(`8ebc573`) and pedsum (`39a56a3`) carried the consumer side of #33,
uncommitted: `create_input` and `epimight-input` counting relatives with
`relatives_per_person` instead of a pair list. fitACE_epimight
`388 passed, 25 deselected`, pedsum `322 passed, 3 skipped`, simACE
`2815 passed, 2 skipped`, fitACE `387 passed`.

`8dff17d` changes only `tools/consumer_gate.py` over the version bump
`003a7e2`: simACE no longer has a Snakefile (simACE ADR 0020), so the
simACE unit's `smoke` step is now `simace run --force small_test` and its
`atlas` step is `gather` (`simace gather test`). The first attempt at
`003a7e2` stopped before the build because the fresh worktree had no pixi
environments; `pixi install --frozen` in pedigree-graph, pedsum and fitACE
fixed that. As for 0.11.0, the gitignored `build-fp32/`, `build-fp64/`
and `ldak6.2.simace` were copied from the main fitACE checkout, whose
HEAD is the worktree's.
