# 15b wheel-site gate for pedigree-graph 0.9.4 (2026-09-23)

The pre-publish gate of 0.9.4, slice 15: inbreeding, the lineage counts and
the Ne prerequisites on the Rust core, and `numba` out of the runtime
dependencies (CHANGELOG "v0.9.4", `gate/15a/NOTES.md`). The wheel is built
from a clean worktree at the local `v0.9.4` tag and every family check unit
runs against it through the routed site, with the consumer locks still at
0.9.3. The post-publish relock is stage `15c`.

## The artifact under test

`tools/pg08_wheel_gate.sh <work> v0.9.4`, run from the simACE umbrella root,
built from `73698935903005d3e00dc3b8ad826d2b98fbbca5`:

| artifact | sha256 |
|---|---|
| `pedigree_graph-0.9.4-cp313-abi3-linux_x86_64.whl` | `75d3868037d413004596a45c201b3867edd8b349e1b2945c1ee4071b95404da9` |
| `pedigree_graph-0.9.4.tar.gz` | `3503750e53339263261528169e6eb88de9e15cee711845b1e02ccff607e0dba5` |

The package's own suite against the installed wheel: `3369 passed, 4
skipped, 10 deselected, 4 warnings in 175.29s`, exit 0.

The wheel's `Requires-Dist` is `numpy>=2.2,<3` and `scipy>=1.14,<2`;
`numba>=0.60,<1` appears only under `extra == 'test'`, and the wheel ships
none of `_inbreeding_kernel`, `_lineage_kernel` or `_kinship_depth` (exit
criterion 4).

## Result

`pixi run --frozen python external/pedigree-graph/tools/pg08_release_gate.py run --stage 15b --routing <work>/wheel-site`
from the umbrella root: `failed units: none`, exit 0. Every routed unit's
`routing` step printed `routed <wheel-site>/pedigree_graph/__init__.py`.

| unit | routing | wall | peak RSS | steps |
|---|---|---:|---:|---|
| `ace_iter_reml` | own manifest | 22.5 s | 104 MiB | fp32-test_laplace_primitives, fp32-test_mcem_step, fp32-test_tmvn, fp64-test_laplace_primitives, fp64-test_mcem_step, fp64-test_tmvn |
| `fitACE` | wheel-site | 17.8 s | 372 MiB | routing, ruff, format, pytest |
| `fitACE_epimight` | wheel-site | 31.1 s | 606 MiB | routing, pytest |
| `fitACE_frailty` | wheel-site | 4.4 s | 211 MiB | routing, pytest |
| `fitACE_iter_reml` | wheel-site | 54.0 s | 344 MiB | routing, pytest |
| `fitACE_pafgrs` | wheel-site | 13.0 s | 399 MiB | routing, pytest |
| `fitACE_pcgc` | wheel-site | 8.0 s | 342 MiB | routing, pytest |
| `fitACE_stan` | wheel-site | 1.3 s | 106 MiB | routing, ruff, format, import |
| `fitACE_tetraher` | wheel-site | 4.7 s | 198 MiB | routing, pytest |
| `pedigree-graph` | own manifest | 155.5 s | 988 MiB | ruff, format, ty, pytest |
| `pedsum` | wheel-site | 31.5 s | 260 MiB | routing, ruff, format, pytest, tsv, cli-smoke |
| `simACE` | wheel-site | 92.4 s | 483 MiB | routing, ruff, format, test, smoke, atlas |
| `tetraher_simace` | own manifest | 0.8 s | 107 MiB | ruff, format, ldak-runs, ldak-is-fork |

Per-unit records: the `.json` beside this file. The step logs were not kept;
each record carries its step's tail.

## What is left

Publishing is the maintainer's: push `main` and the `v0.9.4` tag, which
builds and uploads the wheels. Stage `15c` then relocks simACE (`dev`),
fitACE and pedsum (whose lock drops numba and llvmlite), reruns this gate
unrouted and the consumer byte-parity probe against
`gate/15a/byte-parity/baseline-0.9.3/`.
