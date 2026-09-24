# 16d wheel-site gate for pedigree-graph 0.10.0 (2026-09-24)

The pre-publish gate of 0.10.0: the R package (slice 16; `../16b/` is its
parity record, `../16c/` its source tarball and offline check) and the merged
`tskit-views-sinks` branch, which adds the public `RelationshipBurden` /
`PedigreeGraph.relationship_burden()` and runs sparse views on a compact
pedigree (CHANGELOG "v0.10.0"). The wheel is built from a clean worktree at
the local `v0.10.0` tag and every family check unit runs against it through
the routed site, with the consumer locks still at 0.9.4. The post-publish
relock is the next stage.

## The artifact under test

`tools/pg08_wheel_gate.sh <work> v0.10.0`, run from the simACE umbrella root,
built from `bc65ad0f0e3d670530810e3a48ec71ec6560149f`:

| artifact | sha256 |
|---|---|
| `pedigree_graph-0.10.0-cp313-abi3-linux_x86_64.whl` | `54de79d691104d3f61f16987db68e94c6d74a2fb6ea0c72357872a2b45b4c0ae` |
| `pedigree_graph-0.10.0.tar.gz` | `7cba56145a4727305e7d685d2a92f34cb8789e161052d636f53eeb88c1df5ed0` |

The package's own suite against the installed wheel: `3390 passed, 4
skipped, 10 deselected, 4 warnings in 174.33s`, exit 0.

## Result

`pixi run --frozen python external/pedigree-graph/tools/pg08_release_gate.py run --stage 16d --routing <work>/wheel-site`
from the umbrella root: `failed units: none`, exit 0. Every routed unit's
`routing` step printed `routed <wheel-site>/pedigree_graph/__init__.py`.

| unit | routing | wall | peak RSS | steps |
|---|---|---:|---:|---|
| `ace_iter_reml` | own manifest | 22.5 s | 106 MiB | fp32-test_laplace_primitives, fp32-test_mcem_step, fp32-test_tmvn, fp64-test_laplace_primitives, fp64-test_mcem_step, fp64-test_tmvn |
| `fitACE` | wheel-site | 16.2 s | 371 MiB | routing, ruff, format, pytest |
| `fitACE_epimight` | wheel-site | 30.4 s | 605 MiB | routing, pytest |
| `fitACE_frailty` | wheel-site | 4.2 s | 212 MiB | routing, pytest |
| `fitACE_iter_reml` | wheel-site | 51.9 s | 356 MiB | routing, pytest |
| `fitACE_pafgrs` | wheel-site | 12.8 s | 398 MiB | routing, pytest |
| `fitACE_pcgc` | wheel-site | 7.6 s | 341 MiB | routing, pytest |
| `fitACE_stan` | wheel-site | 1.3 s | 108 MiB | routing, ruff, format, import |
| `fitACE_tetraher` | wheel-site | 4.4 s | 197 MiB | routing, pytest |
| `pedigree-graph` | own manifest | 153.3 s | 984 MiB | ruff, format, ty, pytest |
| `pedsum` | wheel-site | 29.7 s | 160 MiB | routing, ruff, format, pytest, tsv, cli-smoke |
| `simACE` | wheel-site | 92.7 s | 481 MiB | routing, ruff, format, test, smoke, atlas |
| `tetraher_simace` | own manifest | 0.8 s | 106 MiB | ruff, format, ldak-runs, ldak-is-fork |

Per-unit records: the `.json` beside this file. The step logs were not kept;
each record carries its step's tail.

## What is left

Publishing is the maintainer's: push `main` and the `v0.10.0` tag, which
builds and uploads the wheels and, new in 0.10.0, builds and checks the R
source tarball offline and attaches it to a GitHub release. Those two jobs
(and CI's `r` job) run `sudo unshare --net` and have not yet run on GitHub.
The post-publish stage then raises the consumers' `pedigree-graph<0.10` caps
(simACE, fitACE, pedsum), relocks, reruns this gate unrouted and the
consumer byte-parity probe (`pg09_byte_parity.sh`) against the 0.9.4
baseline.
