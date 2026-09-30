# 0.11.0 consumer gate (2026-09-30)

Release gate for 0.11.0: `relationship_moments` (#28, ADR 0013) and MZ
co-twins in sibling groups (#29). Plan: simACE
`plans/relationship-moments-v10.md`, step 4.

## Build stage

`pixi run --frozen python external/pedigree-graph/tools/consumer_gate.py run --stage 0.11.0 --wheel-ref 6ace53d`
from the umbrella root of the `relationship-moments` worktree (`build.json`):

| artifact | sha256 |
|---|---|
| `pedigree_graph-0.11.0-cp313-abi3-linux_x86_64.whl` | `a13db5f37eeafed753c54e9d6b19c5fc4d02d3be9a5f280ade3b98ee07de8158` |
| `pedigree_graph-0.11.0.tar.gz` | `5dee5c1ad35e73c243c11b3857a344bad21743b0b6bc8bc3edd2b38f3ba628cc` |

Both installs pass the import check (14 root names; `RelationshipMoments`
is the new one). The package's own fast suite against the installed wheel:
`3217 passed, 2 skipped, 12 deselected in 230.40s`, exit 0.

## Consumer units

All 13 units ok, every routed unit resolving `pedigree_graph` under
`target/consumer-gate/0.11.0/wheel-site`. simACE and fitACE ran with the
consumer side of the change (simACE#25, the moments-based stats; commit M
of the plan) in their worktrees, uncommitted: simACE `1385 passed, 2
skipped`, fitACE `383 passed`.

The first run failed `ace_iter_reml`, `fitACE_tetraher` and
`tetraher_simace` in under 4 s each: the fresh fitACE worktree had no
`build-fp32/`, `build-fp64/` or `ldak6.2.simace`, all gitignored build
outputs, and none of these units imports `pedigree_graph`. The outputs were
copied from the main fitACE checkout, whose sources match the worktree's
(`diff -rq`, ruff caches aside), and the three units re-ran with
`--routing target/consumer-gate/0.11.0/wheel-site --unit ace_iter_reml
fitACE_tetraher tetraher_simace`: `failed units: none`, exit 0. Their
records here are from that re-run.

The `0.11.0-rc` gate (not recorded here) built `ea2154f`, before the
review follow-ups `3068e12`, `ae31568` and `1b5ac84`, and also passed all
13 units.
