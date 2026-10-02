# 0.12.1 consumer gate (2026-10-02)

Release gate for 0.12.1: `progress=` and Ctrl-C on the long relationship
calls in Python (#37) and R (#39), both under ADR 0017, and the generation
kinship summary behind `mean_kinship_by_generation`, `ne_coancestry` and
`ne_group_coancestry` moved off the kinship DP onto per-cohort backward
sweeps (#38).

## Build stage

`pixi run --frozen python external/pedigree-graph/tools/consumer_gate.py run --stage 0.12.1 --wheel-ref 4fc7a4c`
from the umbrella root of the main simACE checkout (`build.json`):

| artifact | sha256 |
|---|---|
| `pedigree_graph-0.12.1-cp313-abi3-linux_x86_64.whl` | `88c2fb52512d2c7f98755f956b112072f043ff7be77a961d981a40e65c8ae856` |
| `pedigree_graph-0.12.1.tar.gz` | `d24154c47a7793b35167709faff7ed9f257076568ff568a0a1fc0e96fd435632` |

Both installs pass the import check (16 root names, one more than 0.12.0's
15 for `RelationshipProgress`). The package's own fast suite against the
installed wheel: `3440 passed, 3 skipped, 16 deselected in 184.70s`, exit 0.

## Consumer units

All 13 units ok, every routed unit resolving `pedigree_graph` under
`target/consumer-gate/0.12.1/wheel-site`. Every consumer was clean: simACE
`dev` at `0d75496`, fitACE `93f3c69`, fitACE_epimight `5807c76`, pedsum
`1913602`. pedigree-graph `3475 passed`, simACE `2865 passed, 2 skipped`,
fitACE `393 passed`, fitACE_epimight `388 passed, 25 deselected`, pedsum
`373 passed, 3 skipped`, fitACE_iter_reml `110 passed, 4 skipped`,
fitACE_pcgc `220 passed, 4 skipped`, fitACE_pafgrs `122 passed`,
fitACE_tetraher `34 passed`, fitACE_frailty `7 passed`; ace_iter_reml's
fp32 and fp64 harnesses, fitACE_stan's import and tetraher_simace's LDAK
checks all pass.

No consumer needs a change: 0.12.1 adds API and changes no public
signature, so the `>=0.12,<0.13` pins take it at the relock.
