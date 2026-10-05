# 0.12.2 consumer gate (2026-10-05)

Release gate for 0.12.2: `no_estimate_code` on every Ne result record
(#42), the inbreeding walk counting distinct ancestors and memoising them
with `F` (#40), and the generation kinship sums behind
`mean_kinship_by_generation`, `ne_coancestry` and `ne_group_coancestry`
reusing the memoised `F` instead of walking again (#41).

## Build stage

`pixi run --frozen python external/pedigree-graph/tools/consumer_gate.py run --stage 0.12.2 --wheel-ref ce93d5a`
from the umbrella root of the main simACE checkout (`build.json`):

| artifact | sha256 |
|---|---|
| `pedigree_graph-0.12.2-cp313-abi3-linux_x86_64.whl` | `0173e8cfb85c70ce63aba641cfbd7c9429c74a1de9f74c407c7ac48c7c0aa397` |
| `pedigree_graph-0.12.2.tar.gz` | `538e3e4ef82a64b2dde125aa74a47e6db2ac2f6a586c95ebe7865eeff8deb181` |

Both installs pass the import check (16 root names, as in 0.12.1). The
package's own fast suite against the installed wheel: `3469 passed,
3 skipped, 16 deselected in 202.85s`, exit 0.

## Consumer units

All 13 units ok, every routed unit resolving `pedigree_graph` under
`target/consumer-gate/0.12.2/wheel-site`. Every consumer was clean: simACE
`dev` at `487f118`, fitACE `721f9d4`, fitACE_epimight `54e36a4`, pedsum
`6e650b7` (branch `published-pedigrees`). pedigree-graph `3504 passed`,
simACE `2868 passed, 2 skipped`, fitACE `393 passed`, fitACE_epimight
`393 passed, 25 deselected`, pedsum `415 passed, 3 skipped`,
fitACE_iter_reml `110 passed, 4 skipped`, fitACE_pcgc `220 passed,
4 skipped`, fitACE_pafgrs `122 passed`, fitACE_tetraher `34 passed`,
fitACE_frailty `7 passed`; ace_iter_reml's fp32 and fp64 harnesses,
fitACE_stan's import and tetraher_simace's LDAK checks all pass.

No consumer needs a change: 0.12.2 adds a defaulted last field to the Ne
records and changes no public signature, so the consumers' `<0.13` pins take
it at the relock.
