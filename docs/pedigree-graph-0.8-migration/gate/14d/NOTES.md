# 14d wheel-site gate for pedigree-graph 0.9.3 (2026-09-23)

The pre-publish gate of the 0.9.3 patch, which fixes the two raw-binding
defects the review of `v0.9.1..v0.9.2` found (`fa48ee1`; CHANGELOG
"v0.9.3"): a parentless row above depth 0 lost its diagonal in the native
DP, and the topology sort could panic under the allocation test seam.
Neither is reachable through `PedigreeGraph`. The wheel is built from a
clean worktree at the `v0.9.3` tag and every family check unit runs against
it through the routed site, with the consumer locks still at 0.9.2. The
post-publish relock is stage `14e`.

## The artifact under test

`tools/pg08_wheel_gate.sh <work> v0.9.3`, run from the simACE umbrella root,
built from `1f7ba43cad7b74d9740d8e21880c47aa13bcd7ac`:

| artifact | sha256 |
|---|---|
| `pedigree_graph-0.9.3-cp313-abi3-linux_x86_64.whl` | `f7745136efe409a46551758495e949cf017af7d9737daff0aff80eaaaabddb52` |
| `pedigree_graph-0.9.3.tar.gz` | `a32d2398a55d7faa227481c46ea46f60b9618df0b2f017d44d263963d2a3c00f` |

The package's own suite against the installed wheel: `3099 passed, 4
skipped, 10 deselected in 180.34s`, exit 0.

## Result

`pixi run --frozen python external/pedigree-graph/tools/pg08_release_gate.py run --stage 14d --routing <work>/wheel-site`
from the umbrella root: `failed units: none`, exit 0. Every routed unit's
`routing` step printed `routed <wheel-site>/pedigree_graph/__init__.py`.

| unit | routing | wall | peak RSS | steps |
|---|---|---:|---:|---|
| `pedigree-graph` | own manifest | 159.5 s | 994 MiB | ruff, format, ty, pytest |
| `simACE` | wheel-site | 92.1 s | 500 MiB | routing, ruff, format, test, smoke, atlas |
| `fitACE` | wheel-site | 16.7 s | 373 MiB | routing, ruff, format, pytest |
| `fitACE_pcgc` | wheel-site | 8.1 s | 339 MiB | routing, pytest |
| `fitACE_iter_reml` | wheel-site | 51.0 s | 344 MiB | routing, pytest |
| `ace_iter_reml` | own manifest | 22.5 s | 105 MiB | six fp32/fp64 test binaries |
| `fitACE_tetraher` | wheel-site | 5.2 s | 247 MiB | routing, pytest |
| `tetraher_simace` | own manifest | 0.8 s | 104 MiB | ruff, format, ldak-runs, ldak-is-fork |
| `fitACE_pafgrs` | wheel-site | 12.9 s | 398 MiB | routing, pytest |
| `fitACE_stan` | wheel-site | 1.6 s | 106 MiB | routing, ruff, format, import |
| `fitACE_frailty` | wheel-site | 4.5 s | 212 MiB | routing, pytest |
| `fitACE_epimight` | wheel-site | 30.6 s | 606 MiB | routing, pytest |
| `pedsum` | wheel-site | 39.3 s | 328 MiB | routing, ruff, format, pytest, tsv, cli-smoke |

Every step exited 0. Per-unit records are the JSON files beside this note;
the raw step logs were dropped as in 13b. No byte-parity baseline is cut
for this patch: nothing a consumer reaches changes (both fixes are on
raw-binding-only paths and bytes on structural depth are unchanged), and
the 0.9.2 relocked cut in `../14c/` is the current reference.
