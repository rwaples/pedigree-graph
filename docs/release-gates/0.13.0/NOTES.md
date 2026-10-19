# 0.13.0 consumer gate (2026-10-19)

Release gate for 0.13.0: a faster pairwise kinship walk with pooled memo
tables; sib reuse in the relationship engine; the Rust core's
`pairwise::Walker::new` taking a shared `KinshipRows`; and
`reduce_pairs_in` with `RowOrder::Families` for host reducers.

## Build stage

`pixi run --frozen python external/pedigree-graph/tools/consumer_gate.py run --stage 0.13.0 --wheel-ref 0a79c4e`
from the umbrella root of the main simACE checkout (`build.json`):

| artifact | sha256 |
|---|---|
| `pedigree_graph-0.13.0-cp313-abi3-linux_x86_64.whl` | `3375fc930b3670e96422381887767bcd730a250f1a34b66b92eb0eae0a27ffef` |
| `pedigree_graph-0.13.0.tar.gz` | `16802d8f45360d462398f32eebc5402a3850f1418f53c707733cc940a55683fb` |

Both installs pass the import check. The package's own fast suite against
the installed wheel: `3471 passed, 3 skipped, 16 deselected`, exit 0.

## Consumer units

14 of 14 units ok, every routed unit resolving `pedigree_graph` under
`target/consumer-gate/0.13.0/wheel-site`. Consumers: simACE `dev` at
`479a25b`, fitACE `c6b595c`, fitACE_epimight `8043ec8`, pedsum `a4fca4f`,
pg-phenotype `daf8e10`, all clean. simACE `3401 passed, 2 skipped` and its
workflow smoke run exit 0, fitACE `405 passed`, fitACE_epimight `396
passed, 25 deselected`, pedsum `769 passed` and its CLI smoke run exit 0,
fitACE_iter_reml `110 passed, 4 skipped`, fitACE_pcgc `220 passed, 4
skipped`, fitACE_pafgrs `47 passed`, fitACE_tetraher `34 passed`,
fitACE_frailty `7 passed`; ace_iter_reml's fp32 and fp64 harnesses,
fitACE_stan's import and tetraher_simace's LDAK checks all pass.

pg-phenotype `test-against-pg` passes against the clean 0.13.0 checkout:
its Rust suite (`193 passed` in the core) and Python suite (`717 passed`).
pg-phenotype `daf8e10` already builds one `KinshipRows` per PA-FGRS prep and
walks its relative-pair SE pass with `reduce_pairs_in` in family order
(pg-phenotype #24), pinned to `e2e5d91` by rev; it re-pins to `v0.13.0`.

## Pins

The gate routes the wheel in by path, so it does not test the consumers'
version pins. Every Python consumer pins `pedigree-graph<0.13` (simACE,
fitACE, fitACE_epimight, pedsum, pg-phenotype) and must widen the pin to
`<0.14` and relock to take 0.13.0. No consumer's Python code changes: 0.13.0
changes no public Python or R signature. The tag `v0.13.0` goes on the bump
commit `0a79c4e`, the gated wheel's ref.
