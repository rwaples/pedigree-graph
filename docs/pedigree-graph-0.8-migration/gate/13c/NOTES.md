# 13c post-publish gate for pedigree-graph 0.9.1 (2026-09-23)

The second half of slice 13's commit 6. The pre-publish half is `../13b/`.
0.9.1 is on PyPI, the three locks that pin it from PyPI are relocked, the
byte-parity probe has been re-run, and the thirteen-unit gate is green at
`--routing locked`.

## Published

`v0.9.1` (`6bd04d3`) was pushed on 2026-09-23 and the tag-triggered `Publish`
workflow (run 35832171426) went nine jobs green, uploading five `cp313-abi3`
wheels and the sdist through trusted publishing. The published sdist hashes
to `2933cfa721c89e4b7c63e75d3d5ce1053434f10b2452dc02715c86be80ffcbdb`, equal
to the sdist the 13b wheel gate built and tested locally, so PyPI serves the
artifact 13b tested rather than a rebuild sharing a commit. The consumers
lock the `manylinux_2_17_x86_64` wheel,
`12ffea35592bb97e06cc2f8b6c1510d9a72780e41985972bcea3a24c7d48b4a5`.

## Caps and locks

The five caps (`pyproject.toml`, `fitACE/pyproject.toml`,
`fitACE/fitACE_epimight/pyproject.toml`, `external/pedsum/pyproject.toml`,
`external/pedsum/environment.yml`) already read `pedigree-graph>=0.9,<0.10`
and admit a patch release, so no cap moved. Relocking is the same three
trees as 12c: simACE, fitACE and pedsum. `pixi update pedigree-graph` in each
reported exactly one change, `pedigree-graph 0.9.0 -> 0.9.1`, and the lock
diffs (19 lines each) touch only that package's name, version, wheel URL and
sha256. 12c's stale-index trap did not recur: no `pedigree-graph.rkyv` was
present under `~/.cache/uv/simple-v24` or `simple-v25`. After
`pixi install --locked`, each environment imports 0.9.1 from its own
site-packages with `_native.core_version() == "0.9.1"`.

## Byte parity

`byte-parity/baseline-0.9.0/` was cut under the 0.9.0 locks immediately
before the relock and `byte-parity/relocked-0.9.1/` immediately after, with
no simACE or fitACE commit between them; both manifests name their
environment's pedigree-graph version.

| artifact | baseline 0.9.0 | relocked 0.9.1 | |
|---|---|---|---|
| `pairwise_relatedness.tsv` | `505a8fd3…` | `505a8fd3…` | identical |
| `report.yaml` | `9c580c90…` | `9c580c90…` | identical |

Both artifacts are byte-identical, and the pair table's hash is the one 12c
recorded for 0.9.0, so the pair contract is unchanged through two releases.
`report.yaml` matching too, where it differed in 12c, is because the two runs
sit on the same simACE commit and the same day, so the recorded
`simace_version` is the same string; the report carries the relationship
correlations and counts that consume `pair_kinship` downstream.

## Result

`failed units: none`, thirteen units at `--routing locked`, exit 0. Every
`routing` step passed, so each consumer resolved `pedigree_graph` from its
own locked environment's site-packages.

| unit | routing | wall | peak RSS | suite |
|---|---|---:|---:|---|
| `pedigree-graph` | own manifest | 159.8 s | 981 MiB | 3023 passed, 9 skipped, 10 deselected |
| `simACE` | locked | 111.4 s | 560 MiB | 1358 passed, 2 skipped (+ smoke, atlas) |
| `fitACE` | locked | 28.3 s | 489 MiB | 383 passed |
| `fitACE_pcgc` | locked | 8.5 s | 349 MiB | 211 passed, 4 skipped |
| `fitACE_iter_reml` | locked | 51.7 s | 348 MiB | 110 passed, 4 skipped |
| `ace_iter_reml` | own manifest | 22.3 s | 110 MiB | six binary test units, all exit 0 |
| `fitACE_tetraher` | locked | 5.3 s | 247 MiB | 34 passed |
| `tetraher_simace` | own manifest | 0.8 s | 109 MiB | ruff, format, ldak runs, ldak is fork |
| `fitACE_pafgrs` | locked | 13.0 s | 405 MiB | 121 passed |
| `fitACE_stan` | locked | 1.6 s | 107 MiB | ruff, format, import beside fitace and simace |
| `fitACE_frailty` | locked | 4.6 s | 211 MiB | 7 passed |
| `fitACE_epimight` | locked | 31.1 s | 607 MiB | 254 passed, 19 deselected |
| `pedsum` | locked | 50.8 s | 385 MiB | 324 passed |

Per-unit records are the JSON files beside this note; step logs were
dropped as in 12b and 12c. The host ran at 2600 MHz under the `performance`
governor throughout.

## Slice 13 closes here

Exit criteria of the plan: layout selected from the 13a record (per-row,
`../13a/NOTES.md`), no scored cell over the 5 percent gate, pk-536k peak RSS
down 44 percent and rkm-536k level with the reason recorded; bit-identical
to 0.9.0 on every parity fixture and on the four study pedigrees; matrices
still bit-match `pair_kinship`; no Python/Numba pairwise production code;
every input-sized native allocation maps to `allocation_failed`; pytest,
`cargo test --release`, clippy, fmt, ruff and ty green; the thirteen-unit
gate green at wheel-site (13b) and after relock (this stage); fitACE's TSV
and simACE's report byte-identical; 0.9.1 tagged and published.
