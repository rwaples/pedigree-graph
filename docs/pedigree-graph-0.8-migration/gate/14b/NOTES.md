# 14b wheel-site gate for pedigree-graph 0.9.2 (2026-09-23)

Slice 14 (`simACE/plans/pedigree-graph-slice-14-kinship-matrix-dp.md`)
commit 6: the pre-publish half. The 0.9.2 wheel is built from a clean
worktree at the `v0.9.2` tag and every family check unit runs against it
through the routed site, with the consumer locks still frozen at 0.9.1. The
post-publish half, the three relocks, the byte compare and the thirteen-unit
gate at `--routing locked`, is stage `14c` and waits on the tag being pushed
and the `Publish` workflow uploading to PyPI.

## The artifact under test

`tools/pg08_wheel_gate.sh <work> v0.9.2`, run from the simACE umbrella root,
built from `2e0ab0914d4bc4eeeecdf68d6eb0812a6b1a7007`:

| artifact | sha256 |
|---|---|
| `pedigree_graph-0.9.2-cp313-abi3-linux_x86_64.whl` | `6ee10ef643b4289424b53c8b6e0d8e4ee91c3a25fde105ae4951d824b6adb21c` |
| `pedigree_graph-0.9.2.tar.gz` | `0c5e91e5136fc81eea8f37a511f0300c972c795fcb1efa3404417e6225bcea48` |

Both install into isolated venvs and resolve `pedigree_graph` from
site-packages with `core_version()` equal to the distribution version. The
package's own suite against the installed wheel: `3098 passed, 4 skipped,
10 deselected in 180.13s`, exit 0 (`wheel-gate.json` in the work
directory; the numbers above are its record).

## Result

`pixi run --frozen python external/pedigree-graph/tools/pg08_release_gate.py run --stage 14b --routing <work>/wheel-site`
from the umbrella root: `failed units: none`, exit 0. Every routed unit's
`routing` step printed `routed <wheel-site>/pedigree_graph/__init__.py`.

| unit | routing | wall | peak RSS | steps |
|---|---|---:|---:|---|
| `pedigree-graph` | own manifest | 158.5 s | 972 MiB | ruff, format, ty, pytest |
| `simACE` | wheel-site | 91.9 s | 500 MiB | routing, ruff, format, test, smoke, atlas |
| `fitACE` | wheel-site | 29.0 s | 373 MiB | routing, ruff, format, pytest |
| `fitACE_pcgc` | wheel-site | 8.2 s | 339 MiB | routing, pytest |
| `fitACE_iter_reml` | wheel-site | 51.0 s | 356 MiB | routing, pytest |
| `ace_iter_reml` | own manifest | 22.4 s | 104 MiB | six fp32/fp64 test binaries |
| `fitACE_tetraher` | wheel-site | 5.3 s | 246 MiB | routing, pytest |
| `tetraher_simace` | own manifest | 0.8 s | 107 MiB | ruff, format, ldak-runs, ldak-is-fork |
| `fitACE_pafgrs` | wheel-site | 13.2 s | 399 MiB | routing, pytest |
| `fitACE_stan` | wheel-site | 1.6 s | 107 MiB | routing, ruff, format, import |
| `fitACE_frailty` | wheel-site | 4.6 s | 211 MiB | routing, pytest |
| `fitACE_epimight` | wheel-site | 30.9 s | 607 MiB | routing, pytest |
| `pedsum` | wheel-site | 39.3 s | 328 MiB | routing, ruff, format, pytest, tsv, cli-smoke |

Every step exited 0. Per-unit records are the JSON files beside this note;
the raw step logs were dropped as in 13b. The gate is a pass/fail record, not
a benchmark; slice 14's timings are in `../14a/`.

## Byte-parity baseline

`byte-parity/baseline-0.9.1/` was cut with `tools/pg09_byte_parity.sh`
under the 0.9.1 consumer locks before the relock, extended this slice with
fitACE's sparse GRM (the `grm_matrix` rule at `grm_min_kinship` 0.001, the
approximate matrix) and the generation kinship summary of the smoke
pedigree as text. The `report.yaml` and `pairwise_relatedness.tsv` hashes
equal the ones 13c recorded for 0.9.1. The 42 MiB `A.grm.sp.bin` and the
gzipped pair table are hashed in `manifest.txt` and not kept; the id file,
the report and the summary are. Stage 14c cuts the relocked half and
compares.

## What this proves and what it does not

The consumers run unchanged against 0.9.2: the three matrix methods kept
their signatures, dtypes and caches, and no consumer reached the deleted
private DP modules. The byte-level comparison of the four consumer
products across the relock is stage 14c.
