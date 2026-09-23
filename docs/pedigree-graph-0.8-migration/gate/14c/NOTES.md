# 14c post-publish gate for pedigree-graph 0.9.2 (2026-09-23)

The second half of slice 14's commit 6. The pre-publish half is `../14b/`.
0.9.2 is on PyPI, the three locks that pin it from PyPI are relocked, the
byte-parity probe has been re-run, and the thirteen-unit gate is green at
`--routing locked`.

## Published

`v0.9.2` (`2e0ab09`) was pushed on 2026-09-23 and the tag-triggered `Publish`
workflow (run 35853429831) went nine jobs green, uploading five `cp313-abi3`
wheels and the sdist through trusted publishing. The published sdist hashes
to `0c5e91e5136fc81eea8f37a511f0300c972c795fcb1efa3404417e6225bcea48`, equal
to the sdist the 14b wheel gate built and tested locally, so PyPI serves the
artifact 14b tested rather than a rebuild sharing a commit. The consumers
lock the `manylinux_2_17_x86_64` wheel,
`2010ff81e2d4f7ead61c5e9d58c34a9cc5b9619e4a536b3e2e5abcf10b3893f0`.

## Caps and locks

The caps already read `pedigree-graph>=0.9,<0.10` and admit a patch release,
so no cap moved. Relocking is the same three trees as 13c: simACE, fitACE
and pedsum. `pixi update pedigree-graph` in each reported exactly one
change, `pedigree-graph 0.9.1 -> 0.9.2`; the pedsum diff touches only that
package's URL, version and sha256, and the simACE and fitACE diffs the same
plus a reorder of one unchanged `sortedcontainers` entry that the new
wheel's URL now sorts before. 12c's stale-index trap did recur, in a
different place: `~/.cache/rattler/cache/uv-cache/simple-v21/pypi/pedigree-graph.rkyv`
made every resolver report the lock already up to date; removing that one
file was enough. After `pixi install --locked`, each environment imports
0.9.2 from its own site-packages with `_native.core_version() == "0.9.2"`.

## Byte parity

`../14b/byte-parity/baseline-0.9.1/` was cut under the 0.9.1 locks before
the relock and `byte-parity/relocked-0.9.2/` immediately after, with no
simACE or fitACE commit between them; both manifests name their
environment's pedigree-graph version. The probe carries two products more
than 13c: fitACE's sparse GRM (the approximate matrix at 0.001) and the
generation kinship summary of the smoke pedigree.

| artifact | baseline 0.9.1 | relocked 0.9.2 | |
|---|---|---|---|
| `report.yaml` | `9c580c90…` | `9c580c90…` | identical |
| `pairwise_relatedness.tsv` | `505a8fd3…` | `505a8fd3…` | identical |
| `A.grm.sp.bin` | `c26e097a…` | `c26e097a…` | identical |
| `A.grm.id` | `80ec2ca3…` | `80ec2ca3…` | identical |
| `mean_kinship_by_generation.txt` | `955aabb9…` | `955aabb9…` | identical |

All five are byte-identical; the report and pair-table hashes are the ones
13c recorded, so the pair contract is unchanged through three releases and
the matrix products through this one. The smoke pedigree's rows stay under
the 0.9.1 slot capacity, so the summary defect 14a found does not show
here (it needs the 536k study pedigree, `../14a/NOTES.md`).

## Result

`failed units: none`, thirteen units at `--routing locked`, exit 0. Every
`routing` step passed, so each consumer resolved `pedigree_graph` from its
own locked environment's site-packages.

| unit | routing | wall | peak RSS | suite |
|---|---|---:|---:|---|
| `pedigree-graph` | own manifest | 159.3 s | 975 MiB | 3125 passed, 9 skipped, 10 deselected |
| `simACE` | locked | 92.2 s | 504 MiB | 1358 passed, 2 skipped (+ smoke, atlas) |
| `fitACE` | locked | 17.3 s | 393 MiB | 383 passed |
| `fitACE_pcgc` | locked | 8.2 s | 339 MiB | 211 passed, 4 skipped |
| `fitACE_iter_reml` | locked | 52.3 s | 344 MiB | 110 passed, 4 skipped |
| `ace_iter_reml` | own manifest | 22.5 s | 108 MiB | six binary test units, all exit 0 |
| `fitACE_tetraher` | locked | 5.3 s | 247 MiB | 34 passed |
| `tetraher_simace` | own manifest | 0.8 s | 108 MiB | ruff, format, ldak runs, ldak is fork |
| `fitACE_pafgrs` | locked | 13.2 s | 401 MiB | 121 passed |
| `fitACE_stan` | locked | 1.6 s | 109 MiB | ruff, format, import beside fitace and simace |
| `fitACE_frailty` | locked | 4.6 s | 211 MiB | 7 passed |
| `fitACE_epimight` | locked | 30.6 s | 606 MiB | 254 passed, 19 deselected |
| `pedsum` | locked | 39.4 s | 328 MiB | 324 passed |

Per-unit records are the JSON files beside this note; step logs were
dropped as in 13c. The host ran at 2600 MHz under the `performance`
governor throughout.

## Slice 14 closes here

Exit criteria of the plan: owned rows selected from the 14a record, no
scored cell over the 5 percent gate on wall or RSS (every cell is 0.11x to
0.28x wall and 0.21x to 0.80x RSS); bytes identical to 0.9.1 on every
parity fixture and on the study pedigrees except the 536k summary, where
0.9.1 was wrong and the record shows why; the matrices still bit-match
`pair_kinship`, the 0.7.1 support parity and the Ne golden stay green; no
Python/Numba DP or allocator production code and no SciPy permutation copy
remain; every input-sized allocation maps to `allocation_failed` and
`csc_index_overflow` is raised from the core; pytest, `cargo test
--release`, clippy, fmt, ruff and ty green; the thirteen-unit gate green at
wheel-site (14b) and after relock (this stage); the four consumer products
byte-identical; 0.9.2 tagged and published.
