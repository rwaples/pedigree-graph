# 15c post-publish gate for pedigree-graph 0.9.4 (2026-09-23)

The post-publish half of 0.9.4 (`../15b/` is the wheel-site half; `../15a/`
the benchmark and parity record). 0.9.4 is on PyPI, the three locks that pin
it are relocked, the byte-parity probe has been re-run against the 0.9.3
baseline, and the thirteen-unit gate is green at `--routing locked`.

## Published

`v0.9.4` (`7369893`) was pushed on 2026-09-23 and the tag-triggered
`Publish` workflow (run 35904924405) went nine jobs green, uploading five
`cp313-abi3` wheels and the sdist through trusted publishing. The published
sdist hashes to
`3503750e53339263261528169e6eb88de9e15cee711845b1e02ccff607e0dba5`, equal to
the sdist the 15b wheel gate built and tested locally. The consumers lock the
`manylinux_2_17_x86_64` wheel,
`d94d0cb8437530159d2674921a0158e05cb6f061ce5659b22e31ef497ebdedcd`.

## Caps and locks

The caps read `pedigree-graph>=0.9,<0.10` and admit 0.9.4, so no cap moved.
After removing the rattler index entry once
(`~/.cache/rattler/cache/uv-cache/simple-v21/pypi/pedigree-graph.rkyv`),
`pixi update pedigree-graph` reported `pedigree-graph 0.9.3 -> 0.9.4` in
simACE and fitACE, and in pedsum also removed `numba 0.67.0` and
`llvmlite 0.49.0`, which only pedigree-graph pulled in there. simACE
(`pixi.toml:23`) and fitACE pin numba themselves and keep it. After
`pixi install --locked`, each environment imports 0.9.4 from its own
site-packages with `_native.core_version() == "0.9.4"`; pedsum's can no
longer find `numba`. pedsum's lock resolution warns that its locked
`polars 1.44.0` is yanked; that pin predates this relock and did not change.

## Byte parity

`../15a/byte-parity/baseline-0.9.3/` is the baseline, cut under the 0.9.3
locks in slice 15's commit 1; `byte-parity/relocked-0.9.4/` was cut right
after this relock. `tools/pg09_compare_floats.py` reads every file as
`identical`:

| artifact | 0.9.3 | 0.9.4 | |
|---|---|---|---|
| `report.yaml` | `9c580c90…` | `9c580c90…` | identical |
| `pairwise_relatedness.tsv` | `505a8fd3…` | `505a8fd3…` | identical |
| `A.grm.sp.bin` | `c26e097a…` | `c26e097a…` | identical |
| `A.grm.id` | `80ec2ca3…` | `80ec2ca3…` | identical |
| `mean_kinship_by_generation.txt` | `955aabb9…` | `955aabb9…` | identical |
| `effective_size.yaml` (simACE) | `1dfd7b0c…` | `1dfd7b0c…` | identical |
| `inbreeding.tsv` (fitACE) | `5411e7b9…` | `5411e7b9…` | identical |

The two slice 15 products, simACE's effective sizes and fitACE's inbreeding
export, are byte-identical, as 15a's study compare predicted. The 42 MiB
`A.grm.sp.bin` and the pair list were deleted after hashing, as in 14b.

## Result

`failed units: none`, thirteen units at `--routing locked`, exit 0. Every
`routing` step printed `routed <env>/site-packages/pedigree_graph/__init__.py`.

| unit | routing | wall | peak RSS | suite |
|---|---|---:|---:|---|
| `ace_iter_reml` | own manifest | 22.5 s | 108 MiB |  |
| `fitACE` | locked | 17.7 s | 375 MiB | 383 passed in 15.39s |
| `fitACE_epimight` | locked | 31.1 s | 606 MiB | 254 passed, 19 deselected in 28.54s |
| `fitACE_frailty` | locked | 4.5 s | 211 MiB | 7 passed in 3.25s |
| `fitACE_iter_reml` | locked | 52.5 s | 344 MiB | 110 passed, 4 skipped in 51.40s |
| `fitACE_pafgrs` | locked | 13.2 s | 402 MiB | 121 passed in 11.75s |
| `fitACE_pcgc` | locked | 8.0 s | 339 MiB | 211 passed, 4 skipped, 53 warnings in 5.69s |
| `fitACE_stan` | locked | 1.3 s | 107 MiB |  |
| `fitACE_tetraher` | locked | 4.7 s | 196 MiB | 34 passed in 3.52s |
| `pedigree-graph` | own manifest | 156.0 s | 974 MiB | 3396 passed, 9 skipped, 10 deselected, 4 warnings in 153.65s |
| `pedsum` | locked | 30.8 s | 158 MiB | 321 passed, 3 skipped, 58 warnings in 27.84s |
| `simACE` | locked | 92.6 s | 509 MiB | 1358 passed, 2 skipped, 8 warnings in 27.98s |
| `tetraher_simace` | own manifest | 0.8 s | 107 MiB |  |

Per-unit records: the `.json` beside this file.

Slice 15 is closed. The Rust-core plan's only remaining step is the R package.
