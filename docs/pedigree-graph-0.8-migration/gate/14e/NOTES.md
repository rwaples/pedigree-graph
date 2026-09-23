# 14e post-publish gate for pedigree-graph 0.9.3 (2026-09-23)

The post-publish half of the 0.9.3 patch (`../14d/` is the wheel-site half).
0.9.3 is on PyPI, the three locks that pin it are relocked, the byte-parity
probe has been re-run against the 0.9.2 cut, and the thirteen-unit gate is
green at `--routing locked`.

## Published

`v0.9.3` (`1f7ba43`) was pushed on 2026-09-23 and the tag-triggered `Publish`
workflow (run 35862266885) went nine jobs green, uploading five `cp313-abi3`
wheels and the sdist through trusted publishing. The published sdist hashes
to `a32d2398a55d7faa227481c46ea46f60b9618df0b2f017d44d263963d2a3c00f`, equal
to the sdist the 14d wheel gate built and tested locally. The consumers lock
the `manylinux_2_17_x86_64` wheel,
`fd91de2b80f7f5a703013854a84fd52c70aaa7b9f4b0e2a769ea7296db48f057`.

## Caps and locks

The caps read `pedigree-graph>=0.9,<0.10` and admit the patch, so no cap
moved. `pixi update pedigree-graph` in simACE, fitACE and pedsum reported
exactly one change each, `pedigree-graph 0.9.2 -> 0.9.3`; the diffs touch
that package's URL, version and sha256 and reorder unchanged entries
(`sortedcontainers` in simACE and fitACE, `mergedeep` in simACE) that the
new URL now sorts before. The stale-index trap of 12c and 14c recurred
twice: removing `~/.cache/rattler/cache/uv-cache/simple-v21/pypi/pedigree-graph.rkyv`
once was not enough, because the refetch landed on a PyPI CDN edge still
serving the pre-0.9.3 index (`cache-control: max-age=600`) and the rebuilt
entry again stopped at 0.9.2; removing it a second time a few minutes later
fetched an index with 0.9.3. After `pixi install --locked`, each environment
imports 0.9.3 from its own site-packages with
`_native.core_version() == "0.9.3"`.

## Byte parity

`../14c/byte-parity/relocked-0.9.2/` is the baseline: it was cut right after
the 0.9.2 relock, and no simACE or fitACE commit landed between it and
`byte-parity/relocked-0.9.3/`, cut immediately after this relock. Both
manifests name their environment's pedigree-graph version.

| artifact | relocked 0.9.2 | relocked 0.9.3 | |
|---|---|---|---|
| `report.yaml` | `9c580c90…` | `9c580c90…` | identical |
| `pairwise_relatedness.tsv` | `505a8fd3…` | `505a8fd3…` | identical |
| `A.grm.sp.bin` | `c26e097a…` | `c26e097a…` | identical |
| `A.grm.id` | `80ec2ca3…` | `80ec2ca3…` | identical |
| `mean_kinship_by_generation.txt` | `955aabb9…` | `955aabb9…` | identical |

All five are byte-identical, as expected: neither 0.9.3 fix is reachable
through `PedigreeGraph`, whose depth is always structural and whose
topology sort is called on arrays the facade has already allocated.

## Result

`failed units: none`, thirteen units at `--routing locked`, exit 0. Every
`routing` step printed `routed <env>/site-packages/pedigree_graph/__init__.py`.

| unit | routing | wall | peak RSS | suite |
|---|---|---:|---:|---|
| `pedigree-graph` | own manifest | 160.9 s | 977 MiB | 3126 passed, 9 skipped, 10 deselected |
| `simACE` | locked | 92.3 s | 509 MiB | 1358 passed, 2 skipped (+ smoke, atlas) |
| `fitACE` | locked | 18.9 s | 394 MiB | 383 passed |
| `fitACE_pcgc` | locked | 8.3 s | 342 MiB | 211 passed, 4 skipped |
| `fitACE_iter_reml` | locked | 49.7 s | 344 MiB | 110 passed, 4 skipped |
| `ace_iter_reml` | own manifest | 22.5 s | 110 MiB | six binary test units, all exit 0 |
| `fitACE_tetraher` | locked | 5.2 s | 247 MiB | 34 passed |
| `tetraher_simace` | own manifest | 0.8 s | 107 MiB | ruff, format, ldak runs, ldak is fork |
| `fitACE_pafgrs` | locked | 13.1 s | 399 MiB | 121 passed |
| `fitACE_stan` | locked | 1.6 s | 106 MiB | ruff, format, import beside fitace and simace |
| `fitACE_frailty` | locked | 4.6 s | 212 MiB | 7 passed |
| `fitACE_epimight` | locked | 30.7 s | 606 MiB | 254 passed, 19 deselected |
| `pedsum` | locked | 39.8 s | 328 MiB | 324 passed |

The pedigree-graph suite is one test up on 14c, the founder-above-depth-0
differential test that holds the first fix. Per-unit records are the JSON
files beside this note; step logs were dropped as in 14c. The host ran at
2600 MHz under the `performance` governor throughout.
