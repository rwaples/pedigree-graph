# 16e post-publish gate for pedigree-graph 0.10.0 (2026-09-24)

The post-publish half of 0.10.0 (`../16d/` is the wheel-site half; `../16b/`
and `../16c/` the R parity and tarball records). 0.10.0 is on PyPI and its R
source tarball on the GitHub release; the consumer caps now admit it, the
locks that pin it are relocked, the byte-parity probe has been re-run against
the 0.9.4 record, and the thirteen-unit gate is green at `--routing locked`.

## Published

`v0.10.0` (tag object `77071c7`, commit `bc65ad0`) was pushed on 2026-09-24.
The tag-triggered `Publish` run 35959179964 uploaded five `cp313-abi3` wheels
and the sdist; the published sdist hashes to
`7cba56145a4727305e7d685d2a92f34cb8789e161052d636f53eeb88c1df5ed0`, equal to
the one the 16d wheel gate built. Its two R jobs failed: the outer
`sudo unshare --net` reset the environment, dropping `R_LIBS_USER` and with it
testthat. `6059470` preserves the environment through both sudos, and CI run
35961219703's `r` job then checked the tarball offline on the runner's R
4.6.1 with `Status: OK`. The GitHub release `v0.10.0` was created by hand with
the job's own command, from a tarball built in a clean worktree at the tag and
checked offline first (exit 0), with the CHANGELOG section as notes; installing
it from the release URL as the README shows gives `pedigreegraph` 0.10.0.
The consumers lock the `manylinux_2_17_x86_64` wheel,
`48c3185ce6911f5bfa2cf831d6a2812502ccd54ecebe570a0aadaedec1667d71`.

## Caps and locks

0.10.0 is outside `pedigree-graph>=0.9,<0.10`, so every cap moved, floor and
ceiling, to `>=0.10,<0.11`: simACE `pyproject.toml`, fitACE
`pyproject.toml`, fitACE_epimight `pyproject.toml`, and pedsum
`pyproject.toml` and `environment.yml` (pedsum's floor is also a real
requirement: its merged burden summary reads `RelationshipBurden`). After
removing the rattler index entry once, `pixi lock` in simACE, fitACE and
pedsum each reported only `pedigree-graph 0.9.4 -> 0.10.0`, and a per-
environment package-set diff of every lock against its previous commit shows
that wheel and nothing else. After `pixi install --locked`, simACE's and
fitACE's environments import 0.10.0 from their own site-packages with
`_native.core_version() == "0.10.0"`.

## Byte parity

`../15c/byte-parity/relocked-0.9.4/` is the baseline;
`byte-parity/relocked-0.10.0/` was cut right after this relock with
`tools/pg09_byte_parity.sh`.

| artifact | 0.9.4 | 0.10.0 | |
|---|---|---|---|
| `report.yaml` | `9c580c90…` | `00428d5a…` | provenance only (below) |
| `pairwise_relatedness.tsv` | `505a8fd3…` | `505a8fd3…` | identical |
| `A.grm.sp.bin` | `c26e097a…` | `c26e097a…` | identical |
| `A.grm.id` | `80ec2ca3…` | `80ec2ca3…` | identical |
| `mean_kinship_by_generation.txt` | `955aabb9…` | `955aabb9…` | identical |
| `effective_size.yaml` (simACE) | `1dfd7b0c…` | `1dfd7b0c…` | identical |
| `inbreeding.tsv` (fitACE) | `5411e7b9…` | `5411e7b9…` | identical |

The two `report.yaml` files differ in one line, `simace_version`
(`2026.9.1` against `2026.9.2.dev20+gd718f61ec.d20260924`): setuptools-scm
reads simACE's own checkout, which was 20 commits past its tag with this
relock uncommitted. Every relationship count, correlation and statistic in it
is byte-identical. The 41 MiB `A.grm.sp.bin` and the pair list were deleted
after hashing, as in 15c.

## Unrouted gate

`pixi run python external/pedigree-graph/tools/pg08_release_gate.py run --stage 16e --routing locked`
from the umbrella root: `failed units: none`, exit 0, 5 min 25 s. Every
locked unit's `routing` step resolved `pedigree_graph` in its own
environment's site-packages.

| unit | routing | wall | peak RSS | steps |
|---|---|---:|---:|---|
| `ace_iter_reml` | own manifest | 21.2 s | 109 MiB | fp32-test_laplace_primitives, fp32-test_mcem_step, fp32-test_tmvn, fp64-test_laplace_primitives, fp64-test_mcem_step, fp64-test_tmvn |
| `fitACE` | locked | 12.6 s | 373 MiB | routing, ruff, format, pytest |
| `fitACE_epimight` | locked | 19.4 s | 609 MiB | routing, pytest |
| `fitACE_frailty` | locked | 3.2 s | 211 MiB | routing, pytest |
| `fitACE_iter_reml` | locked | 48.7 s | 356 MiB | routing, pytest |
| `fitACE_pafgrs` | locked | 9.0 s | 399 MiB | routing, pytest |
| `fitACE_pcgc` | locked | 6.2 s | 342 MiB | routing, pytest |
| `fitACE_stan` | locked | 0.8 s | 109 MiB | routing, ruff, format, import |
| `fitACE_tetraher` | locked | 3.3 s | 197 MiB | routing, pytest |
| `pedigree-graph` | own manifest | 118.6 s | 985 MiB | ruff, format, ty, pytest |
| `pedsum` | locked | 18.9 s | 160 MiB | routing, ruff, format, pytest, tsv, cli-smoke |
| `simACE` | locked | 61.9 s | 484 MiB | routing, ruff, format, test, smoke, atlas |
| `tetraher_simace` | own manifest | 0.4 s | 110 MiB | ruff, format, ldak-runs, ldak-is-fork |

Per-unit records: the `.json` beside this file; step logs not kept.
