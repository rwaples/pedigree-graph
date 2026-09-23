# 13b wheel-site gate for pedigree-graph 0.9.1 (2026-09-23)

Slice 13 (`simACE/plans/pedigree-graph-slice-13-pair-kinship.md`) commit 6:
the pre-publish half. The 0.9.1 wheel is built from a clean worktree at the
`v0.9.1` tag and every family check unit runs against it through the routed
site, with the consumer locks still frozen at 0.9.0. The post-publish half,
the three relocks, the byte compare and the thirteen-unit gate at
`--routing locked`, is stage `13c` and waits on the tag being pushed and the
`Publish` workflow uploading to PyPI.

## The artifact under test

`tools/pg08_wheel_gate.sh <work> v0.9.1`, run from the simACE umbrella root,
built from `6bd04d3ced042a2f4a4e04ad4975bc0bd8643f98`:

| artifact | sha256 |
|---|---|
| `pedigree_graph-0.9.1-cp313-abi3-linux_x86_64.whl` | `d4e8d6802238f3b0a21318730f8685391c848d901219fa0567cfd44f614d532c` |
| `pedigree_graph-0.9.1.tar.gz` | `2933cfa721c89e4b7c63e75d3d5ce1053434f10b2452dc02715c86be80ffcbdb` |

Both install into isolated venvs and resolve `pedigree_graph` from
site-packages with `py.typed`, `_native.pyi`, and `core_version()` equal to
the distribution version; the sdist recompiles the Rust core (rustc 1.98.0)
rather than repacking the wheel. The package's own suite against the
installed wheel: `2998 passed, 4 skipped, 10 deselected in 192.41s`, exit 0
(`wheel-gate.json`).

A first run of the wheel gate, launched from inside the pedigree-graph
checkout, built the same sdist hash but failed its import check: the venv
interpreter, reading the check from stdin with the checkout as its working
directory, resolved `pedigree_graph` from the source tree. The gate's README
says it runs from the umbrella root; the run above does. The wheel hash
differs between the two runs only through the per-build SBOM, as 12b
documented for 0.9.0.

## Result

`pixi run --frozen python external/pedigree-graph/tools/pg08_release_gate.py run --stage 13b --routing <work>/wheel-site`
from the umbrella root: `failed units: none`, exit 0. Every routed unit's
`routing` step printed `routed <wheel-site>/pedigree_graph/__init__.py`.

| unit | routing | wall | peak RSS | steps |
|---|---|---:|---:|---|
| `pedigree-graph` | own manifest | 161.4 s | 996 MiB | ruff, format, ty, pytest |
| `simACE` | wheel-site | 113.2 s | 560 MiB | routing, ruff, format, test, smoke, atlas |
| `fitACE` | wheel-site | 17.9 s | 386 MiB | routing, ruff, format, pytest |
| `fitACE_pcgc` | wheel-site | 9.0 s | 347 MiB | routing, pytest |
| `fitACE_iter_reml` | wheel-site | 52.9 s | 348 MiB | routing, pytest |
| `ace_iter_reml` | own manifest | 22.3 s | 105 MiB | six fp32/fp64 test binaries |
| `fitACE_tetraher` | wheel-site | 5.1 s | 247 MiB | routing, pytest |
| `tetraher_simace` | own manifest | 0.8 s | 106 MiB | ruff, format, ldak-runs, ldak-is-fork |
| `fitACE_pafgrs` | wheel-site | 13.2 s | 402 MiB | routing, pytest |
| `fitACE_stan` | wheel-site | 1.7 s | 106 MiB | routing, ruff, format, import |
| `fitACE_frailty` | wheel-site | 4.6 s | 211 MiB | routing, pytest |
| `fitACE_epimight` | wheel-site | 30.7 s | 606 MiB | routing, pytest |
| `pedsum` | wheel-site | 51.1 s | 385 MiB | routing, ruff, format, pytest, tsv, cli-smoke |

Every step exited 0. Per-unit records are the JSON files beside this note;
the raw step logs were dropped as in 12b. The gate is a pass/fail record, not
a benchmark; slice 13's timings are in `../13a/`.

## What this proves and what it does not

The consumers run unchanged against 0.9.1: `pair_kinship` kept its three
call forms, dtype and values, and no consumer reached the retired
`memo_capacity_exceeded` or the deleted private kernel. The byte-level
comparison of fitACE's pair export and simACE's report against the
0.9.0-locked run (`tools/pg09_byte_parity.sh`) is stage 13c, after the
relock.
