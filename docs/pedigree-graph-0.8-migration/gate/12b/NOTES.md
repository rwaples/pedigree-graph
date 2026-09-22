# 12b wheel-site gate for pedigree-graph 0.9.0 (2026-09-19)

Slice 12 (`plans/pedigree-graph-slice-12-relationship-pairs-v2.md`) commit 8:
the pre-publish half. The 0.9.0 wheel is built from a clean worktree at the
`v0.9.0` tag and every family check unit runs against it through `PYTHONPATH`,
with the consumer locks still frozen at 0.8.3. The post-publish half — the four
dependency caps, the four relocks and the byte compare — is stage `12c`.

## The artifact under test

`tools/pg08_wheel_gate.sh <work> v0.9.0`, built from `a4bacb7a2ee233c7002fb26843a02c24b82789e6`:

| artifact | sha256 |
|---|---|
| `pedigree_graph-0.9.0-cp313-abi3-linux_x86_64.whl` | `6f2ec0322dcba25aab0435da8692b222aff3ca685c734ca0873bcfac2c7b1ee6` |
| `pedigree_graph-0.9.0.tar.gz` | `45013ae13b1285e08223f53a7e17fd53d076455002c556046a10df9bf9550684` |

Both install into isolated venvs and resolve `pedigree_graph` from
site-packages with `py.typed`, `_native.pyi`, and `core_version()` equal to the
distribution version; the sdist recompiles the Rust core rather than repacking
the wheel. The package's own suite against the installed wheel:
`2935 passed, 3 skipped, 9 deselected in 207.20s`, exit 0.

## The tag moved, the artifact did not

The first build of this stage was `2f7e66421dab80c16207c545a544ca0ba3c0077d`,
the changelog commit. Two test-only commits then landed on `main` closing
issues #21 and #26, and rather than leave the release tag behind them the
`v0.9.0` tag and the `v0.9` branch were moved to `a4bacb7`, which is why the
row above names a different commit and a different wheel hash than the first
run of this gate recorded.

Nothing shipped changed, and that is measured rather than assumed. Neither
distribution carries `tests/`: `tar tzf` finds zero entries under
`pedigree_graph-0.9.0/tests/` in the sdist and the wheel has none either. The
rebuilt sdist is byte-identical, hashing to the same
`45013ae13b1285e08223f53a7e17fd53d076455002c556046a10df9bf9550684` the first
build produced. The wheel hash moved only because `diff -r` over the two
unpacked wheels reports exactly two differing members, the CycloneDX SBOM,
whose serial number, timestamp and embedded `/tmp/build-via-sdist-*` paths are
per-build, and the `RECORD` line that hashes it. Every other member is
identical, including the compiled extension: `_native.abi3.so` is
`cba22e01d2dc1a4e009087f0d7343b26` in both wheels.

What the two commits did change is the suite this gate runs, since
`pg08_wheel_gate.sh` runs `$CLEAN/tests` from the tagged worktree. Collection
went from 2916 to 2938 under `-m "not slow"`, 22 added and none removed. One is
`test_parity_v071.py::test_exemptions_are_values_only`. The other 21 are the
single `deep_inbred_60g` line added to `_fixtures()`, which reaches further
than the file it sits in: 10 cases in `test_relationship_pairs.py`, 10 in
`test_view_relationship_pairs.py`, and one in `test_close_relative_counts.py`.

The thirteen-unit table under **Result** was produced against the first build's
`wheel-site` and was not re-run. It stands: the consumers import the wheel, the
two wheels differ only in the SBOM, and neither of the two commits touches
anything a consumer can import.

## Published

`v0.9.0` was pushed on 2026-09-19 and the tag-triggered `Publish` workflow
(run 35474075740) went nine jobs green in 2m51s, uploading five `cp313-abi3`
wheels and the sdist to PyPI through trusted publishing. The local
`linux_x86_64` wheel in the table above is a gate artifact and is never
uploaded; PyPI would reject that platform tag, so the shipped x86_64 Linux
wheel is the workflow's `manylinux_2_17` build.

The published sdist hashes to
`45013ae13b1285e08223f53a7e17fd53d076455002c556046a10df9bf9550684`, identical
to the local one in the table, so the source distribution on PyPI is the exact
artifact this gate tested rather than a rebuild that merely shares a commit.

## Thirteen units, not eleven

The gate ran eleven units through slice 11. `tools/family_repos.py` lists
thirteen, and the two it never covered — `fitACE_stan` and `tetraher_simace` —
had joined the family without anything proving they still passed their own
checks under a new pedigree-graph. Both are test-less units, so they gate on
ruff, formatting, and the thing that would actually break: `fitace_stan`
importing alongside `fitace` and `simace`, and the LDAK fork's binary running
and carrying its own `--simace-grouping` flag. Numerical equivalence of the
fork with upstream stays where it was, in the `fitACE` unit's
`tests/tetraher/test_fork_equivalence.py`.

`tests/test_release_gate_covers_family.py` now holds the gate's unit set equal
to the manifest's, so the next family member cannot go missing the same way.

## Result

`failed units: none`, exit 0. Per-unit records and full step logs are the JSON
files and directories beside this one.

| unit | wall | peak RSS |
|---|---|---|
| pedigree-graph | 175.0 s | 980.2 MiB |
| simACE | 114.7 s | 493.9 MiB |
| fitACE | 23.6 s | 414.0 MiB |
| fitACE_pcgc | 12.8 s | 342.2 MiB |
| fitACE_iter_reml | 59.8 s | 348.1 MiB |
| ace_iter_reml | 28.8 s | 106.2 MiB |
| fitACE_tetraher | 9.2 s | 293.2 MiB |
| tetraher_simace | 1.1 s | 103.0 MiB |
| fitACE_pafgrs | 15.1 s | 717.4 MiB |
| fitACE_stan | 1.8 s | 107.4 MiB |
| fitACE_frailty | 6.6 s | 211.5 MiB |
| fitACE_epimight | 58.8 s | 737.6 MiB |
| pedsum | 212.0 s | 333.6 MiB |

Walls are not comparable with 11a's: these ran interleaved with other work on
the box. The gate is a pass/fail record, not a benchmark; slice 12's timings
are in `../12a/`.

## What the first run caught

The first attempt failed two units, neither of them the release.

`tetraher_simace` failed because its `ldak` step asserted a zero exit and
LDAK's usage path exits 1 — the manual check that chose the step had read
`head`'s exit code through a pipe rather than the binary's. Replaced by the two
probes described above.

`simACE` failed ruff and format on untracked probe scripts under `scratchpad/`,
now gitignored, and failed 22 tests on the 0.9 effective-size changes. Those 22
were confirmed to be 0.9's doing and not the working tree's: the same five
files under the locked 0.8.3 env gave `137 passed`. They are the counterpart of
pedsum's own migration, and simACE followed with `9385970`.

## Known state at this stage

simACE's `dev` is green against 0.9.0 and red against the 0.8.3 its lock still
pins, because `ne_group_coancestry` does not exist in 0.8.3. That is the same
position pedsum took deliberately, and `12c` resolves it for both.
