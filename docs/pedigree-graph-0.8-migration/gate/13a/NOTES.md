# 13a evidence for slice 13: pairwise kinship on the Rust core (2026-09-22/23)

Plan: `simACE/plans/pedigree-graph-slice-13-pair-kinship.md` (locked
2026-09-22). Tree measured: `c50be63` (commit 3 of the slice, both memo
layouts present and selectable through `PEDIGREE_GRAPH_KINSHIP_LAYOUT`) with
the uncommitted benchmark and tool files of commit 4. Baseline: the 0.9.0 PyPI
wheel as simACE's pixi env installs it (`core_version() == "0.9.0"`,
`site-packages/pedigree_graph`).

## Selection: layout B, one small table per lower row

Every scored cell prefers the per-row layout on both metrics, and the flat
port of the 0.9.0 table loses to it on every cell:

| cell | 0.9.0 wheel | rows | flat | rows / wheel | flat / wheel |
|---|---:|---:|---:|---:|---:|
| random_30k cold, degree 3 | 82.5 s, 762 MiB | 5.32 s, 515 MiB | 62.1 s, 720 MiB | 0.065, 0.68 | 0.75, 0.94 |
| random_30k matrix, degree 3 | 79.9 s, 781 MiB | 5.98 s, 523 MiB | 60.0 s, 729 MiB | 0.075, 0.67 | 0.75, 0.93 |
| random_300k cold, degree 3 | did not finish in 3600 s (6.8 GiB and growing) | 934 s, 14,076 MiB | not run (see below) | | |
| baseline10K cold (53,466 rows) | 6.82 s, 865 MiB | 2.06 s, 462 MiB | 5.45 s, 831 MiB | 0.30, 0.53 | 0.80, 0.96 |
| baseline10K self pairs | 0.73 s, 391 MiB | 0.43 s, 276 MiB | 0.62 s, 366 MiB | 0.59, 0.70 | 0.84, 0.94 |
| baseline100K cold (536,036 rows) | 31.3 s, 5,421 MiB | 24.8 s, 3,022 MiB | 27.4 s, 5,260 MiB | 0.79, 0.56 | 0.88, 0.97 |
| baseline100K self pairs | 6.75 s, 1,526 MiB | 5.12 s, 896 MiB | 5.84 s, 1,475 MiB | 0.76, 0.59 | 0.87, 0.97 |
| baseline100K matrix, degree 3 | 49.1 s, 3,100 MiB | 32.7 s, 3,125 MiB | 35.8 s, 5,393 MiB | 0.67, 1.008 | 0.73, 1.74 |
| dev_mean_n10k degree 5 (20,400 rows) | 21.1 s, 864 MiB | 2.35 s, 456 MiB | 16.1 s, 821 MiB | 0.11, 0.53 | 0.77, 0.95 |

Medians of three interleaved reps in fresh single-threaded subprocesses, peak
RSS as kernel `VmHWM` reset at region start, wheel column from
`pair_kinship_0.9.0_perf.json` (see "Governor" below). Full tables with
spans: `comparison.md`; per-report renders: `report_*.md`; raw reports beside
them. Every cell's checksum is identical across the three builds, so the
three are returning the same bits while they differ in time and memory.

The flat port matching the wheel within 12 to 25 percent says the move to
Rust on its own is worth little here; the layout is the win. A row's ancestor
states sit in one small table that stays in cache while the walk fills it,
where the flat table's identity-hashed probes miss on nearly every step.

The flat layout is deleted in commit 4; `Layout`, the `--layout` switch and
`PEDIGREE_GRAPH_KINSHIP_LAYOUT` go with it.

## Gate verdicts (5 percent rule, fresh cells)

- Wall: every scored cell is between 0.065x and 0.79x the wheel. No cell
  blocks.
- Peak RSS: every scored cell is between 0.53x and 0.70x the wheel except
  `baseline100K/matrix` at 1.008x, inside the gate.
- The plan's second target, pk-536k and rkm-536k peak RSS down at least 30
  percent: **met on pk-536k** (0.56x, a 44 percent cut) and **not met on
  rkm-536k** (1.008x). The target was priced on the rehash copy the 0.9.0
  pairs path pays, and the wheel's matrix path never paid it: it streamed
  `2**20`-pair chunks through the retained memo, and the 1 GiB retention
  limit discarded that memo once it outgrew the limit, so later chunks
  re-walked from small cold tables. That bounded its peak at the cost of
  wall time, which is why the wheel's matrix cell is 49 s at 3.1 GiB while
  its pairs cell is 31 s at 5.4 GiB on the same closure. The native support
  walk holds one memo for the whole support (3.1 GiB) and is 0.67x the
  wall. The peak of the rows layout on this closure, 3.0 GiB for the pairs
  cell and 3.1 GiB for the matrix, is the memo itself; there is no copy left
  to remove.

## Accepted warm-call cost (D1, recorded, not scored)

| cell | 0.9.0 wheel | rows |
|---|---:|---:|
| random_30k, the same degree-3 query again on the same graph | 0.45 s | 5.39 s |
| random_30k, matrix after a degree-3 `pair_kinship` | 1.20 s | 5.77 s |

The plan estimated about 83 s for each, the cost of a cold wheel walk. With
the native walk a second call is a second 5 s walk. No consumer makes the
second call (consumer survey in the plan, D1).

## random_300k

The 0.9.0 wheel did not finish the degree-3 walk within the 3600 s cell
timeout; its process was at 6.8 GiB resident and growing when killed. The
rows layout finishes in 934 s (spread 0.2 percent over three reps) at
14.1 GiB. Its memo holds on the order of a billion states on this fixture,
so the flat layout was not run: at 12 bytes a slot it would need about
21 GiB for the same slot count and, holding the predecessor table through
its last doubling, about 31 GiB, the whole host. The cell therefore has no
wheel or flat comparison; it is recorded as a capability the wheel lacked.

## Study pedigree bit check (exit criterion 2)

`tools/pg13_study_kinship.py capture` ran under the simACE env on the 0.9.0
wheel and `compare` under this env on the source build; `study/capture.json`
and `study/comparison.json` are the record (`*.npz` local only).

| pedigree | rows | queries | pairs | differing elements (both orders) | max ULP |
|---|---:|---:|---:|---:|---:|
| dev_mean_n10k/rep1 | 20,400 | degree 3 and 5 blocks plus self | 1,930,268 | 0 | 0 |
| baseline10K/rep1 | 53,466 | degree 3 blocks plus self | 1,108,115 | 0 | 0 |
| dev_laplace_am_strong_50k/rep1 | 102,000 | degree 3 blocks plus self | 2,032,175 | 0 | 0 |
| baseline100K/rep1 | 536,036 | degree 3 blocks plus self | 10,963,816 | 0 | 0 |

Every pair block's SHA-256 matched before values were compared, so the
comparison is over identical pairs. The permanent golden lock on the repo
fixtures (`tests/test_pair_kinship_golden.py`) passes on the same build.

## Governor

The first wheel sweep (`pair_kinship_0.9.0.json`) ran with the CPU governor at
`powersave` capped at 2250 MHz; between it and the rows sweep the host moved
to `performance` at 2600 MHz, and the flat sweep and the rerun of the wheel's
ten short cells (`pair_kinship_0.9.0_perf.json`) ran there. The rerun
reproduced the first sweep within its spread on every cell (random_30k cold
82.50 s against 82.86 s, baseline100K cold 31.28 s against 32.03 s), so the
governor did not move the wheel's numbers and the table above uses the
rerun. The clock was read from `/proc/cpuinfo` under load before the sweeps
(2200 MHz under `powersave`, 2600 MHz under `performance`) and after the flat
sweep (2600 MHz on every core); no PROCHOT clamp was in effect.

## Deviations from the plan

- `_upper_support_chunks`, `_write_symmetric_values` and
  `_EXACT_VALUE_CHUNK_SIZE` stay in `_kinship_matrix.py`: the plan listed
  them for deletion but the approximate-support path still consumes them.
  Only `_exactify_support` was rewired.
- pk-300k has no wheel baseline (timeout) and no flat measurement (memory),
  as above.
- The rkm-536k RSS target, as above.

`tools/pg13_study_kinship.py` was removed in the 1.0 stabilization cleanup (2026-09-24); it was last present at `3adf97d`.
