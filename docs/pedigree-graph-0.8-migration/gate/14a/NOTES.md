# 14a evidence for slice 14: the kinship matrix DP on the Rust core (2026-09-23)

Plan: `simACE/plans/pedigree-graph-slice-14-kinship-matrix-dp.md` (locked
2026-09-23). Tree measured: `2d8b958` (commit 3 of the slice, both row
layouts present and selectable through `PEDIGREE_GRAPH_KINSHIP_ROWS`) with
the uncommitted files of commit 4. Baseline: the 0.9.1 PyPI wheel as
simACE's pixi env installs it (`core_version() == "0.9.1"`,
`site-packages/pedigree_graph`), run under that env's interpreter in the
same interleaved sweep as the two source arms (`benchmarks/_harness.py`
gained a per-arm interpreter and environment for this).

## Selection: layout A, one owned vector pair per row

Every scored cell prefers the owned layout on both metrics, and the arena
(the 0.9.1 slab allocator ported as it was) loses to it on every cell:

| cell | 0.9.1 wheel | owned | arena | owned / wheel | arena / wheel |
|---|---:|---:|---:|---:|---:|
| km-1k, `random_1k` complete | 0.15 s, 170 MiB | 0.03 s, 136 MiB | 0.03 s, 138 MiB | 0.18, 0.80 | 0.19, 0.81 |
| km-20k, `dev_mean_n10k` complete (20,400 rows) | 3.42 s, 715 MiB | 0.83 s, 452 MiB | 0.95 s, 565 MiB | 0.24, 0.63 | 0.28, 0.79 |
| km-30k, `random_30k` complete | 97.1 s, 13,855 MiB | 18.9 s, 4,723 MiB | 21.4 s, 6,268 MiB | 0.19, 0.34 | 0.22, 0.45 |
| km-53k, `baseline10K` complete (53,466 rows) | 10.1 s, 1,500 MiB | 2.85 s, 961 MiB | 3.38 s, 1,548 MiB | 0.28, 0.64 | 0.34, 1.03 |
| akm-20k, `dev_cont_n10k` approximate 0.001 | 6.30 s, 804 MiB | 1.09 s, 373 MiB | 1.20 s, 503 MiB | 0.17, 0.46 | 0.19, 0.63 |
| akm-30k, `random_30k` approximate 0.001 | 75.8 s, 6,928 MiB | 10.5 s, 1,546 MiB | 12.7 s, 4,431 MiB | 0.14, 0.22 | 0.17, 0.64 |
| akm-53k, `baseline10K` approximate 0.001 | 18.0 s, 1,500 MiB | 3.57 s, 786 MiB | 3.66 s, 845 MiB | 0.20, 0.52 | 0.20, 0.56 |
| mkg-30k, `random_30k` summary | 52.1 s, 6,298 MiB | 5.61 s, 1,340 MiB | 7.33 s, 4,226 MiB | 0.11, 0.21 | 0.14, 0.67 |
| mkg-536k, `baseline100K` summary (536,036 rows) | 89.3 s, 14,758 MiB | 15.8 s, 3,604 MiB | 16.4 s, 8,551 MiB | 0.18, 0.24 | 0.18, 0.58 |

Medians of three interleaved reps in fresh single-threaded subprocesses
(`kinship_matrix.json`), peak RSS as kernel `VmHWM` reset at region start.
Full tables with spans and per-arm verdicts: `comparison.md`; the
harness render: `report.md`. The wheel-only shakedown pass that decided
every cell fits the timeout is `kinship_matrix_wheel_pass.json`. Every
cell's checksum is identical across the three builds except mkg-536k,
where the two source arms agree with each other and the wheel is wrong
(below).

The arena's one near-miss, 1.03x the wheel's peak on km-53k, is the
`n * init_cap` pre-allocation the 0.9.1 layout starts from; the owned
layout has no such floor and is 0.64x there. Where retirement frees the
most (the summary and the approximate matrix on `random_30k`, the 536k
summary) the arena's free list keeps every slot it ever carved while the
owned layout returns each retired row to the allocator, which is why the
arena sits at 0.58x to 0.67x of the wheel and the owned layout at 0.21x to
0.24x. The wheel's 13.9 GiB on km-30k is the eager `n * 4096` slab
(`_kinship_allocator._suggest_init_cap_per_row` at depth 8); the
permutation round trip through SciPy that the plan set out to delete is a
second copy on top of it, and both are gone.

The arena layout, `Layout`, the `PEDIGREE_GRAPH_KINSHIP_ROWS` switch and the
benchmark's third arm are deleted in commit 4.

## Gate verdicts (5 percent rule, all nine cells scored)

- Wall: every scored cell is between 0.11x and 0.28x the wheel. No cell
  blocks.
- Peak RSS: every scored cell is between 0.21x and 0.80x the wheel. No
  cell blocks.
- The plan's unverified claim that owned rows beat the arena on RSS in
  CSC mode is measured: 0.63x against 0.79x on km-20k, 0.34x against
  0.45x on km-30k, 0.64x against 1.03x on km-53k.

## Study pedigree byte check (exit criterion 2)

`tools/pg14_study_matrix.py capture` ran under the simACE env on the 0.9.1
wheel and `compare` under this env on the source build (owned rows);
`study/capture.json` and `study/comparison.json` are the record. Each
product ran on its own freshly built graph in its own process, so the
summary never took the cached-matrix route.

| pedigree | rows | `kinship_matrix()` | `approximate_kinship_matrix(0.001)` | `mean_kinship_by_generation()` |
|---|---:|---|---|---|
| dev_mean_n10k/rep1 | 20,400 | byte-identical | byte-identical | byte-identical |
| dev_cont_n10k/rep1 | 20,400 | byte-identical | byte-identical | byte-identical |
| baseline10K/rep1 | 53,466 | byte-identical | byte-identical | byte-identical |
| random_30k | 30,300 | byte-identical | byte-identical | byte-identical |
| baseline100K/rep1 | 536,036 | not run (plan: summary only) | not run | **differs, deepest bucket only** |

"Byte-identical" is equality of nnz and of the SHA-256 of `indptr`,
`indices` and the uint32 view of `data`, or of the float64 bits of the
summary's means with its pair counts. The arena arm, run through the same
tool with `PEDIGREE_GRAPH_KINSHIP_ROWS=arena`, gives the same bits as the
owned arm on the 536k summary.

## The 536k summary: 0.9.1 is wrong, and why

The deepest generation's mean under the wheel is `2.96781502e-05`; the
native value is `2.96785613e-05`, 1.4e-5 higher in relative terms. The
other five buckets and every pair count agree to the bit.

The cause is in the 0.9.1 allocator, `_kinship_allocator._append_entry`
under retirement. A child's merge walk reads its parents' rows through
offsets captured before the walk. When the write of `(parent, child)` fills
the parent's slot mid-walk, the row relocates and the old slot is pushed
onto the size-bucketed free list; a later append in the same walk to
another row that is full at the same capacity pops that slot back and
copies its own entries over it, and the walk keeps reading the old offset.
The garbage entries change the child's row and the inline sum. It needs a
row to outgrow its first slot, `2 ** (max_depth + 4)` entries clamped to
[16, 4096]; no parity fixture does (`random_30k` at depth 8 starts at 4096
and no row reaches it), which is why the differential tests and the Ne
golden lock were silent. `baseline100K` at depth 5 starts at 512 and its
deepest rows pass it.

Three independent checks, all reproducible from the oracle package
`tests/oracle/kinship_dp/` (the 0.9.1 DP verbatim):

1. A random search over 150 seeded pedigrees of 60 to 400 rows with the
   slot capacity forced to 2 and 16: the retiring DP (lazy or eager
   allocation) disagrees with the non-retiring DP's post-hoc walk in 177
   of 300 cases, by up to 3.8 in a sum of order 100; the non-retiring DP
   with tiny slots agrees with itself at 4096 in all 300, and the native
   sums agree with the non-retiring walk in all 300.
2. The same search on a copy of the oracle whose `_freelist_pop` never
   returns a slot: zero disagreements in 300. Slot reuse is the mechanism.
3. The verbatim oracle run on `baseline100K` under this env, with the
   genome-node labels the public method uses, reproduces the wheel's six
   means bit for bit, wrong bucket included; so the oracle move is faithful
   and the defect is the wheel's.

`tests/test_native_kinship_matrix.py::test_native_sums_are_free_of_the_0_9_1_slot_reuse_hazard`
keeps case 1 as a test: on a 349-row pedigree with slots of 2 the oracle's
retiring path disagrees with its relocation-free walk while the native
sums match it to 1e-12. The native DP stages each walk's relatives in a
scratch vector before writing either row, so no write can move a row the
walk is reading.

`approximate_kinship_matrix` ran the same retiring pass in 0.9.1 to
capture its values. A corrupted row there would have broken the sorted
merge against the candidate list and failed the "complete DP did not emit
every approximate-support candidate" assertion rather than returned a
wrong value; the three study pedigrees that run it stay below their slot
capacity and matched byte for byte.

## Deviations from the plan

- D8 named a new validation code `kinship_depth_not_structural` for a
  depth that is negative or not above a parent's. The check exists and
  runs before any allocation, but it raises the existing
  `value_out_of_range` on field `depth` (position the row, minimum the
  parent's depth plus one) instead of a new code: the public path never
  reaches it, and a new code would have touched the error registry, its
  size test and the ADR 0006 code table for a raw-binding guard.
- D8's `value_out_of_range` on `threshold` is a usage error instead
  (`ValueError`, class Usage, no code): the registry's out-of-range fields
  are integers and the public facade validates the threshold first.
- Exit criterion 2 holds on every parity fixture and on twelve of the
  thirteen study products; the thirteenth is the summary above, where the
  0.9.1 value is the wrong one. `study/comparison.json` therefore records
  `byte_identical: false` with the single differing product named.
- The lazy/eager allocation distinction of the 0.9.1 DP was not ported:
  with owned rows an unwritten row costs nothing, so every founder's
  diagonal is stored and the childless-founder special cases in the MZ
  pass and the capture disappear. The bytes are unchanged (the study
  table and the differential tests).

## Governor

The host ran at 2600 MHz under `performance` before, during and after the
sweep (`/proc/cpuinfo` under a busy loop, no PROCHOT clamp); the first
one-second probe of the day read 800 MHz while the cores ramped and was
repeated over four seconds before anything was timed.
