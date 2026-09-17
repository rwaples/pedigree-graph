# Slice 12 stage A: screening the pair emitters

Measured 2026-09-16 on the 12-core, 30 GiB workstation (i7-9750H, `powersave`
governor, cores at about 2.2 GHz under load, not the 800 MHz clamp), release
build at `a1731b5` plus the uncommitted slice 12 working tree, through
`benchmarks/bench_pair_emitters.py --repeat 3 --threads 1 6`.  Graph
receiver, all 23 categories at `max_degree=5`, three interleaved fresh
processes per cell.  Wall is the `pair_blocks` call only.  Engine RSS is the
process `VmHWM` after the call minus `VmHWM` after loading the columns, so it
is the emitter's own peak above the input.  The raw report is
`benchmarks/reports/pair_emitters_screening.json` (untracked).

Before timing, every emitter's 23 blocks matched the Python matrix oracle
(`relationship_pairs(max_degree=5)`) on `random_30k`, count and digest, for
the graph and for a seeded reordered half view; `random_1k` likewise.  All
36 timed runs produced identical digests across emitters and thread counts.

| fixture | threads | emitter | pairs | wall median (s) | wall range | engine RSS median (MiB) | engine RSS max |
|---|---|---|---|---|---|---|---|
| random_30k | 1 | buffered | 3,067,768 | 1.291 | 1.270 to 1.430 | 49.5 | 49.6 |
| random_30k | 1 | two_pass | 3,067,768 | 2.493 | 2.438 to 2.503 | 26.1 | 26.1 |
| random_30k | 1 | bounded_wave | 3,067,768 | 1.285 | 1.260 to 1.361 | 37.8 | 37.8 |
| random_30k | 6 | buffered | 3,067,768 | 0.325 | 0.324 to 0.344 | 50.8 | 53.6 |
| random_30k | 6 | two_pass | 3,067,768 | 0.631 | 0.627 to 0.652 | 26.9 | 26.9 |
| random_30k | 6 | bounded_wave | 3,067,768 | 0.320 | 0.315 to 0.356 | 45.2 | 45.8 |
| random_300k | 1 | buffered | 30,820,305 | 16.569 | 16.479 to 17.434 | 528.9 | 528.9 |
| random_300k | 1 | two_pass | 30,820,305 | 31.336 | 31.007 to 31.861 | 261.2 | 261.3 |
| random_300k | 1 | bounded_wave | 30,820,305 | 16.028 | 16.028 to 16.727 | 327.8 | 327.9 |
| random_300k | 6 | buffered | 30,820,305 | 3.941 | 3.913 to 4.006 | 537.4 | 538.4 |
| random_300k | 6 | two_pass | 30,820,305 | 7.174 | 7.037 to 7.261 | 264.6 | 264.8 |
| random_300k | 6 | bounded_wave | 30,820,305 | 4.323 | 4.256 to 4.523 | 410.3 | 422.4 |

`bounded_wave` ran with four tasks per thread per wave.

## Reading

The raw payload of two `int32` arrays at 300k is 235 MiB (`8P`).

* `two_pass` sits at the payload plus about 26 MiB of engine state at both
  thread counts: no copy of the result exists.  It pays the second
  classification pass in full, 1.8 to 1.9 times the wall of the others.
* `buffered` holds every chunk plus the block being copied, and its engine
  RSS at 300k is 2.2 times the payload, matching the `8P + 8Pmax` accounting
  of the plan rather than the 1.1x claim of the superseded draft.  It is the
  fastest cell at six threads on 300k, by 9 percent over `bounded_wave`.
* `bounded_wave` is within noise of `buffered` single-threaded and slightly
  slower at six threads; its RSS lies between the other two and grows with
  the thread count because a wave holds more chunks.  The `Vec` doubling
  slack of the growing blocks is visible: 328 MiB against 261 MiB for the
  same payload.

## Verdict

No candidate is dropped under the stage A rule (another candidate no slower
and no larger on every cell): `two_pass` is the smallest everywhere and the
slowest everywhere, `buffered` is the fastest at six threads on 300k and the
largest everywhere, and `bounded_wave` is faster single-threaded but slower
at six threads than `buffered`.  Stage B (graph and view, degrees 3 and 5,
five repeats, against the 0.8.4 wheel) and the 2M scale run decide the
`speed` mapping between `buffered` and `bounded_wave`; `two_pass` is the
`memory` candidate unless stage B contradicts this.

# Stage B: qualification against the PyPI 0.8.4 wheel

Measured 2026-09-17 on the same machine and governor, through
`benchmarks/bench_pair_qualification.py --repeat 5` with the 0.8.4 wheel in
a plain venv as `--baseline-python`.  Cells are fixture x receiver (graph,
seeded reordered half view) x degree (3, 5) x threads (1, 6); five
interleaved fresh processes per arm and cell.  The baseline arm is
`relationship_pairs(max_degree=D)` on the 0.8.4 wheel with
`PEDIGREE_GRAPH_THREADS` set to the cell's threads.  Engine RSS is `VmHWM`
after the call minus before it in each process.  The raw report is
`benchmarks/reports/pair_qualification.json` (untracked).

Before timing, every emitter's 23 blocks were compared element for element
with the wheel's arrays on all eight (fixture, receiver, degree) cells: equal
everywhere.  All 320 timed runs agreed on every digest.  The single-thread
degree-5 300k cells carry one slow outlier per arm (the machine was
suspended once during repetition 3); the medians are unaffected.

| fixture | receiver | degree | threads | arm | pairs | wall median (s) | wall range | wall / 0.8.4 | engine RSS median (MiB) | RSS / 0.8.4 |
|---|---|---|---|---|---|---|---|---|---|---|
| random_300k | graph | 3 | 1 | bounded_wave | 5,695,235 | 2.769 | 2.727 to 2.785 | 0.918 | 96.2 | 0.196 |
| random_300k | graph | 3 | 1 | buffered | 5,695,235 | 2.796 | 2.699 to 2.880 | 0.927 | 117.2 | 0.239 |
| random_300k | graph | 3 | 1 | python_0.8.4 | 5,695,235 | 3.015 | 3.006 to 3.022 | 1.000 | 490.2 | 1.000 |
| random_300k | graph | 3 | 1 | two_pass | 5,695,235 | 5.112 | 4.942 to 5.206 | 1.696 | 69.3 | 0.141 |
| random_300k | graph | 3 | 6 | bounded_wave | 5,695,235 | 0.783 | 0.754 to 0.824 | 0.335 | 120.5 | 0.171 |
| random_300k | graph | 3 | 6 | buffered | 5,695,235 | 0.771 | 0.759 to 0.840 | 0.330 | 124.1 | 0.176 |
| random_300k | graph | 3 | 6 | python_0.8.4 | 5,695,235 | 2.334 | 2.307 to 2.344 | 1.000 | 706.4 | 1.000 |
| random_300k | graph | 3 | 6 | two_pass | 5,695,235 | 1.281 | 1.246 to 1.284 | 0.549 | 72.5 | 0.103 |
| random_300k | graph | 5 | 1 | bounded_wave | 30,820,305 | 17.152 | 16.380 to 26.521 | 0.787 | 327.8 | 0.152 |
| random_300k | graph | 5 | 1 | buffered | 30,820,305 | 17.569 | 16.920 to 26.820 | 0.806 | 528.8 | 0.245 |
| random_300k | graph | 5 | 1 | python_0.8.4 | 30,820,305 | 21.796 | 21.366 to 24.829 | 1.000 | 2161.6 | 1.000 |
| random_300k | graph | 5 | 1 | two_pass | 30,820,305 | 32.823 | 31.451 to 41.030 | 1.506 | 261.1 | 0.121 |
| random_300k | graph | 5 | 6 | bounded_wave | 30,820,305 | 3.888 | 3.695 to 4.126 | 0.300 | 403.0 | 0.143 |
| random_300k | graph | 5 | 6 | buffered | 30,820,305 | 3.751 | 3.491 to 3.833 | 0.290 | 539.0 | 0.192 |
| random_300k | graph | 5 | 6 | python_0.8.4 | 30,820,305 | 12.942 | 12.314 to 13.008 | 1.000 | 2814.6 | 1.000 |
| random_300k | graph | 5 | 6 | two_pass | 30,820,305 | 6.933 | 6.468 to 7.400 | 0.536 | 264.6 | 0.094 |
| random_300k | view | 3 | 1 | bounded_wave | 1,424,204 | 1.598 | 1.516 to 1.808 | 0.483 | 47.6 | 0.097 |
| random_300k | view | 3 | 1 | buffered | 1,424,204 | 1.625 | 1.520 to 1.778 | 0.492 | 53.1 | 0.108 |
| random_300k | view | 3 | 1 | python_0.8.4 | 1,424,204 | 3.306 | 3.182 to 3.443 | 1.000 | 491.9 | 1.000 |
| random_300k | view | 3 | 1 | two_pass | 1,424,204 | 2.864 | 2.676 to 3.278 | 0.866 | 39.5 | 0.080 |
| random_300k | view | 3 | 6 | bounded_wave | 1,424,204 | 0.494 | 0.488 to 0.510 | 0.188 | 52.1 | 0.074 |
| random_300k | view | 3 | 6 | buffered | 1,424,204 | 0.484 | 0.474 to 0.507 | 0.184 | 57.0 | 0.080 |
| random_300k | view | 3 | 6 | python_0.8.4 | 1,424,204 | 2.626 | 2.513 to 2.661 | 1.000 | 708.2 | 1.000 |
| random_300k | view | 3 | 6 | two_pass | 1,424,204 | 0.779 | 0.719 to 0.829 | 0.297 | 43.2 | 0.061 |
| random_300k | view | 5 | 1 | bounded_wave | 7,701,698 | 9.354 | 8.922 to 9.508 | 0.399 | 133.0 | 0.062 |
| random_300k | view | 5 | 1 | buffered | 7,701,698 | 9.464 | 9.003 to 9.553 | 0.404 | 178.3 | 0.083 |
| random_300k | view | 5 | 1 | python_0.8.4 | 7,701,698 | 23.443 | 22.502 to 23.627 | 1.000 | 2154.4 | 1.000 |
| random_300k | view | 5 | 1 | two_pass | 7,701,698 | 17.763 | 16.800 to 17.853 | 0.758 | 108.6 | 0.050 |
| random_300k | view | 5 | 6 | bounded_wave | 7,701,698 | 2.266 | 2.111 to 2.280 | 0.156 | 156.3 | 0.056 |
| random_300k | view | 5 | 6 | buffered | 7,701,698 | 2.195 | 2.034 to 2.200 | 0.151 | 183.9 | 0.065 |
| random_300k | view | 5 | 6 | python_0.8.4 | 7,701,698 | 14.535 | 13.771 to 14.636 | 1.000 | 2808.7 | 1.000 |
| random_300k | view | 5 | 6 | two_pass | 7,701,698 | 3.863 | 3.596 to 3.963 | 0.266 | 112.0 | 0.040 |
| random_30k | graph | 3 | 1 | bounded_wave | 566,714 | 0.206 | 0.203 to 0.210 | 0.708 | 10.9 | 0.198 |
| random_30k | graph | 3 | 1 | buffered | 566,714 | 0.209 | 0.205 to 0.209 | 0.718 | 11.8 | 0.215 |
| random_30k | graph | 3 | 1 | python_0.8.4 | 566,714 | 0.291 | 0.283 to 0.306 | 1.000 | 55.0 | 1.000 |
| random_30k | graph | 3 | 1 | two_pass | 566,714 | 0.385 | 0.378 to 0.385 | 1.323 | 6.9 | 0.125 |
| random_30k | graph | 3 | 6 | bounded_wave | 566,714 | 0.057 | 0.054 to 0.062 | 0.249 | 14.7 | 0.206 |
| random_30k | graph | 3 | 6 | buffered | 566,714 | 0.059 | 0.055 to 0.061 | 0.258 | 12.6 | 0.177 |
| random_30k | graph | 3 | 6 | python_0.8.4 | 566,714 | 0.229 | 0.228 to 0.231 | 1.000 | 71.3 | 1.000 |
| random_30k | graph | 3 | 6 | two_pass | 566,714 | 0.091 | 0.088 to 0.094 | 0.397 | 7.2 | 0.101 |
| random_30k | graph | 5 | 1 | bounded_wave | 3,067,768 | 1.314 | 1.311 to 1.328 | 0.694 | 37.8 | 0.170 |
| random_30k | graph | 5 | 1 | buffered | 3,067,768 | 1.320 | 1.314 to 1.337 | 0.697 | 49.6 | 0.223 |
| random_30k | graph | 5 | 1 | python_0.8.4 | 3,067,768 | 1.894 | 1.834 to 1.926 | 1.000 | 222.4 | 1.000 |
| random_30k | graph | 5 | 1 | two_pass | 3,067,768 | 2.549 | 2.544 to 2.556 | 1.346 | 26.1 | 0.117 |
| random_30k | graph | 5 | 6 | bounded_wave | 3,067,768 | 0.301 | 0.290 to 0.308 | 0.248 | 44.6 | 0.116 |
| random_30k | graph | 5 | 6 | buffered | 3,067,768 | 0.293 | 0.276 to 0.325 | 0.242 | 50.6 | 0.132 |
| random_30k | graph | 5 | 6 | python_0.8.4 | 3,067,768 | 1.212 | 1.179 to 1.228 | 1.000 | 383.9 | 1.000 |
| random_30k | graph | 5 | 6 | two_pass | 3,067,768 | 0.572 | 0.510 to 0.591 | 0.472 | 26.9 | 0.070 |
| random_30k | view | 3 | 1 | bounded_wave | 142,920 | 0.120 | 0.119 to 0.123 | 0.379 | 5.1 | 0.096 |
| random_30k | view | 3 | 1 | buffered | 142,920 | 0.120 | 0.120 to 0.121 | 0.379 | 5.4 | 0.102 |
| random_30k | view | 3 | 1 | python_0.8.4 | 142,920 | 0.317 | 0.309 to 0.325 | 1.000 | 53.2 | 1.000 |
| random_30k | view | 3 | 1 | two_pass | 142,920 | 0.216 | 0.212 to 0.217 | 0.681 | 4.0 | 0.075 |
| random_30k | view | 3 | 6 | bounded_wave | 142,920 | 0.040 | 0.039 to 0.042 | 0.159 | 6.3 | 0.088 |
| random_30k | view | 3 | 6 | buffered | 142,920 | 0.039 | 0.038 to 0.042 | 0.155 | 5.9 | 0.082 |
| random_30k | view | 3 | 6 | python_0.8.4 | 142,920 | 0.252 | 0.251 to 0.255 | 1.000 | 71.9 | 1.000 |
| random_30k | view | 3 | 6 | two_pass | 142,920 | 0.061 | 0.057 to 0.064 | 0.242 | 4.4 | 0.061 |
| random_30k | view | 5 | 1 | bounded_wave | 771,754 | 0.719 | 0.714 to 0.725 | 0.350 | 14.6 | 0.066 |
| random_30k | view | 5 | 1 | buffered | 771,754 | 0.723 | 0.719 to 0.727 | 0.352 | 15.2 | 0.068 |
| random_30k | view | 5 | 1 | python_0.8.4 | 771,754 | 2.053 | 2.028 to 2.124 | 1.000 | 222.5 | 1.000 |
| random_30k | view | 5 | 1 | two_pass | 771,754 | 1.350 | 1.343 to 1.366 | 0.658 | 10.7 | 0.048 |
| random_30k | view | 5 | 6 | bounded_wave | 771,754 | 0.176 | 0.174 to 0.198 | 0.130 | 18.4 | 0.049 |
| random_30k | view | 5 | 6 | buffered | 771,754 | 0.172 | 0.157 to 0.189 | 0.127 | 17.4 | 0.047 |
| random_30k | view | 5 | 6 | python_0.8.4 | 771,754 | 1.358 | 1.335 to 1.375 | 1.000 | 373.3 | 1.000 |
| random_30k | view | 5 | 6 | two_pass | 771,754 | 0.308 | 0.283 to 0.334 | 0.227 | 11.2 | 0.030 |

## Reading

* Against the wheel, `buffered` and `bounded_wave` are 0.69 to 0.93x the
  wall single-threaded on graph receivers and 0.24 to 0.34x at six threads;
  on views 0.35 to 0.49x and 0.13 to 0.19x.  Their engine memory is 0.05 to
  0.25x the wheel's.  `two_pass` is 1.3 to 1.7x the wheel single-threaded on
  graph receivers, faster everywhere else, and always the smallest.
* `buffered` and `bounded_wave` are within 4 percent of each other on every
  cell, with overlapping ranges: `buffered` ahead at six threads on the
  300k graph (3.75 s against 3.89 s), `bounded_wave` ahead single-threaded
  (17.15 s against 17.57 s).  `bounded_wave` uses 25 to 38 percent less
  engine memory on the degree-5 graph cells (328 against 529 MiB at one
  thread, 403 against 539 MiB at six).
* `two_pass` is 1.5 to 1.9x the wall of the other two and 0.5 to 0.8x
  their memory; at 300k degree 5 it sits 26 MiB above the 235 MiB payload.

# Stage C step 1: `pedsum_2M`, graph, degree 5, six threads, once

Measured 2026-09-17, one fresh process per emitter under `/usr/bin/time -v`,
input dumped from `results/bench_pedsum/pedsum_2M/rep1/pedigree.full.parquet`
in simACE (2,000,000 rows).  212,626,359 pairs, identical blocks and digests
from all three emitters, and the same total the slice 11 count run recorded.
The raw payload is 1,622 MiB.  RSS before the call was 56 MiB.

| emitter | wall (s) | engine RSS (MiB) | RSS / payload | process max RSS (MiB) |
|---|---|---|---|---|
| buffered | 16.3 | 3,721 | 2.29 | 3,777 |
| two_pass | 30.5 | 1,811 | 1.12 | 1,866 |
| bounded_wave | 18.5 | 2,063 | 1.27 | 2,119 |

# Selection

Applying the plan's rule (fastest survivor is `speed`, lowest peak is
`memory`, `bounded_wave` only if it wins one of them):

* `speed` = `buffered`: fastest at six threads on every graph cell of 300k
  and larger (4 percent at 300k, 13 percent at 2M), within noise elsewhere.
  Its peak is about 2.3x the payload, as the `8P + 8Pmax` accounting says.
* `memory` = `two_pass`: the payload plus engine state everywhere (1.12x at
  2M), at 1.6 to 1.9x the wall of `buffered`.
* `bounded_wave` wins neither and is dropped.  For the record it sits at
  1.27x payload and 13 percent slower than `buffered` at 2M, so a
  maintainer who later wants one mode with both properties has this
  measurement.

Extrapolating `two_pass` to the 20M degree-5 count of 2,124,650,324 pairs
gives a 15.8 GiB payload plus engine state, within the 30 GiB box; `buffered`
would need about 36 GiB and cannot serve degree 5 at 20M.  The degree-3
capability gate and the degree-5 attempt remain to be run.

# Stage C step 2: `pedsum_20M`, graph, degree 3, `two_pass`, six threads

Measured 2026-09-17, one fresh process under `/usr/bin/time -v`, input
dumped from `results/bench_pedsum/pedsum_20M/rep1/pedigree.full.parquet`
(20,000,000 rows).  439,217,825 pairs; the 1C count of 47,215,037 equals
the slice 11 count record for this pedigree.

| quantity | value |
|---|---|
| engine wall | 70.9 s (77.9 s process, TSV read included) |
| RSS before the call | 536 MiB |
| process peak RSS | 5,711 MiB |
| engine RSS above the columns | 5,175 MiB |
| raw payload | 3,351 MiB |
| engine RSS / payload | 1.54 |

The 1.8 GiB above the payload is the engine's own state, which the slice
11 count run put at 2.0 GiB single-threaded and 2.86 GiB at twelve threads
on this pedigree; no copy of the output exists.  The gate (degree 3 in
memory mode within 30 GiB) passes with room to spare.  The degree-5
attempt would need the 15.8 GiB payload plus the same engine state, about
18 GiB, and has not been run.

# Stage C step 3: `pedsum_20M`, graph, degree 5, `two_pass`, six threads

Measured 2026-09-17, one fresh process under `/usr/bin/time -v`, same input,
with 18 GiB available before the run.  It completed.

| quantity | value |
|---|---|
| pairs | 2,124,650,324 (equal to the slice 11 count record) |
| engine wall | 291 s (302 s process) |
| process peak RSS | 18.1 GiB |
| raw payload | 15.8 GiB |
| engine RSS above the columns | 17.6 GiB, 1.11x payload |
| major page faults | 0 |

The 20M degree-5 pair query is therefore served in memory mode within the
30 GiB box; `buffered` would need about 36 GiB and cannot.

# Commit 3 gate: fallible engine against the prototype

Measured 2026-09-17 on `random_300k`, graph, degree 5, three interleaved
fresh processes per arm: the prototype binary built from `fff2e04` against
the fallible engine with `bounded_wave` removed and the arms renamed
`speed` (was `buffered`) and `memory` (was `two_pass`).  Digests identical
across all sixteen runs of each thread count.

| threads | arm | wall median (s) | wall range | engine RSS median (MiB) |
|---|---|---|---|---|
| 1 | old buffered | 13.898 | 13.850 to 13.996 | 529.7 |
| 1 | new speed | 14.422 | 14.216 to 14.513 | 529.6 |
| 1 | old two_pass | 26.301 | 26.153 to 26.598 | 261.2 |
| 1 | new memory | 26.916 | 26.915 to 27.235 | 257.6 |
| 6 | old buffered | 3.030 | 3.020 to 3.032 | 536.3 |
| 6 | new speed | 3.099 | 3.094 to 3.126 | 535.9 |
| 6 | old two_pass | 5.533 | 5.530 to 5.557 | 264.8 |
| 6 | new memory | 5.646 | 5.587 to 5.768 | 265.2 |

Wall ratios new over old: speed 1.038 and 1.023, memory 1.023 and 1.020;
RSS ratios 0.986 to 1.002.  The wall cost is the `Result` plumbing and the
per-reservation check through every row set, chunk push, and block; it
stays under the ADR 0007 five percent rule.  A first cut that pushed
element by element in `alloc::extend` measured 1.020 to 1.048 and was
replaced by a bulk path for exact-size iterators before this record.
The benchmark drivers now spell the arms `speed` and `memory`; the stage A
and B tables above keep the prototype names they were measured under.
