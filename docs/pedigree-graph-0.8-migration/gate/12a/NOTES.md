# 12a benchmark record for pedigree-graph 0.9.0 (2026-09-16 to 2026-09-17)

Slice 12 (`plans/pedigree-graph-slice-12-relationship-pairs-v2.md`):
`PedigreeGraph.relationship_pairs` and `PedigreeView.relationship_pairs` move
from the Python/SciPy matrix extractor onto the Rust row-streaming engine,
behind a public `execution="speed" | "memory"` keyword (ADRs 0006, 0007 and
0010 as amended in pedigree-graph). This directory is the evidence the plan's
commit 6 asks for, copied from the runs that selected the two modes. The
narrative with every table is pedigree-graph
`benchmarks/bench_pair_emitters.md` at `96e020e`; this file carries the
headline numbers and the raw outputs.

All runs on the 12-core, 30 GiB workstation (i7-9750H), clock checked
before each session (about 2.2 GHz under load, not the 800 MHz clamp seen
in 11a). Six-thread cells use `PEDIGREE_GRAPH_THREADS=6`; the Rust arms run
through the `pgr-bench-pairs` binary in a fresh process per measurement.

## Files

- `pair_emitters_screening.{json,md}`: stage A, three candidate emitters
  (`buffered`, `two_pass`, `bounded_wave`), `random_30k` and `random_300k`,
  graph, degree 5, one and six threads, three interleaved repeats.
- `pair_qualification.{json,md}`: stage B, the same emitters against the
  PyPI 0.8.4 wheel in a plain venv, fixture x receiver (graph, seeded
  reordered half view) x degree (3, 5) x threads (1, 6), five interleaved
  repeats, 320 timed runs.
- `scale/`: stage C, `pedsum_2M` (three emitters) and `pedsum_20M`
  (`two_pass`, degrees 3 and 5) under `/usr/bin/time -v`; the `.json` is
  the binary's report (wall, RSS, per-code counts and digests), the `.err`
  the `time` output, the `.sh` the exact invocation.
- `commit3-ab/`: the commit 3 gate, prototype binary at `fff2e04` against
  the fallible engine, `random_300k` graph degree 5, three interleaved
  repeats per arm and thread count.

## Parity before timing

Every emitter's 23 blocks matched the Python matrix oracle
(`relationship_pairs(max_degree=5)` on the 0.8.4 wheel) element for element
on all eight (fixture, receiver, degree) cells of stage B, and the count and
digest checks passed on `random_1k` and `random_30k` in stage A. Every timed
run in stages A, B and C and in the commit 3 gate produced the same digests
across emitters and thread counts. The 2M and 20M totals equal the slice 11
count records for those pedigrees (212,626,359; 439,217,825 at degree 3 and
2,124,650,324 at degree 5).

## Stage B headline (medians, wall and engine RSS relative to the 0.8.4 wheel)

| cell | buffered wall | two_pass wall | buffered RSS | two_pass RSS |
|---|---|---|---|---|
| random_300k graph degree 5, 1 thread | 0.81 | 1.51 | 0.25 | 0.12 |
| random_300k graph degree 5, 6 threads | 0.29 | 0.54 | 0.19 | 0.09 |
| random_300k view degree 5, 1 thread | 0.40 | 0.76 | 0.08 | 0.05 |
| random_300k view degree 5, 6 threads | 0.15 | 0.27 | 0.07 | 0.04 |
| random_30k graph degree 5, 1 thread | 0.70 | 1.35 | 0.22 | 0.12 |
| random_30k graph degree 5, 6 threads | 0.24 | 0.47 | 0.13 | 0.07 |

The full 64-row table with `bounded_wave` and the degree-3 cells is in
`pair_qualification.md`. `two_pass` single-threaded on graph receivers is the
only place a Rust arm is slower than the wheel (1.3 to 1.7x); everywhere else
both arms are faster, and engine memory is 0.03 to 0.25x the wheel's.

## Stage C: scale

`pedsum_2M`, graph, degree 5, six threads, once per emitter:

| emitter | wall (s) | process max RSS (MiB) | RSS / payload |
|---|---|---|---|
| buffered | 16.3 | 3,777 | 2.29 |
| two_pass | 30.5 | 1,866 | 1.12 |
| bounded_wave | 18.5 | 2,119 | 1.27 |

`pedsum_20M`, graph, `two_pass`, six threads, once per degree:

| degree | pairs | engine wall (s) | process peak RSS | major faults |
|---|---|---|---|---|
| 3 | 439,217,825 | 70.9 | 5,711 MiB | 0 |
| 5 | 2,124,650,324 | 291 | 18,572 MiB (18.1 GiB) | 0 |

The degree-3 run is the plan's capability gate (memory mode within the 30 GiB
box): passed. The degree-5 run is the attempt the plan left optional: it
completed at 1.11x the 15.8 GiB payload. `buffered` was not attempted at this
size; extrapolating its 2M ratio of 2.29 puts it near 36 GiB, which is an
estimate rather than a measurement.

## Selection

By the plan's rule (fastest survivor is `speed`, lowest peak is `memory`,
`bounded_wave` kept only if it wins one): `speed` = `buffered` (fastest at
six threads on every graph cell of 300k and larger, 13 percent ahead at 2M),
`memory` = `two_pass` (payload plus engine state everywhere, 1.6 to 1.9x the
wall of `buffered`). `bounded_wave` won neither and was deleted in commit 3.

## Commit 3 gate: fallible engine

Wall ratios of the fallible engine over the `fff2e04` prototype on
`random_300k` graph degree 5: `speed` 1.038 (1 thread) and 1.023 (6 threads),
`memory` 1.023 and 1.020; RSS ratios 0.986 to 1.002. Under the ADR 0007 five
percent rule. A first cut that pushed element by element measured 1.020 to
1.048 and was replaced by a bulk path for exact-size iterators before this
record (`commit3-ab/ab.jsonl` is the final run).

## Not in this record

The wheel-site gate across the 13 consumer units, the publish handoff, the
consumer relocks and their byte-compares are commit 8 of the plan and will be
recorded under `12b/` once 0.9.0 is on PyPI.
