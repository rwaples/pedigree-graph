# Relationship moments against pairs plus NumPy (ADR 0013, issue #28)

`bench_relationship_moments.py`, measured 2026-09-30 on the `pedsum_2M`
pedigree of `benchmarks/relationship_counts_rust.md` (2,000,000 rows, eight
recorded generations), three interleaved repetitions per cell in fresh
pinned processes. Both arms compute the same table over the seven simACE
analysis categories (`MZ FS MO FO MHS PHS 1C`, so the engine runs at degree
3): per category and per cell of (first member's generation and sex, second
member's sex, same household), the pair count and the sums, sums of squares
and diagonal cross sums of `liability1` and `liability2`; 64 cells,
10,990,130 pairs. `moments` is one `relationship_moments` pass;
`pairs_numpy` is `relationship_pairs` followed by `np.bincount` folds. The
checksum is over the per-cell pair counts and agrees between the arms.

- commit `9c99f285f7` on `relationship-moments`, working tree dirty (the
  uncommitted 0.11.0 work: issues #29 and #28)
- Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz, 12 logical CPUs, 31.0 GiB RAM,
  kernel 7.1.5-76070105-generic; `platform_profile` was `quiet`, which
  clamps every core to 2200 MHz (checked in `/proc/cpuinfo` under load), so
  absolute walls are above the 2.6 GHz base-clock figures elsewhere in this
  directory and the spreads are wide
- Python 3.13.15, pixi lock `72261700f47a04e2`, harness `88facd9acb07679f`
- every backend pinned to 1 thread; the `12t` arms set
  `PEDIGREE_GRAPH_THREADS=12`
- peak RSS is kernel VmHWM, reset via /proc/self/clear_refs at region
  start; the process held the graph and the parquet columns at 538 MiB
  before every region

| input | strategy | reps | wall (median) | spread | peak RSS (median) | checksum |
|---|---|---:|---:|---:|---:|---|
| `pedsum_2M/rep1` (2,000,000 rows) | `relationship_moments`, 1 thread | 3 | 15.63 s | 16.4% | 756 MiB | `13505313454495710835` |
| `pedsum_2M/rep1` (2,000,000 rows) | `relationship_pairs` + `np.bincount`, 1 thread | 3 | 16.94 s | 9.3% | 1,133 MiB | `13505313454495710835` |
| `pedsum_2M/rep1` (2,000,000 rows) | `relationship_moments`, 12 threads | 3 | 3.93 s | 26.2% | 843 MiB | `13505313454495710835` |
| `pedsum_2M/rep1` (2,000,000 rows) | `relationship_pairs` + `np.bincount`, 12 threads | 3 | 5.12 s | 21.6% | 1,239 MiB | `13505313454495710835` |

Per repetition (wall s, peak MiB): moments 1t 17.8/756, 15.24/756,
15.63/756; pairs 1t 16.94/1133, 15.89/1133, 17.47/1133; moments 12t
3.58/843, 4.61/866, 3.93/843; pairs 12t 5.12/1239, 5.94/1238, 4.83/1239.
Two earlier sweeps of the same script (before its checksum layout was
aligned between the arms) gave medians of 15.69 / 15.87 / 3.67 / 5.44 s and
17.66 / 16.94 / 3.82 / 5.37 s in the same order, so the ratios below hold
across three sweeps even though single cells move by up to a quarter under
the clock clamp.

## Reading

* Memory above the 538 MiB baseline: 218 MiB for the moments pass at one
  thread and 305 MiB at twelve, against 595 MiB and 701 MiB for the pair
  list plus its NumPy folds. The moments accumulators themselves are
  140 KB (one lane) and 1.0 MB (twelve lanes) by the budget estimate the
  call reports; the rest is the engine's own state (`relationship_counts_rust.md`
  puts it at about 0.3 GiB on this pedigree) and the quantized value
  columns (2,000,000 × 2 × int64, 32 MiB).
* Wall: the pass is the classification. At one thread the moments arm is
  within the spread of the pair arm (the pair list is cheap to write once
  classified); at twelve threads it is 0.77x, since the NumPy folds are
  serial and the pair blocks have to be assembled. The lane count reported
  was 12 at 12 threads and 1 at one, under the default 1 GiB budget.
* Both arms scale the same way with threads because both are the engine
  pass; the moments arm has no result to assemble.

## pedsum_20M

Measured 2026-09-30 at twelve threads only, three interleaved repetitions
per cell in fresh pinned processes, with a third arm: `counts_12t` is
`relationship_counts` over the same seven categories, the engine pass with
no sink work, which issue #28 set as the wall target. Same table, 64 cells,
109,794,669 pairs in every arm (the seven categories are a quarter of the
439,217,825-pair degree-3 set, `bench_pair_emitters.md`).

- commit `39bd76448b` on `main`, working tree dirty (this note, the
  `counts_12t` arm, and the harness change that lets `--only` pick cells
  outside the default sweep); native extension rebuilt at that commit with
  `pixi run build-dev` (`core_version` 0.11.0; the env's stale
  `pedigree_graph-0.10.0.dist-info` makes `package_version` read 0.10.0)
- same machine; `platform_profile` `balanced`, all 12 cores held 2600 MHz
  under a busy loop checked in `/proc/cpuinfo` before the run, so these
  walls are not comparable with the clamped 2M walls above
- peak RSS is kernel VmHWM as above; the process held the graph and the
  parquet columns at 3,947 to 4,658 MiB before each region

| input | strategy | reps | wall (median) | spread | peak RSS (median) | checksum |
|---|---|---:|---:|---:|---:|---|
| `pedsum_20M/rep1` (20,000,000 rows) | `relationship_moments`, 12 threads | 3 | 32.77 s | 10.4% | 7,637 MiB | `5868940489448471493` |
| `pedsum_20M/rep1` (20,000,000 rows) | `relationship_pairs` + `np.bincount`, 12 threads | 3 | 55.67 s | 9.1% | 9,660 MiB | `5868940489448471493` |
| `pedsum_20M/rep1` (20,000,000 rows) | `relationship_counts`, 12 threads | 3 | 32.14 s | 8.4% | 6,815 MiB | `3905718629` |

Per repetition (wall s, peak MiB): moments 36.14/6977, 32.77/7637,
32.74/7683; pairs 55.67/9660, 57.27/9654, 52.19/9692; counts 30.00/6343,
32.69/6815, 32.14/7051. The counts arm's checksum is over per-category
totals, so it differs from the other two by construction; its pair total
matches.

* Wall: moments is within 2% of the count pass (32.77 s against 32.14 s,
  inside either arm's spread), which meets #28's target, and 0.59x the
  pair list plus NumPy folds.
* Memory above each region's baseline (median): 3,027 MiB for moments,
  2,396 MiB for counts, 5,047 MiB for the pair arm. The accumulators are
  1.7 MiB across 12 lanes by the call's own estimate; the rest of the
  630 MiB over counts is the quantized value columns (20,000,000 × 2 ×
  int64, 305 MiB) and the float64 widening of one column at a time.

The command, from this directory:

```bash
SIMACE_RESULTS=/data/Documents/simACE/results \
  pixi run python benchmarks/bench_relationship_moments.py --repeat 3 \
  --only pedsum_20M/moments_12t pedsum_20M/pairs_numpy_12t pedsum_20M/counts_12t \
  --out /tmp/moments_20M.json
```

## Reproduce

```bash
SIMACE_RESULTS=/data/Documents/simACE/results \
  pixi run python benchmarks/bench_relationship_moments.py --repeat 3 --out /tmp/moments.json
pixi run python benchmarks/bench_relationship_moments.py --render /tmp/moments.json
```

`SIMACE_RESULTS` points a checkout at the simACE `results/` directory that
holds `bench_pedsum/pedsum_{2M,20M}/rep1/pedigree.full.parquet`; without it
the script looks under the umbrella checkout this repository sits in.
