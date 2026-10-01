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

## Region baseline for issue #30

Measured 2026-10-01 on `main` at `4931d4ad06` (the 0.11.1 code; the
working tree held only this script and the harness's tree-peak meter),
before core takes over the boundary arithmetic (ADR 0015). Five
interleaved repetitions per cell in fresh pinned processes, each child
alone in a `systemd-run --user --scope`. Every region arm runs the engine
at twelve threads; the derived regions run the call in their untimed setup
and time one step. `c64` is the 64-cell table above, `c16k` the
`(7, 36, 2, 2, 2, 2, 2, 2)` table: 16,128 cells, stride 11, 177,408
accumulators, 10,990,130 pairs.

- Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz, 12 logical CPUs,
  `platform_profile` `balanced`, `no_turbo` 0, `max_perf_pct` 100, but
  `scaling_max_freq` 2600 MHz on every core; a busy loop held every core
  at 2600 MHz in `/proc/cpuinfo` before the sweep. Other sessions ran test
  bursts during it (load average 4.3 at the start), which is the spread
  of the engine cells.
- Python 3.14.7, pixi lock `80bb1e2c78b5c931`, harness `53f0d336036f7432`,
  native extension built by `pixi run build-dev` at this commit
- wall is the region's `perf_counter`; `rss+` is VmHWM above the region's
  starting RSS (`clear_refs`); `tree+` is the scope cgroup's
  `memory.peak` above its usage at region start (per-fd reset). The two
  meters are never compared with each other.

| cell | wall median (ms) | wall range (ms) | rss+ median (MiB) | tree+ median (MiB) | checksum |
|---|---:|---:|---:|---:|---|
| `c64_engine` | 2,668.45 | 2,474.12 to 4,615.67 | 304.2 | 306.4 | `80102806278128935` |
| `c64_mean` | 0.27 | 0.27 to 0.59 | 0.0 | 0.0 | `12053625478470716356` |
| `c64_m2_first` | 0.53 | 0.53 to 0.90 | 0.0 | 0.0 | `14358246025188411139` |
| `c64_comoment` | 0.59 | 0.58 to 0.73 | 0.0 | 0.0 | `2878769951903852980` |
| `c64_cross` | 0.49 | 0.48 to 0.51 | 0.0 | 0.0 | `1599632063308298045` |
| `c64_pearson` | 0.71 | 0.70 to 0.71 | 0.0 | 0.0 | `17250025275507118495` |
| `c64_export` | 6.97 | 5.56 to 9.47 | 0.0 | 0.0 | `15624322561619599958` |
| `c64_sum` | 0.47 | 0.45 to 0.78 | 0.0 | 0.0 | `1333705551228726269` |
| `c64_merge` | 0.76 | 0.68 to 1.92 | 0.2 | 0.2 | `10611122335786047183` |
| `c16k_engine` | 3,200.48 | 3,014.73 to 3,369.24 | 381.3 | 382.6 | `12344696441518330834` |
| `c16k_mean` | 4.51 | 4.45 to 4.69 | 0.0 | 0.0 | `17885692230901075073` |
| `c16k_m2_first` | 16.61 | 16.07 to 18.58 | 1.7 | 1.8 | `1339972897945427418` |
| `c16k_comoment` | 15.97 | 15.70 to 19.25 | 1.1 | 1.2 | `8480704721490216858` |
| `c16k_cross` | 15.90 | 14.87 to 25.38 | 0.3 | 0.2 | `11543833117043843022` |
| `c16k_pearson` | 24.33 | 24.25 to 28.93 | 1.9 | 1.8 | `11036575727330612972` |
| `c16k_export` | 192.90 | 179.31 to 213.49 | 1.9 | 1.8 | `6850338734063697698` |
| `c16k_sum` | 4.62 | 4.48 to 9.04 | 0.1 | 0.0 | `1333705551228726269` |
| `c16k_merge` | 20.03 | 19.45 to 28.92 | 7.8 | 8.8 | `4373962563598087776` |

The checksums are over the exact integers (engine, `sum`, `merge`) or the
float bits (every view), so the revision must reproduce each one. The two
`sum` cells agree because both fold to the same seven per-category totals.
Every derived region but `c16k_merge` stays within 2 MiB of its starting
memory; the baseline process sits at 790 to 850 MiB (graph, parquet
columns and the result), so the gate compares the growth columns, not the
absolute peaks the harness table prints.

### After core takes over the boundary arithmetic (ADR 0015)

The same sweep, same day, same machine and clock (2600 MHz on every core
under a busy loop), at `4931d4ad06` with the ADR 0015 working tree: core
packs, quantizes, folds, merges and derives every float, and the result
holds core's encoded table instead of Python ints. Ratios are new median
over baseline median; the process peak ratio is the whole child's
`ru_maxrss`.

| cell | wall median (ms) | ratio to baseline | wall range (ms) | rss+ median (MiB) | tree+ median (MiB) | process peak ratio | checksum equal |
|---|---:|---:|---:|---:|---:|---:|---|
| `c64_engine` | 2,562.44 | 0.96 | 2,506.72 to 2,906.31 | 310.0 | 312.7 | 1.007 | yes |
| `c64_mean` | 0.07 | 0.26 | 0.07 to 0.16 | 0.0 | 0.0 | 1.018 | yes |
| `c64_m2_first` | 0.13 | 0.25 | 0.12 to 0.28 | 0.0 | 0.0 | 1.018 | yes |
| `c64_comoment` | 0.13 | 0.23 | 0.13 to 0.17 | 0.0 | 0.0 | 1.018 | yes |
| `c64_cross` | 0.17 | 0.34 | 0.12 to 0.19 | 0.0 | 0.0 | 1.018 | yes |
| `c64_pearson` | 0.25 | 0.35 | 0.24 to 0.26 | 0.0 | 0.0 | 1.018 | yes |
| `c64_export` | 1.26 | 0.18 | 1.22 to 1.71 | 0.0 | 0.0 | 1.018 | yes |
| `c64_sum` | 0.22 | 0.46 | 0.20 to 0.51 | 0.0 | 0.0 | 1.018 | yes |
| `c64_merge` | 0.47 | 0.62 | 0.44 to 0.67 | 0.0 | 0.0 | 1.019 | yes |
| `c16k_engine` | 3,108.81 | 0.97 | 2,858.89 to 3,188.13 | 380.7 | 383.1 | 0.999 | yes |
| `c16k_mean` | 1.01 | 0.22 | 0.82 to 2.41 | 0.0 | 0.0 | 1.030 | yes |
| `c16k_m2_first` | 2.08 | 0.13 | 2.06 to 3.05 | 0.0 | 0.0 | 1.028 | yes |
| `c16k_comoment` | 2.24 | 0.14 | 2.19 to 2.95 | 0.0 | 0.0 | 1.028 | yes |
| `c16k_cross` | 3.42 | 0.22 | 2.10 to 3.93 | 0.0 | 0.0 | 1.030 | yes |
| `c16k_pearson` | 7.82 | 0.32 | 7.80 to 21.23 | 0.0 | 0.0 | 1.027 | yes |
| `c16k_export` | 36.36 | 0.19 | 34.17 to 37.51 | 0.0 | 0.0 | 1.027 | yes |
| `c16k_sum` | 2.44 | 0.53 | 2.40 to 3.02 | 0.0 | 0.0 | 1.029 | yes |
| `c16k_merge` | 10.89 | 0.54 | 10.34 to 11.81 | 0.0 | 0.0 | 1.022 | yes |

* Every checksum equals the baseline's: the exact integers and every float
  bit are unchanged.
* Wall: every region is faster, the views 3 to 8 times (no Python-int
  loop), `sum` and `merge` about twice (no object arrays). The engine
  call is within its spread (0.96 and 0.97).
* Memory: no derived region grows any more (it held 0.3 to 8.8 MiB of
  Python ints before). The engine call's growth is +1.9% (`c64`) and
  -0.2% (`c16k`). Whole-process peaks of the derived arms are 1.8% to
  3.0% higher. They start 14 MiB (`c64`) and 24 MiB (`c16k`) above the
  baseline after the same setup call, although the result they hold is
  smaller; the cause is not established. Every figure is inside the 5%
  gate.

The command, from this directory:

```bash
SIMACE_RESULTS=/data/Documents/simACE/results \
  pixi run python benchmarks/bench_relationship_moments.py --repeat 5 --out /tmp/moments_regions.json
```

The default sweep is these eighteen region cells; the four pair-arm cells
above and the 20M cells run through `--only`.

## Reproduce

```bash
SIMACE_RESULTS=/data/Documents/simACE/results \
  pixi run python benchmarks/bench_relationship_moments.py --repeat 3 \
  --only pedsum_2M/moments_1t pedsum_2M/pairs_numpy_1t pedsum_2M/moments_12t pedsum_2M/pairs_numpy_12t \
  --out /tmp/moments.json
pixi run python benchmarks/bench_relationship_moments.py --render /tmp/moments.json
```

`SIMACE_RESULTS` points a checkout at the simACE `results/` directory that
holds `bench_pedsum/pedsum_{2M,20M}/rep1/pedigree.full.parquet`; without it
the script looks under the umbrella checkout this repository sits in.
