# Relatives per person against pairs plus `np.add.at` (issue #33)

`bench_relatives_per_person.py` measures `relatives_per_person` against the
pair list the two consumers build today, on the `pedsum_2M` and `pedsum_20M`
pedigrees of `bench_relationship_moments.md`. It is acceptance item 4 of the
issue #33 plan (simACE `plans/pedigree-graph-issue-33-relatives-per-person.md`).

Measured 2026-09-30 at `bf646dd` on `pedsum_2M` and `pedsum_20M`. Every
arm of a configuration printed one checksum at both sizes, so the engine's
folded counts equal the pair-list kernels' on both pedigrees. At 20M the
engine used 0.50x the pairs arm's peak RSS for pedsum's totals and 0.60x
for epimight's aligned counts, and every engine repetition was faster than
every pairs repetition.

## What each arm computes

Every arm folds the nine registry codes `MO FO FS MZ MHS PHS Av GP 1C` into
the eight EPIMIGHT kinds `PO FS HS mHS pHS Av 1G 1C`
(`PO = MO + FO`, `FS = FS + MZ`, `HS = MHS + PHS`, `mHS = MHS`,
`pHS = PHS`, `Av`, `1G = GP`, `1C`). `PO`, `Av` and `1G` credit only the
junior member of a pair; the other kinds credit both. Each fold is cast to
int32, as the consumers do.

Two configurations:

- `totals` is pedsum's `epimight-input` skeleton (`pedsum/epimight.py`), the
  `relatives` column per kind. The engine arm is one
  `relatives_per_person(categories=<nine codes>)` call and eight
  `RelativesPerPerson.sum` folds. The pairs arm is
  `relationship_pairs(max_degree=3)` and pedsum's `count_total_relatives`
  (`np.add.at` into int64) per kind.
- `epimight` is fitACE_epimight's `create_input` with aligned counts. Two
  synthetic traits (seed 33, prevalence 0.1 and 0.2, onset uniform on
  [0, 80) as float32) and one float64 cutoff uniform on [0, 100) are drawn
  in the arm's untimed setup, so both arms see identical arrays. The engine
  arm passes two threshold columns,
  `trait: (np.where(affected, onset, np.float64(nan)), cutoff)`, which share one cutoff
  array, then folds the total once per kind and each trait column per
  kind. The pairs arm is `relationship_pairs(max_degree=3)` and fitACE's
  `count_total_relatives` and aligned `count_affected_relatives` per trait
  and kind, the loop of `create_input.build_pipeline_input_frame`.

The kernels are copied into the script, not imported, so the benchmark does
not depend on either consumer's checkout.

simACE writes `t_observed` as float32. `np.where(affected, onset, np.nan)`
would keep float32, and the boundary would then copy each column to float64
(320 MB for two columns at 20M rows). A float64 NaN makes `np.where` build
float64 in the allocation it makes anyway, so the boundary borrows the
columns and nothing is copied. Widening float32 to float64 is exact, so the
counts do not change. The epimight migration builds its columns the same
way.

The checksum is SHA-256 over the stacked int32 folds: `[8, rows]` for
`totals`, `[2 traits × 8 kinds × (total, diagnosed), rows]` for `epimight`.
Every arm of one configuration must print the same checksum. That equality
is the correctness check. `--render`, and a sweep run with `--out`, print
`CHECKSUM MISMATCH` and exit 1 when two arms of one configuration disagree
or one arm's checksum changes across repetitions.

| arm | configuration | threads |
|---|---|---:|
| `totals_engine_12t` | `totals` | 12 |
| `totals_pairs_12t` | `totals` | 12 |
| `totals_engine_1t` | `totals` | 1 |
| `epimight_engine_12t` | `epimight` | 12 |
| `epimight_pairs_12t` | `epimight` | 12 |
| `epimight_engine_1t` | `epimight` | 1 |

The default sweep is all six arms on `pedsum_2M`. `pedsum_20M` runs only
through `--only`, at twelve threads. `random_1k` is the parity pedigree,
declared so the script can be run end to end in seconds.

## Methodology

**Clock check.** Do this before every sweep. This host has clamped its
clock three ways: PROCHOT at 800 MHz, `platform_profile=quiet` at a flat
2200 MHz, and `scaling_max_freq` at 2600 MHz with turbo on. Run a busy
loop and read the clock under load, then read the settings:

```bash
timeout 3 sh -c 'while :; do :; done' & sleep 2; grep MHz /proc/cpuinfo; wait
cat /sys/devices/system/cpu/intel_pstate/no_turbo \
    /sys/devices/system/cpu/intel_pstate/max_perf_pct \
    /sys/firmware/acpi/platform_profile
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_max_freq | sort | uniq -c
uptime; free -g
```

Record in this note the MHz under load, `no_turbo`, `max_perf_pct`,
`platform_profile`, `scaling_max_freq`, and the load average. Do not
measure while another job holds memory or cores. The `pairs` arms at 20M
hold a degree-3 pair list of 443,033,800 pairs at `bf646dd` (the
439,217,825 of `bench_pair_emitters.md`, Stage C step 2, predates 0.11.0's
MZ co-twins in sibling groups), about 3.4 GB of int32 payload.

**Repetitions.** Report medians over at least 3 repetitions
(`--repeat 3`). Each repetition is a fresh child process with every backend
pinned to one thread except `PEDIGREE_GRAPH_THREADS`, which the `12t` arms
set to 12. Cells run interleaved: one repetition of every cell, then the
next round. Give the per-repetition wall and peak for every cell, as
`bench_relationship_moments.md` does.

**Peak RSS.** Peak RSS is the kernel's `VmHWM` for the timed region. It is
reset through `/proc/self/clear_refs` at region entry, after the graph is
built and the trait inputs are drawn. The table reports the median peak.
Give the region's baseline (`baseline_rss_mib` in the JSON) too, since
growth above the baseline is what compares with the D3 table: counts of
`4 · rows · 9 · (1 + K)` bytes, engine workspaces, and an `8 · rows` int64
buffer per `sum()`.

## Commands

From this directory's parent (the pedigree-graph checkout):

```bash
SIMACE_RESULTS=/data/Documents/simACE/results \
  pixi run python benchmarks/bench_relatives_per_person.py --repeat 3 \
  --out /tmp/relatives_per_person_2M.json
pixi run python benchmarks/bench_relatives_per_person.py --render /tmp/relatives_per_person_2M.json

SIMACE_RESULTS=/data/Documents/simACE/results \
  pixi run python benchmarks/bench_relatives_per_person.py --repeat 3 \
  --only pedsum_20M/totals_engine_12t pedsum_20M/totals_pairs_12t \
         pedsum_20M/epimight_engine_12t pedsum_20M/epimight_pairs_12t \
  --out /tmp/relatives_per_person_20M.json
pixi run python benchmarks/bench_relatives_per_person.py --render /tmp/relatives_per_person_20M.json
```

The smoke check runs every arm in process on `random_1k` and requires one
checksum per configuration:

```bash
pixi run pytest benchmarks/tests/test_bench_relatives_per_person.py -q
```

## Environment

- commit `bf646dd993` on `main`, clean; native extension built at `10d3b03`
  with `pixi run build-dev` (`core_version` 0.11.0; the env's stale
  `pedigree_graph-0.10.0.dist-info` makes `package_version` read 0.10.0)
- Intel Core i7-9750H, 12 logical CPUs, 31.0 GiB RAM, kernel
  7.1.5-76070105-generic, Python 3.13.15
- `platform_profile` `balanced`, `no_turbo` 0, `max_perf_pct` 100,
  `scaling_max_freq` 2600000 on every core (the hardware maximum is
  4500000; the cap was left in place). All 12 cores read 2600 MHz under
  load before the 2M sweep and before the 20M sweep. After the run, idle
  cores read 800 MHz and the busy-loop core 2600 MHz: that is the powersave
  governor idling, not PROCHOT, and thermal zones read 20 to 38 °C.
- load average 1.95 at the start, with no other test or pipeline job
  running; memory 21 GiB available
- walls are at the 2600 MHz cap, so they compare only within this note

## Results: pedsum_2M

Six arms, three interleaved repetitions, 44,331,896 degree-3 pairs.

| input | strategy | reps | wall (median) | spread | peak RSS (median) | checksum |
|---|---|---:|---:|---:|---:|---|
| `pedsum_2M/rep1` (2,000,000 rows) | totals: `relatives_per_person` + `sum`, 12 threads | 3 | 2.95 s | 5.1% | 832 MiB | `9144566782418774010` |
| `pedsum_2M/rep1` (2,000,000 rows) | totals: `relationship_pairs` + `np.add.at`, 12 threads | 3 | 3.23 s | 10.8% | 1,510 MiB | `9144566782418774010` |
| `pedsum_2M/rep1` (2,000,000 rows) | totals: `relatives_per_person` + `sum`, 1 thread | 3 | 13.70 s | 4.4% | 727 MiB | `9144566782418774010` |
| `pedsum_2M/rep1` (2,000,000 rows) | epimight: `relatives_per_person` + `sum`, K=2, 12 threads | 3 | 3.65 s | 9.0% | 1,157 MiB | `7726069907300045332` |
| `pedsum_2M/rep1` (2,000,000 rows) | epimight: `relationship_pairs` + `np.add.at`, 12 threads | 3 | 4.63 s | 6.9% | 1,779 MiB | `7726069907300045332` |
| `pedsum_2M/rep1` (2,000,000 rows) | epimight: `relatives_per_person` + `sum`, K=2, 1 thread | 3 | 15.51 s | 7.5% | 1,052 MiB | `7726069907300045332` |

Per repetition (wall s, peak MiB): totals engine 12t 3.02/832, 2.95/832,
2.87/832; totals pairs 3.23/1510, 3.00/1511, 3.34/1508; totals engine 1t
13.67/727, 13.70/727, 14.27/727; epimight engine 12t 3.74/1156, 3.41/1157,
3.65/1157; epimight pairs 4.63/1781, 4.42/1779, 4.74/1779; epimight engine
1t 15.51/1052, 14.74/1052, 15.89/1052. Region baselines were 475 MiB
(totals) and 510 MiB (epimight, which also holds the trait inputs).

* Totals wall: 0.91x the pairs arm, but the arms' repetitions overlap
  (engine up to 3.02 s, pairs from 3.00 s), so at 2M the engine is not
  measurably faster. Epimight wall: 0.79x, with no overlap.
* Memory above the baseline (median): 357 against 1,035 MiB for totals,
  647 against 1,269 MiB for epimight.
* Twelve threads against one: 4.6x for totals, 4.2x for epimight.

## Results: pedsum_20M

Four arms at twelve threads, three interleaved repetitions, 443,033,800
degree-3 pairs.

| input | strategy | reps | wall (median) | spread | peak RSS (median) | checksum |
|---|---|---:|---:|---:|---:|---|
| `pedsum_20M/rep1` (20,000,000 rows) | totals: `relatives_per_person` + `sum`, 12 threads | 3 | 29.83 s | 4.6% | 6,751 MiB | `3390612664691627121` |
| `pedsum_20M/rep1` (20,000,000 rows) | totals: `relationship_pairs` + `np.add.at`, 12 threads | 3 | 34.35 s | 12.2% | 13,495 MiB | `3390612664691627121` |
| `pedsum_20M/rep1` (20,000,000 rows) | epimight: `relatives_per_person` + `sum`, K=2, 12 threads | 3 | 38.05 s | 10.3% | 8,770 MiB | `3013490223177829198` |
| `pedsum_20M/rep1` (20,000,000 rows) | epimight: `relationship_pairs` + `np.add.at`, 12 threads | 3 | 53.94 s | 61.8% | 14,707 MiB | `3013490223177829198` |

Per repetition (wall s, peak MiB): totals engine 29.83/6751, 30.85/6711,
29.48/6757; totals pairs 36.48/13508, 34.35/13495, 32.31/13462; epimight
engine 40.57/8765, 38.05/8776, 36.66/8770; epimight pairs 84.60/14707,
53.94/14679, 51.29/14763. The epimight pairs arm's first repetition is the
outlier behind its 61.8% spread; its median and the two others agree.
Region baselines were 3,620 to 3,673 MiB (totals) and 3,960 to 4,014 MiB
(epimight).

* Totals: wall 0.87x and peak 0.50x the pairs arm; above the baseline,
  3,084 against 9,841 MiB (0.31x). Every engine repetition (at most
  30.85 s) beat every pairs repetition (at least 32.31 s).
* Epimight: wall 0.71x and peak 0.60x; above the baseline, 4,761 against
  10,747 MiB (0.44x).
* The engine's growth is made of row-proportional terms. For totals: the
  counts, `20M × 9 × 4` bytes (687 MiB); twelve engine workspaces, about
  9 bytes per graph row each (about 2,060 MiB, ADR 0013); and the int64
  `sum()` buffer and int32 folds the arm keeps for its checksum. The pairs
  arm adds the pair list and its assembly, which scale with the pair count.
  Rows and pairs both grow tenfold between these two pedigrees, so the
  2M-to-20M ratio (8.6x for engine totals) cannot separate the two by
  itself; the breakdown is what shows no pair-proportional term.
* Epimight's engine growth over totals, 1,677 MiB, is the counts growing
  to three columns (1,373 MiB more) plus the two float64 `np.where`
  columns (305 MiB). No float64 copy of the columns appears, as the
  float64 NaN intends.
