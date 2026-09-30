# Relatives per person against pairs plus `np.add.at` (issue #33)

`bench_relatives_per_person.py` measures `relatives_per_person` against the
pair list the two consumers build today, on the `pedsum_2M` and `pedsum_20M`
pedigrees of `bench_relationship_moments.md`. It is acceptance item 4 of the
issue #33 plan (simACE `plans/pedigree-graph-issue-33-relatives-per-person.md`).

**Status: pending, not yet measured.** The script and its smoke test exist;
no number below has been recorded.

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
hold a degree-3 pair list of 439,217,825 pairs (3,351 MiB of payload and
5,711 MiB process peak at six threads, `bench_pair_emitters.md`,
Stage C step 2).

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

## Results: pedsum_2M

pending: not yet measured

| input | strategy | reps | wall (median) | spread | peak RSS (median) | checksum |
|---|---|---:|---:|---:|---:|---|

## Results: pedsum_20M

pending: not yet measured

| input | strategy | reps | wall (median) | spread | peak RSS (median) | checksum |
|---|---|---:|---:|---:|---:|---|
