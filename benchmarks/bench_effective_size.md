# Slice 6c-3 effective-size orchestration

Whether removing the effective-size worker pool (ADR 0007) costs anything.
`estimate_effective_sizes` runs the eight estimators and their lazily built
prerequisites serially; the `compute_all_ne` adapter keeps the 0.7.1 shape,
every prerequisite built eagerly and the formulas dispatched on a
`ThreadPoolExecutor` when `n_threads > 1`.

## Method

Every figure below was produced by `benchmarks/bench_effective_size.py`:

```bash
cd external/pedigree-graph
pixi run python benchmarks/bench_effective_size.py --repeat 5 --out benchmarks/reports/effective_size.json
pixi run python benchmarks/bench_effective_size.py --render benchmarks/reports/effective_size.json
```

Run these from inside the checkout; `pixi run --manifest-path <path>` selects
the environment but leaves the working directory at the umbrella root.

Each cell runs in a fresh subprocess pinned to one thread per backend, the
three arms are interleaved so host drift lands on all of them, and every cell
is repeated five times; the table quotes the median with the observed spread.
Peak RSS is the kernel's `VmHWM`, reset through `/proc/self/clear_refs` at the
start of the timed region, which is the orchestration call plus a checksum
over the eight scalar estimates. Fixture construction is excluded.

The three arms share every formula, since slice 6c-1 routed the adapter
through the evaluators the final surface uses, so the checksum must agree
across arms; it does on both fixtures. Only the orchestration differs.
`adapter_pool4` is the baseline, `estimate_serial` the gated subject under the
family 5% rule, and `adapter_serial` shows the cost of the pool itself.

`benchmarks/.gitignore` excludes `reports/`, so the raw JSON is local-only and
this note carries the environment and spread inline.

## Inputs

Closed-parentage Wright-Fisher pedigrees with dense generation labels, sex,
and birth years, built by `tests/parity/generate_ne_baseline.py` (the slice-6c
parity generator, `random_mating` plus `with_birth_years`), so every estimator
runs, Hill's birth-year branch included. The parity corpus is not used: its
random pedigrees carry external parents, which the founder-based estimators
refuse since slice 6c-2 (`incomplete_parentage`), and the adapter re-raises
that refusal.

A 90,000-row fixture (10,000 per generation) was tried first and dropped: one
cell costs 16 minutes and 21 GiB, almost all of it the coancestry DP, which is
the same on every arm and only hides the orchestration cost this suite is
about.

## Verdict

Both gated cells pass. The serial path is within the noise of the pooled
adapter on wall time and identical on memory:

| input | wall ratio (serial / pool) | subject span | baseline span | peak RSS ratio |
|---|---:|---:|---:|---:|
| `wf_n2000_g8` (18,000 rows) | 0.981 | 64.57 s to 67.93 s | 64.90 s to 66.97 s | 0.9995 |
| `wf_n5000_g8` (45,000 rows) | 1.008 | 285.12 s to 298.91 s | 282.95 s to 287.34 s | 0.9999 |

The pool never bought anything here: the adapter builds F, the founder
means, the Caballero-Toro accumulators, and the streamed kinship summary
before any worker starts, and those prerequisites are the whole cost. The
formulas the workers dispatch are per-cohort reductions over arrays already
in memory. `adapter_serial` and `adapter_pool4` are within 1.5% of each other
on both fixtures, which is the direct measurement of that.

Peak RSS is the same to within 2 MiB across arms because the same
prerequisites are live at the same time on every arm. The lazy memo saves
memory only when an estimator is unselected, which this suite does not
measure; it runs all eight.

## Environment

- commit `75e601fd71` on `v0.8`, working tree dirty (the slice 6c-3 changes)
- Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz, 12 logical CPUs at up to 2600 MHz (performance), 31.0 GiB RAM, kernel 7.0.11-76070011-generic
- Python 3.13.15, pixi lock `07cf6c93c3038a0a`, harness `1ba28ac06978cfd3`
- every backend pinned to 1 thread
- peak RSS is kernel VmHWM, reset via /proc/self/clear_refs at region start

## Results

| input | strategy | reps | wall (median) | spread | peak RSS (median) | checksum |
|---|---|---:|---:|---:|---:|---|
| `wf_n2000_g8` (seed 31, 2000 per generation, 8 generations) | `compute_all_ne(n_threads=4)`, eager prerequisites + pool | 5 | 66.28 s | 3.1% | 1,744 MiB | `4179101552` |
| `wf_n2000_g8` (seed 31, 2000 per generation, 8 generations) | `compute_all_ne(n_threads=1)`, eager prerequisites, serial | 5 | 65.24 s | 3.5% | 1,745 MiB | `4179101552` |
| `wf_n2000_g8` (seed 31, 2000 per generation, 8 generations) | `estimate_effective_sizes`, lazy prerequisites, serial | 5 | 65.00 s | 5.2% | 1,743 MiB | `4179101552` |
| `wf_n5000_g8` (seed 37, 5000 per generation, 8 generations) | `compute_all_ne(n_threads=4)`, eager prerequisites + pool | 5 | 283.25 s | 1.5% | 7,382 MiB | `2192726221` |
| `wf_n5000_g8` (seed 37, 5000 per generation, 8 generations) | `compute_all_ne(n_threads=1)`, eager prerequisites, serial | 5 | 284.22 s | 6.4% | 7,382 MiB | `2192726221` |
| `wf_n5000_g8` (seed 37, 5000 per generation, 8 generations) | `estimate_effective_sizes`, lazy prerequisites, serial | 5 | 285.49 s | 4.8% | 7,381 MiB | `2192726221` |
