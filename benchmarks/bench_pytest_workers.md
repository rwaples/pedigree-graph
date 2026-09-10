# pytest-xdist worker sweep (issue #14)

Which `pytest -n <N> --dist <scheduler>` configuration this repository's test
suite should run under, chosen by measurement rather than by copying simACE's
`-n 6 --dist worksteal`.

## Method

Every figure that will appear below is produced by
`benchmarks/bench_pytest_workers.py`, which is committed so the numbers can be
reproduced and challenged:

```bash
cd external/pedigree-graph
pixi run python benchmarks/bench_pytest_workers.py --list
pixi run python benchmarks/bench_pytest_workers.py --only not_slow/serial not_slow/n2_worksteal not_slow/n3_worksteal \
    not_slow/n4_worksteal not_slow/n6_worksteal not_slow/n8_worksteal not_slow/n6_loadfile not_slow/n6_loadscope \
    not_slow/n6_unpinned --repeat 3 --out benchmarks/reports/pytest_workers_not_slow.json
pixi run python benchmarks/bench_pytest_workers.py --only full/serial full/<winner> \
    --repeat 3 --out benchmarks/reports/pytest_workers_full.json
pixi run python benchmarks/bench_pytest_workers.py --render benchmarks/reports/pytest_workers_not_slow.json
```

Run these from inside the checkout. `pixi run --manifest-path <path>` selects
the environment but leaves the working directory at the umbrella root, so the
relative paths above do not resolve under it.

The suite has two fixtures and nine arms. A fixture is a test selection, not a
pedigree: `not_slow` is `-m "not slow"`, the cheap pass that picks a worker
count, and `full` is the whole suite, which confirms the winner against
`serial`. An arm is one worker configuration. Each repetition is a fresh
harness child that spawns one `python -m pytest` process for the cell and
returns the outcome digest as its checksum. Cells are interleaved
(`RunOrder.INTERLEAVED`) so host drift lands on every arm equally.

The checksum is `checksum_ints` over the passed, failed, skipped, xfailed,
xpassed and error counts parsed from pytest's summary line. Two cells that ran
different tests, or disagreed on an outcome, therefore render different
checksums and are visibly not comparable. A pytest exit status other than 0 or
1 (collection error, internal error, usage error, no tests collected) raises in
the child, so the harness refuses the record and the run cannot enter a median.
Exit 1 is a test failure, which still ran the suite; it enters with
`failed > 0` and a checksum that differs from a clean run.

Every cell, including `serial`, runs with `-v --durations=0 --durations-min=0`
so the per-worker load can be reconstructed: the `[gwN]` verbose lines map each
test to its worker and the durations lines give each test's setup, call and
teardown seconds. The cells are therefore comparable to each other and not to a
bare `pytest` run, whose output is cheaper to produce. Per-run facts recorded
in the JSON:

- `load_spread`, the ratio of the busiest worker's summed test seconds to the
  idlest worker's. Wall time alone cannot tell a scheduling fix from a lucky
  packing; `serial` is 1.0 by construction. A worker that ran no test writes no
  verbose line and is invisible in the output, so `idle_workers` counts
  `workers_requested` (the `-n` value) against `workers_seen`, and
  `load_spread` is `null` whenever a requested worker has zero seconds.
- `test_seconds_total`, the sum of every test's setup, call and teardown
  seconds. `tests/test_relationship_pairs.py` and
  `tests/test_view_relationship_pairs.py` each hold a module-scoped
  `full_results` fixture that builds `relationship_pairs(max_degree=5)` over
  every parity fixture. A per-test scheduler (`worksteal`, `load`) rebuilds a
  module-scoped fixture on every worker that draws from the module, so this
  total rising above `serial` is that rebuild cost made visible, and is why
  `loadfile` and `loadscope` are in the sweep even though `loadscope` lost in
  simACE.
- `unmatched_durations` and `unmatched_verbose`, the two directions in which
  the join between a `[gwN]` verbose line and a durations line can fail. Both
  are zero unless pytest's output format has drifted; a non-zero value means
  `load_spread` is not trustworthy for that run.
- `collected` and `selection_sha256`, the collected test count and a digest of
  the sorted node ids from `pytest --collect-only -q`, so a selection change
  between runs is visible.

Memory is a first-class constraint here, which the issue does not mention. A
single serial pytest process peaked at 7548.7 MiB on this 31 GiB box (measured
by the maintainer on 2026-09-10 at `3fa702b`, outside this benchmark), so every
cell must report RSS and not wall time alone. Read the two RSS figures
carefully. The harness's `peak_rss_mib` is the `VmHWM` of the harness child,
which is a small parent process in this suite and says nothing about the
workers. `ru_maxrss_mib` is the largest single process in the reaped tree,
because `wait4` folds reaped descendants into the child's rusage the way
`/usr/bin/time` would; it understates whole-box memory by roughly the worker
count, since each xdist worker is its own process holding its own fixtures.
Neither is an aggregate, which is why the gate's ratio metric is wall time only
and the memory judgement is made by hand from `ru_maxrss_mib` times `-n`.

## Thread environment per arm

The harness pins seven variables in every child it spawns (`PINNED_ENV`,
`_harness.py`), including `PEDIGREE_GRAPH_THREADS=1`. The pytest process does
not inherit that set unchanged:

- `PYTEST_ADDOPTS` is removed for every arm, so the operator's shell cannot
  add flags that change what a cell measures without appearing in its
  recorded `pytest_args`.
- `PEDIGREE_GRAPH_THREADS` is removed for every arm. Production callers get the
  package default of 1 unless they call `configure_threads` or set the
  variable, and several tests assert that default or `monkeypatch.delenv` it.
  Pinning it would measure a configuration nobody runs and could fight the
  tests.
- Pinned arms re-apply the remaining six (OMP, MKL, OpenBLAS, NumExpr, Numba,
  Polars) so each xdist worker gets one BLAS and one Polars thread.
- `n6_unpinned` applies none of them, so the workers use each library's own
  default and oversubscribe the 12 cores. It is in the sweep to show what the
  pins are worth, not as a candidate.

None of this touches Rayon or Numba parallelism. At `3fa702b` no
`parallel=True` kernel remains in `pedigree_graph/` (ADR 0007's BFS engine was
removed by #7), and the Rust pool is built explicitly from `thread_budget()`
rather than from `RAYON_NUM_THREADS`, so the only thread knob the package
itself reads is the one the tests already exercise.

The rendered environment block says "every backend pinned to 1 thread". That
line describes the harness child and is what `Environment.threads_pinned`
records; the pytest process's actual pins are the per-run `pinned` fact.

## There is no gate, on purpose

`Gate(baseline="serial")` with `gated` empty. This is a sweep for choosing a
configuration; ratios against `serial` are computed and nothing blocks. The
5% rule (`_harness.GATE`) applies to A/B regression gates between two
implementations, which this is not.

## Environment

Both sweeps ran on 2026-09-10 at `3fa702b` with the clock clamp cleared.

| field | value |
|---|---|
| cpu | Intel(R) Core(TM) i7-9750H @ 2.60GHz, 12 logical |
| `cpu_max_mhz` / governor | 2600 / `powersave` |
| memory | 31.0 GiB |
| kernel / python | 7.0.11-76070011-generic / 3.13.15 |
| `threads_pinned` | 1 |
| `pixi_lock_sha256` | `598740a05f17c1e1` |
| `harness_sha256` | `e7096039ac31f844` |
| `suite_sha256` | `16b50c088590eff5` |

Earlier the same day the host was clamped (`platform_profile=quiet`,
`no_turbo=1`, `max_perf_pct=50`, all 12 cores flat at 2200 MHz against a
2.6 GHz base). It was cleared to `balanced`, `no_turbo=0`, `max_perf_pct=100`,
all 12 cores at 2600 MHz under sustained all-core load, and every number here
was taken in that state. Check the clock before running: busy-loop while
reading `/proc/cpuinfo` MHz, and read `intel_pstate/no_turbo`,
`intel_pstate/max_perf_pct` and `/sys/firmware/acpi/platform_profile`. A value
that is flat in both the idle and loaded states is a clamp, and the recorded
`cpu_max_mhz` is what distinguishes a clamped sweep from this one.

## Inputs

| fixture | pytest args | role |
|---|---|---|
| `not_slow` | `-m "not slow"` | picks the worker count |
| `full` | (none) | confirms the winner against `serial` |

Issue #14's scale figures (1802 tests, 9 `slow`, about 800 s serial) are
stale. Measured by the maintainer on 2026-09-10 at `3fa702b` with the clamp
cleared, one serial full run gave 2323 passed and 7 skipped (2330 collected)
in 717.22 s, with 712.2 s of summed test time over the 2188 tests that had
timing entries, 9 tests still marked `slow`. The longest single test,
`tests/test_parity_v071.py::test_random_30k_matches_the_frozen_baseline`,
took 324.51 s, 45.6% of the summed time; it is one `capture()` call over the
30,300-row `random_30k` fixture (`tests/test_parity_v071.py:225-240`) and no
scheduler can split it. The nine `slow` tests together are 548.3 s, 77% of the
total.

The arm list follows from the perfect-packing floor `max(longest_test,
summed / N)`. On the full suite that floor is 356.1 s at `-n 2` and 324.5 s
from `-n 3` upward, so the ceiling is 717 / 324.5 = 2.21x and nothing above
`-n 3` can help. On `not_slow` (summed 163.9 s, longest 23.55 s) the floor is
82.0 s at `-n 2`, 54.6 s at `-n 3`, 41.0 s at `-n 4`, 27.3 s at `-n 6`, and
23.6 s from `-n 8` upward. The worksteal arms `-n 2, 3, 4, 6, 8` bracket both
floors; `-n 12` is provably pointless in both regimes and is not in the sweep.
The benchmark's own `collected` fact and `serial` medians supersede these
figures once a sweep exists.

**The floor prediction was half wrong, and the Results table is what corrected
it.** The 2.21x ceiling held, but "nothing above `-n 3` can help" did not.
A perfect-packing floor assumes the longest task starts first. xdist dispatches
in collection order, `test_parity_v071.py` is collected late, and so the 324.51 s
test starts late and the run waits behind it. Only `-n 6` clears the earlier
files fast enough to start it promptly, which is why `-n 3` and `-n 4` land 140
and 127 seconds above the floor they were predicted to reach.

## Results

`not_slow`, 3 reps per cell, medians with min-max ranges. `ru_maxrss` is the
pytest child; `peak_rss` is the driver's timed region and is not worker memory.

| input | strategy | reps | wall (median) | range | speedup | spread | ru_maxrss |
|---|---|---:|---:|---|---:|---:|---:|
| `not_slow` | `serial` | 3 | 164.5 s | 160.2-166.0 | 1.00x | 1.00 | 990 MiB |
| `not_slow` | `-n 2 worksteal` | 3 | 91.5 s | 87.6-109.9 | 1.80x | 1.05 | 974 MiB |
| `not_slow` | `-n 3 worksteal` | 3 | 66.6 s | 66.5-71.1 | 2.47x | 1.32 | 964 MiB |
| `not_slow` | `-n 4 worksteal` | 3 | 67.0 s | 65.4-68.5 | 2.46x | 2.27 | 961 MiB |
| `not_slow` | **`-n 6 worksteal`** | 3 | **61.6 s** | 60.7-61.8 | **2.67x** | 3.19 | 951 MiB |
| `not_slow` | `-n 8 worksteal` | 3 | 66.1 s | 48.7-71.3 | 2.49x | 5.38 | 960 MiB |
| `not_slow` | `-n 6 loadfile` | 3 | 97.3 s | 96.1-100.4 | 1.69x | 7.20 | 946 MiB |
| `not_slow` | `-n 6 loadscope` | 3 | 96.9 s | 93.8-98.0 | 1.70x | 6.71 | 928 MiB |
| `not_slow` | `-n 6 worksteal, unpinned` | 3 | 62.3 s | 61.8-63.4 | 2.64x | 2.97 | 965 MiB |

`full`, 3 reps per cell.

| input | strategy | reps | wall (median) | range | speedup | spread | ru_maxrss |
|---|---|---:|---:|---|---:|---:|---:|
| `full` | `serial` | 3 | 703.3 s | 684.5-708.0 | 1.00x | 1.00 | 7458 MiB |
| `full` | `-n 3 worksteal` | 3 | 464.9 s | 457.3-467.1 | 1.51x | 3.94 | 7418 MiB |
| `full` | `-n 4 worksteal` | 3 | 451.6 s | 446.3-455.5 | 1.56x | 7.12 | 7477 MiB |
| `full` | **`-n 6 worksteal`** | 3 | **333.7 s** | 331.0-340.8 | **2.11x** | 17.18 | 7380 MiB |
| `full` | `-n 6 worksteal, unpinned` | 3 | 331.9 s | 331.7-341.9 | 2.12x | 17.67 | 7419 MiB |

Every cell in both sweeps ran the identical test count with zero idle workers,
so the cells are comparable to each other. They are not comparable to a bare
`pytest` run, because every cell carries `-v --durations=0 --durations-min=0`
so the per-worker spread can be parsed.

## What the numbers show

**`-n 6 --dist worksteal`, and it is now `[tasks.test]` in `pixi.toml`.** The
full suite goes from 703.3 s to 333.7 s, a 2.11x speedup that reaches 95% of
the 2.21x ceiling. On `not_slow` it is 164.5 s to 61.6 s, 2.67x.

The result is confident by the standard `_harness.GATE` sets for a blocking
comparison, disjoint ranges over at least three reps, even though this sweep
does not block. On `full`, `-n 6` spans 331.0-340.8 s against `-n 4` at
446.3-455.5 s. On `not_slow`, `-n 6` spans 60.7-61.8 s against `-n 3` at
66.5-71.1 s. `-n 8` is the counter-example that shows why the range matters:
its 48.7-71.3 s span overlaps everything, so its single lucky 48.7 s packing is
noise and its median is worse than `-n 6`.

**Coarse packing loses badly here.** `loadfile` at 97.3 s and `loadscope` at
96.9 s are both about 35 s behind worksteal, with spreads above 6.7 against
3.19. Those arms existed because two module-scoped fixtures build
`relationship_pairs(max_degree=5)` over every parity fixture
(`tests/test_relationship_pairs.py:100`, `tests/test_view_relationship_pairs.py:70`)
and per-test schedulers rebuild them once per worker. Stranding a large file on
one worker costs more than those rebuilds save.

**Thread pinning is inert, so `[tasks.test]` sets no thread variables.**
`-n 6` unpinned is 331.9 s against 333.7 s pinned on `full`, and 62.3 s against
61.6 s on `not_slow`; the ranges overlap in both. Issue #14 proposed pinning six
variables per worker by analogy with simACE. That analogy does not hold: simACE
pins because its phenotype kernels are numba `parallel=True`, while here #7
removed the last such kernel in `f743e62` (`grep -rn "parallel=True\|prange"
pedigree_graph/` is empty), the package thread budget already defaults to 1
(`_threads.py`), and rayon builds an explicit pool from that budget rather than
reading `RAYON_NUM_THREADS` (`crates/python/src/lib.rs:282`).

**Memory does not constrain the choice.** `ru_maxrss` is flat across worker
counts, 7380-7477 MiB on `full`, with `-n 6` the lowest of the four. The peak
is set by a single heavy test rather than by accumulation, so it does not
multiply with workers. Across the whole two-hour `full` sweep the minimum
available system RAM was 11673 MiB of 31 GiB, with no OOM.

**The scheduler is not the lever that matters most.** The 2.21x ceiling exists
because one test is 45.6% of the summed time. Cutting
`test_random_30k_matches_the_frozen_baseline` would beat any worker count, and
the per-worker spread of 17.18 at `-n 6` on `full` is that test showing up as
imbalance rather than as a scheduling defect. Attributing its 324.51 s is the
next question, and #12's 264.13 s figure for `approximate_kinship_matrix` on
`random_30k` is the obvious first suspect. It is not the memo of #13, which
already persists across calls.
