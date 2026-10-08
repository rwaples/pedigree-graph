# Benchmarks

Every benchmark declares a `Suite` in `_harness.py` terms and calls `main`.
The harness owns process spawning, thread pinning, memory measurement,
repetition, aggregation, gating and rendering, so a benchmark file contains
only its inputs and its timed operations.

```bash
python benchmarks/matrix_exactification.py --repeat 5 --out benchmarks/reports/sweep.json
python benchmarks/matrix_exactification.py --render benchmarks/reports/sweep.json
python benchmarks/bench_estimate_counts.py --repeat 5 --out benchmarks/reports/counts.json
```

## The contract

- **Measure through the harness.** Do not write a driver into `/tmp`. A number
  whose method is not committed cannot be re-derived, and
  `benchmarks/tests/test_benchmark_contract.py` fails a tracked note that cites one.
- **Never sample RSS from a Python thread.** A native kernel that holds the
  GIL for its whole call starves the sampler; the measurement that showed it
  was a `@numba.njit` kernel. Measured against a 2.77 s
  kernel it ran for 1 of an expected 553 ticks and reported 358 MiB against a
  true 759 MiB. Use `PeakRss`, which reads the kernel's `VmHWM`.
- **Fixture parameters come from `tests/parity/pedigrees.py`.** `parity_fixture`
  takes no parameters argument, so a benchmark cannot re-declare a seed and
  drift from the parity definitions.
- **Every arm returns a checksum.** `Measurement.checksum` is required, so a
  benchmark proves correctness rather than only speed. Agreement across arms is
  what shows two implementations compute the same values.
- **The 5% rule lives in `_harness.GATE`,** once. Blocking needs *confidence*
  as ADR 0007 states it: at least three repetitions and disjoint ranges.
  Anything less is `INCONCLUSIVE`, not a pass and not a block.

## Gating one build against another

`--baseline-python <python>` turns any suite into a gate of this checkout's
build against the build another interpreter imports, usually a second
worktree's `.pixi/envs/default/bin/python` (`_harness.build_pair`). Every arm
that runs in this env gains a `@base` twin with the same fixtures, setup,
thread environment and timed region; arms pinned to another interpreter (a
released wheel) are dropped. The twins interleave, need five repetitions to
pass or block, and must print one checksum. `--only` names candidate cells
and their twins come along:

```bash
export SIMACE_RESULTS=/data/Documents/simACE/results   # study and pedsum fixtures, from a worktree
python benchmarks/bench_kinship_matrix.py --baseline-python <pg-base>/.pixi/envs/default/bin/python \
    --only mkg-30k/source mkg-536k/source --out benchmarks/reports/mkg-gate.json
```

Every run records the build it imported (`package_facts`: the interpreter,
`_native.__file__`, its SHA-256, the source tree's `git describe`), and the
rendered note lists them per arm. A gated arm that imported its baseline's
native library, or an arm that saw two builds, is a **wrong build** and exits 1,
as do a block and a checksum mismatch.

`bench_pair_emitters.py` times a Rust binary rather than a suite, so it takes
`--baseline-binary <other checkout>/target/release/pgr-bench-pairs` (and
`--binary` for the candidate, default this checkout's) and gates through the
same `_harness.compare` rule.

## Two memory scopes, on purpose

`PeakRss` measures one region inside a process, which is how cost is attributed
to a phase. `ru_maxrss` from the child covers the whole process including
fixture construction, which is what an A/B comparison of two arms wants. Both
are recorded per run. They are not interchangeable.

## Choosing an order

`RunOrder.INTERLEAVED` runs one repetition of every cell before the next round,
so host drift lands on both arms equally. Use it for anything gated.
`RunOrder.GROUPED` finishes a cell before starting the next, so an interrupted
sweep still has complete cells. Use it when a single cell runs for hours.

## Not a benchmark

`profile_pedigree_graph.py` is an exploratory profiler from before the Rust
core: it attributes cost across scipy, numba and Python layers in one process,
and the first two are no longer in the package, so its layer split is a record
of that era rather than a current attribution. It has no baseline and no gate,
so it borrows `PeakRss` and nothing else.

`threshold_structure/` is a frozen investigation, kept as evidence.
