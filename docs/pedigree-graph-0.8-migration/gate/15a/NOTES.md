# 15a evidence for slice 15: inbreeding, lineage and the Ne prerequisites on the Rust core (2026-09-23)

Plan: `simACE/plans/pedigree-graph-slice-15-inbreeding-lineage-ne.md`
(locked 2026-09-23). Tree measured: `61406fc` (`v0.9.3-7-g61406fc`), clean.
Baseline: the 0.9.3 PyPI wheel as simACE's pixi env installs it
(`core_version() == "0.9.3"`, `site-packages/pedigree_graph`), run under that
env's interpreter in the same interleaved sweep as the source build.

## Machine state

The CPU was capped at 2.6 GHz for the whole slice: `scaling_max_freq` is
2,600,000 against a `cpuinfo_max_freq` of 4,500,000, the governor and
platform profile read `performance`, and a busy loop reads a flat 2,600 MHz.
The cause was not found and root was not available to lift it. Both arms ran
interleaved under the same cap, so the ratios stand; absolute times are those
of a 2.6 GHz part (the reports record `up to 2600 MHz`).

## Results

Medians of three interleaved reps in fresh single-threaded subprocesses,
peak RSS as kernel `VmHWM` reset at region start. Full tables:
`inbreeding.md`, `distinct_ancestors.md`, `lineage_ne.md`; raw runs in the
matching `.json`. Every cell's checksum is identical across the two arms.

| cell | fixture | wheel | source | wall | RSS |
|---|---|---:|---:|---:|---:|
| inb-60g | `deep_inbred_60g` | 22.1 ms, 144 MiB | 4.29 ms, 59 MiB | 0.19x | 0.41x |
| inb-300k | `random_300k` | 2.73 s, 181 MiB | 0.88 s, 89 MiB | 0.32x | 0.49x |
| inb-536k | `baseline100K/rep1` | 1.56 s, 359 MiB | 0.45 s, 268 MiB | 0.29x | 0.75x |
| da | `closed_w200_g8` | 3.03 ms, 145 MiB | 0.93 ms, 60 MiB | 0.31x | 0.41x |
| da | `closed_w1000_g8` | 17.0 ms, 149 MiB | 5.39 ms, 62 MiB | 0.32x | 0.42x |
| da | `closed_w2000_g8` | 35.1 ms, 152 MiB | 11.2 ms, 63 MiB | 0.32x | 0.42x |
| da | `closed_w128_g8` | 1.78 ms, 145 MiB | 0.56 ms, 60 MiB | 0.31x | 0.41x |
| da | `closed_w128_g16` | 13.2 ms, 147 MiB | 4.06 ms, 61 MiB | 0.31x | 0.41x |
| da | `closed_w128_g32` | 53.4 ms, 153 MiB | 21.8 ms, 62 MiB | 0.41x | 0.40x |
| da | `closed_w128_g60` | 161 ms, 160 MiB | 88.3 ms, 65 MiB | 0.55x | 0.40x |
| da | `random_1k` | 0.53 ms, 144 MiB | 0.24 ms, 60 MiB | 0.46x | 0.41x |
| da | `deep_inbred_60g` | 1.62 ms, 144 MiB | 1.00 ms, 59 MiB | 0.62x | 0.41x |
| da | `random_30k` | 43.6 ms, 156 MiB | 13.5 ms, 65 MiB | 0.31x | 0.42x |
| da | `random_300k` | 455 ms, 248 MiB | 165 ms, 103 MiB | 0.36x | 0.42x |
| da | `baseline100K/rep1` | 255 ms, 393 MiB | 145 ms, 272 MiB | 0.57x | 0.69x |
| ltc-18k | `wf_n2000_g8` | 5.02 ms, 197 MiB | 3.12 ms, 100 MiB | 0.62x | 0.51x |
| idf-45k | `wf_n5000_g8` | 586 ms, 208 MiB | 162 ms, 115 MiB | 0.28x | 0.55x |
| ltc-45k | `wf_n5000_g8` | 10.5 ms, 210 MiB | 9.63 ms, 114 MiB | 0.92x | 0.55x |
| ne-45k | `wf_n5000_g8` | 10.02 s, 2,248 MiB | 9.27 s, 2,151 MiB | 0.93x | 0.96x |
| dp-300k | `random_300k` | 1.41 ms, 173 MiB | 1.83 ms, 88 MiB | **1.30x** | 0.51x |
| dp-536k | `baseline100K/rep1` | 4.27 ms, 344 MiB | 5.07 ms, 253 MiB | **1.19x** | 0.74x |

Exit criterion 1 holds on every cell but the two descendant-path cells,
which the maintainer accepted on 2026-09-23 (below) and the gate records as
`accepted`. Distinct-ancestor peak RSS is at or below the wheel on all twelve
cells, the width and depth series included, so issue #1 can close on this
record (plan D6); it is left open until this record is pushed.

## Three blocks in the first sweep, and what closed them

The first sweep (tree `011258d`) blocked three cells; `superseded/` holds the
JSON of the intermediate sweep described below.

**ne-45k, 12.37 s to 13.74 s.** Split by estimator, all of it was
`ne_group_coancestry`, whose generation-summary DP this slice does not change.
Timed alone on `wf_n5000_g8`, interleaved in fresh processes under one
interpreter, that DP took 9.41 s in 0.9.3 built locally from its tag and
10.51 s in the source build: identical DP source, 12% slower. Building both
with one codegen unit gave 9.01 s for 0.9.3 and 8.76 s for the source, so the
new modules had repartitioned the crate's sixteen codegen units and moved the
DP's inlining. `Cargo.toml`'s release profile now sets `codegen-units = 1`;
ne-45k is 0.93x the wheel.

**dp-300k and dp-536k, 9.0 to 15.0 ms and 20.3 to 31.7 ms.** Three causes,
each measured in isolation:

* The binding took `graph.depth`, which the graph computes lazily (9 ms on the
  536k pedigree), where 0.9.3's facade swept rows it already knew were
  parents-first and never touched depth. The lineage and EqG sweeps now take
  depth as optional, walk graph rows when construction found them
  parents-first, and read (and check) depth only when the rows need sorting.
* The timed region also hashed the result with SHA-256, about 9 ms on 300k
  int64 values, in each arm's own environment; the source env's NumPy 2.5.2
  path ran it 0.26 ms slower than the wheel env's 2.4.6. The harness now
  resolves a `Measurement`'s checksum and facts after the region.
* Re-validating the parent columns the construction had validated cost a
  pass. The bindings now pass construction's `rows_topological` and trust
  its range check (`ParentColumns::validated`); a broken promise can give a
  panic or wrong counts, never undefined behaviour.

That left 1.30x and 1.19x. In fresh-process Rust microbenchmarks on the real
`random_300k` columns, an unchecked reverse sweep is 1.33 ms warm; checking
every add for int64 overflow, as plan D7 requires and the 0.9.3 loop did not
(it wrapped), costs about 0.2 ms even branch-free, and the facade and binding
the rest. The maintainer accepted the two cells with D7 kept (2026-09-23): 0.4
and 0.8 ms on a result memoised once per graph, with peak RSS halved at 300k.

## Correctness

* Differential tests (`tests/test_native_{inbreeding,lineage,generations}.py`)
  hold each binding to the verbatim 0.9.3 kernel on every parity fixture in
  three row orders. Counts are asserted byte-equal; F, EqG and the founder
  means within `rtol 1e-9, atol 1e-12`, and all 144 float cases recorded
  `bit_identical=True`.
* Study pedigrees (`study/comparison.json`, `tools/pg15_study_lineage.py`):
  `dev_mean_n10k/rep1`, `dev_cont_n10k/rep1`, `baseline10K/rep1`,
  `random_30k`, `baseline100K/rep1`; both count vectors, F, EqG and the Ne
  records (`ne_coancestry` excluded at 536k): all 25 products byte-identical
  to the 0.9.3 capture (`all_within_tolerance=True`, `max_abs=0`).
* Full suite at `d91260e` (`61406fc` changes only a benchmark gate): 3,406
  passed, 9 skipped; `cargo test --release`
  120 core tests plus the doc and integration binaries; clippy, rustfmt, ruff,
  ty clean.

## Consumer parity

`byte-parity/baseline-0.9.3/` is the 0.9.3 baseline under the current consumer
locks, taken in commit 1: the five products of 14e unchanged, plus fitACE's
`exports/inbreeding.tsv` and simACE's `effective_size.yaml`. The comparison
run happens after the relock (15c), since the consumers install the wheel.

## Deviations from the plan

* D3/D11/D12: the three parents-first sweeps take `depth` as optional and
  check it only when they sort; construction's validated columns and
  `rows_topological` are trusted rather than re-checked. Inbreeding and the
  founder means always take depth and check it, as planned.
* D6: the sets are `Box<[u32]>` (16 bytes per row) rather than `Vec<i32>`.
* D11's "the core's order costs the same either way" did not hold for the
  millisecond sweeps; see above.
* `codegen-units = 1` in the release profile, not in the plan.
* The harness change moves verification hashing out of the timed region for
  every benchmark that opts in; the three slice 15 benchmarks do.

`tools/pg15_study_lineage.py` was removed in the 1.0 stabilization cleanup (2026-09-24); it was last present at `3adf97d`.
