# Distinct ancestor count scaling

Issue #1 replaced `PedigreeGraph.distinct_ancestor_counts()`'s sparse boolean
transitive closure with one retiring Numba DP. This benchmark freezes the
removed closure as its baseline and compares the production DP on controlled
width and depth series plus the repository's standard random fixtures.

## Slice 15 (0.9.4): the sweep on the Rust core

The retiring DP moved to the core with one exactly sized set per live row,
and this driver's baseline arm is now the 0.9.3 wheel instead of the frozen
closure. On all twelve fixtures the core measured 0.31x to 0.62x the wheel's
median wall and 0.40x to 0.69x its peak RSS, counts byte-identical
(`gate/15a/distinct_ancestors.md`, `gate/15a/NOTES.md`). The sections below
record the closure-to-Numba change of 2026-09-16.

## Method

Every result was produced by the committed driver:

```bash
cd external/pedigree-graph
pixi run python benchmarks/bench_distinct_ancestors.py \
  --repeat 5 --out benchmarks/reports/distinct-ancestors-retiring.json
pixi run python benchmarks/bench_distinct_ancestors.py \
  --render benchmarks/reports/distinct-ancestors-retiring.json
```

The shared harness runs every cell in a fresh process, interleaves the two arms,
and pins each numerical backend to one thread. The timed region computes all
counts, checksums the complete int32 result, and calculates output summaries.
Peak RSS is the kernel's `VmHWM`, reset at timed-region entry. Fixture
construction and each arm's warm-up are excluded. Whole-process peak RSS still
includes the runtime loaded by that warm-up.

The baseline freezes the deleted SciPy implementation's operations in the
benchmark. It repeatedly multiplies and combines a sparse parent closure. The
production arm makes one topological pass. Each live row owns a sorted closed
ancestor set, meaning its strict ancestors plus itself. The arm merges parent
sets in two passes, first to size the union and then to write it, and returns a
row's power-of-two slot after its last represented child.

`total_ancestor_links` is the sum of the returned counts and the number of
entries in the baseline's final closure. Candidate telemetry reports peak live
slot capacity, pool high-water, final pool allocation, parent-set entries
scanned, and reused slots.

All checksums agreed between arms on every fixture and repetition.

## Inputs

The controlled width series uses closed parentage, eight generated generations,
and equal founder and per-generation counts of 200, 1,000, and 2,000. These
fixtures contain 1,800, 9,000, and 18,000 rows.

The controlled depth series keeps 128 founders and 128 rows per generated
generation, then varies generated depth over 8, 16, 32, and 60. These fixtures
contain 1,152 to 7,808 rows.

The suite also runs `random_1k`, `deep_inbred_60g`, `random_30k`, and
`random_300k` from the shared parity generators.

## CPU results

| input | rows | closure wall | retiring wall | speedup |
|---|---:|---:|---:|---:|
| `closed_w200_g8` | 1,800 | 0.017 s | 0.003 s | 5.7x |
| `closed_w1000_g8` | 9,000 | 0.091 s | 0.017 s | 5.4x |
| `closed_w2000_g8` | 18,000 | 0.178 s | 0.035 s | 5.1x |
| `closed_w128_g8` | 1,152 | 0.011 s | 0.002 s | 5.9x |
| `closed_w128_g16` | 2,176 | 0.254 s | 0.013 s | 19.3x |
| `closed_w128_g32` | 4,224 | 3.60 s | 0.052 s | 68.5x |
| `closed_w128_g60` | 7,808 | 31.68 s | 0.157 s | 202x |
| `random_1k` | 1,020 | 0.004 s | 0.001 s | 7.2x |
| `deep_inbred_60g` | 728 | 0.217 s | 0.002 s | 135x |
| `random_30k` | 30,300 | 0.257 s | 0.043 s | 5.9x |
| `random_300k` | 303,000 | 3.42 s | 0.455 s | 7.5x |

The DP is faster on every fixture. Its advantage grows with depth because it
merges each row once instead of multiplying the complete accumulated closure
once per additional ancestral step. The largest wall-time spread was 10.6% on
the 35-millisecond `closed_w2000_g8` DP cell. The 60-generation closure and DP
spreads were 0.7% and 1.5%.

## RAM results

| input | closure peak | retiring peak | closure timed growth | retiring timed growth |
|---|---:|---:|---:|---:|
| `closed_w200_g8` | 128 MiB | 168 MiB | 3.3 MiB | 0.8 MiB |
| `closed_w1000_g8` | 144 MiB | 171 MiB | 17.6 MiB | 3.0 MiB |
| `closed_w2000_g8` | 165 MiB | 174 MiB | 37.0 MiB | 5.0 MiB |
| `closed_w128_g8` | 127 MiB | 167 MiB | 1.9 MiB | 0.4 MiB |
| `closed_w128_g16` | 146 MiB | 170 MiB | 20.4 MiB | 2.9 MiB |
| `closed_w128_g32` | 245 MiB | 176 MiB | 119.8 MiB | 8.6 MiB |
| `closed_w128_g60` | 532 MiB | 183 MiB | 405.6 MiB | 14.8 MiB |
| `random_1k` | 125 MiB | 167 MiB | 0.6 MiB | 0.1 MiB |
| `deep_inbred_60g` | 130 MiB | 167 MiB | 4.8 MiB | 0.2 MiB |
| `random_30k` | 173 MiB | 178 MiB | 43.7 MiB | 7.6 MiB |
| `random_300k` | 586 MiB | 269 MiB | 434.2 MiB | 76.2 MiB |

The DP's timed allocation is lower on every fixture. A fresh DP process starts
about 42 MiB higher because its warm-up loads the Numba runtime. That fixed cost
makes total peak RSS worse on small fixtures. The gate blocked seven such cells.
The one-method design accepts those blocks rather than adding a size or depth
dispatch threshold. At 30,300 rows the total peak difference is 5 MiB; at
303,000 rows the DP saves 317 MiB.

The largest allocator states were:

| input | final closure links | peak live capacity | pool high-water | reused slots |
|---|---:|---:|---:|---:|
| `closed_w128_g60` | 19,576,085 | 1,236,992 | 2,414,511 | 4,986 |
| `random_30k` | 2,137,260 | 712,897 | 1,159,271 | 3,729 |
| `random_300k` | 21,893,616 | 7,187,744 | 11,663,888 | 37,134 |

On the 60-generation fixture, the pool high-water is 12% of the old closure.
On `random_300k` it is 53%. Exact-capacity reuse does not promise one-generation
memory: parents with late last children stay live, and unused bucket sizes can
leave fragmentation. It does remove cumulative storage on the tested shapes.

## Decision

Ship the retiring Numba DP as the sole implementation. It preserves every
checksum, improves CPU time from 5.1x to 202x, and cuts peak RSS on the two
large-memory fixtures. The accepted tradeoff is approximately 42 MiB of fixed
Numba runtime RSS in a fresh process, which dominates small pedigrees whose
closure previously allocated only a few MiB.

## Environment

- commit `e4365e5450` on `main`, working tree dirty with the issue #1 changes
- Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz, 12 logical CPUs at up to 2600 MHz, performance governor, 31.0 GiB RAM, kernel 7.1.5-76070105-generic
- Python 3.13.15, pixi lock `598740a05f17c1e1`, harness `e7096039ac31f844`
- every backend pinned to one thread
- peak RSS is kernel `VmHWM`, reset through `/proc/self/clear_refs` at timed-region entry
