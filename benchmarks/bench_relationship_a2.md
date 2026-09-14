# Issue #24: `_A2` demand during half-sibling extraction

What a maternal-half-sibling-only relationship query costs when it builds the
2-hop parent adjacency `_A2`, compared with leaving matrix demand to the
selected relationship codes.

## Method

The measurements come from `benchmarks/bench_relationship_a2.py`:

```bash
cd external/pedigree-graph
pixi run python benchmarks/bench_relationship_a2.py \
  --repeat 5 \
  --out benchmarks/reports/relationship_a2.json
```

The harness runs each cell in a fresh single-threaded subprocess. Repetitions
are interleaved, and the timed region uses the kernel's `VmHWM` for peak RSS.
Fixture construction and warm-up happen before the timed region.

Both arms call the public
`relationship_pairs(categories=["MHS"])` endpoint. `eager_mhs` first reads
`graph._A2`, freezing the old behavior in the benchmark. `selected_mhs` does
not touch a matrix itself, so the production selector decides what to build.
Both arms checksum the returned MHS count and must agree.

Before the source change, a five-repetition run confirmed that both arms still
built `_A2`. On `random_300k` their median peak RSS was 226.8 and 227.0 MiB,
and their median wall time was 0.244 and 0.245 seconds. Those absolute values
are not compared with the final sweep because the host rebooted onto another
kernel and used a different configured CPU maximum between sweeps. The final
result is the interleaved same-sweep comparison below.

## Environment

- commit `4dbe708a27` on `main`, working tree dirty
- Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz, 12 logical CPUs at a configured maximum of 2600 MHz, governor `performance`
- 31.0 GiB RAM, kernel 7.1.5-76070105-generic
- Python 3.13.15, pixi lock `598740a05f17c1e1`, harness `e7096039ac31f844`
- every backend pinned to 1 thread
- peak RSS is kernel `VmHWM`, reset through `/proc/self/clear_refs` at region start

The usual `/proc/cpuinfo` probe reported a stale 800 MHz value after reboot.
A separate OpenSSL SHA-256 load pinned to CPU 0 held both
`scaling_cur_freq` and `cpuinfo_avg_freq` at 2.60 GHz for four consecutive
one-second samples. The benchmark therefore ran at the configured maximum,
not under the host's known 800 MHz PROCHOT clamp.

## Results

| input | strategy | reps | wall median [range] | wall ratio | peak RSS median [range] | RSS ratio | checksum |
|---|---|---:|---:|---:|---:|---:|---|
| `random_30k` | force `_A2`, then select MHS | 5 | 0.0316 s [0.0307, 0.0345] | baseline | 130.41 MiB [128.22, 130.62] | baseline | `24406421` |
| `random_30k` | select MHS on demand | 5 | 0.0281 s [0.0269, 0.0297] | 0.889x | 128.80 MiB [128.71, 129.23] | 0.988x | `24406421` |
| `random_300k` | force `_A2`, then select MHS | 5 | 0.3790 s [0.3739, 0.3837] | baseline | 216.84 MiB [216.55, 217.05] | baseline | `2539657552` |
| `random_300k` | select MHS on demand | 5 | 0.3350 s [0.3323, 0.3423] | 0.884x | 203.29 MiB [202.94, 203.68] | 0.938x | `2539657552` |

No gate blocks. At 300k rows, skipping `_A2` removes 13.56 MiB from median
peak RSS and 0.044 seconds from median wall time. Both ranges are disjoint.
The checksums match in every cell.

## Scope

PHS follows the same sibling-group path and has the same absence of adjacency
reads. The regression test covers both MHS and PHS and retains a GP control.
GP remains the first code in the dependency closure that consumes `_A2`, so GP
and every higher selection keep the old pre-trigger timing. The degree-3 1C
calculation still populates the half-1C pair cache before degree-4 H1C reads it.
