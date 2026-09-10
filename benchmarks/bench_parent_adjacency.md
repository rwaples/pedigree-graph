# Issue #18: reaching the parent adjacency

What it costs to build a graph and reach `_A`, before and after the eager
`_Am`/`_Af` pair was replaced by one lazily assembled COO.

Two claims are measured separately, because they are different claims.
Construction got cheaper for everyone, including callers who never look at a
relationship path. Reaching `_A` got cheaper too, but by much less, and only
after a second fix that the first measurement is what found.

## Method

Every figure below was produced by `benchmarks/bench_parent_adjacency.py`,
which is committed so these numbers can be reproduced and challenged:

```bash
cd external/pedigree-graph
pixi run python benchmarks/bench_parent_adjacency.py --repeat 5 --out benchmarks/reports/parent_adjacency.json
pixi run python benchmarks/bench_parent_adjacency.py --render benchmarks/reports/parent_adjacency.json
```

Run these from inside the checkout. `pixi run --manifest-path <path>` selects
the environment but leaves the working directory at the umbrella root, so the
relative paths above do not resolve under it.

Each cell runs in a fresh single-threaded subprocess, five times, interleaved,
and the tables quote the median with the observed spread. Peak RSS is the
kernel's `VmHWM`, reset through `/proc/self/clear_refs` at the start of the
timed region. Fixture construction happens before the region; each arm then
builds its own graph from that fixture's columns inside the region, so the
numbers cover construction and whatever the arm does next.

The suite has three arms. `eager` is `PedigreeGraph._build_parent_csr` as it
stood at `93f6c97`, copied verbatim into the benchmark, then summed. `lazy`
reads `pg._A`. Both end holding the same matrix and must agree on its checksum,
which is a SHA-256 of `indptr`, `indices` and `data` with their dtypes, in
layout order and without canonicalising first. An order-insensitive digest
would let a differently sorted matrix match, and canonicity is load-bearing:
`_streaming_counter.py` reads child counts straight off `_A.tocsc().indptr`.

`construct` builds a graph and stops. Nothing inside one commit can A/B that,
so it is the cell compared across commits below, and its checksum covers the
edge structure rather than `_A` because it deliberately never builds one.

`benchmarks/.gitignore` excludes `reports/`, so the raw JSON is local-only and
this note carries the environment and spread inline.

## Environment

- commit `93f6c978f4` on `main`, working tree dirty
- Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz, 12 logical CPUs at up to 4500 MHz (performance), 31.0 GiB RAM, kernel 7.0.11-76070011-generic
- Python 3.13.15, pixi lock `598740a05f17c1e1`, harness `e7096039ac31f844`
- every backend pinned to 1 thread
- peak RSS is kernel VmHWM, reset via /proc/self/clear_refs at region start

Both sweeps report the same commit and a dirty tree, which is worth stating
plainly. The "before" sweep ran on `93f6c97` with only the new, untracked
benchmark file added, so nothing in the measured library differed from that
commit. The "after" sweep ran on the same commit with this issue's source
change applied and not yet committed. The CPU was confirmed unclamped before
each sweep, by busy-loop and `/proc/cpuinfo`: 4.1 GHz against a 4.5 GHz
maximum, governor `performance`. This box can otherwise sit at 800 MHz under an
external PROCHOT clamp while every governor still reads `performance`.

## Construction, for a caller who never reads the adjacency

| input | metric | before | after | ratio |
|---|---|---:|---:|---:|
| `random_30k` | peak RSS | 135.92 MiB [135.90, 136.27] | 133.78 MiB [133.73, 134.03] | 0.984x |
| `random_30k` | wall | 0.004 s [0.004, 0.005] | 0.003 s [0.003, 0.003] | 0.714x |
| `random_300k` | peak RSS | 183.57 MiB [182.98, 184.15] | 162.63 MiB [161.83, 162.83] | **0.886x** |
| `random_300k` | wall | 0.073 s [0.060, 0.082] | 0.059 s [0.055, 0.064] | 0.807x |

Every peak RSS range is disjoint, and so is `random_30k` wall. `random_300k`
wall overlaps, so its 0.807x is a direction, not a confident figure.

At 300k rows that is 20.9 MiB off construction. The two matrices themselves are
only 6.6 MiB of it; the rest is the transient the two separate COO-to-CSR
conversions allocate and then free. The arm's `caches_adjacency` fact reads
`false`, which is the structural half of the same claim: a future change that
made construction eager again would flip it and the arm would stop measuring
what its label says.

## Reaching `_A`

| input | metric | before | after | ratio |
|---|---|---:|---:|---:|
| `random_30k` | peak RSS | 136.07 MiB [135.98, 136.23] | 135.20 MiB [135.00, 135.25] | 0.994x |
| `random_30k` | wall | 0.005 s [0.005, 0.007] | 0.005 s [0.005, 0.006] | 1.000x |
| `random_300k` | peak RSS | 183.71 MiB [183.36, 184.80] | 172.41 MiB [172.34, 173.64] | **0.938x** |
| `random_300k` | wall | 0.085 s [0.081, 0.089] | 0.087 s [0.081, 0.097] | 1.021x |

Both peak RSS ranges are disjoint. Both wall ranges overlap, so reaching `_A`
costs the same time as it did; the saving is memory only.

The checksum is identical before and after on every cell, so the new `_A` is
byte-identical to `_Am + _Af`, not merely equal in value.

## The gated A/B, and the regression it caught

`eager` is the gate baseline and `lazy` is gated against it, both on the same
lean post-change graph, so this table isolates the two assembly strategies from
the construction saving above.

| input | strategy | reps | wall (median) | spread | peak RSS (median) | checksum |
|---|---|---:|---:|---:|---:|---|
| `random_30k` | two CSRs then their sum | 5 | 0.01 s | 18.4% | 135 MiB | `1758600907` |
| `random_30k` | one COO | 5 | 0.01 s | 24.2% | 135 MiB | `1758600907` |
| `random_30k` | construction only, adjacency never read | 5 | 0.00 s | 11.3% | 134 MiB | `3815860957` |
| `random_300k` | two CSRs then their sum | 5 | 0.09 s | 20.0% | 177 MiB | `2452044484` |
| `random_300k` | one COO | 5 | 0.09 s | 18.4% | 172 MiB | `2452044484` |
| `random_300k` | construction only, adjacency never read | 5 | 0.06 s | 15.0% | 163 MiB | `60948367` |

`lazy` against `eager` on `random_300k` is 0.974x peak RSS with disjoint ranges
and 1.012x wall with overlapping ranges. No cell blocks.

Five repetitions is not always enough to settle the wall column here. A later
sweep put `random_300k/lazy` at 1.095x wall on a visibly bimodal sample,
`[0.075, 0.078, 0.095, 0.095, 0.105]`, which the harness correctly reported as
inconclusive rather than a block. Nine repetitions resolved it to 0.998x with
overlapping ranges, against 0.977x peak RSS with disjoint ranges on the same
sweep. Peak RSS is the stable column at this size; if the wall column reads
inconclusive, re-run with `--repeat 9` before drawing any conclusion from it.

The first version of the change did not read this way. It indexed the children
with `np.where`, which returns `intp`, and scipy widens a COO to its widest
input dtype, so both index arrays became `int64` even though `mother_rows` is
already `int32`. One COO over all 555,898 edges then allocated a taller
transient than two sequential conversions over half as many edges each.
`lazy` measured 1.026x peak RSS against `eager` with disjoint ranges: a real
regression, and one that would have passed unremarked under the 5% gate.
Building the child index in `mother_rows.dtype` removed it and turned it into
the 0.974x above. The checksum did not move, which is what confirmed the fix
changed only the transient.

That is the case for gating a change nobody expected to regress. The end-to-end
comparison in the two tables above was a win in both directions either way; only
the in-process A/B against the frozen old builder separated "this is faster than
what it replaced" from "this is as good as what it replaced could be".

## What this does not measure

The saving is linear in row count and modest per row, around 23 bytes of
retained matrix and a similar transient. At 300k rows it is 21 MiB against a
163 MiB process. Nothing here says what it is worth at the tens-of-millions of
rows `pedsum` benchmarks, where the F and inbreeding phases dominate anyway.
The fixtures stop at `random_300k` because that is the largest parity fixture.
