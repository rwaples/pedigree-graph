# Row-streaming relationship counts (ADR 0010, issue #11)

Measured 2026-09-04 on the 12-core, 30 GiB workstation with `/usr/bin/time -v`,
release build of `crates/core` (`pgr-count`), all 23 categories at
`max_degree=5`. Inputs are the simACE `config/bench_pedsum.yaml` pedigrees
(`pedsum_2M`: N=250k per generation, `pedsum_20M`: N=2.5M, 8 recorded
generations each), dumped with
`tests/parity/dump_relationship_inputs.py --only-parquet --no-oracle`.
Wall is the engine only; reading the 20M-row TSV adds 3.3 s.

| rows | threads | wall | peak RSS |
|---|---|---|---|
| 2M | 1 | 41.1 s | 209 MiB |
| 2M | 12 | 6.7 s | 302 MiB |
| 20M | 1 | 498 s | 2.00 GiB |
| 20M | 12 | 83.3 s | 2.86 GiB |

Counts are identical at every thread count.

## Against the other engines

* Python matrix engine (`count_pairs`): bit-identical on every parity fixture
  and on simACE pedigrees of 120k and 300k rows (the largest that fit;
  300k rows took 3.0 GiB in Python). Extrapolated to 20M rows it needs
  150 to 560 GiB (issue #11).
* Scalar counter (`count_pairs_streaming`) on the same 20M pedigree: 57 s,
  10.1 GiB peak in a fresh process, all ten exact codes identical to the Rust
  counts, and the approximate codes off by up to a factor of two:

| code | streaming | exact |
|---|---|---|
| 1C | 46,381,471 | 47,215,037 |
| H1C | 38,793,439 | 47,309,721 |
| 1C1R | 196,993,770 | 197,523,187 |
| H1C1R | 291,591,401 | 157,689,995 |
| 1C2R | 312,138,306 | 316,129,515 |
| 2C | 329,625,241 | 157,387,915 |

The avuncular family is within 0.01 percent on this twin-having pedigree.

Those `count_pairs_streaming` figures were measured at commit `aa71c35`, before
the call was deleted with the 0.7.1 adapters; `estimate_relationship_counts` is
the scalar counter on the 0.8 surface, and its values are fold-aware, so the
streaming numbers above are not reproducible through it.

## Reproduce

```bash
pixi run cargo build --release
pixi run python tests/parity/dump_relationship_inputs.py --out /tmp/bench \
    --only-parquet --no-oracle --parquet <pedigree.full.parquet>
/usr/bin/time -v target/release/pgr-count /tmp/bench/pedigree.full.tsv --threads 12
```

## Through the Python call (slice 11, 0.8.3)

`PedigreeGraph.relationship_counts(max_degree=5)` on the 0.8.3 wheel, fresh
process per measurement, graph already built, measured 2026-09-09 with the
CPU at its 2.6 GHz base clock (`docs/pedigree-graph-0.8-migration/gate/11a/
capability-bench.tsv` in simACE). Peak RSS is the whole process, graph
included (`rss_before` is the process before the call).

| rows | threads | wall | RSS before | peak RSS | pairs |
|---|---|---|---|---|---|
| 2M | 1 | 65.6 s | 632 MiB | 677 MiB | 212,626,359 |
| 2M | 12 | 9.6 s | 651 MiB | 738 MiB | 212,626,359 |
| 20M | 1 | 714 s | 2.90 GiB | 3.35 GiB | 2,124,650,324 |
| 20M | 12 | 102 s | 2.90 GiB | 4.22 GiB | 2,124,650,324 |

Against the `pgr-count` table at the top of this file (41 s / 6.7 s at 2M,
498 s / 83 s at 20M) the Python path is 1.2 to 1.6 times slower; part of that
is the fold and the graph resident in the same process, part is clock (that
table was measured on a day the same machine reached turbo). The engine's
memory above the graph is what ADR 0010 reported: about 0.45 GiB at 20M
single-threaded, 1.3 GiB at twelve threads.

`estimate_relationship_counts(max_degree=5)` on the same 2M graph: 2.0 s,
1.28 GiB peak, and 1C1R 19,731,270 against the exact 15,927,760 (24 percent
high). The 20M estimate was not re-measured; the section above records 57 s
and 10.1 GiB.

A note on measurement: the first pass of these runs happened while the host
was clamped to 800 MHz by an external PROCHOT assertion (package at 39 C,
governor `performance`, every core at the floor under load). Those numbers
are kept as `*-throttled-800mhz.tsv` next to the real ones; version ratios
were the same, absolute walls were 3 to 7 times higher. Check
`/proc/cpuinfo` clocks under load before trusting any wall time from this box.
