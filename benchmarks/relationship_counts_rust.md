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
