# Slice 5b candidate-value exactification profile

Profiled 2026-09-06 before implementing the matrix exact-value pass, as
required by the 0.8.0 slice plan. The input was the fitACE-shaped simACE result
`results/dev/dev_cont_n10k/rep1/pedigree.parquet`: 20,400 full-pedigree rows and
4,991,524 upper-triangle entries (including diagonal) in the propagation-pruned
`0.001` candidate matrix.

Host: Intel i7-9750H (12 logical CPUs), 30 GiB RAM. Runs were sequential fresh
processes under the pedigree-graph Pixi manifest. Each strategy evaluated the
same candidate coordinates with `_pairwise_kinship_with_stats`; the equal XOR
checksum (`32915456`) checked returned bits.

| strategy | chunks | exact-value wall | evaluation RSS delta | largest memo capacity |
|---|---:|---:|---:|---:|
| one shared memo | 1 | 117.5 s | 960 MiB | 67,108,864 |
| 256-column chunks | 80 | 404.1 s | 69 MiB | 16,777,216 |
| fixed 262,144-pair chunks | 20 | 380.5 s | 40 MiB | 16,777,216 |
| fixed 1,048,576-pair chunks | 5 | 207.3 s | 353 MiB | 33,554,432 |

Candidate-support construction itself took 2.7–2.8 s. Peak-process RSS includes
the graph, candidate CSC, and profiling-only materialisation of every upper
coordinate; the delta isolates the strategy comparison better than total RSS.

## Initial decision (superseded for approximate support)

The first implementation used deterministic fixed-size pair chunks. They bound
the recurrence memo independently of total candidate support, unlike one shared
memo, and avoid the uneven work and slightly higher wall time of fixed column
spans. A 1,048,576-pair internal chunk was the measured wall/RSS compromise.
Chunking cannot alter values because ADR 0009 fixes recurrence evaluation within
each pair; all four runs produced the same checksum, and the test suite compares
complete data bytes across chunk sizes 1, 7, and 1,048,576.

The 30k result below showed that a bound on one chunk's memo did not bound
repeated work: each chunk discarded dependencies needed again by later chunks.
Pair chunks remain appropriate for sparse relationship-selected support, but no
longer exactify dense approximate support.

## 30k integration observation

The fixed 1,048,576-pair strategy was also run over the generated
`random_30k` fixture (30,300 rows; seed/parameters in
`tests/parity/pedigrees.py`; input SHA-256
`f4d558a6957b9c12efe4a28778448a9b981769a5998c039849bbcec316d0b80d`;
26,924,109 upper-triangle `0.001` candidates including diagonal). Exactifying a
prebuilt candidate CSC completed in 6,781.3 s (1 h 53 min), with process RSS
rising from 577 MiB to 2,956 MiB. A one-shared-memo run did not complete within
a 30-minute observation window. This result triggered the fused-DP follow-up.

## Fused complete-DP candidate capture

The existing retiring threshold-zero DP was then profiled as the available exact
working-set computation. It took 2.87 s / 666 MiB peak process RSS on the
20,400-row fitACE pedigree and 54.6 s / 6,343 MiB on `random_30k`. The optimized
path merges each complete exact DP row against the propagation-pruned candidate
row and writes only matching values to the output CSC; complete rows continue to
retire after their last direct child and no complete CSC is materialized.

Final fresh-process end-to-end timings, including candidate-support construction,
topology-space candidate indexing, exact capture, symmetric fill, and CSC freeze:

| input | upper candidates | wall | peak process RSS | upper-value XOR |
|---|---:|---:|---:|---:|
| fitACE `dev_cont_n10k/rep1` (20,400 rows) | 4,991,524 | 6.48 s | 848 MiB | 32,915,456 |
| generated `random_30k` (30,300 rows) | 26,924,109 | 76.44 s | 6,963 MiB | 963,002,880 |

The fitACE checksum is unchanged from every pairwise strategy above. Small,
MZ/inbred/deep, and row-permuted fixtures compare every retained value directly
with `pair_kinship`. A dedicated 30k matrix integration test runs the public
fused path; the differential parity test separately checks the frozen support
hash without repeating exactification inside its already-large operation bundle.

A matching final-source complete retiring-DP baseline took 2.89 s on the fitACE
input (the earlier `random_30k` baseline was 54.6 s). The fused operation thus
adds candidate indexing, capture, and symmetric-output overhead without repeating
pairwise dependency discovery across chunks. Both final runs satisfy the locked
under-10 s and under-120 s wall-time gates.

The profiling drivers were temporary (`/tmp/profile_slice5b.py`,
`/tmp/profile_slice5b_random30k.py`, `/tmp/profile_complete_streaming_dp.py`,
and `/tmp/profile_fused_approximate.py`) and are not release artifacts. Commands
had the form:

```bash
pixi run --manifest-path external/pedigree-graph/pixi.toml \
  python /tmp/profile_slice5b.py <shared|columns|pairs> \
  results/dev/dev_cont_n10k/rep1/pedigree.parquet
```

The earlier support/downstream investigation remains under
`benchmarks/threshold_structure/`; in particular, changing support rather than
correcting retained values moved PCGC h2 by 0.033 on the 10,200-person phenotype
subset.
