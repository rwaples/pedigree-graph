# 9a gate notes (2026-09-09)

Run: `EPIMIGHT_CONDA_ENV=epimight-master pixi run --frozen python tools/pg08_release_gate.py run --stage 9a --routing source --slow`.
Every consumer unit's `routing` step resolved `pedigree_graph` under `external/pedigree-graph/`.

## fitACE_epimight: default and slow suites pass

The first 9a run found five pre-existing slow-suite failures against epimight R
1.0.1. Four tests still invoked the absent `epimight` conda env directly. The
CLI test reached R, but its 60-person fixture had independent per-person A/C/E
values and therefore no expected familial enrichment. Epimight 1.0.1 discarded
its non-positive h2 rows, then failed when its exact-time join had no rows for
both disorders.

The repair routes every test subprocess through `EPIMIGHT_CONDA_ENV`, gives the
pedigree fixture inherited additive effects and shared maternal-household
effects, and uses deterministic enrichment in the hand-built schema fixture.
The master-schema R branch also writes unavailable random-effect and Rubin
diagnostic columns as `NA`, preserving the 12-file schema used by the v2.0
branch.

The rerun passed both gate steps: 256 default tests and 18 slow tests. The
routing check resolved `pedigree_graph` under `external/pedigree-graph/`.

This remains independent of pedigree-graph 0.8. The emitter's only
pedigree-graph call is `relationship_pairs(max_degree=3)`
(`fitace_epimight/create_input.py:84`), and the original failing fixture had
identical category pair sets under the locked 0.7.1 wheel and the 0.8 source
checkout: `MO` 40, `FO` 40, `FS` 20, `GP` 80, `Av` 40, `1C` 20, with all
others empty.

## ace_iter_reml

`ctest` is not installed anywhere on the box; the unit runs the six test
executables directly. `pedsum` `cli-smoke` reads a TSV written from the smoke
pedigree parquet.

## 30k integration

Not rerun in 9a. The slice 8f `random_30k` library snapshot
(`../../migrated/library.json`, written 2026-09-09 03:36 with
`pg08_migration_diff.py snapshot --source`) was taken against pedigree-graph
`c8c963a`, which is still the `v0.8` head at gate time, so its comparison with
the 0.7.1 baseline (`../../report.md`, `library.random_30k.*` rows) is the
9a 30k evidence: pair sets, kinship values, and candidate-matrix support at 30k,
with the approximate matrix's DP-pass cost (75.7 s, 6938 MiB) recorded there.

## 300k release cases (recorded, not gated)

`random_300k` from `tests/parity/pedigrees.RELEASE_FIXTURES` (seed 300000,
15000 founders, 8 generations, 36000 per generation), pedigree-graph
`c8c963a`, one thread, one cell per child process. Reports:
`external/pedigree-graph/benchmarks/reports/release_300k{,_inbreeding,_counts}.json`.

| operation | wall | peak RSS | note |
|---|---:|---:|---|
| `relationship_pairs(max_degree=3)` | 2.57 s | 846 MiB | 5.8M pairs (MO 277854, FO 278044, GP 925918, GGP 1502119, HAv 2086186, MHS 302281, PHS 300508, MZ 5506, 1C 16480) |
| `connected_component_ids()` | 0.07 s | 212 MiB | 142 components |
| `inbreeding()` | 1.71 s | 211 MiB | mean F 3.7e-5, max F 0.127 |
| `estimate_relationship_counts(max_degree=3)` | 0.29 s | 327 MiB | |
| `pair_kinship` over the degree-3 pairs | **timeout at 3600 s** | 6.9 GiB | unverified at 300k |
| complete / approximate kinship matrices | not attempted | | unverified at 300k; the DP pass costs 6.9 GiB at 30k |

fitACE `write_pedigree(min_kinship=0.001)` on a generated 101k-row pedigree
(seed 100000, `gate/9a/write_pedigree_100k.json`): 293.9 s, peak RSS
21.97 GiB, 2.7 GB sparse GRM. Runnable on the 30 GiB box; the RSS is the
approximate matrix's complete DP pass and is the practical ceiling here.
