# Construction with the parent-role check (issue #58)

`bench_construction.py` gates `PedigreeGraph.from_arrays` on this branch
against main (`fff1a0f`), built from a `git archive` of main with
`pixi run build-dev`, through `--baseline-python`. Five interleaved
repetitions per arm, one thread, on 2026-11-03 (the host also ran another
session's jobs; spreads below).

ADR 0019's check marks each resolved parent row's role while the parent
columns resolve and merges the distinct external parent ids of the two roles.
On sparse ids the merged lookup lists those ids in order as it walks, so the
check costs a scan of one byte a row. On dense ids the lookup is a table, and
the external ids are sorted afterwards.

| input | strategy | reps | wall (median) | spread | peak RSS (median) | tree peak (median) | checksum |
|---|---|---:|---:|---:|---:|---:|---|
| `random_300k` | source build (this env) [base] | 5 | 0.04 s | 9.2% | 95 MiB | 66 MiB | `17997675665764432824` |
| `random_300k` | source build (this env) [candidate] (inconclusive) | 5 | 0.04 s | 7.5% | 95 MiB | 67 MiB | `17997675665764432824` |
| `rows_5m` (5,000,000 rows, sparse ids, 0% external parents) | source build (this env) [base] | 5 | 0.68 s | 3.2% | 502 MiB | 474 MiB | `13782437209077309668` |
| `rows_5m` (5,000,000 rows, sparse ids, 0% external parents) | source build (this env) [candidate] | 5 | 0.69 s | 1.0% | 506 MiB | 479 MiB | `13782437209077309668` |
| `external_5m` (5,000,000 rows, sparse ids, 60% external parents) | source build (this env) [base] | 5 | 0.65 s | 2.5% | 502 MiB | 474 MiB | `18347597176251524691` |
| `external_5m` (5,000,000 rows, sparse ids, 60% external parents) | source build (this env) [candidate] | 5 | 0.67 s | 2.8% | 513 MiB | 486 MiB | `18347597176251524691` |
| `dense_external_5m` (5,000,000 rows, dense ids, 60% external parents) | source build (this env) [base] | 5 | 0.17 s | 1.2% | 502 MiB | 474 MiB | `2486748808945678517` |
| `dense_external_5m` (5,000,000 rows, dense ids, 60% external parents) | source build (this env) [candidate] **BLOCK** | 5 | 0.35 s | 2.6% | 506 MiB | 479 MiB | `2486748808945678517` |

`dense_external_5m` blocks: its 6M external parent references (3M per role,
from 1M distinct ids per role) are sorted, which main never did. Inputs with
no external parents, as simACE writes and pedsum compacts, and inputs on
sparse ids pass. An earlier version that sorted every external reference as
an `(id, role)` pair blocked `external_5m` too (0.77 s to 1.02 s), and one
that marked roles in a pass after resolution blocked `rows_5m` (0.69 s to
0.73 s).
