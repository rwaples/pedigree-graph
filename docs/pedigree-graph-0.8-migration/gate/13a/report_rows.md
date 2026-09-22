- commit `c50be63ff9` on `main`, working tree dirty
- Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz, 12 logical CPUs at up to 2600 MHz (performance), 31.0 GiB RAM, kernel 7.1.5-76070105-generic
- Python 3.13.15, pixi lock `598740a05f17c1e1`, harness `43358db31bf52f34`
- every backend pinned to 1 thread
- peak RSS is kernel VmHWM, reset via /proc/self/clear_refs at region start

| input | strategy | reps | wall (median) | spread | peak RSS (median) | checksum |
|---|---|---:|---:|---:|---:|---|
| `random_30k` | degree-3 pair_kinship, fresh graph | 3 | 5.32 s | 1.7% | 515 MiB | `11925760` |
| `random_30k` | the same query again on the same graph | 3 | 5.39 s | 4.8% | 515 MiB | `11925760` |
| `random_30k` | relationship_kinship_matrix(max_degree=3), fresh graph | 3 | 5.98 s | 3.5% | 523 MiB | `12021248` |
| `random_30k` | the same matrix after a degree-3 pair_kinship | 3 | 5.77 s | 0.6% | 528 MiB | `12021248` |
| `random_300k` | degree-3 pair_kinship, fresh graph | 3 | 934 s (16 min) | 0.2% | 14,076 MiB | `1029931008` |
| `baseline10K/rep1` (53,466 rows) | degree-3 pair_kinship, fresh graph | 3 | 2.06 s | 5.7% | 462 MiB | `1026097152` |
| `baseline10K/rep1` (53,466 rows) | pair_kinship over every self pair, fresh graph | 3 | 0.43 s | 5.3% | 276 MiB | `212992` |
| `baseline100K/rep1` (536,036 rows) | degree-3 pair_kinship, fresh graph | 3 | 24.84 s | 3.2% | 3,022 MiB | `24461312` |
| `baseline100K/rep1` (536,036 rows) | pair_kinship over every self pair, fresh graph | 3 | 5.12 s | 0.3% | 896 MiB | `999424` |
| `baseline100K/rep1` (536,036 rows) | relationship_kinship_matrix(max_degree=3), fresh graph | 3 | 32.68 s | 0.7% | 3,125 MiB | `24772608` |
| `dev_mean_n10k/rep1` (20,400 rows) | degree-5 pair_kinship, fresh graph | 3 | 2.35 s | 2.0% | 456 MiB | `1011253248` |

