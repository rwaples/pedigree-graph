- commit `c50be63ff9` on `main`, working tree dirty
- Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz, 12 logical CPUs at up to 2600 MHz (performance), 31.0 GiB RAM, kernel 7.1.5-76070105-generic
- Python 3.13.15, pixi lock `598740a05f17c1e1`, harness `43358db31bf52f34`
- every backend pinned to 1 thread
- peak RSS is kernel VmHWM, reset via /proc/self/clear_refs at region start

| input | strategy | reps | wall (median) | spread | peak RSS (median) | checksum |
|---|---|---:|---:|---:|---:|---|
| `random_30k` | degree-3 pair_kinship, fresh graph | 3 | 82.50 s | 0.1% | 762 MiB | `11925760` |
| `random_30k` | the same query again on the same graph | 3 | 0.45 s | 0.4% | 569 MiB | `11925760` |
| `random_30k` | relationship_kinship_matrix(max_degree=3), fresh graph | 3 | 79.86 s | 0.2% | 781 MiB | `12021248` |
| `random_30k` | the same matrix after a degree-3 pair_kinship | 3 | 1.20 s | 3.1% | 596 MiB | `12021248` |
| `baseline10K/rep1` (53,466 rows) | degree-3 pair_kinship, fresh graph | 3 | 6.82 s | 0.1% | 865 MiB | `1026097152` |
| `baseline10K/rep1` (53,466 rows) | pair_kinship over every self pair, fresh graph | 3 | 0.73 s | 2.3% | 391 MiB | `212992` |
| `baseline100K/rep1` (536,036 rows) | degree-3 pair_kinship, fresh graph | 3 | 31.28 s | 0.3% | 5,421 MiB | `24461312` |
| `baseline100K/rep1` (536,036 rows) | pair_kinship over every self pair, fresh graph | 3 | 6.75 s | 1.2% | 1,526 MiB | `999424` |
| `baseline100K/rep1` (536,036 rows) | relationship_kinship_matrix(max_degree=3), fresh graph | 3 | 49.10 s | 2.5% | 3,100 MiB | `24772608` |
| `dev_mean_n10k/rep1` (20,400 rows) | degree-5 pair_kinship, fresh graph | 3 | 21.06 s | 0.2% | 864 MiB | `1011253248` |

