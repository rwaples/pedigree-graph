- commit `c50be63ff9` on `main`, working tree dirty
- Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz, 12 logical CPUs at up to 2250 MHz (powersave), 31.0 GiB RAM, kernel 7.1.5-76070105-generic
- Python 3.13.15, pixi lock `598740a05f17c1e1`, harness `43358db31bf52f34`
- every backend pinned to 1 thread
- peak RSS is kernel VmHWM, reset via /proc/self/clear_refs at region start

| input | strategy | reps | wall (median) | spread | peak RSS (median) | checksum |
|---|---|---:|---:|---:|---:|---|
| `random_30k` | degree-3 pair_kinship, fresh graph | 3 | 82.86 s | 21.9% | 762 MiB | `11925760` |
| `random_30k` | the same query again on the same graph | 3 | 0.45 s | 20.9% | 569 MiB | `11925760` |
| `random_30k` | relationship_kinship_matrix(max_degree=3), fresh graph | 3 | 80.67 s | 18.4% | 781 MiB | `12021248` |
| `random_30k` | the same matrix after a degree-3 pair_kinship | 3 | 1.24 s | 18.5% | 593 MiB | `12021248` |
| `random_300k` | degree-3 pair_kinship, fresh graph | 0 | did not finish within 3600 s | n/a | n/a | n/a |
| `baseline10K/rep1` (53,466 rows) | degree-3 pair_kinship, fresh graph | 3 | 6.92 s | 1.8% | 865 MiB | `1026097152` |
| `baseline10K/rep1` (53,466 rows) | pair_kinship over every self pair, fresh graph | 3 | 0.76 s | 5.9% | 392 MiB | `212992` |
| `baseline100K/rep1` (536,036 rows) | degree-3 pair_kinship, fresh graph | 3 | 32.03 s | 2.6% | 5,425 MiB | `24461312` |
| `baseline100K/rep1` (536,036 rows) | pair_kinship over every self pair, fresh graph | 3 | 6.80 s | 1.7% | 1,526 MiB | `999424` |
| `baseline100K/rep1` (536,036 rows) | relationship_kinship_matrix(max_degree=3), fresh graph | 3 | 49.64 s | 1.8% | 3,100 MiB | `24772608` |
| `dev_mean_n10k/rep1` (20,400 rows) | degree-5 pair_kinship, fresh graph | 3 | 21.17 s | 1.1% | 865 MiB | `1011253248` |

