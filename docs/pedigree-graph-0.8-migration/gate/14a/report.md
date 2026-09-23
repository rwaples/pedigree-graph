- commit `2d8b95863b` on `main`, working tree dirty
- Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz, 12 logical CPUs at up to 2600 MHz (performance), 31.0 GiB RAM, kernel 7.1.5-76070105-generic
- Python 3.13.15, pixi lock `598740a05f17c1e1`, harness `5e71657e88c6db6e`
- every backend pinned to 1 thread
- peak RSS is kernel VmHWM, reset via /proc/self/clear_refs at region start

| input | strategy | reps | wall (median) | spread | peak RSS (median) | checksum |
|---|---|---:|---:|---:|---:|---|
| `random_1k`, `kinship_matrix()` | 0.9.1 wheel (simACE env) | 3 | 0.15 s | 1.9% | 170 MiB | `324086416` |
| `random_1k`, `kinship_matrix()` | source, owned rows | 3 | 0.03 s | 28.3% | 136 MiB | `324086416` |
| `random_1k`, `kinship_matrix()` | source, arena rows | 3 | 0.03 s | 2.8% | 138 MiB | `324086416` |
| `dev_mean_n10k/rep1` (20,400 rows), `kinship_matrix()` | 0.9.1 wheel (simACE env) | 3 | 3.42 s | 4.0% | 715 MiB | `3504479156` |
| `dev_mean_n10k/rep1` (20,400 rows), `kinship_matrix()` | source, owned rows | 3 | 0.83 s | 5.5% | 452 MiB | `3504479156` |
| `dev_mean_n10k/rep1` (20,400 rows), `kinship_matrix()` | source, arena rows | 3 | 0.95 s | 3.0% | 565 MiB | `3504479156` |
| `random_30k`, `kinship_matrix()` | 0.9.1 wheel (simACE env) | 3 | 97.14 s | 2.8% | 13,855 MiB | `29977759` |
| `random_30k`, `kinship_matrix()` | source, owned rows | 3 | 18.86 s | 7.2% | 4,723 MiB | `29977759` |
| `random_30k`, `kinship_matrix()` | source, arena rows | 3 | 21.44 s | 3.5% | 6,268 MiB | `29977759` |
| `baseline10K/rep1` (53,466 rows), `kinship_matrix()` | 0.9.1 wheel (simACE env) | 3 | 10.09 s | 3.5% | 1,500 MiB | `462728527` |
| `baseline10K/rep1` (53,466 rows), `kinship_matrix()` | source, owned rows | 3 | 2.85 s | 4.5% | 960 MiB | `462728527` |
| `baseline10K/rep1` (53,466 rows), `kinship_matrix()` | source, arena rows | 3 | 3.38 s | 3.5% | 1,548 MiB | `462728527` |
| `dev_cont_n10k/rep1` (20,400 rows), `approximate_kinship_matrix(0.001)` | 0.9.1 wheel (simACE env) | 3 | 6.30 s | 1.8% | 804 MiB | `926247057` |
| `dev_cont_n10k/rep1` (20,400 rows), `approximate_kinship_matrix(0.001)` | source, owned rows | 3 | 1.09 s | 5.4% | 373 MiB | `926247057` |
| `dev_cont_n10k/rep1` (20,400 rows), `approximate_kinship_matrix(0.001)` | source, arena rows | 3 | 1.20 s | 2.5% | 503 MiB | `926247057` |
| `random_30k`, `approximate_kinship_matrix(0.001)` | 0.9.1 wheel (simACE env) | 3 | 75.82 s | 1.3% | 6,928 MiB | `3312513764` |
| `random_30k`, `approximate_kinship_matrix(0.001)` | source, owned rows | 3 | 10.54 s | 8.9% | 1,546 MiB | `3312513764` |
| `random_30k`, `approximate_kinship_matrix(0.001)` | source, arena rows | 3 | 12.67 s | 2.3% | 4,431 MiB | `3312513764` |
| `baseline10K/rep1` (53,466 rows), `approximate_kinship_matrix(0.001)` | 0.9.1 wheel (simACE env) | 3 | 17.95 s | 1.4% | 1,500 MiB | `485143250` |
| `baseline10K/rep1` (53,466 rows), `approximate_kinship_matrix(0.001)` | source, owned rows | 3 | 3.57 s | 1.5% | 786 MiB | `485143250` |
| `baseline10K/rep1` (53,466 rows), `approximate_kinship_matrix(0.001)` | source, arena rows | 3 | 3.66 s | 0.7% | 845 MiB | `485143250` |
| `random_30k`, `mean_kinship_by_generation()` | 0.9.1 wheel (simACE env) | 3 | 52.13 s | 0.9% | 6,298 MiB | `1343094455` |
| `random_30k`, `mean_kinship_by_generation()` | source, owned rows | 3 | 5.61 s | 4.9% | 1,340 MiB | `1343094455` |
| `random_30k`, `mean_kinship_by_generation()` | source, arena rows | 3 | 7.33 s | 1.8% | 4,226 MiB | `1343094455` |
| `baseline100K/rep1` (536,036 rows), `mean_kinship_by_generation()` | 0.9.1 wheel (simACE env) | 3 | 89.33 s | 1.2% | 14,758 MiB | `286067239` |
| `baseline100K/rep1` (536,036 rows), `mean_kinship_by_generation()` | source, owned rows | 3 | 15.79 s | 3.5% | 3,604 MiB | `4221489846` |
| `baseline100K/rep1` (536,036 rows), `mean_kinship_by_generation()` | source, arena rows | 3 | 16.41 s | 2.3% | 8,551 MiB | `4221489846` |

