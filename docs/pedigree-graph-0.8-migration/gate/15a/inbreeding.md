- commit `61406fca75` on `main`
- Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz, 12 logical CPUs at up to 2600 MHz (performance), 31.0 GiB RAM, kernel 7.1.5-76070105-generic
- Python 3.13.15, pixi lock `01f17bb501182fec`, harness `83ca5eaea419a9df`
- every backend pinned to 1 thread
- peak RSS is kernel VmHWM, reset via /proc/self/clear_refs at region start

| input | strategy | reps | wall (median) | spread | peak RSS (median) | checksum |
|---|---|---:|---:|---:|---:|---|
| `deep_inbred_60g` | 0.9.3 wheel (simACE env) | 3 | 0.02 s | 0.8% | 144 MiB | `9414880614268284619` |
| `deep_inbred_60g` | source build (this env) | 3 | 0.00 s | 2.0% | 59 MiB | `9414880614268284619` |
| `random_300k` | 0.9.3 wheel (simACE env) | 3 | 2.73 s | 1.8% | 181 MiB | `12031751213449802473` |
| `random_300k` | source build (this env) | 3 | 0.88 s | 3.8% | 89 MiB | `12031751213449802473` |
| `baseline100K/rep1` (536,036 rows) | 0.9.3 wheel (simACE env) | 3 | 1.56 s | 5.5% | 359 MiB | `6821534888781537935` |
| `baseline100K/rep1` (536,036 rows) | source build (this env) | 3 | 0.45 s | 11.4% | 268 MiB | `6821534888781537935` |

