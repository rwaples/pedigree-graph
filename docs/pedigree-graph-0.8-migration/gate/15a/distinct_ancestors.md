- commit `61406fca75` on `main`
- Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz, 12 logical CPUs at up to 2600 MHz (performance), 31.0 GiB RAM, kernel 7.1.5-76070105-generic
- Python 3.13.15, pixi lock `01f17bb501182fec`, harness `83ca5eaea419a9df`
- every backend pinned to 1 thread
- peak RSS is kernel VmHWM, reset via /proc/self/clear_refs at region start

| input | strategy | reps | wall (median) | spread | peak RSS (median) | checksum |
|---|---|---:|---:|---:|---:|---|
| `closed_w200_g8` (1,800 rows, 200 founders, 8 generated generations, 200/generation) | 0.9.3 wheel (simACE env) | 3 | 0.00 s | 0.4% | 145 MiB | `12468895593382563718` |
| `closed_w200_g8` (1,800 rows, 200 founders, 8 generated generations, 200/generation) | source build (this env) | 3 | 0.00 s | 20.1% | 60 MiB | `12468895593382563718` |
| `closed_w1000_g8` (9,000 rows, 1,000 founders, 8 generated generations, 1,000/generation) | 0.9.3 wheel (simACE env) | 3 | 0.02 s | 3.0% | 149 MiB | `12508086900247589243` |
| `closed_w1000_g8` (9,000 rows, 1,000 founders, 8 generated generations, 1,000/generation) | source build (this env) | 3 | 0.01 s | 11.1% | 62 MiB | `12508086900247589243` |
| `closed_w2000_g8` (18,000 rows, 2,000 founders, 8 generated generations, 2,000/generation) | 0.9.3 wheel (simACE env) | 3 | 0.04 s | 3.0% | 152 MiB | `12524142418444581327` |
| `closed_w2000_g8` (18,000 rows, 2,000 founders, 8 generated generations, 2,000/generation) | source build (this env) | 3 | 0.01 s | 8.2% | 63 MiB | `12524142418444581327` |
| `closed_w128_g8` (1,152 rows, 128 founders, 8 generated generations, 128/generation) | 0.9.3 wheel (simACE env) | 3 | 0.00 s | 1.7% | 145 MiB | `9047697374352796410` |
| `closed_w128_g8` (1,152 rows, 128 founders, 8 generated generations, 128/generation) | source build (this env) | 3 | 0.00 s | 4.0% | 60 MiB | `9047697374352796410` |
| `closed_w128_g16` (2,176 rows, 128 founders, 16 generated generations, 128/generation) | 0.9.3 wheel (simACE env) | 3 | 0.01 s | 2.0% | 147 MiB | `10912113437002054624` |
| `closed_w128_g16` (2,176 rows, 128 founders, 16 generated generations, 128/generation) | source build (this env) | 3 | 0.00 s | 0.8% | 61 MiB | `10912113437002054624` |
| `closed_w128_g32` (4,224 rows, 128 founders, 32 generated generations, 128/generation) | 0.9.3 wheel (simACE env) | 3 | 0.05 s | 21.3% | 153 MiB | `5534745263277103867` |
| `closed_w128_g32` (4,224 rows, 128 founders, 32 generated generations, 128/generation) | source build (this env) | 3 | 0.02 s | 5.2% | 62 MiB | `5534745263277103867` |
| `closed_w128_g60` (7,808 rows, 128 founders, 60 generated generations, 128/generation) | 0.9.3 wheel (simACE env) | 3 | 0.16 s | 2.3% | 160 MiB | `3544957112422719649` |
| `closed_w128_g60` (7,808 rows, 128 founders, 60 generated generations, 128/generation) | source build (this env) | 3 | 0.09 s | 1.0% | 65 MiB | `3544957112422719649` |
| `random_1k` | 0.9.3 wheel (simACE env) | 3 | 0.00 s | 6.8% | 144 MiB | `2023886073456858142` |
| `random_1k` | source build (this env) | 3 | 0.00 s | 2.7% | 60 MiB | `2023886073456858142` |
| `deep_inbred_60g` | 0.9.3 wheel (simACE env) | 3 | 0.00 s | 0.9% | 144 MiB | `12828496093552940807` |
| `deep_inbred_60g` | source build (this env) | 3 | 0.00 s | 2.2% | 59 MiB | `12828496093552940807` |
| `random_30k` | 0.9.3 wheel (simACE env) | 3 | 0.04 s | 4.8% | 156 MiB | `5600220856912005827` |
| `random_30k` | source build (this env) | 3 | 0.01 s | 5.1% | 65 MiB | `5600220856912005827` |
| `random_300k` | 0.9.3 wheel (simACE env) | 3 | 0.45 s | 1.8% | 248 MiB | `2929118677118745316` |
| `random_300k` | source build (this env) | 3 | 0.16 s | 3.1% | 103 MiB | `2929118677118745316` |
| `baseline100K/rep1` (536,036 rows) | 0.9.3 wheel (simACE env) | 3 | 0.26 s | 4.2% | 393 MiB | `2278874049866584189` |
| `baseline100K/rep1` (536,036 rows) | source build (this env) | 3 | 0.14 s | 10.3% | 272 MiB | `2278874049866584189` |

