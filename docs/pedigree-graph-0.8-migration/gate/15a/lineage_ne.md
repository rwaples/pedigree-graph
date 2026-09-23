- commit `61406fca75` on `main`
- Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz, 12 logical CPUs at up to 2600 MHz (performance), 31.0 GiB RAM, kernel 7.1.5-76070105-generic
- Python 3.13.15, pixi lock `01f17bb501182fec`, harness `83ca5eaea419a9df`
- every backend pinned to 1 thread
- peak RSS is kernel VmHWM, reset via /proc/self/clear_refs at region start

| input | strategy | reps | wall (median) | spread | peak RSS (median) | checksum |
|---|---|---:|---:|---:|---:|---|
| `wf_n2000_g8` (seed 31, 2000 per generation, 8 generations), `ne_long_term_contributions` (founder means) | 0.9.3 wheel (simACE env) | 3 | 0.01 s | 6.1% | 197 MiB | `16750263830317407774` |
| `wf_n2000_g8` (seed 31, 2000 per generation, 8 generations), `ne_long_term_contributions` (founder means) | source build (this env) | 3 | 0.00 s | 0.7% | 100 MiB | `16750263830317407774` |
| `wf_n5000_g8` (seed 37, 5000 per generation, 8 generations), `ne_individual_delta_f` (F and EqG) | 0.9.3 wheel (simACE env) | 3 | 0.59 s | 2.1% | 208 MiB | `5381999641013868900` |
| `wf_n5000_g8` (seed 37, 5000 per generation, 8 generations), `ne_individual_delta_f` (F and EqG) | source build (this env) | 3 | 0.16 s | 6.8% | 115 MiB | `5381999641013868900` |
| `wf_n5000_g8` (seed 37, 5000 per generation, 8 generations), `ne_long_term_contributions` (founder means) | 0.9.3 wheel (simACE env) | 3 | 0.01 s | 4.4% | 210 MiB | `6034935606833952854` |
| `wf_n5000_g8` (seed 37, 5000 per generation, 8 generations), `ne_long_term_contributions` (founder means) | source build (this env) | 3 | 0.01 s | 12.3% | 114 MiB | `6034935606833952854` |
| `wf_n5000_g8` (seed 37, 5000 per generation, 8 generations), `estimate_effective_sizes` without `ne_coancestry` | 0.9.3 wheel (simACE env) | 3 | 10.02 s | 0.9% | 2,248 MiB | `12415791647294131217` |
| `wf_n5000_g8` (seed 37, 5000 per generation, 8 generations), `estimate_effective_sizes` without `ne_coancestry` | source build (this env) | 3 | 9.27 s | 2.1% | 2,151 MiB | `12415791647294131217` |
| `random_300k`, `descendant_path_counts()` | 0.9.3 wheel (simACE env) | 3 | 0.00 s | 12.3% | 173 MiB | `10493215501704456571` |
| `random_300k`, `descendant_path_counts()` | source build (this env) | 3 | 0.00 s | 3.2% | 88 MiB | `10493215501704456571` |
| `baseline100K/rep1` (536,036 rows), `descendant_path_counts()` | 0.9.3 wheel (simACE env) | 3 | 0.00 s | 1.3% | 344 MiB | `444860848713942354` |
| `baseline100K/rep1` (536,036 rows), `descendant_path_counts()` | source build (this env) | 3 | 0.01 s | 1.0% | 253 MiB | `444860848713942354` |

