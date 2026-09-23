# The kinship matrix products on the Rust core

`bench_kinship_matrix.py` times `kinship_matrix()`,
`approximate_kinship_matrix(0.001)` and `mean_kinship_by_generation()` on
the parity and simACE study pedigrees, each product on a fresh graph in a
fresh single-threaded process, interleaving a released wheel run under the
simACE env's interpreter with this checkout's build. The slice 14 sweep that
selected the DP's row storage and gated the port is recorded in
`docs/pedigree-graph-0.8-migration/gate/14a/NOTES.md`; this note keeps the
headline numbers of that sweep so the benchmark's own file says what it last
showed.

Sweep of 2026-09-23 (`gate/14a/kinship_matrix.json`, commit `2d8b958`, three
reps, medians, 0.9.1 wheel against the owned-rows source build):

| cell | wheel | source | ratio |
|---|---:|---:|---:|
| `random_30k`, complete | 97.1 s, 13,855 MiB | 18.9 s, 4,723 MiB | 0.19, 0.34 |
| `baseline10K/rep1` (53,466 rows), complete | 10.1 s, 1,500 MiB | 2.85 s, 961 MiB | 0.28, 0.64 |
| `random_30k`, approximate 0.001 | 75.8 s, 6,928 MiB | 10.5 s, 1,546 MiB | 0.14, 0.22 |
| `baseline10K/rep1`, approximate 0.001 | 18.0 s, 1,500 MiB | 3.57 s, 786 MiB | 0.20, 0.52 |
| `random_30k`, summary | 52.1 s, 6,298 MiB | 5.61 s, 1,340 MiB | 0.11, 0.21 |
| `baseline100K/rep1` (536,036 rows), summary | 89.3 s, 14,758 MiB | 15.8 s, 3,604 MiB | 0.18, 0.24 |

Every checksum agrees between the two builds except the 536k summary, where
0.9.1's retiring DP read storage its own merge walk had freed
(`gate/14a/NOTES.md`, "The 536k summary").

The wheel arm needs the simACE umbrella env at `../../.pixi/envs/default`
with its locked `pedigree-graph`; the harness records which package each
child imported, and `gate/14a/compare_arms.py` lays the arms side by side
with ratios and verdicts.
