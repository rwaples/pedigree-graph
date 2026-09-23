# 16b R kernel parity gate (2026-09-23)

Slice 16b (simACE `plans/pedigree-graph-slice-16-r-package.md`) gives the R
package its four kernels: `relationship_pairs`, `pair_kinship`,
`kinship_matrix` and `inbreeding`. The R and Python bindings call the same
core, so the gate asks whether the R binding hands the core's results over
intact: no lost 1-based offset, no reordering, no lossy promotion, no dropped
category, no id-type drift.

## Goldens (ship in the tarball)

`tools/r_golden.py` writes, from the Python package, the pedigree frame,
degree-5 pairs, the upper triangle of the kinship matrix (hex floats) and F
for the 14 hand-built core fixtures of at most 20 rows plus `inbred0`, and the
Python registry: 1.0 MB, 129 KB gzipped. `r/tests/testthat/test-golden.R`
compares every product with `expect_identical`, no tolerance.
`tests/test_r_goldens.py` fails when the committed goldens differ from what the
current package writes. `pixi run -e r r-test`:
`[ FAIL 0 | WARN 0 | SKIP 0 | PASS 413 ]`.

## Differential (`r_parity.tsv`)

`tools/r_parity.py` ran 16 cells through both hosts on 2026-09-23 and compared
raw little-endian arrays byte for byte: the 11 core fixtures above the golden
size (degree-5 pairs, complete matrix, F, pairwise kinship of every pair up to
degree 3), `dev_mean_n10k` (degree 5), `dev_cont_n10k` and `baseline10K`
(degree 3, matrix), and `dev_laplace_am_strong_50k` and `baseline100K`
(536,036 rows; degree-3 pairs and F, no complete matrix, as in slice 14).
Result: **120/120 products identical**. The Python side was the 0.9.4 build
from before the core change below; the R side the working tree. Whole run
3:02 wall, 9.3 GiB peak RSS (`/usr/bin/time -v`).

R wall per product, for the record, not gated (the R path does extra work:
the 1-based pass, the `code` column and the float32 to double promotion):
`random_30k` pairs 1.03 s, pairwise 5.13 s, matrix 13.34 s; `baseline100K`
pairs 3.22 s, F 0.52 s.

## The core change

`kinship_csc_upper` assembles only row <= column of the complete DP, so the
int32 cap applies to the upper entries. The DP's row store is the full
symmetric matrix either way, so the core peak falls from about 16 to about
12 bytes per full nonzero, not to half. `crates/core/tests/kinship.rs` holds
it equal to the upper triangle of `kinship_csc` on every fixture and checks
the lowered cap. The complete product's path now runs through the same
`assemble` with `Triangle::Full`; after `pixi run maturin develop --release`
from that core, the Python kinship suites passed (222 passed) and
`r_golden.py --check` still matched goldens written by the earlier build.
