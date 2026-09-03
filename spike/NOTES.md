# Rust port spike (throwaway)

Question: does the matrix pair engine (`MatrixPairExtractor`, degree 5, min_kinship 0)
translate to Rust without a sparse-matrix library and reproduce the Python output exactly?

Answer (2026-09-03): yes. `rust/src/main.rs` is a ~450-line single-file port with its own
CSR type and a Gustavson sparse product. `compare.py` dumps `PedigreeGraph` arrays to TSV,
runs both engines, and diffs every code as a set of pairs.

Results, all 23 codes identical:

| pedigree | rows | python | rust compute | rust incl. writing 28M lines |
|---|---|---|---|---|
| tests/data/small_pedigree.parquet | 3,000 | 0.018 s | 0.004 s | |
| 5 random inbred, overlapping generations, twins, missing parents | ~490 | 0.012 s | 0.002 s | |
| simace run_simulation N=5000 G_ped=6 | 30,000 | 0.29 s | | 0.37 s |
| simace run_simulation N=50000 G_ped=6 | 300,000 | 2.66 s | ~2.4 s | 4.0 s |

Rust is serial; Python overlaps codes with a thread pool over scipy's C SpGEMM.

Not covered: subsample mask / remap, min_kinship gating, kinship matrix, Ne estimators,
any binding layer.

Rerun:

    cd rust && pixi run cargo build --release
    pixi run python external/pedigree-graph-rust-spike/spike/compare.py small inbred sim big
