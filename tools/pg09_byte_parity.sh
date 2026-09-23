#!/usr/bin/env bash
# Byte-parity probe across a pedigree-graph relock (0.8.3 -> 0.9.0 and on).
#
#   external/pedigree-graph/tools/pg09_byte_parity.sh <out-dir>
#
# 0.9.0 moved relationship_pairs onto the Rust engine and promised element-for-
# element identical blocks, so unlike the 0.7.1 -> 0.8.0 migration the consumer
# outputs must match byte for byte. Two artifacts carry the pair contract
# furthest into consumer space: simACE's curated report.yaml (relationship
# correlations and counts) and fitACE's pairwise_relatedness.tsv (the canonical
# pair list itself). Slice 14 (0.9.2) moved the kinship matrix DP, so two
# more products join: fitACE's sparse GRM (the grm_matrix rule at its default
# grm_min_kinship of 0.001, the approximate matrix) and the generation kinship
# summary of the same pedigree, written as text. Slice 15 (0.9.4) moved
# inbreeding, the lineage counts, EqG and the founder means, so two more join:
# fitACE's exports/inbreeding.tsv (F > 0 rows) and simACE's effective_size.yaml
# (the eight Ne records, an opt-in target outside report.yaml's chain). All
# are rebuilt from scratch on the smoke scenario and hashed. Float products may
# legitimately move within rtol 1e-9; tools/pg09_compare_floats.py parses two
# manifest dirs and reports the largest relative difference when hashes differ.
#
# Run it once under the old locks and once after the relock, then diff the two
# manifests. Each run uses whichever pedigree-graph the pixi env resolves; the
# caller sets that, this script only builds and hashes.
set -euo pipefail

OUT="$(realpath -m "${1:?out dir}")"
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"   # the simACE umbrella checkout this repo sits inside
SIMACE_REPORT="results/test/small_test/rep1/report.yaml"
SIMACE_NE="results/test/small_test/rep1/effective_size.yaml"
FITACE_TSV="results/test/small_test/rep1/exports/pairwise_relatedness.tsv"
FITACE_GRM_BIN="results/test/small_test/rep1/grm/A.grm.sp.bin"
FITACE_GRM_ID="results/test/small_test/rep1/grm/A.grm.id"
FITACE_INBREEDING="results/test/small_test/rep1/exports/inbreeding.tsv"
PEDIGREE="results/test/small_test/rep1/pedigree.parquet"

mkdir -p "$OUT"

# simACE rebuilds the whole simulate -> phenotype -> analyze chain; fitACE then
# forces only the export rule, off the pedigree simACE just wrote (fitACE/results
# is a symlink to the same tree), so the two artifacts describe one pedigree.
( cd "$ROOT" && pixi run --frozen snakemake --cores 4 --forceall "$SIMACE_REPORT" "$SIMACE_NE" )
( cd "$ROOT/fitACE" && pixi run --frozen snakemake --cores 4 -f "$FITACE_TSV" "$FITACE_GRM_BIN" "$FITACE_GRM_ID" "$FITACE_INBREEDING" )

cp "$ROOT/$SIMACE_REPORT" "$OUT/report.yaml"
cp "$ROOT/fitACE/$FITACE_TSV" "$OUT/pairwise_relatedness.tsv"
cp "$ROOT/fitACE/$FITACE_GRM_BIN" "$OUT/A.grm.sp.bin"
cp "$ROOT/fitACE/$FITACE_GRM_ID" "$OUT/A.grm.id"
cp "$ROOT/$SIMACE_NE" "$OUT/effective_size.yaml"
cp "$ROOT/fitACE/$FITACE_INBREEDING" "$OUT/inbreeding.tsv"

# The generation kinship summary has no consumer artifact on the smoke
# scenario, so it is computed here under the simACE env's pedigree-graph and
# written as the float64 bits of its means beside the counts.
( cd "$ROOT" && pixi run --frozen python - "$PEDIGREE" "$OUT/mean_kinship_by_generation.txt" <<'PY'
import sys
import numpy as np
import polars as pl
from pedigree_graph import PedigreeGraph

frame = pl.read_parquet(sys.argv[1], columns=["id", "mother", "father", "twin", "sex"])
summary = PedigreeGraph.from_frame(frame).mean_kinship_by_generation()
with open(sys.argv[2], "w") as out:
    for generation, mean, count in zip(summary.generations, summary.mean_kinship, summary.pair_counts, strict=True):
        out.write(f"{int(generation)}\t{np.float64(mean).view(np.uint64)}\t{int(count)}\n")
    out.write(f"unlabelled\t{summary.unlabelled_individual_count}\n")
PY
)

{
  ( cd "$ROOT" && pixi run --frozen python -c \
      "import importlib.metadata as m; print('simACE env pedigree-graph', m.version('pedigree-graph'))" )
  ( cd "$ROOT/fitACE" && pixi run --frozen python -c \
      "import importlib.metadata as m; print('fitACE env pedigree-graph', m.version('pedigree-graph'))" )
  ( cd "$OUT" && sha256sum report.yaml pairwise_relatedness.tsv A.grm.sp.bin A.grm.id mean_kinship_by_generation.txt \
      effective_size.yaml inbreeding.tsv )
} | tee "$OUT/manifest.txt"

# The hashes are the gate; the copies are kept so a mismatch can be diffed
# after the fact, when the other side's environment no longer exists. The pair
# table is 2 MiB of repeated relationship codes and compresses about 7x, so it
# is stored gzipped -- gunzip it to diff.
gzip -9 -f "$OUT/pairwise_relatedness.tsv"
