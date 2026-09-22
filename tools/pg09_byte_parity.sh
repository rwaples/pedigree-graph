#!/usr/bin/env bash
# Byte-parity probe for the pedigree-graph 0.8.3 -> 0.9.0 relock.
#
#   external/pedigree-graph/tools/pg09_byte_parity.sh <out-dir>
#
# 0.9.0 moves relationship_pairs onto the Rust engine and promises element-for-
# element identical blocks, so unlike the 0.7.1 -> 0.8.0 migration the consumer
# outputs must match byte for byte. Two artifacts carry the pair contract
# furthest into consumer space: simACE's curated report.yaml (relationship
# correlations and counts) and fitACE's pairwise_relatedness.tsv (the canonical
# pair list itself). Both are rebuilt from scratch on the smoke scenario and
# hashed.
#
# Run it once under the 0.8.3 locks and once after the relock, then diff the two
# manifests. Each run uses whichever pedigree-graph the pixi env resolves; the
# caller sets that, this script only builds and hashes.
set -euo pipefail

OUT="$(realpath -m "${1:?out dir}")"
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"   # the simACE umbrella checkout this repo sits inside
SIMACE_REPORT="results/test/small_test/rep1/report.yaml"
FITACE_TSV="results/test/small_test/rep1/exports/pairwise_relatedness.tsv"

mkdir -p "$OUT"

# simACE rebuilds the whole simulate -> phenotype -> analyze chain; fitACE then
# forces only the export rule, off the pedigree simACE just wrote (fitACE/results
# is a symlink to the same tree), so the two artifacts describe one pedigree.
( cd "$ROOT" && pixi run --frozen snakemake --cores 4 --forceall "$SIMACE_REPORT" )
( cd "$ROOT/fitACE" && pixi run --frozen snakemake --cores 4 -f "$FITACE_TSV" )

cp "$ROOT/$SIMACE_REPORT" "$OUT/report.yaml"
cp "$ROOT/fitACE/$FITACE_TSV" "$OUT/pairwise_relatedness.tsv"

{
  ( cd "$ROOT" && pixi run --frozen python -c \
      "import importlib.metadata as m; print('simACE env pedigree-graph', m.version('pedigree-graph'))" )
  ( cd "$ROOT/fitACE" && pixi run --frozen python -c \
      "import importlib.metadata as m; print('fitACE env pedigree-graph', m.version('pedigree-graph'))" )
  ( cd "$OUT" && sha256sum report.yaml pairwise_relatedness.tsv )
} | tee "$OUT/manifest.txt"

# The hashes are the gate; the copies are kept so a mismatch can be diffed
# after the fact, when the other side's environment no longer exists. The pair
# table is 2 MiB of repeated relationship codes and compresses about 7x, so it
# is stored gzipped -- gunzip it to diff.
gzip -9 -f "$OUT/pairwise_relatedness.tsv"
