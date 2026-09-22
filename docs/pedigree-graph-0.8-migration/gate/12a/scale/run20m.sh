set -e
cd /data/Documents/simACE/external/pedigree-graph
if [ ! -f /tmp/claude-1000/-data-Documents-simACE/929d0db5-ba2d-4ea4-9bee-8ec7c889a87e/scratchpad/scale/pedsum_20M.tsv ]; then
pixi run python - <<'PY'
import sys, polars as pl
sys.path.insert(0, "benchmarks")
from pathlib import Path
from _pair_fixtures import dump_engine_columns
from pedigree_graph import PedigreeGraph
g = PedigreeGraph.from_frame(pl.read_parquet("/data/Documents/simACE/results/bench_pedsum/pedsum_20M/rep1/pedigree.full.parquet"))
print("n", g.n_individuals, flush=True)
dump_engine_columns(g, Path("/tmp/claude-1000/-data-Documents-simACE/929d0db5-ba2d-4ea4-9bee-8ec7c889a87e/scratchpad/scale/pedsum_20M.tsv"))
PY
fi
/usr/bin/time -v target/release/pgr-bench-pairs /tmp/claude-1000/-data-Documents-simACE/929d0db5-ba2d-4ea4-9bee-8ec7c889a87e/scratchpad/scale/pedsum_20M.tsv --emitter two_pass --threads 6 --max-degree 3 > /tmp/claude-1000/-data-Documents-simACE/929d0db5-ba2d-4ea4-9bee-8ec7c889a87e/scratchpad/scale/20m.two_pass.d3.json 2> /tmp/claude-1000/-data-Documents-simACE/929d0db5-ba2d-4ea4-9bee-8ec7c889a87e/scratchpad/scale/20m.two_pass.d3.err
grep -E "Elapsed|Maximum resident" /tmp/claude-1000/-data-Documents-simACE/929d0db5-ba2d-4ea4-9bee-8ec7c889a87e/scratchpad/scale/20m.two_pass.d3.err
echo GATE_DONE
