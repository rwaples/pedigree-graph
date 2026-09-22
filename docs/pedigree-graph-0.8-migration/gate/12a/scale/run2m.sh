set -e
cd /data/Documents/simACE/external/pedigree-graph
pixi run python - <<'PY'
import sys, polars as pl
sys.path.insert(0, "benchmarks")
from pathlib import Path
import bench_pair_emitters as b
from pedigree_graph import PedigreeGraph
g = PedigreeGraph.from_frame(pl.read_parquet("/data/Documents/simACE/results/bench_pedsum/pedsum_2M/rep1/pedigree.full.parquet"))
print("n", g.n_individuals, flush=True)
b.dump(g, "pedsum_2M", Path("/tmp/claude-1000/-data-Documents-simACE/929d0db5-ba2d-4ea4-9bee-8ec7c889a87e/scratchpad/scale"))
PY
for e in buffered two_pass bounded_wave; do
  /usr/bin/time -v target/release/pgr-bench-pairs /tmp/claude-1000/-data-Documents-simACE/929d0db5-ba2d-4ea4-9bee-8ec7c889a87e/scratchpad/scale/pedsum_2M.tsv --emitter $e --threads 6 --max-degree 5 > /tmp/claude-1000/-data-Documents-simACE/929d0db5-ba2d-4ea4-9bee-8ec7c889a87e/scratchpad/scale/2m.$e.json 2> /tmp/claude-1000/-data-Documents-simACE/929d0db5-ba2d-4ea4-9bee-8ec7c889a87e/scratchpad/scale/2m.$e.err
  grep -E "Elapsed|Maximum resident" /tmp/claude-1000/-data-Documents-simACE/929d0db5-ba2d-4ea4-9bee-8ec7c889a87e/scratchpad/scale/2m.$e.err | sed "s/^/$e: /"
done
echo SCALE_DONE
