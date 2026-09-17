r"""Time ``relationship_pairs`` in whichever ``pedigree_graph`` this interpreter has.

Run by ``bench_pair_qualification.py`` under the PyPI 0.8.4 wheel's
interpreter.  Prints the same JSON record as ``pgr-bench-pairs``::

    python _pair_baseline_child.py --columns fx.npz --receiver graph|view \\
        --degree 5 --threads 6 [--dump DIR]

The thread budget is committed through ``PEDIGREE_GRAPH_THREADS`` before the
package is imported.  RSS is ``VmHWM`` before and after the call.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _pair_common import block_summary, dump_blocks, half_view_rows, peak_rss_mib, write_json


def main() -> None:
    """Build the receiver, time the call, print the record."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--columns", type=Path, required=True)
    ap.add_argument("--receiver", choices=["graph", "view"], required=True)
    ap.add_argument("--degree", type=int, required=True)
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--dump", type=Path, default=None)
    args = ap.parse_args()
    os.environ["PEDIGREE_GRAPH_THREADS"] = str(args.threads)
    from pedigree_graph import PedigreeGraph

    with np.load(args.columns) as npz:
        columns = {key: npz[key] for key in npz.files}
    graph = PedigreeGraph.from_frame(columns)
    n = graph.n_individuals
    receiver = graph if args.receiver == "graph" else graph.view(rows=half_view_rows(n))
    baseline = peak_rss_mib()
    t0 = time.perf_counter()
    pairs = receiver.relationship_pairs(max_degree=args.degree)
    seconds = time.perf_counter() - t0
    peak = peak_rss_mib()
    if args.dump is not None:
        dump_blocks(pairs, args.dump)
    write_json(
        {
            "n": n,
            "execution": "python_0.8.4",
            "threads": args.threads,
            "max_degree": args.degree,
            "view": args.receiver == "view",
            "seconds": round(seconds, 3),
            "baseline_rss_mib": round(baseline, 1),
            "peak_rss_mib": round(peak, 1),
            "pairs": sum(len(b) for b in pairs.values()),
            "blocks": block_summary(pairs),
        }
    )


if __name__ == "__main__":
    main()
