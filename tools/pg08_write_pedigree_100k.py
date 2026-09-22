#!/usr/bin/env python
r"""Time fitACE's ``write_pedigree(min_kinship=...)`` on a generated 100k pedigree.

Slice 8 flagged that pedigree-graph 0.8's approximate kinship matrix pays the
complete coancestry DP pass, so the sparse-GRM build fitACE runs at scale is a
slice 9a performance case.  This records it; it does not gate.  Run from the
fitACE manifest with the routing the gate uses::

    PYTHONPATH=external/pedigree-graph pixi run --manifest-path fitACE/pixi.toml --frozen \\
        python external/pedigree-graph/tools/pg08_write_pedigree_100k.py --out external/pedigree-graph/docs/pedigree-graph-0.8-migration/gate/9a/write_pedigree_100k.json

The pedigree is ``tests/parity/pedigrees.random_pedigree`` with the parameters
recorded in the output (seed, founders, generations, per-generation rows) and
the parity module's ``input_hash`` of the arrays, so a rerun is comparable.
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import polars as pl

PG_SOURCE = Path(__file__).resolve().parents[1]
ROOT = PG_SOURCE.parents[1]  # the simACE umbrella checkout this repo sits inside
sys.path.insert(0, str(PG_SOURCE / "tests" / "parity"))

import pedigrees  # noqa: E402
from fitace.kinship.grm_io import AceGrmArtifact  # noqa: E402

PARAMS = {"seed": 100_000, "n_founders": 5_000, "n_generations": 8, "per_generation": 12_000}


def main(argv: list[str] | None = None) -> int:
    """Generate, build, write, and record."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-kinship", type=float, default=0.001)
    parser.add_argument("--per-generation", type=int, default=PARAMS["per_generation"])
    args = parser.parse_args(argv)

    params = {**PARAMS, "per_generation": args.per_generation}
    fx = pedigrees.random_pedigree(params["seed"], **{k: v for k, v in params.items() if k != "seed"})
    frame = pl.DataFrame(
        {
            "id": fx["ids"],
            "mother": fx["mother"],
            "father": fx["father"],
            "twin": fx["twin"],
            "sex": fx["sex"],
        }
    )
    import pedigree_graph

    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="pg08-100k-") as tmp:
        artifact = AceGrmArtifact(Path(tmp) / "A")
        artifact.write_pedigree(frame, min_kinship=args.min_kinship)
        wall = time.perf_counter() - started
        bin_bytes = artifact.bin_path.stat().st_size
    peak_rss_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    record = {
        "pedigree_graph_file": pedigree_graph.__file__,
        "params": params,
        "n_rows": len(frame),
        "input_hash": pedigrees.input_hash(fx),
        "min_kinship": args.min_kinship,
        "wall_s": round(wall, 1),
        "peak_rss_mib": round(peak_rss_mib, 1),
        "grm_bin_bytes": int(bin_bytes),
        "n_twin_links": int(np.count_nonzero(fx["twin"] >= 0)),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
