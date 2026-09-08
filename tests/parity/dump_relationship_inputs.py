"""Dump relationship-engine inputs and oracle counts for the Rust core's parity tests.

For each fixture, writes ``<out>/<name>.tsv`` with one row per individual in
graph-space order (columns ``mother father twin orig_mother orig_father``:
parent and co-twin rows, ``-1`` when absent, plus the original parent ids,
``-1`` when missing) and ``<out>/<name>.counts.json`` with
``count_pairs(max_degree=5, scope="full")`` from the Python matrix engine.

Fixtures come from :mod:`pedigrees` (motifs, ``random_1k``,
``deep_inbred_60g``, ``random_30k``), the shipped ``small_pedigree.parquet``,
and any ``--parquet`` files named on the command line (columns ``id``,
``mother``, ``father``, optional ``twin``)::

    pixi run python tests/parity/dump_relationship_inputs.py --out crates/core/tests/fixtures

Frozen at base commit ``aa71c35``: it calls the 0.7.1 API the 0.8 branch
deleted, runs only against a pre-slice-7 checkout, and imports nothing from
the 0.8 surface at test time.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pedigrees  # noqa: E402

from pedigree_graph import PedigreeGraph  # noqa: E402

MAX_DEGREE = 5


def _graph_from_fixture(fx: dict[str, np.ndarray]) -> PedigreeGraph:
    return PedigreeGraph.from_arrays(
        ids=fx["ids"],
        mothers=fx["mother"],
        fathers=fx["father"],
        twins=fx["twin"],
        sex=fx["sex"],
    )


def _graph_from_parquet(path: Path) -> PedigreeGraph:
    df = pl.read_parquet(path)
    return PedigreeGraph(df)


def dump(name: str, pg: PedigreeGraph, out: Path, oracle: bool = True) -> dict[str, int]:
    """Write the engine inputs, and the oracle counts unless *oracle* is off; return the counts."""
    pl.DataFrame(
        {
            "mother": pg.mother.astype(np.int64),
            "father": pg.father.astype(np.int64),
            "twin": pg.twin.astype(np.int64),
            "orig_mother": pg._orig_mother.astype(np.int64),
            "orig_father": pg._orig_father.astype(np.int64),
        }
    ).write_csv(out / f"{name}.tsv", separator="\t")
    if not oracle:
        return {}
    counts = {code: int(v) for code, v in pg.count_pairs(max_degree=MAX_DEGREE, scope="full").items()}
    (out / f"{name}.counts.json").write_text(json.dumps({"n": int(pg.n), "counts": counts}, indent=1) + "\n")
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--parquet", type=Path, nargs="*", default=[], help="extra pedigree parquet files")
    ap.add_argument("--skip-large", action="store_true", help="skip random_30k")
    ap.add_argument("--no-oracle", action="store_true", help="write inputs only (matrix engine too large)")
    ap.add_argument("--only-parquet", action="store_true", help="skip the built-in fixtures")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    graphs: dict[str, PedigreeGraph] = {}
    if not args.only_parquet:
        for name, fx in pedigrees.motif_fixtures().items():
            graphs[name] = _graph_from_fixture(fx)
        for name, params in pedigrees.RANDOM_FIXTURES.items():
            graphs[name] = _graph_from_fixture(pedigrees.build_random(name, params))
        if not args.skip_large:
            for name, params in pedigrees.LARGE_FIXTURES.items():
                graphs[name] = _graph_from_fixture(pedigrees.build_random(name, params))
        graphs["small_pedigree"] = _graph_from_parquet(HERE.parent / "data" / "small_pedigree.parquet")
    for path in args.parquet:
        graphs[path.stem] = _graph_from_parquet(path)

    for name, pg in graphs.items():
        counts = dump(name, pg, args.out, oracle=not args.no_oracle)
        print(f"{name}: n={pg.n} pairs={sum(counts.values())}")


if __name__ == "__main__":
    main()
