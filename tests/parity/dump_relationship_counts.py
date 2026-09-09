"""Write the closest-category oracle counts for the Rust engine's parity fixtures.

For every ``<fixtures>/<name>.tsv`` (columns ``mother father twin orig_mother
orig_father``, graph-space rows, written once by the frozen
``dump_relationship_inputs.py``) this rebuilds the pedigree through the 0.8
API and writes ``<name>.counts.json`` with
``PedigreeGraph.relationship_counts(max_degree=5)``, the counts
``crates/core/tests/parity.rs`` asserts.

The TSV carries rows, not the individuals' own ids, so the graph is rebuilt
with synthetic ids: every row gets ``row + offset`` where ``offset`` exceeds
every original parent id.  A parent with a row maps to that row's synthetic
id; an external parent keeps its original id, so the sibships that unresolved
parents define survive; a co-twin maps by row.  The engine's input is then the
same as the TSV's::

    pixi run python tests/parity/dump_relationship_counts.py crates/core/tests/fixtures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from pedigree_graph import PedigreeGraph

MAX_DEGREE = 5


def graph_from_tsv(path: Path) -> PedigreeGraph:
    """Rebuild the pedigree a fixture TSV describes; see the module docstring."""
    df = pl.read_csv(path, separator="\t")
    n = df.height
    mother = df["mother"].to_numpy()
    father = df["father"].to_numpy()
    twin = df["twin"].to_numpy()
    orig_mother = df["orig_mother"].to_numpy().astype(np.int64)
    orig_father = df["orig_father"].to_numpy().astype(np.int64)
    offset = int(max(orig_mother.max(initial=-1), orig_father.max(initial=-1))) + 1
    ids = np.arange(n, dtype=np.int64) + offset

    def parent_ids(rows: np.ndarray, orig: np.ndarray) -> np.ndarray:
        return np.where(rows >= 0, rows.astype(np.int64) + offset, orig)

    twin_ids = np.where(twin >= 0, twin.astype(np.int64) + offset, -1)
    return PedigreeGraph.from_arrays(
        ids=ids,
        mother_ids=parent_ids(mother, orig_mother),
        father_ids=parent_ids(father, orig_father),
        twin_ids=twin_ids,
    )


def dump(tsv: Path) -> dict[str, int]:
    graph = graph_from_tsv(tsv)
    counts = {code: int(v) for code, v in graph.relationship_counts(max_degree=MAX_DEGREE).items()}
    tsv.with_suffix(".counts.json").write_text(
        json.dumps({"n": graph.n_individuals, "counts": counts}, indent=1) + "\n"
    )
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("fixtures", type=Path, help="directory of <name>.tsv fixtures")
    ap.add_argument("--only", nargs="*", default=None, help="fixture names to (re)write; default all")
    args = ap.parse_args()
    for tsv in sorted(args.fixtures.glob("*.tsv")):
        if args.only is not None and tsv.stem not in args.only:
            continue
        counts = dump(tsv)
        print(f"{tsv.stem}: pairs={sum(counts.values())}")


if __name__ == "__main__":
    main()
