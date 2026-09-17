"""On-disk forms of a parity fixture for the slice 12 pair drivers.

Needs the repository environment (``tests/parity``, polars, the editable
package); the 0.8.4 baseline child must not import this.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "tests"))
sys.path.insert(0, str(REPO / "tests" / "parity"))

import pedigrees  # noqa: E402
from _pair_common import half_view_rows  # noqa: E402
from conftest import parity_columns  # noqa: E402

from pedigree_graph import PedigreeGraph  # noqa: E402

BINARY = REPO / "target" / "release" / "pgr-bench-pairs"
EXECUTIONS = ("speed", "memory")


def fixture_params(name: str) -> dict:
    """Parameters of a ``tests/parity`` fixture, never re-declared here."""
    for table in (pedigrees.RANDOM_FIXTURES, pedigrees.LARGE_FIXTURES, pedigrees.RELEASE_FIXTURES):
        if name in table:
            return table[name]
    raise SystemExit(f"unknown parity fixture {name!r}")


class Inputs:
    """One fixture's on-disk forms: constructor columns, engine TSV, half-view map.

    ``graph`` is kept so a driver can run the in-tree oracle on it; drop the
    instance before timing anything memory-sensitive in this process.
    """

    def __init__(self, name: str, work: Path) -> None:
        columns = parity_columns(pedigrees.build_random(name, fixture_params(name)))
        self.graph = PedigreeGraph.from_frame(columns)
        self.n = self.graph.n_individuals
        self.columns = work / f"{name}.npz"
        np.savez(self.columns, **columns)
        self.tsv = work / f"{name}.tsv"
        dump_engine_columns(self.graph, self.tsv)
        self.view_rows = half_view_rows(self.n)
        self.view = work / f"{name}.view.tsv"
        np.savetxt(self.view, self.graph.view(rows=self.view_rows)._graph_to_view(), fmt="%d")


def dump_engine_columns(graph: PedigreeGraph, tsv: Path) -> None:
    """Write the five engine columns as the TSV ``PedigreeColumns::read_tsv`` reads."""
    pl.DataFrame(
        {
            "mother": graph.mother_rows.astype(np.int64),
            "father": graph.father_rows.astype(np.int64),
            "twin": graph.twin_rows.astype(np.int64),
            "orig_mother": graph.mother_ids.astype(np.int64),
            "orig_father": graph.father_ids.astype(np.int64),
        }
    ).write_csv(tsv, separator="\t")
