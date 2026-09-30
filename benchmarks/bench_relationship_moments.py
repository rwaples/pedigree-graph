"""``relationship_moments`` against ``relationship_pairs`` plus NumPy on the simACE pedsum pedigrees (ADR 0013).

    pixi run python benchmarks/bench_relationship_moments.py --repeat 3 --out /tmp/moments.json
    pixi run python benchmarks/bench_relationship_moments.py --render /tmp/moments.json

Both arms compute the same statistics over the same seven categories the
simACE analysis reads (``simace.core.relationships.RELATIONSHIP_TYPES``):
per category and per cell of (first member's generation and sex, second
member's sex, same household), the pair count and the sums, sums of squares
and diagonal cross sums of the two liabilities.  ``moments`` is one engine
pass with no pair list; ``pairs_numpy`` materialises the pair blocks, then
folds them with ``np.bincount``.  Both run at one thread and at twelve, the
counts benchmark's two points (``relationship_counts_rust.md``).

The checksum is over the pair counts per cell, which both arms produce
exactly; the float sums are compared by the ADR 0013 tests, not here.  The
fixtures are the ``bench_pedsum`` pedigrees generated in the simACE umbrella
(``results/bench_pedsum/pedsum_{2M,20M}/rep1/pedigree.full.parquet``,
or under ``$SIMACE_RESULTS``) and are unavailable elsewhere.  ``pedsum_20M`` is declared but not measured by
default: pass ``--only pedsum_20M/...`` cells to run it.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _harness import UMBRELLA, Arm, Fixture, Measurement, Prepared, RunOrder, Suite, checksum_array, main

CATEGORIES = ("MZ", "FS", "MO", "FO", "MHS", "PHS", "1C")
PEDSUM = {
    "pedsum_2M": ("bench_pedsum/pedsum_2M/rep1/pedigree.full.parquet", "`pedsum_2M/rep1` (2,000,000 rows)"),
    "pedsum_20M": ("bench_pedsum/pedsum_20M/rep1/pedigree.full.parquet", "`pedsum_20M/rep1` (20,000,000 rows)"),
}
RESULTS = Path(os.environ.get("SIMACE_RESULTS", UMBRELLA / "results"))
"""The simACE ``results/`` directory; ``SIMACE_RESULTS`` points a worktree at the main checkout's outputs."""


_COLUMNS: dict[int, dict[str, np.ndarray]] = {}
"""The per-row columns of every built pedsum graph, by graph identity, for :func:`_columns`."""


def pedsum_fixture(name: str) -> Fixture:
    """The graph of one pedsum pedigree; its per-row columns reach the arms through :func:`_columns`."""
    relative, label = PEDSUM[name]
    path = RESULTS / relative

    def build() -> Any:
        import pyarrow.parquet as pq

        from pedigree_graph import PedigreeGraph

        table = pq.read_table(path)
        columns = {name: table[name].to_numpy() for name in table.column_names}
        graph = PedigreeGraph.from_frame(
            {"id": columns["id"], "mother": columns["mother"], "father": columns["father"], "twin": columns["twin"]}
        )
        _COLUMNS[id(graph)] = columns
        return graph

    def provenance() -> str:
        import hashlib

        return f"{label}; sha256 {hashlib.sha256(path.read_bytes()).hexdigest()[:16]}"

    return Fixture(name, label, build, provenance, available=path.exists)


def _columns(graph: Any) -> Prepared:
    """The graph's pedsum columns, or synthetic ones for the harness's warm-up graph."""
    columns = _COLUMNS.get(id(graph))
    if columns is None:
        rng = np.random.default_rng(0)
        n = graph.n_individuals
        columns = {
            "generation": rng.integers(0, 4, n),
            "sex": rng.integers(0, 2, n),
            "household_id": rng.integers(-1, n // 2, n),
            "liability1": rng.normal(size=n),
            "liability2": rng.normal(size=n),
        }
    return Prepared(payload=columns)


def _inputs(columns: dict[str, np.ndarray]) -> dict[str, Any]:
    return {
        "first": {"generation": columns["generation"], "sex": columns["sex"]},
        "second": {"sex": columns["sex"]},
        "values": {"liability1": columns["liability1"], "liability2": columns["liability2"]},
        "same": {"household": columns["household_id"]},
    }


def _moments(graph: Any, columns: dict[str, np.ndarray]) -> Measurement:
    result = graph.relationship_moments(categories=list(CATEGORIES), **_inputs(columns))
    return Measurement(
        lambda: checksum_array(result.counts.reshape(len(CATEGORIES), -1)),
        lambda: {
            "pairs": int(result.counts.sum()),
            "cells": int(np.prod(result.shape[1:])),
            "lanes": result.lanes,
            "estimated_peak_bytes": result.estimated_peak_bytes,
        },
    )


def _pairs_numpy(graph: Any, columns: dict[str, np.ndarray]) -> Measurement:
    inputs = _inputs(columns)
    pairs = graph.relationship_pairs(categories=list(CATEGORIES))
    generation = np.unique(inputs["first"]["generation"], return_inverse=True)[1]
    sex = np.unique(inputs["first"]["sex"], return_inverse=True)[1]
    n_gen, n_sex = int(generation.max()) + 1, int(sex.max()) + 1
    household = inputs["same"]["household"]
    values = [np.asarray(v, dtype=np.float64) for v in inputs["values"].values()]
    cells = n_gen * n_sex * n_sex * 2
    counts = np.zeros((len(CATEGORIES), cells), dtype=np.int64)
    # Registry order, as the moments result lays its category axis out.
    for ci, code in enumerate(code for code in pairs if code in CATEGORIES):
        first, second = pairs[code]
        same = (household[first] >= 0) & (household[first] == household[second])
        cell = ((generation[first] * n_sex + sex[first]) * n_sex + sex[second]) * 2 + same
        counts[ci] = np.bincount(cell, minlength=cells)
        for column in values:
            a, b = column[first], column[second]
            for weights in (a, b, a * a, b * b, a * b):
                np.bincount(cell, weights=weights, minlength=cells)
    return Measurement(lambda: checksum_array(counts), {"pairs": int(counts.sum()), "cells": cells})


SUITE = Suite(
    name="relationship_moments",
    note=Path(__file__).with_suffix(".md"),
    fixtures=(pedsum_fixture("pedsum_2M"), pedsum_fixture("pedsum_20M")),
    arms=(
        Arm("moments_1t", _moments, label="`relationship_moments`, 1 thread", setup=_columns),
        Arm("pairs_numpy_1t", _pairs_numpy, label="`relationship_pairs` + `np.bincount`, 1 thread", setup=_columns),
        Arm(
            "moments_12t",
            _moments,
            label="`relationship_moments`, 12 threads",
            setup=_columns,
            env={"PEDIGREE_GRAPH_THREADS": "12"},
        ),
        Arm(
            "pairs_numpy_12t",
            _pairs_numpy,
            label="`relationship_pairs` + `np.bincount`, 12 threads",
            setup=_columns,
            env={"PEDIGREE_GRAPH_THREADS": "12"},
        ),
    ),
    cells=tuple(f"pedsum_2M/{arm}" for arm in ("moments_1t", "pairs_numpy_1t", "moments_12t", "pairs_numpy_12t")),
    gate=None,
    order=RunOrder.INTERLEAVED,
    timeout_s=7200.0,
)

if __name__ == "__main__":
    main(SUITE)
