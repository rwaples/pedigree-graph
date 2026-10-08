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
``counts_12t`` is ``relationship_counts`` over the same categories, the
engine pass with no sink work, which issue #28 sets as the wall target.

The checksum is over the pair counts per cell, which both arms produce
exactly; the float sums are compared by the ADR 0013 tests, not here.

The region arms (issue #30) time one step of the moments result each, in two
specs: ``c64`` is the 64-cell table above, ``c16k`` the 288 × 8 label table
of ADR 0013 (36 random generations × sex × two random flags for the first
member, sex × the first flag for the second, same household; 16,128 cells,
177,408 accumulators).  ``engine`` is the call with its conversion; every
other region runs the call in its untimed setup and times one derived view,
``sum`` over every non-category axis, ``merge`` of the result with itself, or
``export``, every float the R ``as.data.frame`` offers.  Their checksums are
over the exact integers or the float bits, so a revision that changes a
result changes the checksum.  The
fixtures are the ``bench_pedsum`` pedigrees generated in the simACE umbrella
(``results/bench_pedsum/pedsum_{2M,20M}/rep1/pedigree.full.parquet``,
or under ``$SIMACE_RESULTS``) and are unavailable elsewhere.  ``pedsum_20M`` is declared but not measured by
default: pass ``--only pedsum_20M/...`` cells to run it.  ``random_1k`` with
synthetic columns is for a quick end-to-end run of the script.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _harness import (
    RESULTS,
    Arm,
    Fixture,
    Measurement,
    Prepared,
    RunOrder,
    Suite,
    checksum_array,
    checksum_ints,
    main,
    parity_fixture,
)

CATEGORIES = ("MZ", "FS", "MO", "FO", "MHS", "PHS", "1C")
PEDSUM = {
    "pedsum_2M": ("bench_pedsum/pedsum_2M/rep1/pedigree.full.parquet", "`pedsum_2M/rep1` (2,000,000 rows)"),
    "pedsum_20M": ("bench_pedsum/pedsum_20M/rep1/pedigree.full.parquet", "`pedsum_20M/rep1` (20,000,000 rows)"),
}


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


HIGH_SHAPE = (len(CATEGORIES), 36, 2, 2, 2, 2, 2, 2)


def _inputs_high(columns: dict[str, np.ndarray]) -> dict[str, Any]:
    n = len(columns["sex"])
    rng = np.random.default_rng(30)
    generation, flag_a, flag_b = rng.integers(0, 36, n), rng.integers(0, 2, n), rng.integers(0, 2, n)
    return {
        "first": {"generation": generation, "sex": columns["sex"], "flag_a": flag_a, "flag_b": flag_b},
        "second": {"sex": columns["sex"], "flag_a": flag_a},
        "values": {"liability1": columns["liability1"], "liability2": columns["liability2"]},
        "same": {"household": columns["household_id"]},
    }


SPECS = {"c64": ("64 cells", _inputs), "c16k": ("16,128 cells", _inputs_high)}


def _call(graph: Any, spec: str, columns: dict[str, np.ndarray]) -> Any:
    result = graph.relationship_moments(categories=list(CATEGORIES), **SPECS[spec][1](columns))
    if spec == "c16k" and id(graph) in _COLUMNS:
        assert result.shape == HIGH_SHAPE, result.shape
    return result


def export(m: Any) -> dict[str, np.ndarray]:
    """Every derived view, the set R's ``as.data.frame`` computes; one function for baseline and revision."""
    out = {
        name: getattr(m, name)
        for name in ("sum_first", "sum_second", "sumsq_first", "sumsq_second", "cross", "m2_first", "m2_second")
    }
    out["comoment"] = m.comoment
    for side in ("first", "second"):
        for column in m.columns:
            out[f"mean_{side}.{column}"] = m.mean(f"{side}.{column}")
    for a, b in m.products:
        out[f"pearson_{a}:{b}"] = m.pearson(a, b)
    return out


def _floats_checksum(arrays: dict[str, np.ndarray]) -> int:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        digest.update(f"{name}={checksum_array(arrays[name])};".encode())
    return int.from_bytes(digest.digest()[:8], "big")


def _table_checksum(m: Any) -> int:
    """The exact integers, exponents and axes of a moments table."""
    digest = hashlib.sha256()
    for axis in m.axes:
        digest.update(f"{axis.name}={[str(level) for level in axis.levels]};".encode())
    digest.update(f"exponents={m.exponents.tolist()};counts={checksum_array(m.counts)};".encode())
    for name in ("q_sum_first", "q_sum_second", "q_sumsq_first", "q_sumsq_second", "q_cross"):
        digest.update(f"{name}={','.join(map(str, getattr(m, name).reshape(-1).tolist()))};".encode())
    return int.from_bytes(digest.digest()[:8], "big")


def _table_facts(m: Any) -> dict[str, Any]:
    return {"pairs": int(sum(int(n) for n in m.counts.reshape(-1).tolist())), "shape": list(m.shape)}


REGIONS: dict[str, tuple[str, Any]] = {
    "mean": ("`mean(first.liability1)`", lambda m: {"mean": m.mean("first.liability1")}),
    "m2_first": ("`m2_first`", lambda m: {"m2_first": m.m2_first}),
    "comoment": ("`comoment`", lambda m: {"comoment": m.comoment}),
    "cross": ("`cross`", lambda m: {"cross": m.cross}),
    "pearson": (
        "`pearson(first.liability1, second.liability1)`",
        lambda m: {"pearson": m.pearson("first.liability1", "second.liability1")},
    ),
    "export": ("complete export", export),
}


def _region_arms(spec: str) -> tuple[Arm, ...]:
    label = SPECS[spec][0]
    threads = {"PEDIGREE_GRAPH_THREADS": "12"}

    def engine(graph: Any, columns: dict[str, np.ndarray]) -> Measurement:
        m = _call(graph, spec, columns)
        return Measurement(lambda: _table_checksum(m), lambda: _table_facts(m))

    def called(graph: Any) -> Prepared:
        return Prepared(payload=_call(graph, spec, _columns(graph).payload))

    def view(fn: Any) -> Any:
        def run(graph: Any, m: Any) -> Measurement:
            arrays = fn(m)
            return Measurement(lambda: _floats_checksum(arrays))

        return run

    def summed(graph: Any, m: Any) -> Measurement:
        folded = m.sum(*(axis.name for axis in m.axes if axis.name != "category"))
        return Measurement(lambda: _table_checksum(folded), lambda: _table_facts(folded))

    def merged(graph: Any, m: Any) -> Measurement:
        both = m.merge(m)
        return Measurement(lambda: _table_checksum(both), lambda: _table_facts(both))

    arms = [Arm(f"{spec}_engine", engine, label=f"{label}: engine call", setup=_columns, env=threads)]
    arms += [
        Arm(f"{spec}_{name}", view(fn), label=f"{label}: {what}", setup=called, env=threads)
        for name, (what, fn) in REGIONS.items()
    ]
    arms += [
        Arm(f"{spec}_sum", summed, label=f"{label}: `sum` over every non-category axis", setup=called, env=threads),
        Arm(f"{spec}_merge", merged, label=f"{label}: `merge` with itself", setup=called, env=threads),
    ]
    return tuple(arms)


REGION_ARMS = (*_region_arms("c64"), *_region_arms("c16k"))


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


def _counts(graph: Any, columns: dict[str, np.ndarray]) -> Measurement:
    counts = graph.relationship_counts(categories=list(CATEGORIES))
    return Measurement(
        lambda: checksum_ints(dict(counts)), lambda: {"pairs": sum(c or 0 for c in dict(counts).values())}
    )


SUITE = Suite(
    name="relationship_moments",
    note=Path(__file__).with_suffix(".md"),
    fixtures=(
        pedsum_fixture("pedsum_2M"),
        pedsum_fixture("pedsum_20M"),
        parity_fixture("random_1k", label="`random_1k` parity pedigree, synthetic columns"),
    ),
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
        Arm(
            "counts_12t",
            _counts,
            label="`relationship_counts`, 12 threads",
            setup=_columns,
            env={"PEDIGREE_GRAPH_THREADS": "12"},
        ),
        *REGION_ARMS,
    ),
    cells=tuple(f"pedsum_2M/{arm.name}" for arm in REGION_ARMS),
    gate=None,
    order=RunOrder.INTERLEAVED,
    timeout_s=7200.0,
)

if __name__ == "__main__":
    main(SUITE)
