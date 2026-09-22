"""``pair_kinship`` and ``relationship_kinship_matrix`` on one graph: the slice 13 benchmark.

A call on ``random_30k`` costs its ancestral closure rather than its pair
count.  Slice 5d kept the memo on the graph so a second call could start from
it; slice 13 moves the walk to the Rust core with one memo per call (ADR
0007), so the two warm cells are recorded as an accepted cost, not scored:

    python benchmarks/bench_pair_kinship.py --repeat 3 --out benchmarks/reports/pair_kinship.json
    python benchmarks/bench_pair_kinship.py --render benchmarks/reports/pair_kinship.json

``cold`` is the first degree-3 query on a fresh graph and ``warm`` the same
query again on the same graph, prepared outside the timed region.  ``matrix``
is ``relationship_kinship_matrix(max_degree=3)`` on a fresh graph and
``matrix_after_pairs`` the same matrix after a degree-3 ``pair_kinship``.
Every arm checksums its values, so the cells also show each path returning
the same bits.

``cold`` and ``matrix`` are the scored cells: the native walk must not slow
the first call or grow its peak RSS beyond the 5% rule against the 0.9.0
wheel.  The memo layout under test is selected per subprocess with
``PEDIGREE_GRAPH_KINSHIP_LAYOUT`` (``rows`` or ``flat``) while the slice 13
bake-off runs.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _harness import (
    Arm,
    Gate,
    Measurement,
    Prepared,
    RunOrder,
    Suite,
    checksum_matrix_upper,
    checksum_values,
    main,
    parity_fixture,
)

MAX_DEGREE = 3


def _layout_facts() -> dict[str, str]:
    from pedigree_graph._kinship_pairwise import _MEMO_LAYOUT

    return {"layout": _MEMO_LAYOUT}


def _pairs_setup(graph) -> Prepared:
    pairs = graph.relationship_pairs(max_degree=MAX_DEGREE)
    return Prepared(payload=pairs, facts={"pairs": sum(len(block) for block in pairs.values())})


def _warm_setup(graph) -> Prepared:
    prepared = _pairs_setup(graph)
    graph.pair_kinship(prepared.payload)
    return prepared


def _pair_kinship(graph, pairs) -> Measurement:
    values = graph.pair_kinship(pairs)
    checksum = 0
    for code in pairs:
        if len(values[code]):
            checksum ^= checksum_values(values[code])
    return Measurement(checksum, _layout_facts())


def _matrix(graph, _prepared) -> Measurement:
    matrix = graph.relationship_kinship_matrix(max_degree=MAX_DEGREE)
    return Measurement(checksum_matrix_upper(matrix), {"nnz": int(matrix.nnz), **_layout_facts()})


SUITE = Suite(
    name="pair_kinship",
    note=Path(__file__).with_suffix(".md"),
    fixtures=(parity_fixture("random_30k", label="`random_30k`"),),
    arms=(
        Arm("cold", _pair_kinship, label="degree-3 pair_kinship, fresh graph", setup=_pairs_setup),
        Arm("warm", _pair_kinship, label="the same query again on the same graph", setup=_warm_setup),
        Arm("matrix", _matrix, label="relationship_kinship_matrix(max_degree=3), fresh graph"),
        Arm(
            "matrix_after_pairs",
            _matrix,
            label="the same matrix after a degree-3 pair_kinship",
            setup=_warm_setup,
        ),
    ),
    gate=Gate(baseline="cold"),
    order=RunOrder.INTERLEAVED,
    timeout_s=3600.0,
)

if __name__ == "__main__":
    main(SUITE)
