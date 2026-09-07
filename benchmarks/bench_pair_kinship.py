"""Cold versus warm ``pair_kinship`` on one graph: the slice 5d memo gate.

A kernel call on ``random_30k`` costs its ancestral closure rather than its
pair count, and before slice 5d every call rebuilt that closure from nothing.
The graph now keeps the memo its last call left behind, so this suite measures
what that buys and what it costs:

    python benchmarks/bench_pair_kinship.py --repeat 5 --out benchmarks/reports/pair_kinship.json
    python benchmarks/bench_pair_kinship.py --render benchmarks/reports/pair_kinship.json

``cold`` is the first degree-3 query on a fresh graph and ``warm`` the same
query again on the same graph, prepared outside the timed region.  ``matrix``
is ``relationship_kinship_matrix(max_degree=3)`` on a fresh graph and
``matrix_after_pairs`` the same matrix after a degree-3 ``pair_kinship`` has
already walked the closure.  Every arm checksums its values, so the four cells
also show the warm path returning the cold path's bits.

``cold`` is the baseline and the regression gate: the memo handoff must not
slow the first call or grow its peak RSS by more than the 5% rule.  Run the
same file against the pre-slice commit for the other side of that comparison;
there the "warm" arms simply measure a second cold call.
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


def _memo_facts(graph) -> dict[str, int]:
    memo = getattr(graph, "_pair_memo", None)
    if memo is None:
        return {"memo_entries": 0, "memo_capacity": 0, "memo_mib": 0}
    return {
        "memo_entries": int(memo.entries),
        "memo_capacity": int(memo.capacity),
        "memo_mib": int(memo.nbytes >> 20),
    }


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
    return Measurement(checksum, _memo_facts(graph))


def _matrix(graph, _prepared) -> Measurement:
    matrix = graph.relationship_kinship_matrix(max_degree=MAX_DEGREE)
    return Measurement(checksum_matrix_upper(matrix), {"nnz": int(matrix.nnz), **_memo_facts(graph)})


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
