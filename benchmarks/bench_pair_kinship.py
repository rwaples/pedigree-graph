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

Every fresh-graph cell is scored: the native walk must not slow the first
call or grow its peak RSS beyond the 5% rule against the 0.9.0 wheel.  The
slice 13 record is ``docs/pedigree-graph-0.8-migration/gate/13a/NOTES.md``;
``compare_reports.py`` beside it lays several reports side by side.
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
    file_fixture,
    main,
    parity_fixture,
)

MAX_DEGREE = 3

# The simACE study pedigrees (ADR 0009 corpus), machine-local under the
# umbrella's results/; absent files record as unavailable.
UMBRELLA = Path(__file__).resolve().parent.parent.parent.parent
STUDY = {
    "dev_mean_n10k": ("results/dev/dev_mean_n10k/rep1/pedigree.parquet", "`dev_mean_n10k/rep1` (20,400 rows)"),
    "baseline10K": ("results/base/baseline10K/rep1/pedigree.parquet", "`baseline10K/rep1` (53,466 rows)"),
    "baseline100K": ("results/base/baseline100K/rep1/pedigree.parquet", "`baseline100K/rep1` (536,036 rows)"),
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
    return Measurement(checksum, {})


def _self_pairs(graph, _prepared) -> Measurement:
    import numpy as np

    rows = np.arange(graph.n_individuals, dtype=np.int32)
    return Measurement(checksum_values(graph.pair_kinship(rows, rows)), {})


def _pairs5_setup(graph) -> Prepared:
    pairs = graph.relationship_pairs(max_degree=5)
    return Prepared(payload=pairs, facts={"pairs": sum(len(block) for block in pairs.values())})


def _matrix(graph, _prepared) -> Measurement:
    matrix = graph.relationship_kinship_matrix(max_degree=MAX_DEGREE)
    return Measurement(checksum_matrix_upper(matrix), {"nnz": int(matrix.nnz)})


SUITE = Suite(
    name="pair_kinship",
    note=Path(__file__).with_suffix(".md"),
    fixtures=(
        parity_fixture("random_30k", label="`random_30k`"),
        parity_fixture("random_300k", label="`random_300k`"),
        *(file_fixture(name, UMBRELLA / rel, label=label) for name, (rel, label) in STUDY.items()),
    ),
    arms=(
        Arm("cold", _pair_kinship, label="degree-3 pair_kinship, fresh graph", setup=_pairs_setup),
        Arm("self", _self_pairs, label="pair_kinship over every self pair, fresh graph"),
        Arm("pairs5", _pair_kinship, label="degree-5 pair_kinship, fresh graph", setup=_pairs5_setup),
        Arm("warm", _pair_kinship, label="the same query again on the same graph", setup=_warm_setup),
        Arm("matrix", _matrix, label="relationship_kinship_matrix(max_degree=3), fresh graph"),
        Arm(
            "matrix_after_pairs",
            _matrix,
            label="the same matrix after a degree-3 pair_kinship",
            setup=_warm_setup,
        ),
    ),
    cells=(
        "random_30k/cold",
        "random_30k/warm",
        "random_30k/matrix",
        "random_30k/matrix_after_pairs",
        "random_300k/cold",
        "baseline10K/cold",
        "baseline10K/self",
        "baseline100K/cold",
        "baseline100K/self",
        "baseline100K/matrix",
        "dev_mean_n10k/pairs5",
    ),
    gate=Gate(baseline="cold"),
    order=RunOrder.INTERLEAVED,
    timeout_s=3600.0,
)

if __name__ == "__main__":
    main(SUITE)
