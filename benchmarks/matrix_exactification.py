"""Why approximate-support values are captured through one complete retiring DP.

Compares the fused capture against pairwise evaluation in bounded chunks, and
records what the relationship-selected support costs on the same inputs.

Regenerates every number in ``benchmarks/matrix_exactification.md``:

    python benchmarks/matrix_exactification.py --repeat 5 --out benchmarks/reports/sweep.json
    python benchmarks/matrix_exactification.py --render benchmarks/reports/sweep.json

Ordering is ``GROUPED`` rather than interleaved: one cell here runs for nearly
two hours, so finishing a cell before starting the next keeps an interrupted
sweep useful.  Nothing here is gated, so ratios against the fused path are shown
and never block.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

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

FITACE_PEDIGREE = Path("/data/Documents/simACE/results/dev/dev_cont_n10k/rep1/pedigree.parquet")
THRESHOLD = 0.001


def _build_candidate(graph) -> sp.csc_matrix:
    from pedigree_graph._kinship_dp import _build_kinship_csc

    indptr, indices, propagated = _build_kinship_csc(
        graph.n_individuals,
        graph.mother_rows,
        graph.father_rows,
        graph.twin_rows,
        graph.depth,
        THRESHOLD,
    )
    return sp.csc_matrix((propagated, indices, indptr), shape=(graph.n_individuals,) * 2, copy=False)


def _upper_coordinates(matrix: sp.csc_matrix) -> tuple[np.ndarray, np.ndarray]:
    upper = sp.triu(matrix, k=0, format="coo")
    order = np.lexsort((upper.row, upper.col))
    return upper.row[order].astype(np.int32, copy=False), upper.col[order].astype(np.int32, copy=False)


def _evaluate_chunks(graph, first: np.ndarray, second: np.ndarray, chunks: list[tuple[int, int]]) -> Measurement:
    from pedigree_graph._kinship_pairwise import _pairwise_kinship_with_stats

    mother, father, twin = graph._topological_parents
    translate = graph._topology.translate
    checksum = 0
    capacity = 0
    for lo, hi in chunks:
        values, stats = _pairwise_kinship_with_stats(
            mother, father, twin, translate(first[lo:hi]), translate(second[lo:hi])
        )
        checksum ^= checksum_values(values)
        capacity = max(capacity, stats["memo_capacity"])
    return Measurement(checksum, {"chunks": len(chunks), "max_memo_capacity": capacity})


def _pairwise_setup(graph) -> Prepared:
    candidate = _build_candidate(graph)
    first, second = _upper_coordinates(candidate)
    return Prepared(payload=(first, second), facts={"upper_candidates": len(first)})


def _shared(graph, prepared) -> Measurement:
    first, second = prepared
    return _evaluate_chunks(graph, first, second, [(0, len(first))])


def _by_pairs(size: int):
    def run(graph, prepared) -> Measurement:
        first, second = prepared
        chunks = [(i, min(i + size, len(first))) for i in range(0, len(first), size)]
        return _evaluate_chunks(graph, first, second, chunks)

    return run


def _by_columns(width: int):
    def run(graph, prepared) -> Measurement:
        first, second = prepared
        edges = np.searchsorted(second, np.arange(0, graph.n_individuals + width, width))
        chunks = [(int(a), int(b)) for a, b in itertools.pairwise(edges) if b > a]
        return _evaluate_chunks(graph, first, second, chunks)

    return run


def _candidate_support(graph, _prepared) -> Measurement:
    candidate = _build_candidate(graph)
    # The support structure is the result here, so its indices are the checksum.
    return Measurement(
        checksum_values(candidate.indices.view(np.float32)),
        {"nnz": int(candidate.nnz), "upper_candidates": (int(candidate.nnz) + graph.n_individuals) // 2},
    )


def _complete_dp(graph, _prepared) -> Measurement:
    from pedigree_graph._kinship_dp import _compute_generation_kinship_summary

    summary = _compute_generation_kinship_summary(
        graph.n_individuals,
        graph.mother_rows,
        graph.father_rows,
        graph.twin_rows,
        graph.depth,
        0.0,
        labels=graph.depth,
    )
    theta = summary.mean_kinship
    return Measurement(
        checksum_values(np.ascontiguousarray(theta, dtype=np.float32)),
        {"reduction": float(np.nansum(theta))},
    )


def _fused(graph, _prepared) -> Measurement:
    matrix = graph.approximate_kinship_matrix(min_propagated_kinship=THRESHOLD)
    return Measurement(
        checksum_matrix_upper(matrix),
        {"nnz": int(matrix.nnz), "upper_candidates": (int(matrix.nnz) + graph.n_individuals) // 2},
    )


def _relationship(max_degree: int):
    def run(graph, _prepared) -> Measurement:
        matrix = graph.relationship_kinship_matrix(max_degree=max_degree)
        return Measurement(
            checksum_matrix_upper(matrix),
            {"nnz": int(matrix.nnz), "upper_candidates": (int(matrix.nnz) + graph.n_individuals) // 2},
        )

    return run


SUITE = Suite(
    name="matrix_exactification",
    note=Path(__file__).with_suffix(".md"),
    fixtures=(
        file_fixture("fitace", FITACE_PEDIGREE, label="fitACE `dev_cont_n10k/rep1`"),
        parity_fixture("random_30k", label="generated `random_30k`"),
    ),
    arms=(
        Arm("candidate_support", _candidate_support, label="propagation-pruned support only"),
        Arm("complete_dp", _complete_dp, label="complete retiring DP (no output CSC)"),
        Arm("fused", _fused, label="fused capture (public path)"),
        Arm("pairwise_shared", _shared, label="one shared memo", setup=_pairwise_setup),
        Arm("pairwise_columns256", _by_columns(256), label="256-column chunks", setup=_pairwise_setup),
        Arm("pairwise_pairs262144", _by_pairs(262144), label="262,144-pair chunks", setup=_pairwise_setup),
        Arm("pairwise_pairs1048576", _by_pairs(1048576), label="1,048,576-pair chunks", setup=_pairwise_setup),
        Arm("relationship_degree3", _relationship(3), label="relationship support, max_degree=3 (public path)"),
        Arm("relationship_degree5", _relationship(5), label="relationship support, max_degree=5 (public path)"),
    ),
    # Cheapest first, so an interrupted sweep still renders a usable table.
    cells=(
        "fitace/candidate_support",
        "fitace/complete_dp",
        "fitace/fused",
        "fitace/relationship_degree3",
        "fitace/relationship_degree5",
        "random_30k/candidate_support",
        "random_30k/complete_dp",
        "random_30k/fused",
        "random_30k/relationship_degree3",
        "random_30k/relationship_degree5",
        "fitace/pairwise_shared",
        "fitace/pairwise_pairs1048576",
        "fitace/pairwise_pairs262144",
        "fitace/pairwise_columns256",
        "random_30k/pairwise_pairs1048576",
        "random_30k/pairwise_shared",
    ),
    gate=Gate(baseline="fused", gated=frozenset()),
    order=RunOrder.GROUPED,
    timeout_s=14400.0,
)

if __name__ == "__main__":
    main(SUITE)
