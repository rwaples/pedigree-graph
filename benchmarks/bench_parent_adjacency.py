"""What the parent adjacency costs to reach, eagerly in two CSRs or lazily in one COO (issue #18).

Before this suite's change, ``_initialize`` eagerly built ``_Am`` and ``_Af``
and the only production read of either was their sum in ``_A``.  Reaching the
adjacency therefore allocated both halves and their sum, and a caller who never
touched a relationship path paid for both halves anyway.

    python benchmarks/bench_parent_adjacency.py --repeat 5 --out benchmarks/reports/parent_adjacency.json

The ``eager`` arm is the deleted builder, copied verbatim, so old and new are
one interleaved sweep in one process rather than two sweeps at two commits.
``eager`` and ``lazy`` end in the same state and must agree on the checksum of
``_A``; that agreement is what makes the wall and RSS comparison mean anything.

``construct`` measures the other half of the claim, the cost a caller pays who
never reaches the adjacency at all.  Nothing inside one commit can A/B that, so
it is the cell to re-run against a pre-change worktree, and its checksum covers
the edge structure rather than ``_A`` because it deliberately never builds one.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _harness import Arm, Gate, Measurement, Prepared, Suite, checksum_ints, main, parity_fixture


def _columns(graph) -> Prepared:
    columns = {
        "id": np.asarray(graph.ids),
        "mother": np.asarray(graph.mother_ids),
        "father": np.asarray(graph.father_ids),
        "twin": np.asarray(graph.twin_ids),
    }
    if graph.sex is not None:
        columns["sex"] = np.asarray(graph.sex)
    return Prepared(payload=columns, facts={"n_rows": int(graph.n_individuals)})


def _matrix_mib(matrix) -> float:
    return (matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes) / 2**20


def _checksum_adjacency(matrix: sp.csr_matrix) -> int:
    """Hash the three CSR arrays and their dtypes, in layout order and without canonicalising.

    An order-insensitive digest would let a differently sorted or
    duplicate-carrying matrix match, and canonicity is load-bearing:
    ``_streaming_counter`` reads child counts straight off ``_A.tocsc().indptr``.
    Two arms agree here only when their matrices are byte-identical.
    """
    digest = hashlib.sha256()
    for part in (matrix.indptr, matrix.indices, matrix.data):
        digest.update(str(part.dtype).encode())
        digest.update(np.ascontiguousarray(part).tobytes())
    return int.from_bytes(digest.digest()[:4], "big")


def _build_parent_csr(graph) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """``PedigreeGraph._build_parent_csr`` as it stood at ``93f6c97``, before deletion.

    Frozen here so the pre-change cost stays measurable after the method is
    gone.  Any drift from the original is a drift in what "before" means, so it
    is copied rather than adapted.
    """
    n = graph.n_individuals
    m_idx = np.where(graph.mother_rows >= 0)[0]
    f_idx = np.where(graph.father_rows >= 0)[0]
    return (
        sp.csr_matrix(
            (np.ones(len(m_idx), dtype=np.int32), (m_idx, graph.mother_rows[m_idx])),
            shape=(n, n),
        ),
        sp.csr_matrix(
            (np.ones(len(f_idx), dtype=np.int32), (f_idx, graph.father_rows[f_idx])),
            shape=(n, n),
        ),
    )


def _eager(_graph, columns) -> Measurement:
    from pedigree_graph import PedigreeGraph

    graph = PedigreeGraph.from_frame(columns)
    mother_csr, father_csr = _build_parent_csr(graph)
    adjacency = mother_csr + father_csr
    return Measurement(
        _checksum_adjacency(adjacency),
        {
            "adjacency_mib": _matrix_mib(adjacency),
            "halves_mib": _matrix_mib(mother_csr) + _matrix_mib(father_csr),
            "resident_mib": _matrix_mib(adjacency) + _matrix_mib(mother_csr) + _matrix_mib(father_csr),
            "canonical": bool(adjacency.has_canonical_format),
            "data_dtype": str(adjacency.data.dtype),
        },
    )


def _lazy(_graph, columns) -> Measurement:
    from pedigree_graph import PedigreeGraph

    graph = PedigreeGraph.from_frame(columns)
    adjacency = graph._A
    return Measurement(
        _checksum_adjacency(adjacency),
        {
            "adjacency_mib": _matrix_mib(adjacency),
            "halves_mib": 0.0,
            "resident_mib": _matrix_mib(adjacency),
            "canonical": bool(adjacency.has_canonical_format),
            "data_dtype": str(adjacency.data.dtype),
        },
    )


def _construct(_graph, columns) -> Measurement:
    from pedigree_graph import PedigreeGraph

    graph = PedigreeGraph.from_frame(columns)
    return Measurement(
        checksum_ints(
            {
                "n_rows": int(graph.n_individuals),
                "mother_edges": int((graph.mother_rows >= 0).sum()),
                "father_edges": int((graph.father_rows >= 0).sum()),
            }
        ),
        {"holds_parent_csr": hasattr(graph, "_Am") or hasattr(graph, "_Af")},
    )


SUITE = Suite(
    name="parent_adjacency",
    note=Path(__file__).with_suffix(".md"),
    fixtures=(
        parity_fixture("random_30k", label="`random_30k`"),
        parity_fixture("random_300k", label="`random_300k`"),
    ),
    arms=(
        Arm("eager", _eager, label="two CSRs then their sum", setup=_columns),
        Arm("lazy", _lazy, label="one COO", setup=_columns),
        Arm("construct", _construct, label="construction only, adjacency never read", setup=_columns),
    ),
    gate=Gate(baseline="eager", gated=frozenset({"lazy"})),
    timeout_s=1800.0,
)

if __name__ == "__main__":
    main(SUITE)
