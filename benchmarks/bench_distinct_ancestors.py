"""A/B benchmark for ``PedigreeGraph.distinct_ancestor_counts`` (issue #1).

The baseline freezes the removed sparse boolean transitive closure. The second
arm measures the production retiring Numba DP across controlled width and depth
series, then on the parity fixtures shared by the other benchmarks.

    pixi run python benchmarks/bench_distinct_ancestors.py \
        --repeat 5 --out benchmarks/reports/distinct_ancestors.json

The retiring arm is gated against the closure. Accepted cells document the
fixed Numba runtime RSS chosen with the one-method design.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests" / "parity"))

from _harness import Arm, Fixture, Gate, Measurement, Suite, checksum_values, main, parity_fixture

from pedigree_graph._lineage_kernel import _compute_n_ancestors_profiled

_WIDTH_FIXTURES = {
    "closed_w200_g8": {"seed": 20_008, "n_founders": 200, "n_generations": 8, "per_generation": 200},
    "closed_w1000_g8": {"seed": 100_008, "n_founders": 1000, "n_generations": 8, "per_generation": 1000},
    "closed_w2000_g8": {"seed": 200_008, "n_founders": 2000, "n_generations": 8, "per_generation": 2000},
}

_DEPTH_FIXTURES = {
    "closed_w128_g8": {"seed": 12_808, "n_founders": 128, "n_generations": 8, "per_generation": 128},
    "closed_w128_g16": {"seed": 12_816, "n_founders": 128, "n_generations": 16, "per_generation": 128},
    "closed_w128_g32": {"seed": 12_832, "n_founders": 128, "n_generations": 32, "per_generation": 128},
    "closed_w128_g60": {"seed": 12_860, "n_founders": 128, "n_generations": 60, "per_generation": 128},
}
_FIXTURES = _WIDTH_FIXTURES | _DEPTH_FIXTURES


def _closed_fixture(name: str) -> Fixture:
    """Build one deterministic closed-parentage scaling fixture."""
    params = _FIXTURES[name]

    def frame() -> dict[str, np.ndarray]:
        import pedigrees

        return pedigrees.deep_inbred_pedigree(**params)

    def build():
        from pedigree_graph import PedigreeGraph

        columns = frame()
        return PedigreeGraph.from_frame(
            {
                "id": columns["ids"],
                "mother": columns["mother"],
                "father": columns["father"],
                "twin": columns["twin"],
                "sex": columns["sex"],
            }
        )

    def provenance() -> str:
        import pedigrees

        return pedigrees.input_hash(frame())

    n = params["n_founders"] + params["n_generations"] * params["per_generation"]
    label = (
        f"`{name}` ({n:,} rows, {params['n_founders']:,} founders, "
        f"{params['n_generations']} generated generations, {params['per_generation']:,}/generation)"
    )
    return Fixture(name=name, label=label, build=build, provenance=provenance)


def _sparse_closure_counts(graph) -> np.ndarray:
    """The deleted SciPy implementation, frozen as the benchmark baseline."""
    n = graph.n_individuals
    if n == 0:
        return np.zeros(0, dtype=np.int32)

    mother, father = graph.mother_rows, graph.father_rows
    mother_mask = mother >= 0
    father_mask = father >= 0
    if not mother_mask.any() and not father_mask.any():
        return np.zeros(n, dtype=np.int32)

    rows = np.concatenate([np.where(mother_mask)[0], np.where(father_mask)[0]]).astype(np.int32)
    columns = np.concatenate([mother[mother_mask], father[father_mask]]).astype(np.int32)
    adjacency = sp.csr_matrix(
        (np.ones(len(rows), dtype=np.int8), (rows, columns)),
        shape=(n, n),
    )

    closure = adjacency.copy()
    previous_nnz = -1
    while closure.nnz != previous_nnz:
        previous_nnz = closure.nnz
        combined = closure + closure @ adjacency
        combined.data = (combined.data > 0).astype(np.int8)
        combined.eliminate_zeros()
        closure = combined
    return np.diff(closure.indptr).astype(np.int32)


def _measurement(counts: np.ndarray, facts: dict[str, int] | None = None) -> Measurement:
    """Checksum the result and describe the final closure it represents."""
    total = int(counts.sum(dtype=np.int64))
    return Measurement(
        checksum_values(counts),
        {
            "total_ancestor_links": total,
            "max_ancestors": int(counts.max(initial=0)),
            "mean_ancestors": float(total / len(counts)) if len(counts) else 0.0,
            **(facts or {}),
        },
    )


def _closure(graph, _prepared) -> Measurement:
    """Measure the frozen pre-change sparse transitive closure."""
    return _measurement(_sparse_closure_counts(graph))


def _retiring(graph, _prepared) -> Measurement:
    """Measure the production retiring DP, including allocator telemetry."""
    topology = None
    if graph._rows_are_topological:
        mother, father = graph.mother_rows, graph.father_rows
    else:
        topology = graph._topology
        mother, father, _ = graph._topological_parents
    counts, peak_live, highwater, allocated, scanned, reused = _compute_n_ancestors_profiled(
        mother, father, graph.n_individuals
    )
    if topology is not None:
        counts = topology.per_row_to_graph(counts)
    return _measurement(
        counts,
        {
            "peak_live_capacity_entries": int(peak_live),
            "pool_highwater_entries": int(highwater),
            "pool_allocated_entries": int(allocated),
            "parent_set_entries_scanned": int(scanned),
            "reused_slots": int(reused),
        },
    )


SUITE = Suite(
    name="distinct_ancestors",
    note=Path(__file__).with_suffix(".md"),
    fixtures=(
        _closed_fixture("closed_w200_g8"),
        _closed_fixture("closed_w1000_g8"),
        _closed_fixture("closed_w2000_g8"),
        _closed_fixture("closed_w128_g8"),
        _closed_fixture("closed_w128_g16"),
        _closed_fixture("closed_w128_g32"),
        _closed_fixture("closed_w128_g60"),
        parity_fixture("random_1k", label="`random_1k`"),
        parity_fixture("deep_inbred_60g", label="`deep_inbred_60g`"),
        parity_fixture("random_30k", label="`random_30k`"),
        parity_fixture("random_300k", label="`random_300k`"),
    ),
    arms=(
        Arm("closure", _closure, label="frozen sparse boolean transitive closure"),
        Arm("retiring", _retiring, label="production retiring sorted-set DP"),
    ),
    gate=Gate(
        baseline="closure",
        gated=frozenset({"retiring"}),
        accepted={
            "closed_w200_g8/retiring": "accepted fixed Numba runtime RSS on a sub-20ms baseline",
            "closed_w1000_g8/retiring": "accepted fixed Numba runtime RSS for the one-method design",
            "closed_w2000_g8/retiring": "accepted 9 MiB peak RSS for a 5.1x wall-time gain",
            "closed_w128_g8/retiring": "accepted fixed Numba runtime RSS on a sub-20ms baseline",
            "closed_w128_g16/retiring": "accepted 24 MiB peak RSS for a 19x wall-time gain",
            "random_1k/retiring": "accepted fixed Numba runtime RSS on a sub-5ms baseline",
            "deep_inbred_60g/retiring": "accepted fixed Numba runtime RSS for a 135x wall-time gain",
        },
    ),
    timeout_s=3600.0,
)

if __name__ == "__main__":
    main(SUITE)
