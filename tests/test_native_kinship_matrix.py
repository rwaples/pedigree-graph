"""The three kinship-matrix bindings against the 0.9.1 numba DP kept under ``tests/oracle``.

Slice 14 moves the depth-major DP behind ``kinship_matrix``,
``approximate_kinship_matrix`` and ``mean_kinship_by_generation`` to the Rust
core (ADR 0007, 0009).  These tests hold the raw bindings to the bytes the
Python DP produced on every parity fixture, in permuted row orders too, and
pin the boundary contract: owned arrays, structured errors, and every matrix
allocation family surfacing as ``allocation_failed``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from conftest import parity_columns, parity_fixtures
from oracle.kinship_dp.dp import KinshipDPConfig, _build_kinship_csc, _run_dp_core, _stream_sum_theta_per_gen
from test_native_relationship_pairs import CHILD_PRELUDE, _run_child
from test_pedigree_graph import _PAIRWISE_FIXTURES

import pedigree_graph
from pedigree_graph import PedigreeGraph, PedigreeValidationError, _native
from pedigree_graph._cohorts import _densify_labels

FIXTURES = parity_fixtures("random_1k", "deep_inbred_60g")
FIXTURE_NAMES = sorted(FIXTURES)
THRESHOLD = 0.001
PERMUTATION_SEEDS = (None, 5, 11)


def _graph(name: str, seed: int | None = None) -> PedigreeGraph:
    columns = parity_columns(FIXTURES[name])
    if seed is not None:
        perm = np.random.default_rng(seed).permutation(len(columns["id"]))
        columns = {key: value[perm] for key, value in columns.items()}
    return PedigreeGraph.from_frame(columns)


def _oracle_csc(graph: PedigreeGraph, threshold: float) -> tuple[bytes, bytes, bytes]:
    """The oracle DP's support at *threshold* with its values recomputed exactly.

    The Python DP's thresholded pass leaves propagated values on the support;
    the 0.9.1 facade then replaced them by a second complete pass.  Here the
    exact values come from ``kinship_support_values`` instead, the pairwise
    walk slice 13 holds to its own oracle, so the structure is the oracle's
    and the values are the pinned recurrence.  At threshold 0 the DP's own
    values are already exact and are compared as they are.
    """
    indptr, indices, data = _build_kinship_csc(
        graph.n_individuals, graph.mother_rows, graph.father_rows, graph.twin_rows, graph.depth, threshold
    )
    if threshold > 0.0:
        data = _native.kinship_support_values(graph._built, graph.depth, indptr.astype(np.int64), indices)
    return indptr.tobytes(), indices.tobytes(), data.tobytes()


def _oracle_sums(
    graph: PedigreeGraph, dense: np.ndarray, n_buckets: int, init_cap_per_row: int | None = None
) -> np.ndarray:
    result = _run_dp_core(
        graph.n_individuals,
        graph.mother_rows,
        graph.father_rows,
        graph.twin_rows,
        graph.depth,
        0.0,
        init_cap_per_row,
        labels=dense,
        config=KinshipDPConfig(retire=True, lazy=True, debug_asserts=False),
    )
    # The oracle sizes its accumulator by the largest label present, so an
    # empty sentinel bucket is absent rather than zero.
    sums = np.zeros(n_buckets, dtype=np.float64)
    sums[: result.sum_theta.shape[0]] = result.sum_theta
    return sums


def _bytes(arrays) -> tuple[bytes, ...]:
    return tuple(array.tobytes() for array in arrays)


@pytest.mark.parametrize("seed", PERMUTATION_SEEDS)
@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_complete_and_approximate_match_the_oracle_bytes(name, seed):
    graph = _graph(name, seed)
    assert _bytes(_native.kinship_csc(graph._built, graph.depth)) == _oracle_csc(graph, 0.0)
    assert _bytes(_native.approximate_kinship_csc(graph._built, graph.depth, THRESHOLD)) == _oracle_csc(
        graph, THRESHOLD
    )


@pytest.mark.parametrize("seed", PERMUTATION_SEEDS)
@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_generation_sums_match_the_oracle_bits(name, seed):
    graph = _graph(name, seed)
    n = graph.n_individuals
    shuffled = np.random.default_rng(3).integers(-1, 4, n).astype(np.int32)
    for labels in (np.asarray(graph.depth, dtype=np.int32), shuffled):
        dense, observed, _ = _densify_labels(labels)
        n_buckets = int(observed.shape[0]) + 1
        got = _native.generation_kinship_sums(graph._built, graph.depth, dense, n_buckets)
        assert got.dtype == np.float64
        assert got.tobytes() == _oracle_sums(graph, dense, n_buckets).tobytes()


@pytest.mark.parametrize("build", _PAIRWISE_FIXTURES, ids=lambda b: b.__name__)
def test_mz_and_inbred_constructions_match_the_oracle(build):
    graph = PedigreeGraph.from_frame(build())
    assert _bytes(_native.kinship_csc(graph._built, graph.depth)) == _oracle_csc(graph, 0.0)
    assert _bytes(_native.approximate_kinship_csc(graph._built, graph.depth, 0.2)) == _oracle_csc(graph, 0.2)


def test_a_parentless_row_above_depth_zero_keeps_its_diagonal():
    """The raw binding accepts any structural depth; a founder raised above 0 is still a founder.

    ``PedigreeGraph`` always passes the true structural depth, so only the
    bindings can reach this; 0.9.2 dropped such a row's diagonal and its
    edges to descendants.
    """
    graph = _graph("random_1k")
    # Doubling keeps every child strictly below its parents; the extra one
    # then puts every founder at an odd depth of at least 1.
    depth = np.asarray(graph.depth, dtype=np.int32) * 2
    founders = (np.asarray(graph.mother_rows) < 0) & (np.asarray(graph.father_rows) < 0)
    depth[founders] += 1

    class Raised:
        n_individuals = graph.n_individuals
        mother_rows = graph.mother_rows
        father_rows = graph.father_rows
        twin_rows = graph.twin_rows
        _built = graph._built

    Raised.depth = depth
    assert _bytes(_native.kinship_csc(graph._built, depth)) == _oracle_csc(Raised, 0.0)
    assert _bytes(_native.approximate_kinship_csc(graph._built, depth, THRESHOLD)) == _oracle_csc(Raised, THRESHOLD)
    dense, observed, _ = _densify_labels(np.asarray(graph.depth, dtype=np.int32))
    n_buckets = int(observed.shape[0]) + 1
    got = _native.generation_kinship_sums(graph._built, depth, dense, n_buckets)
    assert got.tobytes() == _oracle_sums(Raised, dense, n_buckets).tobytes()


def test_a_retired_row_cannot_be_resurrected():
    """A founder whose last child is at depth 1 retires; a depth-2 write to it dissolves.

    Every row shares one bucket, so a resurrected founder row would change
    the sum the oracle produces.
    """
    graph = PedigreeGraph.from_frame(
        {
            "id": np.arange(5),
            "mother": np.array([-1, -1, 0, -1, 2]),
            "father": np.array([-1, -1, 1, -1, 3]),
        }
    )
    dense = np.zeros(5, dtype=np.int32)
    got = _native.generation_kinship_sums(graph._built, graph.depth, dense, 1)
    assert got.tobytes() == _oracle_sums(graph, dense, 1).tobytes()
    matrix = graph.kinship_matrix()
    assert matrix[0, 4] == 0.125
    assert got[0] == 4 * 0.25 + 2 * 0.125


def _relocation_free_walk(graph: PedigreeGraph, dense: np.ndarray, n_buckets: int) -> np.ndarray:
    """The oracle's non-retiring DP with slots no row outgrows, summed after the fact.

    No row ever relocates and the free list is never used, so this walk is
    free of the 0.9.1 hazard below and is the reference the retiring paths
    are held to.
    """
    result = _run_dp_core(
        graph.n_individuals,
        graph.mother_rows,
        graph.father_rows,
        graph.twin_rows,
        graph.depth,
        0.0,
        4096,
        labels=dense,
        config=KinshipDPConfig(retire=False, lazy=False, debug_asserts=False),
    )
    return _stream_sum_theta_per_gen(
        result.cols,
        result.vals,
        result.row_start,
        result.row_count,
        result.labels,
        result.tw_idx,
        np.int32(n_buckets - 1),
    )[:n_buckets]


def test_native_sums_are_free_of_the_0_9_1_slot_reuse_hazard():
    """The 0.9.1 retiring DP could read a slot its own merge walk had just freed.

    When a parent row relocated during a child's merge walk, its old slot went
    to the free list and a later append in the same walk could take it back
    and overwrite what the walk was still reading.  It needs a row to outgrow
    its first slot, which no parity fixture does at the default capacity;
    ``baseline100K/rep1`` did, and its deepest-generation mean under the
    0.9.1 wheel is wrong by 1.4e-5 relative (gate 14a).  Forcing tiny slots
    reproduces it on a small pedigree.  The native DP stages each walk's
    relatives before writing, so it matches the relocation-free walk.
    """
    rng = np.random.default_rng(0)
    n = 349
    mother = np.full(n, -1)
    father = np.full(n, -1)
    for i in range(12, n):
        low = max(0, i - 60)
        mother[i], father[i] = rng.integers(low, i), rng.integers(low, i)
        if mother[i] == father[i]:
            father[i] = -1
    graph = PedigreeGraph.from_frame({"id": np.arange(n), "mother": mother, "father": father})
    dense, observed, _ = _densify_labels(np.asarray(graph.depth, dtype=np.int32))
    n_buckets = int(observed.shape[0]) + 1
    reference = _relocation_free_walk(graph, dense, n_buckets)
    native = _native.generation_kinship_sums(graph._built, graph.depth, dense, n_buckets)
    assert np.abs(native - reference).max() <= 1e-12
    hazard = _oracle_sums(graph, dense, n_buckets, init_cap_per_row=2)
    assert np.abs(hazard - reference).max() > 1e-3, "the 0.9.1 DP no longer shows the hazard; retire this test"


class TestBoundary:
    def test_the_arrays_are_owned_and_contiguous(self):
        graph = _graph("random_1k")
        indptr, indices, data = _native.kinship_csc(graph._built, graph.depth)
        for array, dtype in ((indptr, np.int32), (indices, np.int32), (data, np.float32)):
            assert array.dtype == dtype
            assert array.ndim == 1
            assert array.flags.c_contiguous
            assert not isinstance(array.base, np.ndarray)
        assert indptr.shape == (graph.n_individuals + 1,)
        assert indices.shape == data.shape == (indptr[-1],)

    def test_an_empty_pedigree_gives_empty_products(self):
        empty = np.zeros(0, np.int64)
        graph = PedigreeGraph.from_frame({"id": empty, "mother": empty, "father": empty})
        indptr, indices, data = _native.kinship_csc(graph._built, graph.depth)
        assert indptr.tolist() == [0]
        assert indices.shape == data.shape == (0,)
        sums = _native.generation_kinship_sums(graph._built, graph.depth, np.zeros(0, np.int32), 1)
        assert sums.tolist() == [0.0]

    def test_a_non_structural_depth_is_rejected(self):
        graph = _graph("nuclear_full_sibs")
        flat = np.zeros(graph.n_individuals, dtype=np.int32)
        with pytest.raises(PedigreeValidationError) as info:
            _native.kinship_csc(graph._built, flat)
        assert info.value.code == "value_out_of_range"
        assert info.value.fields["field"] == "depth"
        with pytest.raises(PedigreeValidationError) as info:
            _native.kinship_csc(graph._built, graph.depth[:-1])
        assert info.value.code == "length_mismatch"
        assert info.value.fields["field"] == "depth"

    @pytest.mark.parametrize("threshold", [-0.1, 1.5, float("nan"), float("inf")])
    def test_a_bad_threshold_is_a_value_error(self, threshold):
        graph = _graph("nuclear_full_sibs")
        with pytest.raises(ValueError, match="min_propagated_kinship"):
            _native.approximate_kinship_csc(graph._built, graph.depth, threshold)

    def test_bad_labels_are_rejected(self):
        graph = _graph("nuclear_full_sibs")
        n = graph.n_individuals
        with pytest.raises(PedigreeValidationError) as info:
            _native.generation_kinship_sums(graph._built, graph.depth, np.full(n, 2, np.int32), 2)
        assert info.value.code == "value_out_of_range"
        assert info.value.fields["field"] == "labels"
        with pytest.raises(PedigreeValidationError) as info:
            _native.generation_kinship_sums(graph._built, graph.depth, np.zeros(n - 1, np.int32), 1)
        assert info.value.code == "length_mismatch"
        with pytest.raises(PedigreeValidationError) as info:
            _native.generation_kinship_sums(graph._built, graph.depth, np.zeros(n, np.int32), 0)
        assert info.value.fields["field"] == "n_buckets"

    def test_the_stub_names_the_three_entries(self):
        stub = (Path(pedigree_graph.__file__).parent / "_native.pyi").read_text()
        for name in ("kinship_csc", "approximate_kinship_csc", "generation_kinship_sums"):
            assert hasattr(_native, name)
            assert f"def {name}(" in stub

    def test_the_public_matrices_are_the_bindings_without_a_copy(self):
        graph = _graph("random_1k")
        matrix = graph.approximate_kinship_matrix(min_propagated_kinship=THRESHOLD)
        assert not isinstance(matrix.data.base, np.ndarray)
        assert not matrix.data.flags.writeable
        assert _bytes((matrix.indptr, matrix.indices, matrix.data)) == _bytes(
            _native.approximate_kinship_csc(graph._built, graph.depth, THRESHOLD)
        )


SEAM_CASES = [
    (family, product)
    for family in ("kinship_rows", "kinship_csc", "kinship_sums", "kinship_scratch")
    for product in ("complete", "approximate", "sums")
    if not (family == "kinship_csc" and product == "sums") and not (family == "kinship_sums" and product != "sums")
]


@pytest.mark.parametrize(("family", "product"), SEAM_CASES, ids=[f"{f}-{p}" for f, p in SEAM_CASES])
def test_a_refused_allocation_raises_a_resource_error(family, product):
    """Each product surfaces every family it reserves as ``ResourceError("allocation_failed")``."""
    body = f"""
        from pedigree_graph import ResourceError
        family, product = {family!r}, {product!r}
        labels = np.zeros(n, dtype=np.int32)
        calls = {{
            "complete": lambda: _native.kinship_csc(graph._built, graph.depth),
            "approximate": lambda: _native.approximate_kinship_csc(graph._built, graph.depth, 0.01),
            "sums": lambda: _native.generation_kinship_sums(graph._built, graph.depth, labels, 1),
        }}
        call = calls[product]
        _native.fail_next_allocation(family, 1)
        try:
            call()
        except ResourceError as e:
            print(e.code, e.fields["operation"], e.fields["requested_elements"] >= 1)
        else:
            print("no error")
        _native.fail_next_allocation(None)
        call()
        print("recovered")
    """
    lines = _run_child(CHILD_PRELUDE, body).strip().splitlines()
    assert lines == [f"allocation_failed {family} True", "recovered"]


def test_allocation_families_list_the_matrix_families():
    names = _native.allocation_families()
    for family in ("kinship_rows", "kinship_csc", "kinship_sums", "kinship_scratch"):
        assert family in names
