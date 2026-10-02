"""The three kinship-matrix bindings against the 0.9.1 numba DP kept under ``tests/oracle``.

The depth-major DP behind ``kinship_matrix`` and ``approximate_kinship_matrix``
runs in the Rust core (ADR 0007, 0009).  These tests hold those bindings to
the bytes the Python DP produced on every parity fixture, in permuted row
orders too.  ``generation_kinship_sums``, behind ``mean_kinship_by_generation``,
no longer runs the DP: it is held to the oracle's sums exactly where float32
holds every kinship, and to rounding elsewhere.  The boundary contract is
pinned for all three: owned arrays, structured errors, and every allocation
family surfacing as ``allocation_failed``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from _support import _PAIRWISE_FIXTURES, CHILD_PRELUDE, _run_child
from conftest import FIXTURE_NAMES, parity_graph
from oracle.kinship_dp.dp import KinshipDPConfig, _build_kinship_csc, _run_dp_core

import pedigree_graph
from pedigree_graph import PedigreeGraph, PedigreeValidationError, _native
from pedigree_graph._cohorts import _densify_labels
from pedigree_graph._threads import thread_budget

THRESHOLD = 0.001
PERMUTATION_SEEDS = (None, 5, 11)
# Fixtures with a kinship float32 cannot hold, where the oracle's float32 sums
# and the float64 sweep part by rounding.
FLOAT32_INEXACT = frozenset({"deep_inbred_60g"})


def _oracle_csc(graph: PedigreeGraph, threshold: float) -> tuple[bytes, bytes, bytes]:
    """The oracle DP's support at *threshold* with its values recomputed exactly.

    The Python DP's thresholded pass leaves propagated values on the support;
    the 0.9.1 facade then replaced them by a second complete pass.  Here the
    exact values come from ``kinship_support_values`` instead, the pairwise
    walk ``test_native_pair_kinship`` holds to its own oracle, so the structure is the oracle's
    and the values are the pinned recurrence.  At threshold 0 the DP's own
    values are already exact and are compared as they are.
    """
    indptr, indices, data = _build_kinship_csc(
        graph.n_individuals, graph.mother_rows, graph.father_rows, graph.twin_rows, graph.depth, threshold
    )
    if threshold > 0.0:
        data = _native.kinship_support_values(
            graph._built, graph.depth, indptr.astype(np.int64), indices, threads=thread_budget()
        )
    return indptr.tobytes(), indices.tobytes(), data.tobytes()


def _oracle_sums(graph: PedigreeGraph, dense: np.ndarray, n_buckets: int) -> np.ndarray:
    result = _run_dp_core(
        graph.n_individuals,
        graph.mother_rows,
        graph.father_rows,
        graph.twin_rows,
        graph.depth,
        0.0,
        None,
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
    graph = parity_graph(name, seed)
    assert _bytes(_native.kinship_csc(graph._built, graph.depth)) == _oracle_csc(graph, 0.0)
    assert _bytes(_native.approximate_kinship_csc(graph._built, graph.depth, THRESHOLD)) == _oracle_csc(
        graph, THRESHOLD
    )


@pytest.mark.parametrize("seed", PERMUTATION_SEEDS)
@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_generation_sums_match_the_oracle(name, seed):
    """The sentinel bucket of the densified labels joins no bucket."""
    graph = parity_graph(name, seed)
    n = graph.n_individuals
    shuffled = np.random.default_rng(3).integers(-1, 4, n).astype(np.int32)
    for labels in (np.asarray(graph.depth, dtype=np.int32), shuffled):
        dense, observed, _ = _densify_labels(labels)
        k = int(observed.shape[0])
        got = _native.generation_kinship_sums(graph._built, graph.depth, dense, k)
        want = _oracle_sums(graph, dense, k + 1)[:k]
        assert got.dtype == np.float64
        if name in FLOAT32_INEXACT:
            np.testing.assert_allclose(got, want, rtol=1e-6, atol=0.0)
        else:
            assert got.tobytes() == want.tobytes()


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
    graph = parity_graph("random_1k")
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
    k = int(observed.shape[0])
    got = _native.generation_kinship_sums(graph._built, depth, dense, k)
    assert got.tobytes() == _oracle_sums(Raised, dense, k + 1)[:k].tobytes()


def test_a_retired_row_cannot_be_resurrected():
    """A founder whose last child is at depth 1 retires; a depth-2 write to it dissolves.

    The approximate product's second pass retires rows, so a resurrected
    founder row would change its bytes against the oracle's.
    """
    graph = PedigreeGraph.from_frame(
        {
            "id": np.arange(5),
            "mother": np.array([-1, -1, 0, -1, 2]),
            "father": np.array([-1, -1, 1, -1, 3]),
        }
    )
    assert _bytes(_native.approximate_kinship_csc(graph._built, graph.depth, 0.0)) == _oracle_csc(graph, 0.0)
    assert graph.kinship_matrix()[0, 4] == 0.125
    dense = np.zeros(5, dtype=np.int32)
    got = _native.generation_kinship_sums(graph._built, graph.depth, dense, 1)
    assert got.tolist() == [4 * 0.25 + 2 * 0.125]


class TestBoundary:
    def test_the_arrays_are_owned_and_contiguous(self):
        graph = parity_graph("random_1k")
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
        graph = parity_graph("nuclear_full_sibs")
        flat = np.zeros(graph.n_individuals, dtype=np.int32)
        with pytest.raises(PedigreeValidationError) as info:
            _native.kinship_csc(graph._built, flat)
        assert info.value.code == "value_out_of_range"
        assert info.value.fields["field"] == "depth"

    @pytest.mark.parametrize("threshold", [-0.1, 1.5, float("nan"), float("inf")])
    def test_a_bad_threshold_is_a_value_error(self, threshold):
        graph = parity_graph("nuclear_full_sibs")
        with pytest.raises(ValueError, match="min_propagated_kinship"):
            _native.approximate_kinship_csc(graph._built, graph.depth, threshold)

    def test_bad_labels_are_rejected(self):
        graph = parity_graph("nuclear_full_sibs")
        n = graph.n_individuals
        with pytest.raises(PedigreeValidationError) as info:
            _native.generation_kinship_sums(graph._built, graph.depth, np.full(n, 3, np.int32), 2)
        assert info.value.code == "value_out_of_range"
        assert info.value.fields["field"] == "labels"
        with pytest.raises(PedigreeValidationError) as info:
            _native.generation_kinship_sums(graph._built, graph.depth, np.full(n, -1, np.int32), 2)
        assert info.value.fields["field"] == "labels"

    def test_a_row_labelled_n_buckets_joins_no_bucket(self):
        graph = parity_graph("nuclear_full_sibs")
        n = graph.n_individuals
        sums = _native.generation_kinship_sums(graph._built, graph.depth, np.zeros(n, np.int32), 0)
        assert sums.shape == (0,)

    def test_the_stub_names_the_three_entries(self):
        stub = (Path(pedigree_graph.__file__).parent / "_native.pyi").read_text()
        for name in ("kinship_csc", "approximate_kinship_csc", "generation_kinship_sums"):
            assert hasattr(_native, name)
            assert f"def {name}(" in stub

    def test_the_public_matrices_are_the_bindings_without_a_copy(self):
        graph = parity_graph("random_1k")
        matrix = graph.approximate_kinship_matrix(min_propagated_kinship=THRESHOLD)
        assert not isinstance(matrix.data.base, np.ndarray)
        assert not matrix.data.flags.writeable
        assert _bytes((matrix.indptr, matrix.indices, matrix.data)) == _bytes(
            _native.approximate_kinship_csc(graph._built, graph.depth, THRESHOLD)
        )


SEAM_CASES = [
    *(
        (family, product)
        for family in ("kinship_rows", "kinship_csc", "kinship_scratch")
        for product in ("complete", "approximate")
    ),
    *((family, "sums") for family in ("inbreeding_walk", "kinship_sums")),
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
