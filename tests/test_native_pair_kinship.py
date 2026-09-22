"""``_native.pair_kinship`` and ``_native.kinship_support_values`` against the pure-Python oracle.

The binding is the seam slice 13 moves the pairwise recurrence across (ADR
0007, 0009): the Rust walk evaluates in graph space with structural depth as
the peel input, and Python keeps the query resolution.  These tests hold the
raw binding against ``tests/oracle/pair_kinship.py`` on every parity fixture,
in both memo layouts and both endpoint orders, and pin the boundary
contract: an owned float32 array, structured errors, and every kinship
allocation family surfacing as ``allocation_failed``.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import parity_columns, parity_fixtures
from oracle.pair_kinship import pair_kinship as oracle_pair_kinship
from test_native_relationship_pairs import CHILD_PRELUDE, _run_child
from test_pedigree_graph import _PAIRWISE_FIXTURES

from pedigree_graph import PedigreeGraph, PedigreeValidationError, _native

FIXTURES = parity_fixtures("random_1k", "deep_inbred_60g")
FIXTURE_NAMES = sorted(FIXTURES)
LAYOUTS = ("flat", "rows")
KINSHIP_FAMILIES = ("kinship_memo", "kinship_stack", "kinship_output")
ENVELOPE_UNIT = 2.0**-25


def _graph(name: str) -> PedigreeGraph:
    return PedigreeGraph.from_frame(parity_columns(FIXTURES[name]))


def _native_kinship(graph: PedigreeGraph, first, second, layout: str = "rows") -> np.ndarray:
    return _native.pair_kinship(
        graph._built, graph.depth, np.asarray(first, dtype=np.int32), np.asarray(second, dtype=np.int32), layout=layout
    )


def _oracle(graph: PedigreeGraph, first, second) -> np.ndarray:
    return oracle_pair_kinship(graph.mother_rows, graph.father_rows, graph.twin_rows, graph.depth, first, second)


def _all_pairs(n: int) -> tuple[np.ndarray, np.ndarray]:
    return np.triu_indices(n)


@pytest.mark.parametrize("name", FIXTURE_NAMES)
@pytest.mark.parametrize("layout", LAYOUTS)
def test_every_pair_matches_the_oracle_in_both_orders(name, layout):
    graph = _graph(name)
    first, second = _all_pairs(graph.n_individuals)
    expected = _oracle(graph, first, second)
    assert _native_kinship(graph, first, second, layout).tobytes() == expected.tobytes()
    assert _native_kinship(graph, second, first, layout).tobytes() == expected.tobytes()


@pytest.mark.parametrize("build", _PAIRWISE_FIXTURES, ids=lambda b: b.__name__)
@pytest.mark.parametrize("layout", LAYOUTS)
def test_mz_and_inbred_constructions_match_the_oracle(build, layout):
    graph = PedigreeGraph.from_frame(build())
    first, second = _all_pairs(graph.n_individuals)
    assert _native_kinship(graph, first, second, layout).tobytes() == _oracle(graph, first, second).tobytes()


def test_self_pairs_encode_inbreeding():
    graph = _graph("deep_inbred_60g")
    rows = np.arange(graph.n_individuals)
    values = _native_kinship(graph, rows, rows).astype(np.float64)
    assert np.abs(2.0 * values - 1.0 - graph.inbreeding()).max() <= 2.0**-22


def test_the_two_layouts_agree_element_for_element():
    graph = _graph("random_1k")
    first, second = _all_pairs(graph.n_individuals)
    assert (
        _native_kinship(graph, first, second, "flat").tobytes()
        == _native_kinship(graph, first, second, "rows").tobytes()
    )


def test_permuted_graphs_stay_inside_the_envelope(capsys):
    fixture = FIXTURES["deep_inbred_60g"]
    reference = _graph("deep_inbred_60g")
    n = reference.n_individuals
    first, second = _all_pairs(n)
    want = _native_kinship(reference, first, second)
    tolerance = 2.0 * (reference.depth[first] + reference.depth[second] + 1) * ENVELOPE_UNIT
    worst = 0
    for seed in (11, 12):
        perm = np.random.default_rng(seed).permutation(n)
        permuted = PedigreeGraph.from_frame({key: value[perm] for key, value in parity_columns(fixture).items()})
        inverse = np.empty(n, dtype=np.intp)
        inverse[perm] = np.arange(n)
        got = _native_kinship(permuted, inverse[first], inverse[second])
        assert got.tobytes() == _oracle(permuted, inverse[first], inverse[second]).tobytes()
        assert np.all(np.abs(want.astype(np.float64) - got.astype(np.float64)) <= tolerance)
        ulp = np.abs(want.view(np.int32).astype(np.int64) - got.view(np.int32).astype(np.int64))
        worst = max(worst, int(ulp.max()))
    with capsys.disabled():
        print(f"native cross-order deep_inbred_60g: max {worst} ulp")


class TestBoundary:
    def test_the_result_is_an_owned_contiguous_float32(self):
        graph = _graph("random_1k")
        values = _native_kinship(graph, [0, 1], [1, 2])
        assert values.dtype == np.float32
        assert values.ndim == 1
        assert values.flags.c_contiguous
        assert not isinstance(values.base, np.ndarray)

    def test_empty_pairs_give_an_empty_float32(self):
        graph = _graph("random_1k")
        empty = np.zeros(0, dtype=np.int32)
        assert _native_kinship(graph, empty, empty).shape == (0,)

    def test_malformed_arguments_raise_structured_errors(self):
        graph = _graph("nuclear_full_sibs")
        n = graph.n_individuals
        with pytest.raises(PedigreeValidationError) as info:
            _native_kinship(graph, [0, n], [0, 0])
        assert info.value.code == "value_out_of_range"
        assert info.value.fields["field"] == "first"
        with pytest.raises(PedigreeValidationError) as info:
            _native_kinship(graph, [0], [0, 1])
        assert info.value.code == "length_mismatch"
        with pytest.raises(PedigreeValidationError) as info:
            _native.pair_kinship(graph._built, graph.depth[:-1], np.zeros(1, np.int32), np.zeros(1, np.int32))
        assert info.value.code == "length_mismatch"
        assert info.value.fields["field"] == "depth"
        with pytest.raises(ValueError, match="layout"):
            _native_kinship(graph, [0], [0], "tree")

    def test_the_stub_names_both_entries(self):
        from pathlib import Path

        import pedigree_graph

        stub = (Path(pedigree_graph.__file__).parent / "_native.pyi").read_text()
        for name in ("pair_kinship", "kinship_support_values"):
            assert hasattr(_native, name)
            assert f"def {name}(" in stub


class TestSupportValues:
    @staticmethod
    def _support(graph: PedigreeGraph, matrix):
        return _native.kinship_support_values(
            graph._built,
            graph.depth,
            np.ascontiguousarray(matrix.indptr, dtype=np.int64),
            np.ascontiguousarray(matrix.indices, dtype=np.int32),
        )

    @pytest.mark.parametrize("name", ["double_first_cousins", "deep_inbred_60g", "random_1k"])
    @pytest.mark.parametrize("layout", LAYOUTS)
    def test_complete_support_reproduces_the_matrix(self, name, layout):
        graph = _graph(name)
        matrix = graph.kinship_matrix()
        values = _native.kinship_support_values(
            graph._built, graph.depth, matrix.indptr.astype(np.int64), matrix.indices, layout=layout
        )
        assert values.dtype == np.float32
        assert values.tobytes() == matrix.data.tobytes()

    def test_relationship_support_matches_pair_kinship(self):
        graph = _graph("random_1k")
        matrix = graph.relationship_kinship_matrix(max_degree=3)
        values = self._support(graph, matrix)
        coo = matrix.tocoo()
        assert values.tobytes() == _native_kinship(graph, coo.row, coo.col).tobytes()

    def test_a_missing_mirror_is_asymmetric(self):
        graph = _graph("nuclear_full_sibs")
        indptr = np.array([0, 1, 2, 3, 4, 6], dtype=np.int64)
        indices = np.array([0, 1, 2, 3, 0, 4], dtype=np.int32)
        with pytest.raises(PedigreeValidationError) as info:
            _native.kinship_support_values(graph._built, graph.depth, indptr, indices)
        assert info.value.code == "kinship_support_asymmetric"
        assert dict(info.value.fields) == {"row": 0, "column": 4}

    def test_an_unsorted_column_is_rejected(self):
        graph = _graph("nuclear_full_sibs")
        indptr = np.array([0, 2, 3, 4, 5, 7], dtype=np.int64)
        indices = np.array([4, 0, 1, 2, 3, 0, 4], dtype=np.int32)
        with pytest.raises(PedigreeValidationError) as info:
            _native.kinship_support_values(graph._built, graph.depth, indptr, indices)
        assert info.value.code == "kinship_support_unsorted"
        assert info.value.fields["column"] == 0


@pytest.mark.parametrize("family", KINSHIP_FAMILIES)
@pytest.mark.parametrize("layout", LAYOUTS)
def test_a_refused_allocation_raises_a_resource_error(family, layout):
    """Both entries surface every kinship family as ``ResourceError("allocation_failed")`` in a fresh process."""
    body = f"""
        from pedigree_graph import ResourceError
        family = {family!r}
        rows = np.arange(n, dtype=np.int32)
        matrix = graph.relationship_kinship_matrix(max_degree=2)
        def pairs():
            return _native.pair_kinship(graph._built, graph.depth, rows, rows, layout={layout!r})
        def support():
            return _native.kinship_support_values(
                graph._built, graph.depth, matrix.indptr.astype(np.int64), matrix.indices, layout={layout!r}
            )
        for label, call in (("pairs", pairs), ("support", support)):
            _native.fail_next_allocation(family, 1)
            try:
                call()
            except ResourceError as e:
                print(label, e.code, e.fields["operation"], e.fields["requested_elements"] >= 1)
            else:
                print(label, "no error")
            _native.fail_next_allocation(None)
            call()
        print("recovered")
    """
    lines = _run_child(CHILD_PRELUDE, body).strip().splitlines()
    assert lines[-1] == "recovered"
    assert lines[:-1] == [f"pairs allocation_failed {family} True", f"support allocation_failed {family} True"]
