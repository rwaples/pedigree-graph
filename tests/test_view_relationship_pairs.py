"""``PedigreeView.relationship_pairs`` / ``relationship_counts`` (slice 4b, ADR 0006).

The done-criterion is stated independently of the implementation: a view
result equals the graph result filtered to pairs with both endpoints selected,
relabelled into view rows, symmetric codes re-canonicalised, and sorted by the
view canonical key.  Fixtures and predicates are those of
``test_relationship_pairs`` / ``relationship_predicates``.
"""

from __future__ import annotations

import zlib
from typing import TYPE_CHECKING

import numpy as np
import pytest
from relationship_predicates import AncestorWalk
from test_relationship_pairs import ASYMMETRIC, CODES, FIXTURE_NAMES, FIXTURES, SYMMETRIC, _graph

import pedigree_graph
from pedigree_graph import RELATIONSHIPS, PedigreeGraph, PedigreeValidationError, RelationshipCountResult
from pedigree_graph._pair_extractor import check_exclusive
from pedigree_graph._pair_utils import canonical_keys
from pedigree_graph._threads import _reset_thread_state, configure_threads
from pedigree_graph._view import CoordinateToken

if TYPE_CHECKING:
    from pedigree_graph import PedigreeView, RelationshipPairs

SELECTIONS = ("identity", "reversed", "shuffled_half", "every_other_id", "single_row", "empty")


def _select(graph: PedigreeGraph, name: str, selection: str) -> PedigreeView:
    n = graph.n_individuals
    if selection == "identity":
        return graph.view(rows=np.arange(n))
    if selection == "reversed":
        return graph.view(rows=np.arange(n)[::-1])
    if selection == "shuffled_half":
        rng = np.random.default_rng(zlib.crc32(name.encode()))
        return graph.view(rows=rng.permutation(n)[: max(1, n // 2)])
    if selection == "every_other_id":
        return graph.view(ids=FIXTURES[name]["ids"][::2])
    if selection == "single_row":
        return graph.view(rows=[n - 1])
    return graph.view(rows=[])


def _graph_to_view(view: PedigreeView) -> np.ndarray:
    table = np.full(view._graph.n_individuals, -1, dtype=np.int64)
    table[view.graph_rows] = np.arange(len(view))
    return table


def _expected(graph_result: RelationshipPairs, view: PedigreeView) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    table = _graph_to_view(view)
    n = len(view)
    expected = {}
    for code, block in graph_result.items():
        first, second = table[block.first_rows], table[block.second_rows]
        keep = (first >= 0) & (second >= 0)
        first, second = first[keep], second[keep]
        if RELATIONSHIPS[code].symmetric:
            first, second = np.minimum(first, second), np.maximum(first, second)
        order = np.argsort(np.minimum(first, second) * n + np.maximum(first, second), kind="stable")
        expected[code] = (first[order], second[order])
    return expected


@pytest.fixture(scope="module")
def full_results() -> dict[str, RelationshipPairs]:
    return {name: _graph(name).relationship_pairs(max_degree=5) for name in FIXTURE_NAMES}


def _assert_equal(result: RelationshipPairs, expected: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    for code in CODES:
        np.testing.assert_array_equal(result[code].first_rows, expected[code][0], err_msg=f"{code}: first_rows")
        np.testing.assert_array_equal(result[code].second_rows, expected[code][1], err_msg=f"{code}: second_rows")


class TestOracleEquality:
    @pytest.mark.parametrize("selection", SELECTIONS)
    @pytest.mark.parametrize("name", FIXTURE_NAMES)
    def test_view_equals_the_filtered_graph_result(self, full_results, name, selection):
        graph = _graph(name)
        view = _select(graph, name, selection)
        result = view.relationship_pairs(max_degree=5)
        _assert_equal(result, _expected(full_results[name], view))
        assert {code for code, block in result.items() if block.requested} == set(CODES)

    @pytest.mark.parametrize("name", FIXTURE_NAMES)
    def test_categories_selector_filters_the_same_way(self, full_results, name):
        graph = _graph(name)
        view = _select(graph, name, "shuffled_half")
        result = view.relationship_pairs(categories=["1C", "Av", "MO"])
        expected = _expected(graph.relationship_pairs(categories=["1C", "Av", "MO"]), view)
        _assert_equal(result, expected)
        assert {code for code, block in result.items() if block.requested} == {"1C", "Av", "MO"}


def _cousins_through_unselected_rows() -> PedigreeGraph:
    # 1, 2 grandparents; 3, 4 their children; 5, 6 founders married in;
    # 7 = (3, 5), 8 = (4, 6) are first cousins.
    return PedigreeGraph.from_frame(
        {
            "id": np.array([1, 2, 3, 4, 5, 6, 7, 8]),
            "mother": np.array([-1, -1, 1, 1, -1, -1, 3, 4]),
            "father": np.array([-1, -1, 2, 2, -1, -1, 5, 6]),
        }
    )


class TestPathsThroughUnselectedRows:
    def test_first_cousins_with_no_selected_ancestor(self):
        view = _cousins_through_unselected_rows().view(ids=[8, 7])
        result = view.relationship_pairs(max_degree=5)
        assert result["1C"].first_rows.tolist() == [0]
        assert result["1C"].second_rows.tolist() == [1]
        assert [code for code, block in result.items() if len(block)] == ["1C"]

    def test_one_selected_endpoint_of_a_parent_offspring_pair_is_dropped(self):
        view = _cousins_through_unselected_rows().view(ids=[3, 8])
        result = view.relationship_pairs(max_degree=5)
        assert len(result["MO"]) == 0
        assert len(result["FO"]) == 0
        assert [code for code, block in result.items() if len(block)] == ["Av"]


class TestOrientationAndOrdering:
    @pytest.mark.parametrize("name", FIXTURE_NAMES)
    def test_asymmetric_blocks_satisfy_their_roles_in_graph_terms(self, name):
        graph = _graph(name)
        view = _select(graph, name, "shuffled_half")
        walk = AncestorWalk(graph)
        rows = view.graph_rows
        for code in ASYMMETRIC:
            block = view.relationship_pairs(categories=[code])[code]
            for first, second in zip(block.first_rows.tolist(), block.second_rows.tolist(), strict=True):
                assert walk.oriented_pair_is_valid(code, int(rows[first]), int(rows[second])), (
                    name,
                    code,
                    first,
                    second,
                )

    @pytest.mark.parametrize("name", FIXTURE_NAMES)
    def test_symmetric_blocks_are_canonical_in_view_rows(self, name):
        view = _select(_graph(name), name, "reversed")
        result = view.relationship_pairs(max_degree=5)
        for code in SYMMETRIC:
            assert np.all(result[code].first_rows < result[code].second_rows), code

    @pytest.mark.parametrize("name", FIXTURE_NAMES)
    def test_blocks_are_sorted_in_range_owned_and_exclusive(self, name):
        view = _select(_graph(name), name, "shuffled_half")
        result = view.relationship_pairs(max_degree=5)
        n = len(view)
        for code, block in result.items():
            keys = canonical_keys(block.first_rows, block.second_rows, n)
            assert np.all(keys[1:] > keys[:-1]), code
            for rows in block:
                assert rows.dtype == np.int32
                assert rows.flags.c_contiguous
                assert not rows.flags.writeable
                assert np.all((rows >= 0) & (rows < n)), code
        check_exclusive(result)

    def test_reversed_view_flips_symmetric_but_not_asymmetric_orientation(self):
        graph = _graph("avuncular_and_cousins")
        n = graph.n_individuals
        forward = graph.view(rows=np.arange(n)).relationship_pairs(max_degree=5)
        backward = graph.view(rows=np.arange(n)[::-1]).relationship_pairs(max_degree=5)
        av = backward["Av"]
        assert set(zip((n - 1 - av.first_rows).tolist(), (n - 1 - av.second_rows).tolist(), strict=True)) == set(
            zip(forward["Av"].first_rows.tolist(), forward["Av"].second_rows.tolist(), strict=True)
        )
        fs = backward["FS"]
        assert set(zip((n - 1 - fs.second_rows).tolist(), (n - 1 - fs.first_rows).tolist(), strict=True)) == set(
            zip(forward["FS"].first_rows.tolist(), forward["FS"].second_rows.tolist(), strict=True)
        )


class TestTokens:
    def test_blocks_carry_the_view_token_not_the_graph_token(self):
        graph = _graph("avuncular_and_cousins")
        view = graph.view(rows=[0, 1, 2, 3])
        for block in view.relationship_pairs(max_degree=5).values():
            assert block._coordinate_token is view._coordinate_token
            assert block._coordinate_token is not graph._coordinate_token

    def test_two_equivalent_views_yield_distinct_tokens(self):
        graph = _graph("avuncular_and_cousins")
        one = graph.view(rows=[0, 1, 2, 3]).relationship_pairs(max_degree=5)
        two = graph.view(rows=[0, 1, 2, 3]).relationship_pairs(max_degree=5)
        assert one["FS"]._coordinate_token is not two["FS"]._coordinate_token

    def test_no_public_attribute_is_a_token(self):
        result = _graph("avuncular_and_cousins").view(rows=[0, 1]).relationship_pairs(max_degree=1)
        for owner in (result, result["FS"]):
            public = [name for name in dir(owner) if not name.startswith("_")]
            assert not any(isinstance(getattr(owner, name), CoordinateToken) for name in public)


@pytest.fixture(params=["populated", "empty"])
def any_view(request) -> PedigreeView:
    graph = _graph("avuncular_and_cousins")
    return graph.view(rows=[0, 1, 2]) if request.param == "populated" else graph.view(rows=[])


class TestSelectorErrors:
    def test_both_selectors(self, any_view):
        with pytest.raises(TypeError):
            any_view.relationship_pairs(max_degree=2, categories=["FS"])

    def test_neither_selector(self, any_view):
        with pytest.raises(TypeError):
            any_view.relationship_pairs()

    def test_bare_string(self, any_view):
        with pytest.raises(TypeError):
            any_view.relationship_pairs(categories="FS")

    def test_unknown_code(self, any_view):
        with pytest.raises(PedigreeValidationError) as info:
            any_view.relationship_pairs(categories=["ZZ", "FS", "AA"])
        assert info.value.code == "unknown_relationship_category"
        assert info.value.fields["codes"] == ("AA", "ZZ")

    @pytest.mark.parametrize("max_degree", [-1, 6])
    def test_max_degree_out_of_range(self, any_view, max_degree):
        with pytest.raises(PedigreeValidationError) as info:
            any_view.relationship_pairs(max_degree=max_degree)
        assert info.value.code == "max_degree_out_of_range"
        assert info.value.fields == {"value": max_degree, "minimum": 0, "maximum": 5}

    def test_counts_raise_the_same_way(self, any_view):
        with pytest.raises(TypeError):
            any_view.relationship_counts()
        with pytest.raises(PedigreeValidationError) as info:
            any_view.relationship_counts(categories=["ZZ"])
        assert info.value.code == "unknown_relationship_category"


class TestThreads:
    @pytest.fixture(autouse=True)
    def reset_thread_state(self):
        _reset_thread_state()
        yield
        _reset_thread_state()

    def test_in_process_budget_of_four_matches_the_one_thread_result(self, monkeypatch):
        monkeypatch.delenv("PEDIGREE_GRAPH_THREADS", raising=False)
        graph = _graph("random_1k")
        rows = np.random.default_rng(4).permutation(graph.n_individuals)[:500]
        reference = graph.view(rows=rows).relationship_pairs(max_degree=5)
        _reset_thread_state()
        configure_threads(4)
        result = _graph("random_1k").view(rows=rows).relationship_pairs(max_degree=5)
        for code in CODES:
            assert result[code].first_rows.tobytes() == reference[code].first_rows.tobytes(), code
            assert result[code].second_rows.tobytes() == reference[code].second_rows.tobytes(), code


class TestCounts:
    @pytest.fixture(params=["graph", "view"])
    def receiver(self, request):
        graph = _graph("random_1k")
        return graph if request.param == "graph" else graph.view(rows=np.arange(graph.n_individuals)[::3])

    def test_counts_are_block_lengths_and_none_when_unrequested(self, receiver):
        pairs = receiver.relationship_pairs(categories=["FS", "1C", "GP"])
        counts = receiver.relationship_counts(categories=["FS", "1C", "GP"])
        assert isinstance(counts, RelationshipCountResult)
        for code in CODES:
            assert counts[code] == (len(pairs[code]) if code in {"FS", "1C", "GP"} else None), code
        assert counts["FS"] > 0

    def test_code_sets(self, receiver):
        counts = receiver.relationship_counts(max_degree=2)
        assert counts.requested == frozenset(code for code in CODES if RELATIONSHIPS[code].degree <= 2)
        assert counts.exact == counts.requested
        assert counts.approximate == frozenset()
        assert counts.clamped == frozenset()

    def test_iterates_all_codes_in_registry_order(self, receiver):
        counts = receiver.relationship_counts(categories=())
        assert list(counts) == list(CODES)
        assert len(counts) == 23
        assert all(count is None for count in counts.values())

    def test_is_immutable(self, receiver):
        counts = receiver.relationship_counts(max_degree=1)
        with pytest.raises(TypeError):
            counts["FS"] = 0  # ty: ignore[invalid-assignment]
        with pytest.raises(AttributeError):
            counts.requested = frozenset()  # ty: ignore[invalid-assignment]

    def test_repr_lists_requested_counts(self, receiver):
        text = repr(receiver.relationship_counts(categories=["MZ", "FS"]))
        assert text.startswith("RelationshipCountResult(")
        assert "MZ=" in text
        assert "FS=" in text
        assert "GP=" not in text

    def test_view_counts_equal_the_filtered_graph_counts(self, full_results):
        graph = _graph("random_1k")
        view = graph.view(rows=np.arange(graph.n_individuals)[::3])
        expected = _expected(full_results["random_1k"], view)
        counts = view.relationship_counts(max_degree=5)
        assert dict(counts) == {code: len(expected[code][0]) for code in CODES}

    def test_root_export(self):
        assert pedigree_graph.RelationshipCountResult is RelationshipCountResult
        assert "RelationshipCountResult" in pedigree_graph.__all__
