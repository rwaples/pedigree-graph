"""Tests for pedigree views and coordinate tokens (ADR 0006, slice 3).

Pins slice 3: ``graph.view(ids=...)`` xor ``graph.view(rows=...)`` builds an
ordered, owned, read-only selection; every malformed selection is one of the
four structured view errors; each graph and each separately built view owns a
distinct opaque token; and constructing views changes no existing pair output.

The pedigree uses ids 10..19 so an id is never accidentally equal to its row.
"""

import gc

import numpy as np
import pytest

from pedigree_graph import PedigreeGraph, PedigreeValidationError, PedigreeView
from pedigree_graph._view import CoordinateToken

SELECTED_IDS = [13, 11, 17]
SELECTED_ROWS = [3, 1, 7]


def _data():
    return {
        "id": np.arange(10, 20, dtype=np.int64),
        "mother": np.array([-1, -1, -1, -1, 10, 10, 12, 12, 14, 16], dtype=np.int64),
        "father": np.array([-1, -1, -1, -1, 11, 11, 13, 13, 15, 17], dtype=np.int64),
    }


@pytest.fixture
def graph():
    return PedigreeGraph.from_frame(_data())


class TestSelection:
    def test_ids_resolve_in_selection_order(self, graph):
        view = graph.view(ids=SELECTED_IDS)
        np.testing.assert_array_equal(view.graph_rows, SELECTED_ROWS)
        np.testing.assert_array_equal(view.ids, SELECTED_IDS)
        assert len(view) == 3
        assert view.n_individuals == 3
        assert type(view.n_individuals) is int

    def test_rows_are_kept_in_selection_order(self, graph):
        view = graph.view(rows=SELECTED_ROWS)
        np.testing.assert_array_equal(view.graph_rows, SELECTED_ROWS)
        np.testing.assert_array_equal(view.ids, graph.ids[SELECTED_ROWS])

    def test_repr_names_the_size(self, graph):
        assert repr(graph.view(rows=SELECTED_ROWS)) == "PedigreeView(n_individuals=3)"

    def test_view_is_a_pedigree_view(self, graph):
        assert isinstance(graph.view(rows=[0]), PedigreeView)

    @pytest.mark.parametrize("selection", [{"ids": []}, {"rows": []}, {"ids": np.asarray([])}])
    def test_empty_selection_gives_an_empty_view(self, graph, selection):
        view = graph.view(**selection)
        assert len(view) == 0
        assert view.ids.dtype == np.int64
        assert view.graph_rows.dtype == np.int32


class TestSelectors:
    def test_both_keywords_is_a_type_error(self, graph):
        with pytest.raises(TypeError):
            graph.view(ids=[13], rows=[3])

    def test_neither_keyword_is_a_type_error(self, graph):
        with pytest.raises(TypeError):
            graph.view()

    def test_positional_selection_is_a_type_error(self, graph):
        with pytest.raises(TypeError):
            graph.view([13])


class TestArrays:
    @pytest.mark.parametrize(("name", "dtype"), [("ids", np.int64), ("graph_rows", np.int32)])
    def test_dtype_contiguity_and_immutability(self, graph, name, dtype):
        array = getattr(graph.view(ids=SELECTED_IDS), name)
        assert array.dtype == dtype
        assert array.flags.c_contiguous
        assert not array.flags.writeable
        with pytest.raises(ValueError, match="read-only"):
            array[0] = 0

    @pytest.mark.parametrize("name", ["ids", "graph_rows"])
    def test_every_access_returns_the_same_object(self, graph, name):
        view = graph.view(ids=SELECTED_IDS)
        assert getattr(view, name) is getattr(view, name)

    def test_mutating_the_id_selection_afterwards_changes_nothing(self, graph):
        selection = np.array(SELECTED_IDS, dtype=np.int64)
        view = graph.view(ids=selection)
        selection[:] = 19
        np.testing.assert_array_equal(view.ids, SELECTED_IDS)
        np.testing.assert_array_equal(view.graph_rows, SELECTED_ROWS)

    def test_mutating_the_row_selection_afterwards_changes_nothing(self, graph):
        selection = np.array(SELECTED_ROWS, dtype=np.int64)
        view = graph.view(rows=selection)
        selection[:] = 9
        np.testing.assert_array_equal(view.graph_rows, SELECTED_ROWS)
        np.testing.assert_array_equal(view.ids, graph.ids[SELECTED_ROWS])


class TestCoercion:
    @pytest.mark.parametrize("field", ["ids", "rows"])
    def test_two_dimensional_selection_names_the_argument(self, graph, field):
        with pytest.raises(PedigreeValidationError) as excinfo:
            graph.view(**{field: [[1, 2]]})
        assert excinfo.value.code == "invalid_shape"
        assert excinfo.value.fields["field"] == field
        assert excinfo.value.fields["expected_ndim"] == 1
        assert excinfo.value.fields["actual_shape"] == (1, 2)

    def test_fractional_float_is_not_a_lossless_integer(self, graph):
        with pytest.raises(PedigreeValidationError) as excinfo:
            graph.view(ids=[13.5])
        assert excinfo.value.code == "invalid_integer_value"
        assert excinfo.value.fields == {"field": "ids", "position": 0, "value": 13.5}

    def test_bool_selection_is_rejected(self, graph):
        with pytest.raises(PedigreeValidationError) as excinfo:
            graph.view(rows=np.array([True, False]))
        assert excinfo.value.code == "invalid_integer_value"
        assert excinfo.value.fields["field"] == "rows"

    @pytest.mark.parametrize("selection", [np.array([13, None], dtype=object), np.array([13.0, np.nan])])
    def test_host_null_is_rejected_as_null(self, graph, selection):
        with pytest.raises(PedigreeValidationError) as excinfo:
            graph.view(ids=selection)
        assert excinfo.value.code == "invalid_integer_value"
        assert excinfo.value.fields == {"field": "ids", "position": 1, "value": "null"}

    def test_integral_float_ids_are_accepted(self, graph):
        np.testing.assert_array_equal(graph.view(ids=[13.0, 11.0]).graph_rows, [3, 1])

    def test_integral_float_rows_are_accepted(self, graph):
        np.testing.assert_array_equal(graph.view(rows=[3.0, 1.0]).ids, [13, 11])


class TestSelectionErrors:
    def test_duplicate_id(self, graph):
        with pytest.raises(PedigreeValidationError) as excinfo:
            graph.view(ids=[13, 11, 13])
        assert excinfo.value.code == "duplicate_view_id"
        assert excinfo.value.fields == {"id": 13, "positions": (0, 2), "duplicate_count": 1}

    def test_duplicate_id_witness_is_the_smallest_repeated_id(self, graph):
        with pytest.raises(PedigreeValidationError) as excinfo:
            graph.view(ids=[17, 13, 17, 13, 13])
        assert excinfo.value.code == "duplicate_view_id"
        assert excinfo.value.fields == {"id": 13, "positions": (1, 3, 4), "duplicate_count": 3}

    def test_unknown_id(self, graph):
        with pytest.raises(PedigreeValidationError) as excinfo:
            graph.view(ids=[13, 99])
        assert excinfo.value.code == "unknown_view_id"
        assert excinfo.value.fields == {"id": 99, "position": 1, "missing_count": 1}

    def test_negative_id_is_unknown(self, graph):
        with pytest.raises(PedigreeValidationError) as excinfo:
            graph.view(ids=[-5, 13])
        assert excinfo.value.code == "unknown_view_id"
        assert excinfo.value.fields == {"id": -5, "position": 0, "missing_count": 1}

    def test_unknown_id_counts_every_unresolved_entry(self, graph):
        with pytest.raises(PedigreeValidationError) as excinfo:
            graph.view(ids=[99, 13, 98])
        assert excinfo.value.code == "unknown_view_id"
        assert excinfo.value.fields == {"id": 99, "position": 0, "missing_count": 2}

    def test_duplicate_row(self, graph):
        with pytest.raises(PedigreeValidationError) as excinfo:
            graph.view(rows=[3, 1, 3])
        assert excinfo.value.code == "duplicate_view_row"
        assert excinfo.value.fields == {"row": 3, "positions": (0, 2), "duplicate_count": 1}

    def test_row_at_n_is_out_of_range(self, graph):
        with pytest.raises(PedigreeValidationError) as excinfo:
            graph.view(rows=[10])
        assert excinfo.value.code == "view_row_out_of_range"
        assert excinfo.value.fields == {"row": 10, "position": 0, "n_individuals": 10}

    def test_negative_row_is_out_of_range(self, graph):
        with pytest.raises(PedigreeValidationError) as excinfo:
            graph.view(rows=[1, -1])
        assert excinfo.value.code == "view_row_out_of_range"
        assert excinfo.value.fields == {"row": -1, "position": 1, "n_individuals": 10}


class TestCoordinateTokens:
    def test_graph_owns_a_token(self, graph):
        assert isinstance(graph._coordinate_token, CoordinateToken)

    def test_view_token_is_not_the_graph_token(self, graph):
        assert graph.view(ids=SELECTED_IDS)._coordinate_token is not graph._coordinate_token

    def test_identical_id_selections_are_distinct_receivers(self, graph):
        first = graph.view(ids=SELECTED_IDS)
        second = graph.view(ids=SELECTED_IDS)
        assert first._coordinate_token != second._coordinate_token

    def test_equivalent_id_and_row_selections_are_distinct_receivers(self, graph):
        by_id = graph.view(ids=SELECTED_IDS)
        by_row = graph.view(rows=SELECTED_ROWS)
        np.testing.assert_array_equal(by_id.graph_rows, by_row.graph_rows)
        assert by_id._coordinate_token != by_row._coordinate_token

    def test_tokens_compare_by_identity(self):
        token = CoordinateToken()
        assert token == token
        assert token != CoordinateToken()

    def test_hash_is_stable(self):
        token = CoordinateToken()
        assert hash(token) == hash(token)
        assert len({token, token, CoordinateToken()}) == 2

    def test_repr_is_opaque(self):
        assert repr(CoordinateToken()) == "CoordinateToken()"

    def test_no_public_token_attribute(self, graph):
        view = graph.view(ids=SELECTED_IDS)
        for receiver in (graph, view):
            assert [name for name in dir(receiver) if "token" in name and not name.startswith("_")] == []


def test_view_outlives_the_graph_name():
    graph = PedigreeGraph.from_frame(_data())
    view = graph.view(ids=SELECTED_IDS)
    del graph
    gc.collect()
    np.testing.assert_array_equal(view.ids, SELECTED_IDS)
    np.testing.assert_array_equal(view.graph_rows, SELECTED_ROWS)
    assert isinstance(view._graph, PedigreeGraph)


class TestPairOutputSurvivesBuildingViews:
    def test_graph_counts_survive_building_views(self, small_pedigree):
        pg = PedigreeGraph.from_frame(small_pedigree)
        before = dict(pg.relationship_counts(max_degree=5))
        assert sum(before.values()) > 0
        ids = small_pedigree["id"].to_numpy()
        pg.view(ids=ids[[4, 5]])
        pg.view(rows=[9, 0])
        pg.view(ids=[])
        assert dict(pg.relationship_counts(max_degree=5)) == before

    def test_view_pairs_survive_building_further_views(self, small_pedigree):
        selected = small_pedigree.tail(60)["id"].to_numpy()
        pg = PedigreeGraph.from_frame(small_pedigree)
        view = pg.view(ids=selected)
        before = view.relationship_pairs(max_degree=2)
        assert any(len(block) for block in before.values())

        pg.view(ids=selected[[4, 6, 5]])
        pg.view(rows=[2, 0])

        after = view.relationship_pairs(max_degree=2)
        assert set(after) == set(before)
        for code, block in before.items():
            np.testing.assert_array_equal(after[code].first_rows, block.first_rows)
            np.testing.assert_array_equal(after[code].second_rows, block.second_rows)


class TestErrorPrecedence:
    def test_unknown_id_is_reported_before_its_duplicate(self, graph):
        with pytest.raises(PedigreeValidationError) as excinfo:
            graph.view(ids=[99, 99])
        assert excinfo.value.code == "unknown_view_id"
        assert excinfo.value.fields["missing_count"] == 2

    def test_out_of_range_row_is_reported_before_its_duplicate(self, graph):
        with pytest.raises(PedigreeValidationError) as excinfo:
            graph.view(rows=[99, 99])
        assert excinfo.value.code == "view_row_out_of_range"

    def test_duplicate_is_reported_once_every_entry_is_valid(self, graph):
        with pytest.raises(PedigreeValidationError) as excinfo:
            graph.view(ids=[13, 11, 13])
        assert excinfo.value.code == "duplicate_view_id"


class TestOverflowFoldsIntoTheViewCodes:
    def test_uint64_row_beyond_int64_is_out_of_range(self, graph):
        with pytest.raises(PedigreeValidationError) as excinfo:
            graph.view(rows=np.array([2**63], dtype=np.uint64))
        assert excinfo.value.code == "view_row_out_of_range"
        assert excinfo.value.fields["row"] == 2**63
        assert excinfo.value.fields["n_individuals"] == 10

    def test_float_row_beyond_int64_is_out_of_range(self, graph):
        with pytest.raises(PedigreeValidationError) as excinfo:
            graph.view(rows=[1e300])
        assert excinfo.value.code == "view_row_out_of_range"
        assert excinfo.value.fields["position"] == 0

    def test_id_beyond_int64_is_unknown(self, graph):
        with pytest.raises(PedigreeValidationError) as excinfo:
            graph.view(ids=[2**64])
        assert excinfo.value.code == "unknown_view_id"
        assert excinfo.value.fields["id"] == 2**64

    @pytest.mark.parametrize("selection", [{"ids": [13, 13]}, {"rows": [3, 3]}, {"ids": [2**64]}, {"rows": [-1]}])
    def test_never_a_fifth_code(self, graph, selection):
        with pytest.raises(PedigreeValidationError) as excinfo:
            graph.view(**selection)
        assert excinfo.value.code in {
            "duplicate_view_id",
            "unknown_view_id",
            "duplicate_view_row",
            "view_row_out_of_range",
        }


class TestIdIndexIsBuiltOnce:
    def test_index_is_memoised_on_the_graph(self, graph):
        assert graph._id_index is graph._id_index

    def test_later_id_views_do_not_sort_again(self, graph, monkeypatch):
        graph.view(ids=[13])

        def boom(*args, **kwargs):
            raise AssertionError("argsort ran on a warm graph")

        monkeypatch.setattr(np, "argsort", boom)
        np.testing.assert_array_equal(graph.view(ids=[17, 11]).graph_rows, [7, 1])
