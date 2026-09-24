"""``RelationshipSelection``: the one parse every relationship endpoint shares.

The selector used to be validated by a private helper of ``_pair_extractor``
that the count and matrix modules imported.  These tests pin the parse itself,
and the two behaviours that parsing once at the boundary buys: a one-shot
``categories`` iterable survives the trip, and equivalent selectors are one
selection.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from conftest import parity_columns, parity_fixtures

from pedigree_graph import RELATIONSHIPS, PedigreeGraph, PedigreeValidationError
from pedigree_graph._selection import RelationshipSelection

FIXTURES = parity_fixtures()


def _graph(name: str = "avuncular_and_cousins") -> PedigreeGraph:
    return PedigreeGraph.from_frame(parity_columns(FIXTURES[name]))


class TestParsing:
    def test_neither_selector_is_rejected_without_naming_an_api(self):
        with pytest.raises(TypeError, match="exactly one of max_degree") as error:
            RelationshipSelection.parse(None, None)
        assert "relationship_pairs" not in str(error.value)

    def test_both_selectors_are_rejected(self):
        with pytest.raises(TypeError, match="exactly one of max_degree"):
            RelationshipSelection.parse(2, ["FS"])

    def test_a_bare_string_is_not_an_iterable_of_codes(self):
        with pytest.raises(TypeError, match="not a single str"):
            RelationshipSelection.parse(None, "FS")

    def test_a_non_string_code_is_rejected(self):
        with pytest.raises(TypeError, match="must be str"):
            RelationshipSelection.parse(None, ["FS", 3])

    @pytest.mark.parametrize("max_degree", [-1, 6])
    def test_an_out_of_range_cutoff_is_rejected(self, max_degree):
        with pytest.raises(PedigreeValidationError, match="max_degree") as info:
            RelationshipSelection.parse(max_degree, None)
        assert info.value.code == "max_degree_out_of_range"
        assert dict(info.value.fields) == {"value": max_degree, "minimum": 0, "maximum": 5}

    def test_an_integer_like_cutoff_is_accepted(self):
        assert RelationshipSelection.parse(np.int64(1), None).top_degree == 1

    @pytest.mark.parametrize("value", [2.9, "3", True, np.bool_(True)])
    def test_a_non_integer_cutoff_is_rejected(self, value):
        with pytest.raises(TypeError):
            RelationshipSelection.parse(value, None)

    def test_unknown_codes_are_rejected_sorted(self):
        with pytest.raises(PedigreeValidationError, match="unknown") as info:
            RelationshipSelection.parse(None, ["FS", "zz", "aa"])
        assert info.value.code == "unknown_relationship_category"
        assert info.value.fields["codes"] == ("aa", "zz")


class TestResolvedCodes:
    def test_a_cutoff_selects_every_category_at_or_below_it(self):
        selection = RelationshipSelection.parse(1, None)
        assert selection.ordered == ("MZ", "MO", "FO", "FS")
        assert selection.codes == frozenset(selection.ordered)

    def test_degree_zero_selects_only_mz(self):
        assert RelationshipSelection.parse(0, None).ordered == ("MZ",)

    def test_codes_are_canonicalised_to_registry_order(self):
        shuffled = RelationshipSelection.parse(None, ["2C", "MZ", "FS"])
        assert shuffled.ordered == ("MZ", "FS", "2C")

    def test_duplicate_codes_collapse(self):
        assert RelationshipSelection.parse(None, ["FS", "FS", "MZ"]).ordered == ("MZ", "FS")

    def test_equivalent_selectors_are_equal_selections(self):
        codes = [code for code, category in RELATIONSHIPS.items() if category.degree <= 2]
        assert RelationshipSelection.parse(2, None) == RelationshipSelection.parse(None, reversed(codes))

    def test_a_one_shot_iterable_is_consumed_once(self):
        selection = RelationshipSelection.parse(None, (code for code in ["FS", "MZ"]))
        assert selection.ordered == ("MZ", "FS")


class TestTopDegree:
    def test_top_degree_is_the_highest_requested_degree(self):
        assert RelationshipSelection.parse(3, None).top_degree == 3
        assert RelationshipSelection.parse(None, ["MZ", "1C"]).top_degree == 3

    def test_top_degree_ignores_the_cutoff_when_codes_stop_short(self):
        assert RelationshipSelection.parse(None, ["MZ"]).top_degree == 0

    def test_an_empty_selection_has_no_top_degree(self):
        empty = RelationshipSelection.parse(None, [])
        assert empty.codes == frozenset()
        assert empty.ordered == ()
        assert empty.top_degree is None


class TestFrozen:
    def test_a_selection_cannot_be_mutated(self):
        selection = RelationshipSelection.parse(1, None)
        with pytest.raises(dataclasses.FrozenInstanceError):
            selection.top_degree = 4


class TestThroughThePublicEndpoints:
    """A one-shot iterable used to need materialising by the caller."""

    @pytest.mark.parametrize(
        "call",
        [
            lambda g, c: g.relationship_pairs(categories=c),
            lambda g, c: g.relationship_counts(categories=c),
            lambda g, c: g.relationship_kinship_matrix(categories=c),
            lambda g, c: g.view(rows=[0, 1, 2]).relationship_pairs(categories=c),
            lambda g, c: g.view(rows=[0, 1, 2]).relationship_counts(categories=c),
        ],
    )
    def test_every_endpoint_accepts_a_generator_of_codes(self, call):
        graph = _graph()
        assert call(graph, (code for code in ["FS", "MZ"])) is not None

    def test_an_empty_selection_requests_nothing(self):
        graph = _graph()
        counts = graph.relationship_counts(categories=[])
        assert all(value is None for value in counts.values())
        assert not graph.relationship_pairs(categories=[])["FS"].requested
