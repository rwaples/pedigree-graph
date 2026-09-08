"""Pins the canonical relationship registry (ADR 0006, slice 2).

``RELATIONSHIPS`` is the source of truth for the code set, its order, the
per-category path shape, and the positional roles.  The table itself carries
the literal values, so these tests re-derive them from the documented formulas
and would catch a typo in any row.
"""

import dataclasses
import math
import typing

import numpy as np
import polars as pl
import pytest

import pedigree_graph
import pedigree_graph.relationships
from pedigree_graph import PAIR_KINSHIP, RELATIONSHIPS, PedigreeGraph, PedigreeValidationError, RelType
from pedigree_graph._registry import REL_PLAN, REL_REGISTRY, categories_up_to_degree, select_categories
from pedigree_graph.relationships import RelationshipRole

REGISTRY_ORDER = (
    "MZ",
    "MO",
    "FO",
    "FS",
    "MHS",
    "PHS",
    "GP",
    "Av",
    "GGP",
    "HAv",
    "GAv",
    "1C",
    "GGGP",
    "HGAv",
    "GGAv",
    "H1C",
    "1C1R",
    "G3GP",
    "HGGAv",
    "G3Av",
    "H1C1R",
    "1C2R",
    "2C",
)

SYMMETRIC_CODES = ("MZ", "FS", "MHS", "PHS", "1C", "H1C", "2C")

ROLE_TABLE = [
    ("MO", "offspring", "mother"),
    ("FO", "offspring", "father"),
    ("GP", "descendant", "ancestor"),
    ("GGP", "descendant", "ancestor"),
    ("GGGP", "descendant", "ancestor"),
    ("G3GP", "descendant", "ancestor"),
    ("Av", "niece_nephew", "aunt_uncle"),
    ("HAv", "niece_nephew", "aunt_uncle"),
    ("GAv", "niece_nephew", "aunt_uncle"),
    ("HGAv", "niece_nephew", "aunt_uncle"),
    ("GGAv", "niece_nephew", "aunt_uncle"),
    ("HGGAv", "niece_nephew", "aunt_uncle"),
    ("G3Av", "niece_nephew", "aunt_uncle"),
    ("1C1R", "junior_cousin", "senior_cousin"),
    ("H1C1R", "junior_cousin", "senior_cousin"),
    ("1C2R", "junior_cousin", "senior_cousin"),
]

DEGREE_PRECEDENCE = {
    0: ("MZ",),
    1: ("MO", "FO", "FS"),
    2: ("MHS", "PHS", "GP", "Av"),
    3: ("GGP", "HAv", "GAv", "1C"),
    4: ("GGGP", "HGAv", "GGAv", "H1C", "1C1R"),
    5: ("G3GP", "HGGAv", "G3Av", "H1C1R", "1C2R", "2C"),
}


def small_pedigree():
    """The three-generation pedigree from ``test_relationship_plan.py``."""
    return pl.DataFrame(
        {
            "id": np.arange(10),
            "mother": np.array([-1, -1, -1, -1, 0, 0, 2, 2, 4, 6]),
            "father": np.array([-1, -1, -1, -1, 1, 1, 3, 3, 5, 7]),
            "twin": np.full(10, -1),
            "sex": np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 0]),
            "generation": np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2]),
        }
    )


class TestRegistryShape:
    def test_holds_exactly_the_23_codes_in_order(self):
        assert tuple(RELATIONSHIPS) == REGISTRY_ORDER
        assert len(RELATIONSHIPS) == 23

    def test_public_facade_is_the_same_object(self):
        assert pedigree_graph.relationships.RELATIONSHIPS is pedigree_graph.RELATIONSHIPS

    @pytest.mark.parametrize("code", REGISTRY_ORDER)
    def test_category_code_matches_its_key(self, code):
        assert RELATIONSHIPS[code].code == code


class TestImmutability:
    def test_item_assignment_is_rejected(self):
        with pytest.raises(TypeError):
            RELATIONSHIPS["FS"] = RELATIONSHIPS["MZ"]

    def test_item_deletion_is_rejected(self):
        with pytest.raises(TypeError):
            del RELATIONSHIPS["FS"]

    def test_category_attributes_are_frozen(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            RELATIONSHIPS["FS"].nominal_kinship = 0.0

    def test_categories_are_hashable(self):
        assert len(set(RELATIONSHIPS.values())) == 23


class TestCategoryInvariants:
    @pytest.mark.parametrize("code", REGISTRY_ORDER)
    def test_first_is_at_least_as_far_from_the_ancestors(self, code):
        category = RELATIONSHIPS[code]
        assert category.up >= category.down

    @pytest.mark.parametrize("code", REGISTRY_ORDER)
    def test_symmetry_equal_paths_and_absent_roles_agree(self, code):
        category = RELATIONSHIPS[code]
        symmetric = category.symmetric
        assert symmetric == (category.up == category.down)
        assert symmetric == (category.first_role is None)
        assert symmetric == (category.second_role is None)
        assert symmetric == (code in SYMMETRIC_CODES)

    @pytest.mark.parametrize("code", [c for c in REGISTRY_ORDER if c != "MZ"])
    def test_nominal_kinship_follows_the_path_formula(self, code):
        category = RELATIONSHIPS[code]
        assert category.nominal_kinship == category.ancestor_count * 0.5 ** (category.up + category.down + 1)

    @pytest.mark.parametrize("code", [c for c in REGISTRY_ORDER if c != "MZ"])
    def test_degree_follows_from_nominal_kinship(self, code):
        category = RELATIONSHIPS[code]
        assert category.degree == round(-1 - math.log2(category.nominal_kinship))

    def test_mz_is_the_documented_special_case(self):
        mz = RELATIONSHIPS["MZ"]
        assert (mz.up, mz.down, mz.ancestor_count) == (0, 0, 0)
        assert mz.nominal_kinship == 0.5
        assert mz.degree == 0


class TestRoles:
    @pytest.mark.parametrize(("code", "first_role", "second_role"), ROLE_TABLE)
    def test_asymmetric_roles_are_the_canonical_literals(self, code, first_role, second_role):
        category = RELATIONSHIPS[code]
        assert (category.first_role, category.second_role) == (first_role, second_role)
        assert not category.symmetric

    @pytest.mark.parametrize("code", SYMMETRIC_CODES)
    def test_symmetric_categories_carry_no_roles(self, code):
        category = RELATIONSHIPS[code]
        assert category.first_role is None
        assert category.second_role is None
        assert category.symmetric

    def test_every_role_in_use_is_in_the_closed_literal_set(self):
        allowed = set(typing.get_args(RelationshipRole))
        used = {role for c in RELATIONSHIPS.values() for role in (c.first_role, c.second_role) if role is not None}
        assert used == allowed

    def test_the_role_table_covers_every_asymmetric_code(self):
        assert {code for code, _, _ in ROLE_TABLE} == set(REGISTRY_ORDER) - set(SYMMETRIC_CODES)


class TestOrderIsSameDegreePrecedence:
    @pytest.mark.parametrize("degree", sorted(DEGREE_PRECEDENCE))
    def test_within_degree_precedence_is_pinned(self, degree):
        found = tuple(c.code for c in RELATIONSHIPS.values() if c.degree == degree)
        assert found == DEGREE_PRECEDENCE[degree]

    def test_degrees_are_non_decreasing_along_iteration(self):
        degrees = [c.degree for c in RELATIONSHIPS.values()]
        assert degrees == sorted(degrees)


class TestSelectors:
    def test_select_categories_returns_registry_order(self):
        selected = select_categories(["2C", "MZ", "FS", "FS"])
        assert tuple(c.code for c in selected) == ("MZ", "FS", "2C")

    def test_select_categories_reports_unknown_codes_sorted(self):
        with pytest.raises(PedigreeValidationError) as excinfo:
            select_categories(["zz", "FS", "aa"])
        assert excinfo.value.code == "unknown_relationship_category"
        assert excinfo.value.fields["codes"] == ("aa", "zz")

    def test_select_categories_rejects_a_non_string_code(self):
        with pytest.raises(TypeError):
            select_categories(["FS", 3])

    def test_categories_up_to_degree_stops_at_the_cutoff(self):
        assert tuple(c.code for c in categories_up_to_degree(1)) == ("MZ", "MO", "FO", "FS")

    def test_categories_up_to_degree_validates_the_cutoff(self):
        with pytest.raises(PedigreeValidationError) as excinfo:
            categories_up_to_degree(6)
        assert excinfo.value.code == "max_degree_out_of_range"


def test_engine_plan_covers_exactly_the_registry():
    assert set(REL_PLAN) == set(RELATIONSHIPS)


class TestCompatibilitySnapshotsAreDetached:
    @pytest.fixture(autouse=True)
    def _rebuild_snapshots(self):
        # monkeypatch undoes a delitem by re-inserting, which moves the key to
        # the end; the snapshots are order-sensitive, so rebuild them in place.
        registry, kinship = dict(REL_REGISTRY), dict(PAIR_KINSHIP)
        yield
        REL_REGISTRY.clear()
        REL_REGISTRY.update(registry)
        PAIR_KINSHIP.clear()
        PAIR_KINSHIP.update(kinship)

    def test_mutating_pair_kinship_leaves_the_registry_alone(self, monkeypatch):
        monkeypatch.setitem(PAIR_KINSHIP, "FS", 0.0)
        assert RELATIONSHIPS["FS"].nominal_kinship == 0.25

    def test_deleting_from_rel_registry_leaves_the_registry_alone(self, monkeypatch):
        monkeypatch.delitem(REL_REGISTRY, "MZ")
        assert "MZ" in RELATIONSHIPS

    def test_engines_ignore_the_mutated_snapshots(self, monkeypatch):
        monkeypatch.setitem(PAIR_KINSHIP, "FS", 0.0)
        monkeypatch.delitem(REL_REGISTRY, "MZ")
        counts = PedigreeGraph.from_frame(small_pedigree()).relationship_counts(max_degree=5)
        assert set(counts) == set(RELATIONSHIPS)


class TestCompatibilitySnapshotContents:
    def test_avuncular_keeps_the_0_7_1_orientation(self):
        assert REL_REGISTRY["Av"] == RelType(1, 2, 2, "Av", "Avuncular")

    def test_lineal_orientation_is_unchanged(self):
        assert REL_REGISTRY["MO"] == RelType(1, 0, 1, "MO", "Mother-offspring")

    def test_pair_kinship_mirrors_the_nominal_values(self):
        assert {code: c.nominal_kinship for code, c in RELATIONSHIPS.items()} == PAIR_KINSHIP

    def test_snapshot_order_matches_the_registry(self):
        assert list(REL_REGISTRY) == list(RELATIONSHIPS)

    def test_snapshots_are_plain_mutable_dicts(self):
        assert type(REL_REGISTRY) is dict
        assert type(PAIR_KINSHIP) is dict
