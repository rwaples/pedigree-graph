"""Construction: the 0.8 entry points, the 0.7.1 compatibility forms, and MZ validation.

Pins slice 1c. ``from_frame`` and the canonical keyword-only ``from_arrays`` apply no
defaults; the loose constructor, ``from_dataframe``, and the ``mothers``/``fathers`` form
keep the 0.7.1 ones. Every entry point parses through :mod:`pedigree_graph._input`, so all
of them raise the same structured errors and enforce the same MZ pair contract.
"""

import numpy as np
import pandas as pd
import polars as pl
import pytest

from pedigree_graph import PedigreeGraph, PedigreeValidationError

IDS = np.array([0, 1, 2], dtype=np.int64)
MOTHERS = np.array([-1, -1, 0], dtype=np.int64)
FATHERS = np.array([-1, -1, 1], dtype=np.int64)


def _trio(**overrides):
    """Two founders and their child, ids equal to rows."""
    base = {"id": IDS.copy(), "mother": MOTHERS.copy(), "father": FATHERS.copy()}
    base.update(overrides)
    return base


def _full(**overrides):
    """Two founders and their MZ twin children, every optional column supplied."""
    base = {
        "id": np.array([10, 11, 12, 13], dtype=np.int64),
        "mother": np.array([-1, -1, 10, 10], dtype=np.int64),
        "father": np.array([-1, -1, 11, 11], dtype=np.int64),
        "twin": np.array([-1, -1, 13, 12], dtype=np.int64),
        "sex": np.array([0, 1, 1, 1], dtype=np.int8),
        "generation": np.array([0, 0, 1, 1], dtype=np.int32),
        "birth_year": np.array([1980, 1980, 2010, 2010], dtype=np.int32),
    }
    base.update(overrides)
    return base


def _twins(**overrides):
    """Two founders and their two children, cross-referencing as MZ co-twins."""
    base = {
        "id": [10, 11, 12, 13],
        "mother": [-1, -1, 10, 10],
        "father": [-1, -1, 11, 11],
        "twin": [-1, -1, 13, 12],
    }
    base.update(overrides)
    return base


def _mz_fields(data, code):
    with pytest.raises(PedigreeValidationError) as info:
        PedigreeGraph.from_frame(data)
    assert info.value.code == code
    return dict(info.value.fields)


def _from_frame(data):
    return PedigreeGraph.from_frame(data)


def _from_arrays_canonical(data):
    return PedigreeGraph.from_arrays(
        ids=data["id"],
        mother_ids=data["mother"],
        father_ids=data["father"],
        twin_ids=data.get("twin"),
        sex=data.get("sex"),
    )


def _from_arrays_legacy(data):
    return PedigreeGraph.from_arrays(
        ids=data["id"],
        mothers=data["mother"],
        fathers=data["father"],
        twins=data.get("twin"),
        sex=data.get("sex"),
    )


def _loose_constructor(data):
    return PedigreeGraph(data)


def _from_dataframe(data):
    return PedigreeGraph.from_dataframe(pl.DataFrame(data))


ENTRY_POINTS = [
    pytest.param(_from_frame, id="from_frame"),
    pytest.param(_from_arrays_canonical, id="from_arrays_canonical"),
    pytest.param(_from_arrays_legacy, id="from_arrays_legacy"),
    pytest.param(_loose_constructor, id="loose_constructor"),
    pytest.param(_from_dataframe, id="from_dataframe"),
]


class TestFromFrame:
    def test_dict_of_arrays(self):
        pg = PedigreeGraph.from_frame(_trio())
        assert pg.n_individuals == 3
        assert pg.mother_rows.tolist() == [-1, -1, 0]

    def test_dict_of_python_lists(self):
        pg = PedigreeGraph.from_frame({"id": [0, 1, 2], "mother": [-1, -1, 0], "father": [-1, -1, 1]})
        assert pg.ids.tolist() == [0, 1, 2]

    def test_polars_frame(self):
        pg = PedigreeGraph.from_frame(pl.DataFrame(_full()))
        assert pg.ids.tolist() == [10, 11, 12, 13]
        assert pg.father_rows.tolist() == [-1, -1, 1, 1]

    def test_pandas_frame(self):
        pg = PedigreeGraph.from_frame(pd.DataFrame(_full()))
        assert pg.ids.tolist() == [10, 11, 12, 13]
        assert pg.father_rows.tolist() == [-1, -1, 1, 1]


class TestFromArraysCanonical:
    def test_required_columns_only(self):
        pg = PedigreeGraph.from_arrays(ids=IDS, mother_ids=MOTHERS, father_ids=FATHERS)
        assert pg.ids.tolist() == [0, 1, 2]
        assert pg.mother_rows.tolist() == [-1, -1, 0]

    def test_every_optional_column(self):
        data = _full()
        pg = PedigreeGraph.from_arrays(
            ids=data["id"],
            mother_ids=data["mother"],
            father_ids=data["father"],
            twin_ids=data["twin"],
            sex=data["sex"],
            generation=data["generation"],
            birth_year=data["birth_year"],
        )
        assert pg.twin_rows.tolist() == [-1, -1, 3, 2]
        assert pg.sex.tolist() == [0, 1, 1, 1]
        assert pg.generation_labels.tolist() == [0, 0, 1, 1]
        assert pg.birth_year.tolist() == [1980, 1980, 2010, 2010]


class TestLegacyForms:
    def test_loose_constructor(self):
        pg = PedigreeGraph(_trio())
        assert pg.mother_rows.tolist() == [-1, -1, 0]

    def test_from_dataframe(self):
        pg = PedigreeGraph.from_dataframe(pl.DataFrame(_trio()))
        assert pg.mother_rows.tolist() == [-1, -1, 0]

    def test_from_arrays_positional(self):
        pg = PedigreeGraph.from_arrays(IDS, MOTHERS, FATHERS)
        assert pg.mother_rows.tolist() == [-1, -1, 0]

    def test_from_arrays_legacy_keywords(self):
        pg = PedigreeGraph.from_arrays(ids=IDS, mothers=MOTHERS, fathers=FATHERS)
        assert pg.mother_rows.tolist() == [-1, -1, 0]

    def test_from_arrays_all_positional_columns(self):
        data = _full()
        pg = PedigreeGraph.from_arrays(
            data["id"],
            data["mother"],
            data["father"],
            data["twin"],
            data["generation"],
            data["birth_year"],
            data["sex"],
        )
        assert pg.twin_rows.tolist() == [-1, -1, 3, 2]
        assert pg.generation.tolist() == [0, 0, 1, 1]
        assert pg.sex.tolist() == [0, 1, 1, 1]


class TestFromArraysDispatch:
    @pytest.mark.parametrize(
        "call",
        [
            pytest.param(
                lambda: PedigreeGraph.from_arrays(ids=IDS, mothers=MOTHERS, mother_ids=MOTHERS, father_ids=FATHERS),
                id="mothers_and_mother_ids",
            ),
            pytest.param(
                lambda: PedigreeGraph.from_arrays(IDS, mother_ids=MOTHERS, father_ids=FATHERS),
                id="canonical_keywords_with_a_positional",
            ),
            pytest.param(lambda: PedigreeGraph.from_arrays(ids=IDS), id="neither_family"),
            pytest.param(
                lambda: PedigreeGraph.from_arrays(ids=IDS, mothers=MOTHERS, fathers=FATHERS, sex_encoding="plink"),
                id="sex_encoding_with_the_legacy_form",
            ),
            pytest.param(lambda: PedigreeGraph.from_arrays(ids=IDS, mothers=MOTHERS), id="legacy_without_fathers"),
            pytest.param(
                lambda: PedigreeGraph.from_arrays(IDS, MOTHERS, FATHERS, twin_ids=np.full(len(IDS), -1)),
                id="twin_ids_with_the_legacy_form",
            ),
            pytest.param(
                lambda: PedigreeGraph.from_arrays(mother_ids=MOTHERS, father_ids=FATHERS),
                id="canonical_without_ids",
            ),
            pytest.param(
                lambda: PedigreeGraph.from_arrays(ids=IDS, mother_ids=MOTHERS),
                id="canonical_without_father_ids",
            ),
            pytest.param(
                lambda: PedigreeGraph.from_arrays(IDS, MOTHERS, FATHERS, None, None, None, None, None),
                id="eight_positional_arguments",
            ),
            pytest.param(
                lambda: PedigreeGraph.from_arrays(IDS, MOTHERS, FATHERS, ids=IDS),
                id="ids_positionally_and_by_keyword",
            ),
        ],
    )
    def test_rejected_call_forms_are_plain_type_errors(self, call):
        with pytest.raises(TypeError):
            call()


class TestCanonicalDefaults:
    def test_omitted_sex_is_absent(self):
        pg = PedigreeGraph.from_arrays(ids=IDS, mother_ids=MOTHERS, father_ids=FATHERS)
        assert pg.sex is None

    def test_omitted_generation_leaves_only_structural_depth(self):
        pg = PedigreeGraph.from_arrays(ids=IDS, mother_ids=MOTHERS, father_ids=FATHERS)
        assert pg.generation_labels is None
        assert pg.depth.tolist() == [0, 0, 1]

    def test_omitted_birth_year_is_absent(self):
        pg = PedigreeGraph.from_arrays(ids=IDS, mother_ids=MOTHERS, father_ids=FATHERS)
        assert pg.birth_year is None

    def test_from_frame_applies_the_same_absences(self):
        pg = PedigreeGraph.from_frame(_trio())
        assert pg.sex is None
        assert pg.generation_labels is None
        assert pg.birth_year is None


class TestLegacyDefaults:
    @pytest.mark.parametrize("build", [_from_arrays_legacy, _loose_constructor, _from_dataframe])
    def test_omitted_sex_is_all_zero_int8(self, build):
        pg = build(_trio())
        assert pg.sex.dtype == np.int8
        np.testing.assert_array_equal(pg.sex, np.zeros(3, dtype=np.int8))

    @pytest.mark.parametrize("build", [_from_arrays_legacy, _loose_constructor, _from_dataframe])
    def test_defaulted_sex_is_read_only_and_stable(self, build):
        pg = build(_trio())
        assert pg.sex.flags.writeable is False
        assert pg.sex is pg.sex
        with pytest.raises(ValueError, match="read-only"):
            pg.sex[0] = 1

    @pytest.mark.parametrize("build", [_from_arrays_legacy, _loose_constructor, _from_dataframe])
    def test_omitted_generation_falls_back_to_depth(self, build):
        pg = build(_trio())
        assert pg.generation_labels is None
        np.testing.assert_array_equal(pg.generation, pg.depth)


class TestSexEncoding:
    def test_plink_through_from_frame(self):
        pg = PedigreeGraph.from_frame(_trio(sex=[1, 2, 0]), sex_encoding="plink")
        assert pg.sex.tolist() == [1, 0, -1]

    def test_plink_through_canonical_from_arrays(self):
        pg = PedigreeGraph.from_arrays(
            ids=IDS,
            mother_ids=MOTHERS,
            father_ids=FATHERS,
            sex=[1, 2, 0],
            sex_encoding="plink",
        )
        assert pg.sex.tolist() == [1, 0, -1]

    def test_simace_is_the_default_and_stores_values_unchanged(self):
        pg = PedigreeGraph.from_frame(_trio(sex=[1, 0, -1]))
        assert pg.sex.tolist() == [1, 0, -1]

    def test_loose_constructor_has_no_sex_encoding_parameter(self):
        with pytest.raises(TypeError):
            PedigreeGraph(_trio(sex=[1, 2, 0]), sex_encoding="plink")

    def test_from_dataframe_has_no_sex_encoding_parameter(self):
        with pytest.raises(TypeError):
            PedigreeGraph.from_dataframe(pl.DataFrame(_trio(sex=[1, 2, 0])), sex_encoding="plink")


_SHARED_PROPERTIES = (
    "ids",
    "mother_ids",
    "father_ids",
    "twin_ids",
    "mother_rows",
    "father_rows",
    "twin_rows",
    "depth",
    "generation_labels",
    "birth_year",
    "n_individuals",
)


def _assert_same_graph(left_graph, right_graph):
    for name in _SHARED_PROPERTIES:
        left = getattr(left_graph, name)
        right = getattr(right_graph, name)
        if left is None or right is None:
            assert left is right, name
        else:
            np.testing.assert_array_equal(left, right, err_msg=name)


class TestCanonicalLegacyEquivalence:
    def test_from_arrays_forms_agree_on_full_data(self):
        data = _full()
        canonical = PedigreeGraph.from_arrays(
            ids=data["id"],
            mother_ids=data["mother"],
            father_ids=data["father"],
            twin_ids=data["twin"],
            sex=data["sex"],
            generation=data["generation"],
            birth_year=data["birth_year"],
        )
        legacy = PedigreeGraph.from_arrays(
            ids=data["id"],
            mothers=data["mother"],
            fathers=data["father"],
            twins=data["twin"],
            sex=data["sex"],
            generation=data["generation"],
            birth_year=data["birth_year"],
        )
        _assert_same_graph(canonical, legacy)
        np.testing.assert_array_equal(canonical.sex, legacy.sex)

    def test_from_arrays_forms_differ_only_in_the_sex_default(self):
        canonical = PedigreeGraph.from_arrays(ids=IDS, mother_ids=MOTHERS, father_ids=FATHERS)
        legacy = PedigreeGraph.from_arrays(ids=IDS, mothers=MOTHERS, fathers=FATHERS)
        _assert_same_graph(canonical, legacy)
        assert canonical.sex is None
        np.testing.assert_array_equal(legacy.sex, np.zeros(3, dtype=np.int8))

    def test_from_frame_agrees_with_the_loose_constructor(self):
        data = _full()
        _assert_same_graph(PedigreeGraph.from_frame(data), PedigreeGraph(data))

    def test_legacy_positional_agrees_with_legacy_keywords(self):
        data = _full()
        positional = PedigreeGraph.from_arrays(
            data["id"],
            data["mother"],
            data["father"],
            data["twin"],
            data["generation"],
            data["birth_year"],
            data["sex"],
        )
        keywords = PedigreeGraph.from_arrays(
            ids=data["id"],
            mothers=data["mother"],
            fathers=data["father"],
            twins=data["twin"],
            generation=data["generation"],
            birth_year=data["birth_year"],
            sex=data["sex"],
        )
        _assert_same_graph(positional, keywords)
        np.testing.assert_array_equal(positional.sex, keywords.sex)


class TestStructuredErrorsReachEveryEntryPoint:
    @pytest.mark.parametrize("build", ENTRY_POINTS)
    def test_duplicate_id(self, build):
        data = {"id": [0, 1, 1], "mother": [-1, -1, -1], "father": [-1, -1, -1]}
        with pytest.raises(PedigreeValidationError) as info:
            build(data)
        assert info.value.code == "duplicate_id"
        assert info.value.fields["id"] == 1
        assert info.value.fields["rows"] == (1, 2)

    @pytest.mark.parametrize("build", ENTRY_POINTS)
    def test_broken_mz_pair(self, build):
        with pytest.raises(PedigreeValidationError) as info:
            build(_twins(twin=[-1, -1, 13, -1]))
        assert info.value.code == "mz_nonreciprocal"


class TestMzValidation:
    def test_valid_represented_pair_constructs(self):
        pg = PedigreeGraph.from_frame(_twins())
        assert pg.twin_ids.tolist() == [-1, -1, 13, 12]
        assert pg.twin_rows.tolist() == [-1, -1, 3, 2]

    def test_external_co_twin_constructs(self):
        pg = PedigreeGraph.from_frame(_twins(twin=[-1, -1, 900, -1]))
        assert pg.twin_ids.tolist() == [-1, -1, 900, -1]
        assert pg.twin_rows.tolist() == [-1, -1, -1, -1]

    def test_self_reference(self):
        assert _mz_fields(_twins(twin=[-1, -1, 12, -1]), "mz_self_reference") == {"row": 2, "id": 12}

    def test_nonreciprocal(self):
        fields = _mz_fields(_twins(twin=[-1, -1, 13, -1]), "mz_nonreciprocal")
        assert fields == {"row": 2, "id": 12, "twin_id": 13}

    def test_three_member_chain_is_nonreciprocal(self):
        data = {
            "id": [10, 11, 12, 13, 14],
            "mother": [-1, -1, 10, 10, 10],
            "father": [-1, -1, 11, 11, 11],
            "twin": [-1, -1, 13, 12, 13],
        }
        assert _mz_fields(data, "mz_nonreciprocal") == {"row": 4, "id": 14, "twin_id": 13}

    def test_parent_mismatch_in_one_role(self):
        data = {
            "id": [10, 11, 12, 13, 14],
            "mother": [-1, -1, -1, 10, 12],
            "father": [-1, -1, -1, 11, 11],
            "twin": [-1, -1, -1, 14, 13],
        }
        fields = _mz_fields(data, "mz_parent_mismatch")
        assert fields == {"row": 3, "id": 13, "twin_id": 14, "parent_roles": ("mother",)}

    def test_shared_external_parent_is_not_a_mismatch(self):
        pg = PedigreeGraph.from_frame(_twins(mother=[-1, -1, 900, 900]))
        assert pg.mother_ids.tolist() == [-1, -1, 900, 900]
        assert pg.mother_rows.tolist() == [-1, -1, -1, -1]

    def test_differing_external_parents_are_a_mismatch(self):
        fields = _mz_fields(_twins(mother=[-1, -1, 900, 901]), "mz_parent_mismatch")
        assert fields == {"row": 2, "id": 12, "twin_id": 13, "parent_roles": ("mother",)}

    def test_sex_mismatch(self):
        fields = _mz_fields(_twins(sex=[0, 1, 0, 1]), "mz_sex_mismatch")
        assert fields == {"row": 2, "id": 12, "twin_id": 13, "sex": 0, "twin_sex": 1}

    @pytest.mark.parametrize("sex", [[0, 1, -1, 1], [0, 1, 0, -1]])
    def test_sex_mismatch_skipped_when_one_member_is_unknown(self, sex):
        pg = PedigreeGraph.from_frame(_twins(sex=sex))
        assert pg.sex.tolist() == sex

    def test_sex_mismatch_skipped_when_sex_is_absent(self):
        pg = PedigreeGraph.from_frame(_twins())
        assert pg.sex is None


# Both fixtures used to fail inside compute_inbreeding; MZ validation now rejects them
# at construction, so no constructor can hand the kernels a broken pair.
_MIGRATED_IDS = [0, 1, 2, 3, 4]
_MIGRATED_MOTHERS = [-1, -1, 0, 0, 1]
_MIGRATED_FATHERS = [-1, -1, 1, 1, 0]


class TestMigratedInbreedingFixtures:
    def test_nonreciprocal_reference(self):
        data = {
            "id": _MIGRATED_IDS,
            "mother": _MIGRATED_MOTHERS,
            "father": _MIGRATED_FATHERS,
            "twin": [-1, -1, 3, -1, -1],
        }
        assert _mz_fields(data, "mz_nonreciprocal") == {"row": 2, "id": 2, "twin_id": 3}

    def test_parent_mismatch_in_both_roles(self):
        data = {
            "id": _MIGRATED_IDS,
            "mother": _MIGRATED_MOTHERS,
            "father": _MIGRATED_FATHERS,
            "twin": [-1, -1, -1, 4, 3],
        }
        fields = _mz_fields(data, "mz_parent_mismatch")
        assert fields == {"row": 3, "id": 3, "twin_id": 4, "parent_roles": ("mother", "father")}
