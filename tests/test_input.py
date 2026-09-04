"""Boundary tests for the validated, owned input model (ADR 0006, slice 1a)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest
from conftest import pedigree_arrays
from hypothesis import given, settings

from pedigree_graph import PedigreeGraph, PedigreeValidationError, ResourceError
from pedigree_graph._input import parse_pedigree_arrays, parse_pedigree_input, validate_id_field

_INT64_MAX = int(np.iinfo(np.int64).max)
_INT32_MAX = int(np.iinfo(np.int32).max)


def _data(**overrides):
    """A three-row pedigree: two founders and their child."""
    base = {
        "id": np.array([0, 1, 2], dtype=np.int64),
        "mother": np.array([-1, -1, 0], dtype=np.int64),
        "father": np.array([-1, -1, 1], dtype=np.int64),
    }
    base.update(overrides)
    return base


def _raises(data, code, **kwargs):
    with pytest.raises(PedigreeValidationError) as info:
        parse_pedigree_input(data, **kwargs)
    assert info.value.code == code
    return info.value


class TestDtypeAcceptance:
    @pytest.mark.parametrize(
        "dtype",
        [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64, np.float32, np.float64],
    )
    def test_id_dtypes_coerce_losslessly(self, dtype):
        parsed = parse_pedigree_input(_data(id=np.array([0, 1, 2], dtype=dtype)))
        assert parsed.ids.dtype == np.int64
        assert parsed.ids.tolist() == [0, 1, 2]

    @pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int32, np.int64, np.float32, np.float64])
    def test_parent_dtypes_coerce_losslessly(self, dtype):
        parsed = parse_pedigree_input(_data(mother=np.array([-1, -1, 0], dtype=dtype)))
        assert parsed.mother_ids.dtype == np.int64
        assert parsed.mother_ids.tolist() == [-1, -1, 0]
        assert parsed.mother_rows.tolist() == [-1, -1, 0]

    def test_object_dtype_integers_are_accepted(self):
        parsed = parse_pedigree_input(_data(mother=np.array([-1, -1, 0], dtype=object)))
        assert parsed.mother_ids.tolist() == [-1, -1, 0]

    def test_object_dtype_integral_floats_are_accepted(self):
        parsed = parse_pedigree_input(_data(mother=np.array([-1.0, -1.0, 0.0], dtype=object)))
        assert parsed.mother_ids.tolist() == [-1, -1, 0]

    def test_python_lists_are_accepted(self):
        parsed = parse_pedigree_input({"id": [0, 1, 2], "mother": [-1, -1, 0], "father": [-1, -1, 1]})
        assert parsed.mother_rows.tolist() == [-1, -1, 0]

    def test_uint64_above_int64_max_is_out_of_range(self):
        error = _raises(_data(id=np.array([0, 1, 2**63], dtype=np.uint64)), "value_out_of_range")
        assert error.fields["field"] == "id"
        assert error.fields["position"] == 2
        assert error.fields["value"] == 2**63
        assert error.fields["maximum"] == _INT64_MAX

    def test_unknown_dict_keys_are_ignored(self):
        parsed = parse_pedigree_input(_data(household=np.array(["a", "b", "c"])))
        assert parsed.n_individuals == 3


class TestFrameInput:
    def test_polars_int32_columns(self):
        frame = pl.DataFrame(_data()).with_columns(pl.col(c).cast(pl.Int32) for c in ("id", "mother", "father"))
        parsed = parse_pedigree_input(frame)
        assert parsed.ids.tolist() == [0, 1, 2]
        assert parsed.mother_rows.tolist() == [-1, -1, 0]

    def test_pandas_nullable_int64_columns(self):
        frame = pd.DataFrame({k: pd.array(v, dtype="Int64") for k, v in _data().items()})
        parsed = parse_pedigree_input(frame)
        assert parsed.ids.tolist() == [0, 1, 2]
        assert parsed.mother_rows.tolist() == [-1, -1, 0]

    @pytest.mark.parametrize("field", ["mother", "father", "twin", "sex", "generation", "birth_year"])
    def test_polars_nulls_become_the_missing_sentinel(self, field):
        columns = {**_data(), "twin": [-1, -1, -1], "sex": [0, 1, 0], "generation": [0, 0, 1]}
        columns["birth_year"] = [1990, 1990, 2020]
        columns[field] = [None, None, None]
        parsed = parse_pedigree_input(pl.DataFrame(columns))
        stored = {
            "mother": parsed.mother_ids,
            "father": parsed.father_ids,
            "twin": parsed.twin_ids,
            "sex": parsed.sex,
            "generation": parsed.generation,
            "birth_year": parsed.birth_year,
        }[field]
        if field in ("sex", "generation", "birth_year"):
            assert stored is None
        else:
            assert stored.tolist() == [-1, -1, -1]

    def test_pandas_na_becomes_the_missing_sentinel(self):
        frame = pd.DataFrame(
            {
                "id": pd.array([0, 1, 2], dtype="Int64"),
                "mother": pd.array([pd.NA, pd.NA, 0], dtype="Int64"),
                "father": pd.array([pd.NA, pd.NA, 1], dtype="Int64"),
                "birth_year": pd.array([1990, pd.NA, 2020], dtype="Int64"),
            }
        )
        parsed = parse_pedigree_input(frame)
        assert parsed.mother_ids.tolist() == [-1, -1, 0]
        assert parsed.birth_year.tolist() == [1990, -1, 2020]

    def test_polars_null_id_is_not_a_missing_value(self):
        error = _raises(pl.DataFrame({**_data(), "id": [0, None, 2]}), "invalid_integer_value")
        assert error.fields["field"] == "id"
        assert error.fields["position"] == 1
        assert error.fields["value"] == "null"

    def test_pandas_na_id_is_not_a_missing_value(self):
        frame = pd.DataFrame({**{k: pd.array(v, dtype="Int64") for k, v in _data().items()}})
        frame["id"] = pd.array([0, pd.NA, 2], dtype="Int64")
        error = _raises(frame, "invalid_integer_value")
        assert error.fields["field"] == "id"
        assert error.fields["value"] == "null"

    def test_extra_frame_columns_are_ignored(self):
        parsed = parse_pedigree_input(pl.DataFrame({**_data(), "household": [1, 1, 2]}))
        assert parsed.n_individuals == 3


class TestStructuralFailures:
    def test_missing_field_order(self):
        assert _raises({}, "missing_field").fields["field"] == "id"
        assert _raises({"id": [0]}, "missing_field").fields["field"] == "mother"
        assert _raises({"id": [0], "mother": [-1]}, "missing_field").fields["field"] == "father"

    def test_invalid_shape_on_id(self):
        error = _raises(_data(id=np.zeros((2, 2), dtype=np.int64)), "invalid_shape")
        assert error.fields["field"] == "id"
        assert error.fields["expected_ndim"] == 1
        assert error.fields["actual_shape"] == (2, 2)

    def test_invalid_shape_on_optional_field(self):
        error = _raises(_data(sex=np.zeros((3, 1), dtype=np.int64)), "invalid_shape")
        assert error.fields["field"] == "sex"
        assert error.fields["actual_shape"] == (3, 1)

    def test_length_mismatch(self):
        error = _raises(_data(mother=np.array([-1, -1])), "length_mismatch")
        assert error.fields["field"] == "mother"
        assert error.fields["expected_length"] == 3
        assert error.fields["actual_length"] == 2

    def test_duplicate_id_reports_smallest_id_and_all_its_rows(self):
        data = {"id": np.array([5, 3, 5, 3, 5]), "mother": np.full(5, -1), "father": np.full(5, -1)}
        error = _raises(data, "duplicate_id")
        assert error.fields["id"] == 3
        assert error.fields["rows"] == (1, 3)
        assert error.fields["duplicate_count"] == 3

    def test_same_parent_id_including_an_external_id(self):
        data = {"id": np.array([0, 1]), "mother": np.array([-1, 99]), "father": np.array([-1, 99])}
        error = _raises(data, "same_parent_id")
        assert error.fields["row"] == 1
        assert error.fields["child_id"] == 1
        assert error.fields["parent_id"] == 99

    def test_pedigree_too_large(self, monkeypatch):
        monkeypatch.setattr("pedigree_graph._input._MAX_ROWS", 2)
        with pytest.raises(ResourceError) as info:
            parse_pedigree_input(_data())
        assert info.value.code == "pedigree_too_large"
        assert info.value.fields["n_individuals"] == 3
        assert info.value.fields["maximum"] == 2


class TestNumericFailures:
    def test_non_integral_float_is_rejected(self):
        error = _raises(_data(id=np.array([0.0, 0.5, 2.0])), "invalid_integer_value")
        assert error.fields["field"] == "id"
        assert error.fields["position"] == 1
        assert error.fields["value"] == 0.5

    def test_infinite_float_is_rejected(self):
        error = _raises(_data(mother=np.array([-1.0, np.inf, 0.0])), "invalid_integer_value")
        assert error.fields["position"] == 1

    def test_bool_dtype_is_never_treated_as_zero_or_one(self):
        error = _raises(_data(sex=np.array([True, False, True])), "invalid_integer_value")
        assert error.fields["field"] == "sex"
        assert error.fields["position"] == 0
        assert error.fields["value"] is True

    def test_object_bool_is_never_treated_as_zero_or_one(self):
        error = _raises(_data(sex=np.array([0, True, 1], dtype=object)), "invalid_integer_value")
        assert error.fields["position"] == 1

    def test_string_column_is_rejected(self):
        error = _raises(_data(mother=np.array(["a", "b", "c"])), "invalid_integer_value")
        assert error.fields["field"] == "mother"
        assert error.fields["position"] == 0
        assert error.fields["value"] == "a"

    def test_object_string_is_rejected(self):
        error = _raises(_data(mother=np.array([-1, "x", 0], dtype=object)), "invalid_integer_value")
        assert error.fields["position"] == 1
        assert error.fields["value"] == "x"

    @pytest.mark.parametrize(
        ("field", "column", "position", "value", "bounds"),
        [
            ("id", [0, -1, 2], 1, -1, (0, _INT64_MAX)),
            ("mother", [-1, -2, 0], 1, -2, (-1, _INT64_MAX)),
            ("father", [-1, -1, -5], 2, -5, (-1, _INT64_MAX)),
            ("twin", [-2, -1, -1], 0, -2, (-1, _INT64_MAX)),
            ("generation", [-2, 0, 1], 0, -2, (-1, _INT32_MAX)),
            ("generation", [_INT32_MAX + 1, 0, 1], 0, _INT32_MAX + 1, (-1, _INT32_MAX)),
            ("birth_year", [-2, 0, 1], 0, -2, (-1, _INT32_MAX)),
            ("birth_year", [0, 0, _INT32_MAX + 1], 2, _INT32_MAX + 1, (-1, _INT32_MAX)),
            ("sex", [0, 1, 2], 2, 2, (-1, 1)),
            ("sex", [-2, 0, 1], 0, -2, (-1, 1)),
        ],
    )
    def test_value_out_of_range_bounds(self, field, column, position, value, bounds):
        error = _raises(_data(**{field: np.array(column, dtype=np.int64)}), "value_out_of_range")
        assert error.fields["field"] == field
        assert error.fields["position"] == position
        assert error.fields["value"] == value
        assert (error.fields["minimum"], error.fields["maximum"]) == bounds


class TestSexEncoding:
    @pytest.mark.parametrize(("raw", "stored"), [(0, -1), (1, 1), (2, 0), (-1, -1)])
    def test_plink_mapping_table(self, raw, stored):
        parsed = parse_pedigree_input(_data(sex=np.array([raw, 1, 1])), sex_encoding="plink")
        assert parsed.sex[0] == stored

    def test_plink_rejects_three(self):
        error = _raises(_data(sex=np.array([0, 1, 3])), "value_out_of_range", sex_encoding="plink")
        assert error.fields["position"] == 2
        assert (error.fields["minimum"], error.fields["maximum"]) == (0, 2)

    def test_simace_rejects_two(self):
        error = _raises(_data(sex=np.array([0, 1, 2])), "value_out_of_range")
        assert (error.fields["minimum"], error.fields["maximum"]) == (-1, 1)

    def test_simace_stores_values_unchanged(self):
        parsed = parse_pedigree_input(_data(sex=np.array([0, 1, -1])))
        assert parsed.sex.tolist() == [0, 1, -1]
        assert parsed.sex.dtype == np.int8

    def test_unknown_encoding_is_plain_api_misuse(self):
        with pytest.raises(ValueError, match="sex_encoding") as info:
            parse_pedigree_input(_data(), sex_encoding="foo")
        assert not isinstance(info.value, PedigreeValidationError)


class TestNormalization:
    def test_omitted_metadata_is_none(self):
        parsed = parse_pedigree_input(_data())
        assert parsed.sex is None
        assert parsed.generation is None
        assert parsed.birth_year is None

    @pytest.mark.parametrize("field", ["sex", "generation", "birth_year"])
    def test_wholly_missing_metadata_normalizes_to_none(self, field):
        parsed = parse_pedigree_input(_data(**{field: np.full(3, -1)}))
        assert getattr(parsed, field) is None

    @pytest.mark.parametrize(("field", "column"), [("sex", [0, -1, -1]), ("generation", [-1, -1, 2])])
    def test_partial_metadata_keeps_the_sentinel(self, field, column):
        parsed = parse_pedigree_input(_data(**{field: np.array(column)}))
        assert getattr(parsed, field).tolist() == column

    def test_birth_year_partial_keeps_the_sentinel(self):
        parsed = parse_pedigree_input(_data(birth_year=np.array([1990, -1, 2020])))
        assert parsed.birth_year.tolist() == [1990, -1, 2020]
        assert parsed.birth_year.dtype == np.int32

    def test_twin_is_always_an_array(self):
        parsed = parse_pedigree_input(_data())
        assert parsed.twin_ids.tolist() == [-1, -1, -1]
        assert parsed.twin_ids.dtype == np.int64
        assert parsed.twin_rows.tolist() == [-1, -1, -1]
        assert parsed.twin_rows.dtype == np.int32

    def test_empty_pedigree_parses(self):
        empty = np.array([], dtype=np.int64)
        parsed = parse_pedigree_input({"id": empty, "mother": empty, "father": empty})
        assert parsed.n_individuals == 0
        assert parsed.twin_rows.dtype == np.int32
        assert parsed.sex is None

    def test_row_dtypes_are_int32(self):
        parsed = parse_pedigree_input(_data())
        assert parsed.mother_rows.dtype == np.int32
        assert parsed.father_rows.dtype == np.int32


class TestExternalReferences:
    def test_unresolved_ids_keep_their_id_and_lose_their_row(self):
        data = {
            "id": np.array([10, 11]),
            "mother": np.array([99, -1]),
            "father": np.array([-1, -1]),
            "twin": np.array([77, -1]),
        }
        parsed = parse_pedigree_input(data)
        assert parsed.mother_ids.tolist() == [99, -1]
        assert parsed.mother_rows.tolist() == [-1, -1]
        assert parsed.twin_ids.tolist() == [77, -1]
        assert parsed.twin_rows.tolist() == [-1, -1]


class TestOwnership:
    def test_arrays_are_read_only(self):
        parsed = parse_pedigree_input(_data(sex=np.array([0, 1, 0]), birth_year=np.array([1, 2, 3])))
        for array in (
            parsed.ids,
            parsed.mother_ids,
            parsed.father_ids,
            parsed.twin_ids,
            parsed.mother_rows,
            parsed.father_rows,
            parsed.twin_rows,
            parsed.sex,
            parsed.birth_year,
        ):
            assert not array.flags.writeable
            with pytest.raises(ValueError, match="read-only"):
                array[0] = 0

    def test_mutating_the_caller_arrays_changes_nothing(self):
        data = _data()
        parsed = parse_pedigree_input(data)
        data["mother"][2] = 1
        data["id"][0] = 7
        assert parsed.mother_ids.tolist() == [-1, -1, 0]
        assert parsed.ids.tolist() == [0, 1, 2]

    def test_strided_and_fortran_views_are_copied(self):
        strided = np.arange(6, dtype=np.int64)[::2]
        fortran = np.asfortranarray(np.array([[-1, 9], [-1, 9], [0, 9]], dtype=np.int64))[:, 0]
        parsed = parse_pedigree_input({"id": strided, "mother": fortran, "father": np.full(3, -1)})
        assert parsed.ids.flags.c_contiguous
        assert parsed.mother_ids.flags.c_contiguous
        assert parsed.ids.tolist() == [0, 2, 4]
        strided[0] = 100
        fortran[0] = 100
        assert parsed.ids.tolist() == [0, 2, 4]
        assert parsed.mother_ids.tolist() == [-1, -1, 0]

    def test_graph_arrays_survive_caller_mutation(self):
        mothers = np.array([-1, -1, 0], dtype=np.int64)
        pg = PedigreeGraph({"id": np.array([0, 1, 2]), "mother": mothers, "father": np.array([-1, -1, 1])})
        mothers[2] = 1
        assert pg.mother.tolist() == [-1, -1, 0]
        assert pg._orig_mother.tolist() == [-1, -1, 0]


class TestCycles:
    def test_three_cycle_witness(self):
        data = {"id": np.array([0, 1, 2]), "mother": np.array([1, 2, 0]), "father": np.full(3, -1)}
        assert _raises(data, "cycle").fields["ids"] == (0, 1, 2)

    def test_two_cycle_witness(self):
        data = {"id": np.array([0, 1]), "mother": np.array([1, -1]), "father": np.array([-1, 0])}
        assert _raises(data, "cycle").fields["ids"] == (0, 1)

    def test_self_parent_row(self):
        data = {"id": np.array([0, 1]), "mother": np.array([-1, 1]), "father": np.array([-1, -1])}
        assert _raises(data, "cycle").fields["ids"] == (1,)

    def test_witness_is_independent_of_row_order(self):
        first = {"id": np.array([10, 20, 30]), "mother": np.array([20, 30, 10]), "father": np.full(3, -1)}
        order = [2, 0, 1]
        second = {key: value[order] for key, value in first.items()}
        assert _raises(first, "cycle").fields["ids"] == _raises(second, "cycle").fields["ids"]
        assert _raises(second, "cycle").fields["ids"] == (10, 20, 30)

    def test_cycle_through_both_parent_roles(self):
        data = {"id": np.array([0, 1, 2]), "mother": np.array([-1, 0, -1]), "father": np.array([2, -1, 1])}
        assert _raises(data, "cycle").fields["ids"] == (0, 2, 1)

    def test_acyclic_reordered_rows_pass_the_parser(self):
        data = {"id": np.array([0, 1]), "mother": np.array([1, -1]), "father": np.array([-1, -1])}
        parsed = parse_pedigree_input(data)
        assert parsed.mother_rows.tolist() == [1, -1]

    def test_acyclic_reordered_rows_construct_with_structural_depth(self):
        data = {"id": np.array([0, 1]), "mother": np.array([1, -1]), "father": np.array([-1, -1])}
        pg = PedigreeGraph(data)
        assert pg.generation.tolist() == [1, 0]


class TestIdFieldValidation:
    def test_returns_an_owned_int64_array(self):
        validated = validate_id_field(np.array([3, 1, 2], dtype=np.int32))
        assert validated.dtype == np.int64
        assert not validated.flags.writeable
        assert validated.tolist() == [3, 1, 2]

    def test_rejects_duplicates(self):
        with pytest.raises(PedigreeValidationError) as info:
            validate_id_field(np.array([1, 1]))
        assert info.value.code == "duplicate_id"

    def test_rejects_negatives(self):
        with pytest.raises(PedigreeValidationError) as info:
            validate_id_field(np.array([-1, 0]))
        assert info.value.code == "value_out_of_range"


class TestParseArraysEntryPoint:
    def test_matches_the_dict_path(self):
        parsed = parse_pedigree_arrays(
            ids=np.array([0, 1, 2]),
            mother_ids=np.array([-1, -1, 0]),
            father_ids=np.array([-1, -1, 1]),
            sex=[0, 1, 0],
        )
        assert parsed.mother_rows.tolist() == [-1, -1, 0]
        assert parsed.sex.tolist() == [0, 1, 0]
        assert parsed.twin_ids.tolist() == [-1, -1, -1]

    def test_omitted_optionals_are_none(self):
        parsed = parse_pedigree_arrays(
            ids=np.array([0]),
            mother_ids=np.array([-1]),
            father_ids=np.array([-1]),
        )
        assert parsed.sex is None
        assert parsed.generation is None
        assert parsed.birth_year is None


_SETTINGS = settings(deadline=None, max_examples=50)


@_SETTINGS
@given(arrays=pedigree_arrays())
def test_parse_round_trips_ids_and_rows(arrays):
    ids, mother, father, sex = arrays
    parsed = parse_pedigree_input({"id": ids, "mother": mother, "father": father, "sex": sex})
    assert parsed.ids.tolist() == ids.tolist()
    assert parsed.mother_ids.tolist() == mother.tolist()
    assert parsed.father_ids.tolist() == father.tolist()
    assert parsed.mother_rows.tolist() == mother.tolist()
    assert parsed.father_rows.tolist() == father.tolist()


@_SETTINGS
@given(arrays=pedigree_arrays())
def test_dict_construction_matches_from_arrays(arrays):
    ids, mother, father, sex = arrays
    from_dict = PedigreeGraph({"id": ids, "mother": mother, "father": father, "sex": sex})
    from_arrays = PedigreeGraph.from_arrays(ids=ids, mothers=mother, fathers=father, sex=sex)
    for attribute in ("mother", "father", "twin", "generation", "sex"):
        np.testing.assert_array_equal(getattr(from_dict, attribute), getattr(from_arrays, attribute))
