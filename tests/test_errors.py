"""Contract tests for the structured error classes (ADR 0006)."""

import pickle

import numpy as np
import pytest

from pedigree_graph import MissingMetadataError, PedigreeValidationError, ResourceError
from pedigree_graph._errors import METADATA_CODES, RESOURCE_CODES, VALIDATION_CODES

_CLASSES = [
    (PedigreeValidationError, VALIDATION_CODES),
    (MissingMetadataError, METADATA_CODES),
    (ResourceError, RESOURCE_CODES),
]

_ALL_CODES = [(cls, registry, code) for cls, registry in _CLASSES for code in registry]


def _sample_fields(required):
    return {name: index for index, name in enumerate(required)}


def test_registry_sizes_match_the_contract_table():
    assert len(VALIDATION_CODES) == 22
    assert len(METADATA_CODES) == 5
    assert len(RESOURCE_CODES) == 6


@pytest.mark.parametrize(("cls", "registry", "code"), _ALL_CODES, ids=[c for _, _, c in _ALL_CODES])
def test_every_code_constructs_with_its_required_fields(cls, registry, code):
    required = registry[code]
    exc = cls(code, "prose", **_sample_fields(required))
    assert exc.code == code
    assert str(exc) == "prose"
    assert set(exc.fields) == set(required)
    assert dict(exc.fields) == _sample_fields(required)


@pytest.mark.parametrize(("cls", "registry", "code"), _ALL_CODES, ids=[c for _, _, c in _ALL_CODES])
def test_every_code_rejects_a_missing_required_field(cls, registry, code):
    required = registry[code]
    fields = _sample_fields(required)
    fields.pop(required[-1])
    with pytest.raises(TypeError, match=required[-1]):
        cls(code, "prose", **fields)


@pytest.mark.parametrize(("cls", "registry"), _CLASSES)
def test_unknown_code_is_a_type_error(cls, registry):
    assert "not_a_real_code" not in registry
    with pytest.raises(TypeError, match="not_a_real_code"):
        cls("not_a_real_code", "prose")


def test_code_from_another_registry_is_rejected():
    with pytest.raises(TypeError, match="pedigree_too_large"):
        PedigreeValidationError("pedigree_too_large", "prose", n_individuals=1, maximum=2)
    with pytest.raises(TypeError, match="missing_field"):
        ResourceError("missing_field", "prose", field="id")


def test_extra_fields_are_kept():
    exc = PedigreeValidationError("missing_field", "prose", field="mother", hint="see docs")
    assert exc.fields["hint"] == "see docs"


def test_fields_mapping_is_read_only():
    exc = PedigreeValidationError("missing_field", "prose", field="mother")
    with pytest.raises(TypeError):
        exc.fields["field"] = "father"
    with pytest.raises(TypeError):
        del exc.fields["field"]


def test_nested_sequences_are_frozen_to_tuples():
    exc = PedigreeValidationError("cycle", "prose", ids=[3, [4, 5], np.array([6, 7])])
    assert exc.fields["ids"] == (3, (4, 5), (6, 7))


def test_numpy_scalars_become_python_scalars():
    exc = PedigreeValidationError(
        "value_out_of_range",
        "prose",
        field="id",
        position=np.int32(2),
        value=np.int64(-1),
        minimum=np.int8(0),
        maximum=np.float32(1.5),
    )
    fields = exc.fields
    assert type(fields["position"]) is int
    assert type(fields["value"]) is int
    assert type(fields["minimum"]) is int
    assert type(fields["maximum"]) is float
    assert fields["value"] == -1


def test_unrepresentable_value_falls_back_to_repr():
    exc = PedigreeValidationError("invalid_integer_value", "prose", field="id", position=0, value={"a": 1})
    assert exc.fields["value"] == "{'a': 1}"


def test_none_and_bool_survive_unchanged():
    exc = PedigreeValidationError("invalid_integer_value", "prose", field="sex", position=0, value=None, flag=True)
    assert exc.fields["value"] is None
    assert exc.fields["flag"] is True


@pytest.mark.parametrize(("cls", "registry", "code"), _ALL_CODES, ids=[c for _, _, c in _ALL_CODES])
def test_pickle_round_trip(cls, registry, code):
    exc = cls(code, "prose message", **_sample_fields(registry[code]), extra=("a", "b"))
    restored = pickle.loads(pickle.dumps(exc))
    assert type(restored) is cls
    assert restored.code == exc.code
    assert str(restored) == "prose message"
    assert dict(restored.fields) == dict(exc.fields)


def test_isinstance_hierarchy():
    validation = PedigreeValidationError("missing_field", "prose", field="id")
    metadata = MissingMetadataError("missing_sex", "prose", operation="ne_sex_ratio", status="absent", missing_count=3)
    resource = ResourceError("arithmetic_overflow", "prose", operation="compute_n_descendants", dtype="int32")
    assert isinstance(validation, ValueError)
    assert isinstance(metadata, ValueError)
    assert isinstance(resource, RuntimeError)
    assert not isinstance(resource, ValueError)
    assert not isinstance(validation, MissingMetadataError)
