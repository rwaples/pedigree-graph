"""Structured exception classes for the 0.8 public surface (ADR 0006).

Three classes carry a stable ``.code`` and an immutable ``.fields`` mapping.
Messages are prose and are not part of the contract; tests and consumers
branch on the code and read the fields.

The code tables below are the contract. Each registry maps a code to the
field names that code must carry, so a raise site cannot invent a code or
omit a field: both are ``TypeError`` at construction, a programming error
rather than a user-facing failure.
"""

from __future__ import annotations

__all__ = [
    "METADATA_CODES",
    "RESOURCE_CODES",
    "VALIDATION_CODES",
    "MissingMetadataError",
    "PedigreeValidationError",
    "ResourceError",
]

from types import MappingProxyType
from typing import TYPE_CHECKING, ClassVar

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping

VALIDATION_CODES: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "missing_field": ("field",),
        "invalid_shape": ("field", "expected_ndim", "actual_shape"),
        "length_mismatch": ("field", "expected_length", "actual_length"),
        "invalid_integer_value": ("field", "position", "value"),
        "value_out_of_range": ("field", "position", "value", "minimum", "maximum"),
        "duplicate_id": ("id", "rows", "duplicate_count"),
        "same_parent_id": ("row", "child_id", "parent_id"),
        "cycle": ("ids",),
        "mz_self_reference": ("row", "id"),
        "mz_nonreciprocal": ("row", "id", "twin_id"),
        "mz_parent_mismatch": ("row", "id", "twin_id", "parent_roles"),
        "mz_sex_mismatch": ("row", "id", "twin_id", "sex", "twin_sex"),
        "birth_year_topology": (
            "parent_role",
            "child_row",
            "parent_row",
            "child_id",
            "parent_id",
            "child_birth_year",
            "parent_birth_year",
            "violation_count",
        ),
        "duplicate_view_id": ("id", "positions", "duplicate_count"),
        "unknown_view_id": ("id", "position", "missing_count"),
        "duplicate_view_row": ("row", "positions", "duplicate_count"),
        "view_row_out_of_range": ("row", "position", "n_individuals"),
        "pair_length_mismatch": ("first_length", "second_length"),
        "pair_row_out_of_range": ("argument", "row", "position", "n_individuals"),
        "unknown_relationship_category": ("codes",),
        "max_degree_out_of_range": ("value", "minimum", "maximum"),
        "coordinate_space_mismatch": ("operation", "receiver_type", "result_type"),
    }
)

METADATA_CODES: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "missing_generation_labels": ("operation", "status", "missing_count"),
        "missing_sex": ("operation", "status", "missing_count"),
        "missing_birth_year": ("operation", "status", "missing_count"),
        "insufficient_parent_age_data": ("operation", "missing_parent_roles"),
    }
)

RESOURCE_CODES: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "pedigree_too_large": ("n_individuals", "maximum"),
        "pair_key_overflow": ("n_individuals", "maximum"),
        "csc_index_overflow": ("nnz", "maximum"),
        "memo_capacity_exceeded": ("operation", "capacity", "maximum"),
        "arithmetic_overflow": ("operation", "dtype"),
        "allocation_failed": ("operation", "requested_elements", "dtype"),
    }
)


def _freeze_field(value: object) -> object:
    """Return an immutable, picklable form of one field value."""
    if value is None or isinstance(value, int | float | str):
        return value
    if isinstance(value, np.generic):
        return _freeze_field(value.item())
    if isinstance(value, np.ndarray):
        return _freeze_field(value.tolist())
    if isinstance(value, tuple | list):
        return tuple(_freeze_field(item) for item in value)
    return repr(value)


def _rebuild(cls: type, code: str, message: str, fields: dict[str, object]) -> object:
    """Reconstruct a structured error during unpickling."""
    return cls(code, message, **fields)


class _StructuredError(Exception):
    """Base holding the code registry, the frozen fields, and pickle support."""

    _CODES: ClassVar[Mapping[str, tuple[str, ...]]] = MappingProxyType({})

    def __init__(self, code: str, message: str, /, **fields: object) -> None:
        required = self._CODES.get(code)
        if required is None:
            raise TypeError(f"{type(self).__name__} has no code {code!r}")
        missing = tuple(name for name in required if name not in fields)
        if missing:
            raise TypeError(f"code {code!r} requires field(s) {', '.join(missing)}")
        super().__init__(message)
        self.code = code
        self.fields: Mapping[str, object] = MappingProxyType({k: _freeze_field(v) for k, v in fields.items()})

    def __reduce__(self) -> tuple[object, ...]:
        return (_rebuild, (type(self), self.code, str(self), dict(self.fields)))


class PedigreeValidationError(_StructuredError, ValueError):
    """Invalid pedigree input, structure, or coordinate use.

    Attributes:
        code: One of the keys of :data:`VALIDATION_CODES`.
        fields: Immutable mapping carrying at least that code's required
            field names.
    """

    _CODES: ClassVar[Mapping[str, tuple[str, ...]]] = VALIDATION_CODES


class MissingMetadataError(_StructuredError, ValueError):
    """An analysis needs optional metadata the pedigree does not carry.

    Attributes:
        code: One of the keys of :data:`METADATA_CODES`.
        fields: Immutable mapping carrying at least that code's required
            field names.
    """

    _CODES: ClassVar[Mapping[str, tuple[str, ...]]] = METADATA_CODES


class ResourceError(_StructuredError, RuntimeError):
    """A capacity, allocation, or fixed-width arithmetic limit was reached.

    Attributes:
        code: One of the keys of :data:`RESOURCE_CODES`.
        fields: Immutable mapping carrying at least that code's required
            field names.
    """

    _CODES: ClassVar[Mapping[str, tuple[str, ...]]] = RESOURCE_CODES
