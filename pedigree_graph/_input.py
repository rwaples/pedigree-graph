"""Host coercion of pedigree input: the numpy side of the construction boundary.

Frame and array input reaches :func:`host_columns` in whatever dtypes the host
supplied.  Presence, shape, length, and the lossless integer form of every
field are settled here, because they are properties of numpy and pandas
representations rather than of pedigrees; every host null becomes the ``-1``
missing sentinel except in ``id``, which needs a value.  The resulting int64
columns cross into ``pedigree_graph._native.build_pedigree``, where every
pedigree-semantic rule lives (ADR 0006, ADR 0007).

The same coercers serve every row or id selection a caller hands to a
receiver — views, pair endpoints, and the effective-size reference
subpopulation — which is why they are shared here rather than owned by the
constructor.
"""

from __future__ import annotations

__all__ = ["HostColumns", "host_columns", "host_columns_from_arrays"]

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np

from pedigree_graph._errors import PedigreeValidationError
from pedigree_graph._frames import _coerce_to_array_dict

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from pedigree_graph._frames import FrameLike

_INT64_MAX = int(np.iinfo(np.int64).max)
_INT64_MIN = int(np.iinfo(np.int64).min)
_INT32_MAX = int(np.iinfo(np.int32).max)


@dataclass(frozen=True, slots=True)
class _FieldSpec:
    """One input field: whether it is required, its range, and its storage dtype."""

    name: str
    required: bool
    minimum: int
    maximum: int
    dtype: type


_FIELDS: tuple[_FieldSpec, ...] = (
    _FieldSpec("id", True, 0, _INT64_MAX, np.int64),
    _FieldSpec("mother", True, -1, _INT64_MAX, np.int64),
    _FieldSpec("father", True, -1, _INT64_MAX, np.int64),
    _FieldSpec("twin", False, -1, _INT64_MAX, np.int64),
    _FieldSpec("sex", False, -1, 1, np.int8),
    _FieldSpec("generation", False, -1, _INT32_MAX, np.int32),
    _FieldSpec("birth_year", False, -1, _INT32_MAX, np.int32),
)
_FIELDS_BY_NAME: Mapping[str, _FieldSpec] = MappingProxyType({spec.name: spec for spec in _FIELDS})
_ID = _FIELDS_BY_NAME["id"]


def _own(values: np.ndarray, dtype: type) -> np.ndarray:
    """Return a contiguous, read-only copy of *values* as *dtype*.

    The single point where the package takes ownership of a derived array
    (view rows, pair blocks), so no stored array can alias a caller's buffer
    whatever the coercion path did or did not copy.
    """
    out = np.array(values, dtype=dtype, copy=True, order="C")
    out.setflags(write=False)
    return out


def _invalid_integer(field: str, position: int, value: object) -> PedigreeValidationError:
    return PedigreeValidationError(
        "invalid_integer_value",
        f"{field!r} value at position {position} is not a lossless integer",
        field=field,
        position=position,
        value=value,
    )


def _out_of_range(field: str, position: int, value: object, minimum: int, maximum: int) -> PedigreeValidationError:
    return PedigreeValidationError(
        "value_out_of_range",
        f"{field!r} value at position {position} is outside [{minimum}, {maximum}]",
        field=field,
        position=position,
        value=value,
        minimum=minimum,
        maximum=maximum,
    )


def _coerce_object(spec: _FieldSpec, arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Coerce an object-dtype column element-wise (pandas nullable, mixed lists)."""
    out = np.empty(len(arr), dtype=np.int64)
    nulls = np.zeros(len(arr), dtype=bool)
    for position, item in enumerate(arr.tolist()):
        if item is None or type(item).__name__ == "NAType":
            nulls[position] = True
            continue
        if isinstance(item, bool | np.bool_):
            raise _invalid_integer(spec.name, position, bool(item))
        if isinstance(item, int | np.integer):
            as_int = int(item)
            if not _INT64_MIN <= as_int <= _INT64_MAX:
                raise _out_of_range(spec.name, position, as_int, spec.minimum, spec.maximum)
            out[position] = as_int
            continue
        if isinstance(item, float | np.floating):
            as_float = float(item)
            if np.isnan(as_float):
                nulls[position] = True
                continue
            if not np.isfinite(as_float) or as_float != int(as_float):
                raise _invalid_integer(spec.name, position, as_float)
            if not _INT64_MIN <= as_float <= _INT64_MAX:
                raise _out_of_range(spec.name, position, as_float, spec.minimum, spec.maximum)
            out[position] = int(as_float)
            continue
        raise _invalid_integer(spec.name, position, item)
    out[nulls] = -1
    return out, nulls


def _coerce_float(spec: _FieldSpec, arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Coerce a float column, treating NaN as a host null."""
    nulls = np.isnan(arr)
    known = ~nulls
    bad = known & (~np.isfinite(arr) | (arr != np.floor(arr)))
    if bad.any():
        position = int(np.argmax(bad))
        raise _invalid_integer(spec.name, position, float(arr[position]))
    outside = known & ((arr < float(_INT64_MIN)) | (arr >= -float(_INT64_MIN)))
    if outside.any():
        position = int(np.argmax(outside))
        raise _out_of_range(spec.name, position, float(arr[position]), spec.minimum, spec.maximum)
    out = np.where(nulls, -1.0, arr).astype(np.int64)
    return out, nulls


def _coerce_to_int64(spec: _FieldSpec, arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Losslessly coerce one field to int64, returning it with its host-null mask.

    Raises:
        PedigreeValidationError: ``invalid_integer_value`` for a value with no
            lossless integer form, ``value_out_of_range`` for one outside the
            int64 range.
    """
    if arr.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=bool)
    kind = arr.dtype.kind
    if kind == "b":
        raise _invalid_integer(spec.name, 0, bool(arr[0]))
    if kind in "iu":
        if arr.dtype == np.uint64:
            over = arr > np.uint64(_INT64_MAX)
            if over.any():
                position = int(np.argmax(over))
                raise _out_of_range(spec.name, position, int(arr[position]), spec.minimum, spec.maximum)
        return arr.astype(np.int64, copy=False), np.zeros(len(arr), dtype=bool)
    if kind == "f":
        return _coerce_float(spec, arr)
    if kind == "O":
        return _coerce_object(spec, arr)
    raise _invalid_integer(spec.name, 0, str(arr[0]))


def _reject_null_ids(nulls: np.ndarray) -> None:
    """Host nulls are the missing sentinel everywhere except ``id``, which needs a value."""
    if nulls.any():
        raise _invalid_integer("id", int(np.argmax(nulls)), "null")


def _check_shape(spec: _FieldSpec, arr: np.ndarray) -> None:
    if arr.ndim != 1:
        raise PedigreeValidationError(
            "invalid_shape",
            f"{spec.name!r} must be a 1-D column, got shape {arr.shape}",
            field=spec.name,
            expected_ndim=1,
            actual_shape=arr.shape,
        )


def _coerce_selection(spec: _FieldSpec, selection: object) -> np.ndarray:
    """Return one row or id selection argument as int64, rejecting bad shapes and host nulls.

    The shared front half of every selection a caller hands to a receiver: the
    view's ``ids=`` / ``rows=`` and the pair endpoints. A value with no lossless
    int64 form surfaces as ``value_out_of_range``; each caller translates that
    into its own not-in-this-receiver code, so a selection can only ever fail
    with that caller's codes plus the two shape/integer ones.

    Args:
        spec: The field being validated; names the argument in every error.
        selection: The caller's array-like.

    Returns:
        The selection as an int64 array, in the order given.

    Raises:
        PedigreeValidationError: ``invalid_shape``, ``invalid_integer_value``,
            or ``value_out_of_range``.
    """
    arr = np.asarray(selection)
    _check_shape(spec, arr)
    values, nulls = _coerce_to_int64(spec, arr)
    if nulls.any():
        raise _invalid_integer(spec.name, int(np.argmax(nulls)), "null")
    return values


def _coerce_row_selection(
    spec: _FieldSpec,
    selection: object,
    n_individuals: int,
    out_of_range: Callable[[object, int], PedigreeValidationError],
) -> np.ndarray:
    """Return one selection as int64 rows, every one inside ``[0, n_individuals)``.

    :func:`_coerce_selection` followed by the range check, with both the
    unrepresentable-value and the out-of-range case reported through
    *out_of_range* so one caller-chosen code covers "not a row of this
    receiver". Checks run as shape, integer form, then range, so a caller sees
    a single-entry failure before any whole-argument one.

    Args:
        spec: The field being validated; names the argument in every error.
        selection: The caller's array-like.
        n_individuals: Exclusive upper bound on a valid row.
        out_of_range: Builds the caller's error from ``(value, position)``.

    Returns:
        The rows as an int64 array, in the order given.

    Raises:
        PedigreeValidationError: ``invalid_shape``, ``invalid_integer_value``,
            or whatever *out_of_range* builds.
    """
    try:
        rows = _coerce_selection(spec, selection)
    except PedigreeValidationError as err:
        if err.code != "value_out_of_range":
            raise
        position = err.fields["position"]
        assert isinstance(position, int)
        raise out_of_range(err.fields["value"], position) from None
    outside = (rows < 0) | (rows >= n_individuals)
    if outside.any():
        position = int(np.argmax(outside))
        raise out_of_range(int(rows[position]), position)
    return rows


def _duplicate_witness(values: np.ndarray) -> tuple[int, tuple[int, ...], int] | None:
    """Name the smallest repeated value, every position it holds, and how many entries repeat.

    ``None`` when *values* are unique. The one witness rule shared by
    ``duplicate_id`` at construction and every ``duplicate_*`` selection code.
    """
    if values.size < 2:
        return None
    ordered = np.sort(values)
    repeats = ordered[1:] == ordered[:-1]
    if not repeats.any():
        return None
    duplicated = int(ordered[int(np.argmax(repeats))])
    positions = tuple(int(position) for position in np.flatnonzero(values == duplicated))
    return duplicated, positions, int(np.count_nonzero(repeats))


def _check_duplicate_rows(rows: np.ndarray, n_individuals: int, code: str, key: str, values: np.ndarray) -> None:
    """Raise *code* when in-range *rows* repeat, naming the smallest repeated entry of *values*.

    Uniqueness is an O(n) mark over the row range; the sort-based witness runs
    only once a repeat is known, so the failure path shares the constructor's
    ``duplicate_id`` rule while the success path never sorts.
    """
    seen = np.zeros(n_individuals, dtype=bool)
    seen[rows] = True
    if int(np.count_nonzero(seen)) == rows.size:
        return
    witness = _duplicate_witness(values)
    assert witness is not None
    duplicated, positions, count = witness
    raise PedigreeValidationError(
        code,
        f"{key} {duplicated} appears at positions {positions}; {count} selected {key}(s) repeat an earlier one",
        **{key: duplicated, "positions": positions, "duplicate_count": count},
    )


@dataclass(frozen=True, slots=True)
class HostColumns:
    """The int64 columns of one pedigree input, ready to cross into the core.

    Each array is contiguous int64 with host nulls already ``-1``; an optional
    column the host did not supply is ``None``.  Nothing about the pedigree
    has been checked yet: that is the core's job.
    """

    ids: np.ndarray
    mother: np.ndarray
    father: np.ndarray
    twin: np.ndarray | None
    sex: np.ndarray | None
    generation: np.ndarray | None
    birth_year: np.ndarray | None

    def as_args(
        self,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None,
    ]:
        """The columns in ``build_pedigree``'s positional order."""
        return (self.ids, self.mother, self.father, self.twin, self.sex, self.generation, self.birth_year)


def host_columns(data: dict[str, np.ndarray] | FrameLike) -> HostColumns:
    """Coerce pedigree input to int64 columns, rejecting what has no such form.

    Args:
        data: ``dict[str, array-like]`` or any :class:`FrameLike` table with
            required ``id``, ``mother``, ``father`` fields and optional
            ``twin``, ``sex``, ``generation``, ``birth_year`` fields. Other
            keys and columns are ignored.

    Returns:
        The coerced columns.

    Raises:
        PedigreeValidationError: ``missing_field``, ``invalid_shape``,
            ``length_mismatch``, ``invalid_integer_value``, or
            ``value_out_of_range`` for a value outside int64.
    """
    arrays = _coerce_to_array_dict(data)
    present = {spec.name: np.asarray(arrays[spec.name]) for spec in _FIELDS if spec.name in arrays}
    for spec in _FIELDS:
        if spec.required and spec.name not in present:
            raise PedigreeValidationError(
                "missing_field",
                f"input is missing the required {spec.name!r} field",
                field=spec.name,
            )

    _check_shape(_ID, present["id"])
    n = len(present["id"])
    for spec in _FIELDS:
        if spec.name == "id" or spec.name not in present:
            continue
        arr = present[spec.name]
        _check_shape(spec, arr)
        if len(arr) != n:
            raise PedigreeValidationError(
                "length_mismatch",
                f"{spec.name!r} has length {len(arr)}, expected {n} from the id field",
                field=spec.name,
                expected_length=n,
                actual_length=len(arr),
            )

    values: dict[str, np.ndarray] = {}
    for spec in _FIELDS:
        if spec.name not in present:
            continue
        coerced, nulls = _coerce_to_int64(spec, present[spec.name])
        if spec.name == "id":
            _reject_null_ids(nulls)
        values[spec.name] = np.ascontiguousarray(coerced, dtype=np.int64)

    return HostColumns(
        ids=values["id"],
        mother=values["mother"],
        father=values["father"],
        twin=values.get("twin"),
        sex=values.get("sex"),
        generation=values.get("generation"),
        birth_year=values.get("birth_year"),
    )


def host_columns_from_arrays(
    *,
    ids: object,
    mother_ids: object,
    father_ids: object,
    twin_ids: object | None = None,
    sex: object | None = None,
    generation: object | None = None,
    birth_year: object | None = None,
) -> HostColumns:
    """:func:`host_columns` over separate array-likes; an omitted column is absent."""
    data: dict[str, np.ndarray] = {
        "id": np.asarray(ids),
        "mother": np.asarray(mother_ids),
        "father": np.asarray(father_ids),
    }
    optional = {"twin": twin_ids, "sex": sex, "generation": generation, "birth_year": birth_year}
    for name, column in optional.items():
        if column is not None:
            data[name] = np.asarray(column)
    return host_columns(data)
