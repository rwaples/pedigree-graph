"""Validated, owned pedigree input: the one boundary every constructor crosses.

Every guard on caller data lives here (ADR 0006). The parser turns a dict or
any :class:`~pedigree_graph._frames.FrameLike` table into a
:class:`PedigreeInput` of owned, read-only arrays, and every failure is a
structured :class:`~pedigree_graph._errors.PedigreeValidationError` or
:class:`~pedigree_graph._errors.ResourceError` from the 0.8 code table. Past
this boundary the engine trusts its arrays: dtypes, ranges, uniqueness, and
row mapping are settled.

Missing and unresolved references are distinct and need no third sentinel.
A relation is missing when its id is ``-1``; it is unresolved (external to
the represented rows) when its id is ``>= 0`` and its row is ``-1``.
"""

from __future__ import annotations

__all__ = [
    "PedigreeInput",
    "parse_pedigree_arrays",
    "parse_pedigree_input",
    "validate_id_field",
]

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np

from pedigree_graph import _native
from pedigree_graph._errors import PedigreeValidationError, ResourceError
from pedigree_graph._frames import _coerce_to_array_dict

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from pedigree_graph._frames import FrameLike

_INT64_MAX = int(np.iinfo(np.int64).max)
_INT64_MIN = int(np.iinfo(np.int64).min)
_INT32_MAX = int(np.iinfo(np.int32).max)

# Row coordinates are int32, so the row count is the hard capacity limit.
_MAX_ROWS = _INT32_MAX


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


@dataclass(frozen=True, slots=True)
class _SexEncoding:
    """Accepted raw sex range, the range reported on failure, and the mapping to store."""

    accepted_minimum: int
    accepted_maximum: int
    reported_minimum: int
    reported_maximum: int
    mapping: Mapping[int, int] | None


_SEX_ENCODINGS: Mapping[str, _SexEncoding] = MappingProxyType(
    {
        "simace": _SexEncoding(-1, 1, -1, 1, None),
        "plink": _SexEncoding(-1, 2, 0, 2, MappingProxyType({-1: -1, 0: -1, 1: 1, 2: 0})),
    }
)


@dataclass(frozen=True, slots=True)
class PedigreeInput:
    """Validated pedigree input owned by the package.

    Every array is a contiguous, read-only copy, so mutating the caller's
    arrays afterwards cannot change a constructed graph.

    Attributes:
        ids: int64 row ids, unique and nonnegative.
        mother_ids: int64 mother ids, ``-1`` when missing. A nonnegative id
            whose ``mother_rows`` entry is ``-1`` is an unresolved external
            reference.
        father_ids: int64 father ids, as ``mother_ids``.
        twin_ids: int64 MZ co-twin ids, as ``mother_ids``. Always an array,
            all ``-1`` when the field was omitted.
        mother_rows: int32 mother row indices, ``-1`` when missing or external.
        father_rows: int32 father row indices, as ``mother_rows``.
        twin_rows: int32 co-twin row indices, as ``mother_rows``.
        sex: int8 ``0`` female / ``1`` male / ``-1`` unknown, or ``None`` when
            the field was omitted or wholly unknown.
        generation: int32 generation labels ``>= -1``, or ``None`` as ``sex``.
        birth_year: int32 birth years ``>= -1``, or ``None`` as ``sex``.
        rows_topological: ``True`` when every parent row precedes its child
            row, so a parents-before-children sweep may run on these rows
            directly.
    """

    ids: np.ndarray
    mother_ids: np.ndarray
    father_ids: np.ndarray
    twin_ids: np.ndarray
    mother_rows: np.ndarray
    father_rows: np.ndarray
    twin_rows: np.ndarray
    sex: np.ndarray | None
    generation: np.ndarray | None
    birth_year: np.ndarray | None
    rows_topological: bool

    @property
    def n_individuals(self) -> int:
        """Number of represented individuals (rows)."""
        return len(self.ids)


def _own(values: np.ndarray, dtype: type) -> np.ndarray:
    """Return a contiguous, read-only copy of *values* as *dtype*.

    The single point where the package takes ownership: everything the parser
    hands to :class:`PedigreeInput` passes through here, so no stored array can
    alias a caller's buffer whatever the coercion path did or did not copy.
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


def _check_range(name: str, values: np.ndarray, minimum: int, maximum: int, reported: tuple[int, int]) -> None:
    """Raise ``value_out_of_range`` for the first value outside ``[minimum, maximum]``."""
    if values.size == 0 or (int(values.min()) >= minimum and int(values.max()) <= maximum):
        return
    position = int(np.argmax((values < minimum) | (values > maximum)))
    raise _out_of_range(name, position, int(values[position]), reported[0], reported[1])


def _reject_null_ids(nulls: np.ndarray) -> None:
    """Host nulls are the missing sentinel everywhere except ``id``, which needs a value."""
    if nulls.any():
        raise _invalid_integer("id", int(np.argmax(nulls)), "null")


def validate_id_field(ids: object) -> np.ndarray:
    """Validate an id column on its own and return it as an owned int64 array.

    Applies the shape, lossless-coercion, range, and uniqueness rules the full
    parser applies to ``id``, for callers that validate an id list outside a
    construction (the legacy subsample entry point).

    Args:
        ids: Array-like of candidate ids.

    Returns:
        A contiguous, read-only int64 copy of *ids*.

    Raises:
        PedigreeValidationError: ``invalid_shape``, ``invalid_integer_value``,
            ``value_out_of_range``, or ``duplicate_id``.
    """
    arr = np.asarray(ids)
    _check_shape(_ID, arr)
    values, nulls = _coerce_to_int64(_ID, arr)
    _reject_null_ids(nulls)
    _check_range(_ID.name, values, _ID.minimum, _ID.maximum, (_ID.minimum, _ID.maximum))
    _check_duplicate_ids(values)
    return _own(values, np.int64)


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
    ``duplicate_id`` at construction and the ``duplicate_view_*`` codes.
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


def _check_duplicate_ids(ids: np.ndarray) -> None:
    """Raise ``duplicate_id`` naming the smallest repeated id and all its rows."""
    witness = _duplicate_witness(ids)
    if witness is None:
        return
    duplicated, rows, count = witness
    raise PedigreeValidationError(
        "duplicate_id",
        f"id {duplicated} appears at rows {rows}; {count} row(s) repeat an earlier id",
        id=duplicated,
        rows=rows,
        duplicate_count=count,
    )


def _check_same_parent(ids: np.ndarray, mother_ids: np.ndarray, father_ids: np.ndarray) -> None:
    """Raise ``same_parent_id`` for the first row naming one id in both parent roles."""
    same = (mother_ids == father_ids) & (mother_ids >= 0)
    if not same.any():
        return
    row = int(np.argmax(same))
    raise PedigreeValidationError(
        "same_parent_id",
        f"row {row} names id {int(mother_ids[row])} as both mother and father",
        row=row,
        child_id=int(ids[row]),
        parent_id=int(mother_ids[row]),
    )


@dataclass(frozen=True, slots=True)
class IdIndex:
    """Sorted id column plus the permutation that built it, for repeated id lookups.

    Built once per graph and memoised, so resolving a selection costs the
    selection's size, not a fresh sort of every id in the pedigree.
    """

    sorted_ids: np.ndarray
    order: np.ndarray

    @classmethod
    def build(cls, ids: np.ndarray) -> IdIndex:
        """Index unique *ids* (the parser validates uniqueness)."""
        ids = np.asarray(ids)
        order = np.argsort(ids, kind="stable")
        return cls(ids[order], order)

    def resolve(self, query_ids: np.ndarray, dtype: np.dtype | type = np.int32) -> np.ndarray:
        """Row per query id, ``-1`` for a negative id or one not in the index."""
        query = np.asarray(query_ids)
        out = np.full(len(query), -1, dtype=dtype)
        if len(self.sorted_ids) == 0 or len(query) == 0:
            return out
        sel = np.where(query >= 0)[0]
        if sel.size == 0:
            return out
        q = query[sel]
        pos = np.clip(np.searchsorted(self.sorted_ids, q), 0, len(self.sorted_ids) - 1)
        found = self.sorted_ids[pos] == q
        out[sel[found]] = self.order[pos[found]].astype(dtype, copy=False)
        return out


def _map_ids_to_rows(
    target_ids: np.ndarray,
    query_ids: np.ndarray,
    dtype: np.dtype | type = np.int32,
) -> np.ndarray:
    """Map each query id to its row position in *target_ids* via searchsorted.

    *target_ids* must be unique (the parser validates that). Negative query
    ids (the ``-1`` "no relation" sentinel) and ids absent from *target_ids*
    both map to ``-1``, so a partial pedigree whose parent or co-twin is not
    represented resolves to an unresolved external reference rather than an
    error.

    Uses ``searchsorted`` over the sorted target ids rather than a dense
    ``max(id)+1`` lookup table, so sparse / very large ids cost O(n log n)
    instead of allocating an array sized to the largest id. Callers that
    resolve against the same ids repeatedly keep an :class:`IdIndex` instead.

    Args:
        target_ids: Unique ids defining the row coordinates.
        query_ids: Ids to resolve.
        dtype: Integer dtype of the returned row array.

    Returns:
        Row indices aligned with *query_ids*, ``-1`` where unresolved.
    """
    return IdIndex.build(target_ids).resolve(query_ids, dtype)


def _check_mz_pairs(
    ids: np.ndarray,
    mother_ids: np.ndarray,
    father_ids: np.ndarray,
    twin_rows: np.ndarray,
    sex: np.ndarray | None,
) -> None:
    """Reject represented MZ references that break the ADR 0006 pair contract.

    A reference is represented when its ``twin_rows`` entry resolved; an
    external co-twin (``twin_ids >= 0``, ``twin_rows == -1``) forms no pair
    here and is not checked. Each kind of violation is swept over every row
    before the next kind runs, so a third row pointing into an otherwise valid
    pair is reported as non-reciprocal rather than compared for parents against
    a row that is not its partner. Reciprocity is what bounds a pair to two
    members. The symmetric checks read only the lower row of each pair, so one
    violation is reported once.

    Parents are compared by id, so co-twins naming the same unrepresented
    parent agree and co-twins naming different unrepresented parents do not.

    Args:
        ids: int64 row ids.
        mother_ids: int64 mother ids, ``-1`` when missing.
        father_ids: int64 father ids, as *mother_ids*.
        twin_rows: int32 co-twin rows, ``-1`` when missing or external.
        sex: sex codes in the stored ``0``/``1``/``-1`` encoding, or ``None``
            when the field was omitted.

    Raises:
        PedigreeValidationError: ``mz_self_reference``, ``mz_nonreciprocal``,
            ``mz_parent_mismatch``, or ``mz_sex_mismatch``.
    """
    rows = np.flatnonzero(twin_rows >= 0)
    if rows.size == 0:
        return
    partner = twin_rows[rows].astype(np.int64, copy=False)

    self_reference = partner == rows
    if self_reference.any():
        row = int(rows[int(np.argmax(self_reference))])
        raise PedigreeValidationError(
            "mz_self_reference",
            f"row {row} names itself as its MZ co-twin",
            row=row,
            id=int(ids[row]),
        )

    nonreciprocal = twin_rows[partner] != rows
    if nonreciprocal.any():
        first = int(np.argmax(nonreciprocal))
        row, twin_row = int(rows[first]), int(partner[first])
        raise PedigreeValidationError(
            "mz_nonreciprocal",
            f"the MZ reference at row {row} is not reciprocated by row {twin_row}",
            row=row,
            id=int(ids[row]),
            twin_id=int(ids[twin_row]),
        )

    lower = rows < partner
    rows, partner = rows[lower], partner[lower]

    mismatched = {
        "mother": mother_ids[partner] != mother_ids[rows],
        "father": father_ids[partner] != father_ids[rows],
    }
    parent_mismatch = mismatched["mother"] | mismatched["father"]
    if parent_mismatch.any():
        first = int(np.argmax(parent_mismatch))
        row, twin_row = int(rows[first]), int(partner[first])
        raise PedigreeValidationError(
            "mz_parent_mismatch",
            f"MZ co-twins at rows {row} and {twin_row} do not name the same parents",
            row=row,
            id=int(ids[row]),
            twin_id=int(ids[twin_row]),
            parent_roles=tuple(role for role, flags in mismatched.items() if flags[first]),
        )

    if sex is None:
        return
    both_known = (sex[rows] != -1) & (sex[partner] != -1)
    sex_mismatch = both_known & (sex[rows] != sex[partner])
    if sex_mismatch.any():
        first = int(np.argmax(sex_mismatch))
        row, twin_row = int(rows[first]), int(partner[first])
        raise PedigreeValidationError(
            "mz_sex_mismatch",
            f"MZ co-twins at rows {row} and {twin_row} have different known sexes",
            row=row,
            id=int(ids[row]),
            twin_id=int(ids[twin_row]),
            sex=int(sex[row]),
            twin_sex=int(sex[twin_row]),
        )


def _normalize_optional(values: np.ndarray | None, dtype: type) -> np.ndarray | None:
    """Own an optional metadata column, collapsing a wholly unknown one to ``None``."""
    if values is None or values.size == 0 or bool(np.all(values == -1)):
        return None
    return _own(values, dtype)


def _apply_sex_encoding(values: np.ndarray, encoding: _SexEncoding) -> np.ndarray:
    """Range-check raw sex values and map them onto the stored 0/1/-1 encoding."""
    _check_range(
        "sex",
        values,
        encoding.accepted_minimum,
        encoding.accepted_maximum,
        (encoding.reported_minimum, encoding.reported_maximum),
    )
    if encoding.mapping is None:
        return values
    offset = encoding.accepted_minimum
    table = np.array(
        [encoding.mapping[raw] for raw in range(offset, encoding.accepted_maximum + 1)],
        dtype=np.int64,
    )
    return table[values - offset]


def parse_pedigree_input(
    data: dict[str, np.ndarray] | FrameLike,
    *,
    sex_encoding: str = "simace",
) -> PedigreeInput:
    """Validate pedigree input and return owned, read-only arrays.

    Args:
        data: ``dict[str, array-like]`` or any :class:`FrameLike` table with
            required ``id``, ``mother``, ``father`` fields and optional
            ``twin``, ``sex``, ``generation``, ``birth_year`` fields. Other
            keys and columns are ignored.
        sex_encoding: ``"simace"`` (``0`` female, ``1`` male, ``-1`` unknown)
            or ``"plink"`` (``2`` female, ``1`` male, ``0`` unknown).

    Returns:
        The validated :class:`PedigreeInput`.

    Raises:
        ValueError: for an unknown *sex_encoding*, which is API misuse rather
            than a pedigree-data failure.
        PedigreeValidationError: for any invalid field, duplicate id, shared
            parent id, cyclic parent reference, or represented MZ reference
            that is self-directed, non-reciprocal, parent-mismatched, or
            sex-mismatched.
        ResourceError: ``pedigree_too_large`` when the row count exceeds the
            int32 row-coordinate capacity.
    """
    encoding = _SEX_ENCODINGS.get(sex_encoding)
    if encoding is None:
        raise ValueError(f"sex_encoding must be one of {sorted(_SEX_ENCODINGS)}, got {sex_encoding!r}")

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
    if n > _MAX_ROWS:
        raise ResourceError(
            "pedigree_too_large",
            f"pedigree has {n:,} rows, exceeding the int32 row-coordinate capacity",
            n_individuals=n,
            maximum=_MAX_ROWS,
        )
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
        values[spec.name] = coerced

    for spec in _FIELDS:
        if spec.name not in values:
            continue
        if spec.name == "sex":
            values["sex"] = _apply_sex_encoding(values["sex"], encoding)
            continue
        _check_range(spec.name, values[spec.name], spec.minimum, spec.maximum, (spec.minimum, spec.maximum))

    ids = values["id"]
    _check_duplicate_ids(ids)
    _check_same_parent(ids, values["mother"], values["father"])

    twin_ids = values.get("twin")
    if twin_ids is None:
        twin_ids = np.full(n, -1, dtype=np.int64)
    mother_rows = _map_ids_to_rows(ids, values["mother"], np.int32)
    father_rows = _map_ids_to_rows(ids, values["father"], np.int32)
    twin_rows = _map_ids_to_rows(ids, twin_ids, np.int32)
    rows_topological = _native.is_topological(mother_rows, father_rows)
    if not rows_topological:
        _native.validate_acyclic(ids, mother_rows, father_rows)
    _check_mz_pairs(ids, values["mother"], values["father"], twin_rows, values.get("sex"))

    return PedigreeInput(
        ids=_own(ids, np.int64),
        mother_ids=_own(values["mother"], np.int64),
        father_ids=_own(values["father"], np.int64),
        twin_ids=_own(twin_ids, np.int64),
        mother_rows=_own(mother_rows, np.int32),
        father_rows=_own(father_rows, np.int32),
        twin_rows=_own(twin_rows, np.int32),
        sex=_normalize_optional(values.get("sex"), np.int8),
        generation=_normalize_optional(values.get("generation"), np.int32),
        birth_year=_normalize_optional(values.get("birth_year"), np.int32),
        rows_topological=rows_topological,
    )


def parse_pedigree_arrays(
    *,
    ids: object,
    mother_ids: object,
    father_ids: object,
    twin_ids: object | None = None,
    sex: object | None = None,
    generation: object | None = None,
    birth_year: object | None = None,
    sex_encoding: str = "simace",
) -> PedigreeInput:
    """Validate pedigree input supplied as separate arrays.

    Args:
        ids: Row ids.
        mother_ids: Mother ids, ``-1`` or a host null when missing.
        father_ids: Father ids, as *mother_ids*.
        twin_ids: MZ co-twin ids, as *mother_ids*; omitted means no twins.
        sex: Sex codes in *sex_encoding*; omitted means wholly unknown.
        generation: Generation labels, ``-1`` when unknown.
        birth_year: Birth years, ``-1`` when unknown.
        sex_encoding: See :func:`parse_pedigree_input`.

    Returns:
        The validated :class:`PedigreeInput`.
    """
    data: dict[str, np.ndarray] = {
        "id": np.asarray(ids),
        "mother": np.asarray(mother_ids),
        "father": np.asarray(father_ids),
    }
    optional = {"twin": twin_ids, "sex": sex, "generation": generation, "birth_year": birth_year}
    for name, column in optional.items():
        if column is not None:
            data[name] = np.asarray(column)
    return parse_pedigree_input(data, sex_encoding=sex_encoding)
