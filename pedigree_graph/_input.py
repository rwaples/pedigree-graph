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

from collections import deque
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np

from pedigree_graph._errors import PedigreeValidationError, ResourceError
from pedigree_graph._frames import _coerce_to_array_dict
from pedigree_graph._kinship_depth import _check_topological

if TYPE_CHECKING:
    from collections.abc import Mapping

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


def _check_duplicate_ids(ids: np.ndarray) -> None:
    """Raise ``duplicate_id`` naming the smallest repeated id and all its rows."""
    if ids.size < 2:
        return
    ordered = np.sort(ids)
    repeats = ordered[1:] == ordered[:-1]
    if not repeats.any():
        return
    duplicated = int(ordered[int(np.argmax(repeats))])
    rows = tuple(int(row) for row in np.flatnonzero(ids == duplicated))
    count = int(np.count_nonzero(repeats))
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
    instead of allocating an array sized to the largest id.

    Args:
        target_ids: Unique ids defining the row coordinates.
        query_ids: Ids to resolve.
        dtype: Integer dtype of the returned row array.

    Returns:
        Row indices aligned with *query_ids*, ``-1`` where unresolved.
    """
    target_ids = np.asarray(target_ids)
    query = np.asarray(query_ids)
    out = np.full(len(query), -1, dtype=dtype)
    if len(target_ids) == 0 or len(query) == 0:
        return out
    order = np.argsort(target_ids, kind="stable")
    sorted_ids = target_ids[order]
    sel = np.where(query >= 0)[0]
    if sel.size == 0:
        return out
    q = query[sel]
    pos = np.clip(np.searchsorted(sorted_ids, q), 0, len(sorted_ids) - 1)
    found = sorted_ids[pos] == q
    out[sel[found]] = order[pos[found]].astype(dtype, copy=False)
    return out


def _remaining_after_kahn(mother_rows: np.ndarray, father_rows: np.ndarray, n: int) -> np.ndarray:
    """Return the mask of rows Kahn's algorithm could not peel, i.e. the cyclic core."""
    children = np.concatenate([np.arange(n, dtype=np.int64), np.arange(n, dtype=np.int64)])
    parents = np.concatenate([mother_rows, father_rows]).astype(np.int64)
    represented = parents >= 0
    parents = parents[represented]
    children = children[represented]
    indegree = np.bincount(children, minlength=n)
    order = np.argsort(parents, kind="stable")
    parents = parents[order]
    children = children[order]
    starts = np.searchsorted(parents, np.arange(n), side="left")
    ends = np.searchsorted(parents, np.arange(n), side="right")
    peeled = np.zeros(n, dtype=bool)
    queue = deque(int(row) for row in np.flatnonzero(indegree == 0))
    while queue:
        row = queue.popleft()
        peeled[row] = True
        for child in children[starts[row] : ends[row]]:
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(int(child))
    return ~peeled


def _check_cycle(ids: np.ndarray, mother_rows: np.ndarray, father_rows: np.ndarray, n: int) -> None:
    """Raise ``cycle`` with one deterministic witness when the parent edges are cyclic.

    The witness is chosen by id, not by row, so the same graph reports the
    same cycle whatever order its rows arrive in.
    """
    remaining = _remaining_after_kahn(mother_rows, father_rows, n)
    if not remaining.any():
        return
    candidates = np.flatnonzero(remaining)
    row = int(candidates[np.argmin(ids[candidates])])
    walk: list[int] = []
    visited: dict[int, int] = {}
    while row not in visited:
        visited[row] = len(walk)
        walk.append(row)
        parents = [int(p) for p in (mother_rows[row], father_rows[row]) if p >= 0 and remaining[p]]
        row = min(parents, key=lambda p: int(ids[p]))
    cycle = walk[visited[row] :]
    start = int(np.argmin([ids[r] for r in cycle]))
    witness = tuple(int(ids[r]) for r in cycle[start:] + cycle[:start])
    raise PedigreeValidationError(
        "cycle",
        f"parent references form a cycle through ids {witness}",
        ids=witness,
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
            parent id, or cyclic parent reference.
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
    rows_topological = bool(_check_topological(mother_rows, father_rows, n))
    if not rows_topological:
        _check_cycle(ids, mother_rows, father_rows, n)

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
