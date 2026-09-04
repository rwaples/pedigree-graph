"""Structural frame protocol and input coercion for pedigree tables.

The package stays frame-library-neutral: pandas and polars DataFrames both
satisfy :class:`FrameLike` structurally, and neither library is imported at
runtime. ``dict[str, np.ndarray]`` input remains accepted everywhere as the
zero-dependency escape hatch.
"""

from __future__ import annotations

__all__ = ["FrameLike"]

from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable

_REQUIRED_COLUMNS: tuple[str, ...] = ("id", "mother", "father")
_OPTIONAL_COLUMNS: tuple[str, ...] = ("twin", "sex", "generation", "birth_year")


class _SupportsToNumpy(Protocol):
    def to_numpy(self) -> np.ndarray: ...


class FrameLike(Protocol):
    """Structural frame protocol accepted wherever a pedigree table is taken.

    Any column-addressable table satisfies it — pandas and polars DataFrames
    alike — while this package imports neither frame library at runtime. The
    contract is deliberately tiny: ``.columns`` (an iterable of column names),
    string ``__getitem__`` returning a column object, and ``.to_numpy()`` on
    that column.
    """

    @property
    def columns(self) -> Iterable[str]: ...

    def __getitem__(self, key: str) -> _SupportsToNumpy: ...


def _null_mask(column: object) -> np.ndarray | None:
    """Host-native null mask for one column, or ``None`` when the host has no nulls.

    Polars spells it ``is_null``, pandas ``isna``; both return a column that
    the same ``.to_numpy()`` protocol materializes. Neither library is
    imported to find out.
    """
    for name in ("is_null", "isna"):
        probe = getattr(column, name, None)
        if callable(probe):
            mask = probe()
            if not isinstance(mask, np.ndarray):
                mask = mask.to_numpy()
            return np.asarray(mask, dtype=bool)
    return None


def _fill_null_positions(arr: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    """Return *arr* as int64 with the missing sentinel at masked positions.

    ``None`` when the unmasked values are not losslessly integral, so the
    parser sees the raw column and reports which value failed.
    """
    known = arr[~mask]
    if known.dtype == object:
        known = np.asarray(known.tolist())
    if known.dtype.kind == "u":
        if known.size and int(known.max()) > np.iinfo(np.int64).max:
            return None
    elif known.dtype.kind == "f":
        if not np.all(np.isfinite(known)) or not np.all(known == np.floor(known)):
            return None
    elif known.dtype.kind != "i":
        return None
    out = np.full(len(arr), -1, dtype=np.int64)
    out[~mask] = known.astype(np.int64)
    return out


def _column_to_numpy(column: _SupportsToNumpy, *, fill_nulls: bool = True) -> np.ndarray:
    """Extract one frame column as a numpy array via the structural protocol.

    Host nulls survive the extraction as the ``-1`` missing sentinel, so a
    nullable pandas or polars column reaches the parser as ordinary integer
    data. The ``id`` column has no missing sentinel, so it is read with
    *fill_nulls* off and its nulls reach the parser as nulls. Pandas nullable
    extension dtypes (e.g. ``Int64``) come out of ``.to_numpy()`` as object
    arrays; re-materialize those so NA-free nullable-integer columns land as
    ordinary numeric arrays.
    """
    arr = column.to_numpy()
    mask = _null_mask(column) if fill_nulls else None
    if mask is not None and mask.any():
        filled = _fill_null_positions(arr, mask)
        return arr if filled is None else filled
    if arr.dtype == object:
        arr = np.asarray(arr.tolist())
    return arr


def _coerce_to_array_dict(data: dict[str, np.ndarray] | FrameLike) -> dict[str, np.ndarray]:
    """Normalize input to a dict of numpy arrays.

    Accepts either a ``dict[str, np.ndarray]`` (returned as-is) or any
    :class:`FrameLike` table. Columns are extracted via ``.to_numpy()``; a
    missing column is left absent here so the parser reports it uniformly.
    """
    if isinstance(data, dict):
        return data
    columns = set(data.columns)
    result = {col: _column_to_numpy(data[col], fill_nulls=col != "id") for col in _REQUIRED_COLUMNS if col in columns}
    for col in _OPTIONAL_COLUMNS:
        if col in columns:
            result[col] = _column_to_numpy(data[col])
    return result
