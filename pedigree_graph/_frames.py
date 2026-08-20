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

_REQUIRED_COLUMNS: tuple[str, ...] = ("id", "mother", "father", "twin", "sex", "generation")
_OPTIONAL_COLUMNS: tuple[str, ...] = ("birth_year",)


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


def _column_to_numpy(column: _SupportsToNumpy) -> np.ndarray:
    """Extract one frame column as a numpy array via the structural protocol.

    Pandas nullable extension dtypes (e.g. ``Int64``) come out of
    ``.to_numpy()`` as object arrays; re-materialize those so NA-free
    nullable-integer columns land as ordinary numeric arrays. Actual missing
    values still fail downstream validation, as they should.
    """
    arr = column.to_numpy()
    if arr.dtype == object:
        arr = np.asarray(arr.tolist())
    return arr


def _coerce_to_array_dict(data: dict[str, np.ndarray] | FrameLike) -> dict[str, np.ndarray]:
    """Normalize input to a dict of numpy arrays.

    Accepts either a ``dict[str, np.ndarray]`` (returned as-is) or any
    :class:`FrameLike` table. Columns are extracted via ``.to_numpy()``; a
    missing required column is left absent here so
    ``_validate_required_columns`` reports it uniformly.
    """
    if isinstance(data, dict):
        return data
    columns = set(data.columns)
    result = {col: _column_to_numpy(data[col]) for col in _REQUIRED_COLUMNS if col in columns}
    for col in _OPTIONAL_COLUMNS:
        if col in columns:
            result[col] = _column_to_numpy(data[col])
    return result
