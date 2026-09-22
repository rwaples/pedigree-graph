"""``pair_kinship`` on graphs and views: query resolution around one native call.

Every value is the pinned float32 recurrence of ADR 0009, evaluated by the
Rust core (``pedigree_graph_core::kinship``) in graph space with structural
depth as the peel input: ``phi(a, a) = (1 + phi(m, f)) / 2`` with a missing
parent contributing 0 and an MZ co-twin taking the self formula; otherwise the
endpoint of greater depth, ties to the greater row, is peeled to its parents.
The core builds one memo per call, shares it across the whole query, and frees
it before returning; nothing is retained on the graph (ADR 0007), so a value
never depends on call history and two calls store the same bits.

What stays host-side is the receiver boundary: the three call forms and their
validation codes, the translation of view rows to graph rows, and splitting a
collection result back per registry code.  The readable statement of the
recurrence is the test oracle ``tests/oracle/pair_kinship.py``.
"""

from __future__ import annotations

__all__ = ["graph_pair_kinship", "view_pair_kinship"]

import os
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np

from pedigree_graph import _native
from pedigree_graph._errors import PedigreeValidationError
from pedigree_graph._input import _INT32_MAX, _coerce_row_selection, _FieldSpec, _own_native
from pedigree_graph._threads import thread_budget
from pedigree_graph._topology import readonly
from pedigree_graph.relationships import RelationshipPairBlock, RelationshipPairs

if TYPE_CHECKING:
    from collections.abc import Mapping

    from pedigree_graph._core import PedigreeGraph
    from pedigree_graph._view import CoordinateToken, PedigreeView

# The memo layout the core runs on, "rows" or "flat", while the slice 13
# bake-off measures the two; the benchmark harness sets the variable in each
# arm's subprocess.  Removed with the losing layout once 13a records the pick.
_MEMO_LAYOUT = os.environ.get("PEDIGREE_GRAPH_KINSHIP_LAYOUT", "rows")


_FIRST = _FieldSpec("first_rows", True, 0, _INT32_MAX, np.int32)
_SECOND = _FieldSpec("second_rows", True, 0, _INT32_MAX, np.int32)


@dataclass(frozen=True, slots=True)
class _PairQuery:
    """A validated pair request in receiver rows.

    Attributes:
        first: int32 first endpoints.
        second: int32 second endpoints, same length.
        block_lengths: How many pairs each registry code contributed, in
            registry order, when the request was a :class:`RelationshipPairs`,
            so the flat result can be split back; ``None`` for a flat request.
    """

    first: np.ndarray
    second: np.ndarray
    block_lengths: dict[str, int] | None


def _row_out_of_range(argument: str, row: object, position: int, n_individuals: int) -> PedigreeValidationError:
    return PedigreeValidationError(
        "pair_row_out_of_range",
        f"{argument} row {row} at position {position} is outside the {n_individuals}-row receiver",
        argument=argument,
        row=row,
        position=position,
        n_individuals=n_individuals,
    )


def _coerce_rows(spec: _FieldSpec, values: object, n_individuals: int) -> np.ndarray:
    """Return one endpoint argument as int32 receiver rows.

    The view's row selection and a pair endpoint share one rule
    (:func:`pedigree_graph._input._coerce_row_selection`), so an endpoint fails
    with ``invalid_shape``, ``invalid_integer_value``, or
    ``pair_row_out_of_range`` and nothing else, in that order.
    """
    rows = _coerce_row_selection(
        spec,
        values,
        n_individuals,
        lambda value, position: _row_out_of_range(spec.name, value, position, n_individuals),
    )
    return readonly(rows.astype(np.int32))


def _join_rows(blocks: list[np.ndarray]) -> np.ndarray:
    """Join the rows of every non-empty block, or return the empty array for none."""
    if not blocks:
        return np.zeros(0, dtype=np.int32)
    return readonly(np.concatenate(blocks))


def _check_token(block: RelationshipPairBlock, token: CoordinateToken, receiver_type: str) -> None:
    if block._coordinate_token is not token:
        raise PedigreeValidationError(
            "coordinate_space_mismatch",
            f"pair_kinship: this {block.code} block was not produced by this {receiver_type}",
            operation="pair_kinship",
            receiver_type=receiver_type,
            result_type=type(block).__name__,
        )


def _resolve_query(
    first: object,
    second: object | None,
    *,
    token: CoordinateToken,
    n_individuals: int,
    receiver_type: str,
) -> _PairQuery:
    """Turn the three call forms into one validated :class:`_PairQuery`.

    Raises:
        TypeError: A block or collection given with a second argument, or a
            first row array without a second.
        PedigreeValidationError: ``coordinate_space_mismatch`` for a block from
            another receiver; ``invalid_shape``, ``invalid_integer_value``, or
            ``pair_row_out_of_range`` per row argument; then
            ``pair_length_mismatch``.
    """
    if isinstance(first, RelationshipPairs):
        if second is not None:
            raise TypeError("pair_kinship(pairs) takes no second argument")
        block_lengths: dict[str, int] = {}
        firsts: list[np.ndarray] = []
        seconds: list[np.ndarray] = []
        for code, block in first.items():
            _check_token(block, token, receiver_type)
            block_lengths[code] = len(block)
            if len(block):
                firsts.append(block.first_rows)
                seconds.append(block.second_rows)
        return _PairQuery(_join_rows(firsts), _join_rows(seconds), block_lengths)
    if isinstance(first, RelationshipPairBlock):
        if second is not None:
            raise TypeError("pair_kinship(block) takes no second argument")
        _check_token(first, token, receiver_type)
        return _PairQuery(first.first_rows, first.second_rows, None)
    if second is None:
        raise TypeError("pair_kinship(first_rows, second_rows) needs both row arrays")
    first_rows = _coerce_rows(_FIRST, first, n_individuals)
    second_rows = _coerce_rows(_SECOND, second, n_individuals)
    if first_rows.shape[0] != second_rows.shape[0]:
        raise PedigreeValidationError(
            "pair_length_mismatch",
            f"first_rows has {first_rows.shape[0]} entries but second_rows has {second_rows.shape[0]}",
            first_length=first_rows.shape[0],
            second_length=second_rows.shape[0],
        )
    return _PairQuery(first_rows, second_rows, None)


def _shape_result(query: _PairQuery, values: np.ndarray) -> np.ndarray | Mapping[str, np.ndarray]:
    """Return *values* flat, or split back per registry code for a collection query."""
    if query.block_lengths is None:
        return values
    offsets = np.cumsum(list(query.block_lengths.values()))[:-1]
    pieces = np.split(values, offsets)
    return MappingProxyType(dict(zip(query.block_lengths, pieces, strict=True)))


def _evaluate(graph: PedigreeGraph, first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """One native walk over graph-row endpoints; the result is owned and frozen without a copy."""
    thread_budget()
    values = _native.pair_kinship(
        graph._built,
        graph.depth,
        np.ascontiguousarray(first, dtype=np.int32),
        np.ascontiguousarray(second, dtype=np.int32),
        layout=_MEMO_LAYOUT,
    )
    return _own_native(values, np.float32)


def graph_pair_kinship(
    graph: PedigreeGraph,
    first: object,
    second: object | None,
) -> np.ndarray | Mapping[str, np.ndarray]:
    """Answer ``PedigreeGraph.pair_kinship`` in graph rows."""
    query = _resolve_query(
        first, second, token=graph._coordinate_token, n_individuals=graph.n_individuals, receiver_type="PedigreeGraph"
    )
    values = _evaluate(graph, query.first, query.second)
    return _shape_result(query, values)


def view_pair_kinship(
    view: PedigreeView,
    first: object,
    second: object | None,
) -> np.ndarray | Mapping[str, np.ndarray]:
    """Answer ``PedigreeView.pair_kinship``: validate in view rows, evaluate in graph rows."""
    query = _resolve_query(
        first, second, token=view._coordinate_token, n_individuals=view.n_individuals, receiver_type="PedigreeView"
    )
    rows = view.graph_rows
    values = _evaluate(view._graph, readonly(rows[query.first]), readonly(rows[query.second]))
    return _shape_result(query, values)
