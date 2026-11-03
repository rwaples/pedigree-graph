"""``pair_kinship`` on graphs and views: query resolution around one native call.

Every value is the pinned float32 recurrence of ADR 0009, evaluated by the
Rust core (``pedigree_graph_core::kinship``) in graph space with structural
depth as the peel input: ``phi(a, a) = (1 + phi(m, f)) / 2`` with a missing
parent contributing 0 and an MZ co-twin taking the self formula; otherwise the
endpoint of greater depth, ties to the greater row, is peeled to its parents.
The core builds one memo per pool worker, shares it across the pairs that
worker takes, and frees it before returning; nothing is retained on the graph
(ADR 0007, 0016), so a value never depends on call history or thread budget and
two calls store the same bits.

What stays host-side is the receiver boundary: the three call forms, coercing
each row argument to int32, and splitting a collection result back per registry
code.  The core checks every endpoint against the receiver, graph or view, and
the two lengths, so ``pair_row_out_of_range`` and ``pair_length_mismatch`` read
alike in Python and R.  The readable statement of the
recurrence is the test oracle ``tests/oracle/pair_kinship.py``.
"""

from __future__ import annotations

__all__ = ["graph_pair_kinship", "view_pair_kinship"]

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np

from pedigree_graph import _native
from pedigree_graph._errors import PedigreeValidationError
from pedigree_graph._input import _INT32_MAX, _INT32_MIN, _coerce_selection, _FieldSpec, _own_native
from pedigree_graph._threads import thread_budget
from pedigree_graph._topology import readonly
from pedigree_graph.relationships import RelationshipPairBlock, RelationshipPairs

if TYPE_CHECKING:
    from collections.abc import Mapping

    from pedigree_graph._core import PedigreeGraph
    from pedigree_graph._view import CoordinateToken, PedigreeView

_FIRST = _FieldSpec("first_rows", True, _INT32_MIN, _INT32_MAX, np.int32)
_SECOND = _FieldSpec("second_rows", True, _INT32_MIN, _INT32_MAX, np.int32)


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
    """Return one endpoint argument as int32, for the core to range-check.

    Fails with ``invalid_shape`` or ``invalid_integer_value``, or with
    ``pair_row_out_of_range`` for an integer with no int32 form, which cannot
    reach the core and is outside every receiver.  That one is a coercion
    failure, so it is reported in this argument's turn, before the core's
    range checks.
    """
    try:
        rows = _coerce_selection(spec, values)
    except PedigreeValidationError as err:
        if err.code != "value_out_of_range":
            raise
        position = err.fields["position"]
        assert isinstance(position, int)
        raise _row_out_of_range(spec.name, err.fields["value"], position, n_individuals) from None
    outside = (rows < _INT32_MIN) | (rows > _INT32_MAX)
    if outside.any():
        position = int(np.argmax(outside))
        raise _row_out_of_range(spec.name, int(rows[position]), position, n_individuals)
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
            another receiver; ``invalid_shape`` or ``invalid_integer_value`` per
            row argument, or ``pair_row_out_of_range`` for a row with no int32
            form.  The core's checks (:func:`_evaluate`) follow.
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
    return _PairQuery(first_rows, second_rows, None)


def _shape_result(query: _PairQuery, values: np.ndarray) -> np.ndarray | Mapping[str, np.ndarray]:
    """Return *values* flat, or split back per registry code for a collection query."""
    if query.block_lengths is None:
        return values
    offsets = np.cumsum(list(query.block_lengths.values()))[:-1]
    pieces = np.split(values, offsets)
    return MappingProxyType(dict(zip(query.block_lengths, pieces, strict=True)))


def _evaluate(graph: PedigreeGraph, query: _PairQuery, graph_rows: np.ndarray | None = None) -> np.ndarray:
    """One native walk; the result is owned and frozen without a copy.

    Endpoints are graph rows, or view rows when *graph_rows* maps each view row
    to its graph row.  The core reports ``pair_row_out_of_range`` per endpoint
    argument, first then second, then ``pair_length_mismatch``, before the
    thread budget is committed.
    """
    n_receiver = graph.n_individuals if graph_rows is None else len(graph_rows)
    _native.check_pair_rows(n_receiver, query.first, query.second)
    values = _native.pair_kinship(
        graph._built,
        graph.depth,
        np.ascontiguousarray(query.first, dtype=np.int32),
        np.ascontiguousarray(query.second, dtype=np.int32),
        threads=thread_budget(),
        graph_rows=graph_rows,
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
    return _shape_result(query, _evaluate(graph, query))


def view_pair_kinship(
    view: PedigreeView,
    first: object,
    second: object | None,
) -> np.ndarray | Mapping[str, np.ndarray]:
    """Answer ``PedigreeView.pair_kinship``: validate in view rows, evaluate in graph rows."""
    query = _resolve_query(
        first, second, token=view._coordinate_token, n_individuals=view.n_individuals, receiver_type="PedigreeView"
    )
    return _shape_result(query, _evaluate(view._graph, query, view.graph_rows))
