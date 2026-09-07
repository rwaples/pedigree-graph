"""Ordered pedigree views and the opaque token that names one receiver.

A view is an explicit, ordered selection of a graph's rows (ADR 0006): view-space
row ``i`` is graph row ``graph_rows[i]``. Selection arguments are validated here,
at the one boundary a caller's ids or rows cross, so a constructed view's arrays
are owned, read-only, unique, and already known to be in range.

Every receiver carries its own :class:`CoordinateToken`: each graph gets one at
construction and each view gets a fresh one. The token is instance identity, not
value identity, so two views built by two ``view(...)`` calls over the same
selection are separate receivers and a result computed against one is never
silently accepted by the other.
"""

from __future__ import annotations

__all__ = ["CoordinateToken", "PedigreeView"]

from typing import TYPE_CHECKING, overload

import numpy as np

from pedigree_graph._errors import PedigreeValidationError
from pedigree_graph._input import (
    _INT32_MAX,
    _INT64_MAX,
    _coerce_row_selection,
    _coerce_selection,
    _duplicate_witness,
    _FieldSpec,
    _own,
)
from pedigree_graph._kinship_pairwise import view_pair_kinship
from pedigree_graph._pair_extractor import view_relationship_pairs
from pedigree_graph.relationships import RelationshipCountResult

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from pedigree_graph._core import PedigreeGraph
    from pedigree_graph.relationships import RelationshipPairBlock, RelationshipPairs

# Named for the keyword each one validates, so a shape or coercion failure names
# the argument the caller wrote.
_IDS = _FieldSpec("ids", True, 0, _INT64_MAX, np.int64)
_ROWS = _FieldSpec("rows", True, 0, _INT32_MAX, np.int32)


class CoordinateToken:
    """Opaque identity of one receiver's coordinate space.

    Tokens compare and hash by identity alone, so a token names exactly the one
    graph or view that made it.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        return "CoordinateToken()"


def _unknown_id(value: object, position: int, missing_count: int) -> PedigreeValidationError:
    return PedigreeValidationError(
        "unknown_view_id",
        f"id {value} at position {position} is not an id of this pedigree",
        id=value,
        position=position,
        missing_count=missing_count,
    )


def _row_out_of_range(value: object, position: int, n_individuals: int) -> PedigreeValidationError:
    return PedigreeValidationError(
        "view_row_out_of_range",
        f"row {value} at position {position} is outside the {n_individuals}-row pedigree",
        row=value,
        position=position,
        n_individuals=n_individuals,
    )


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


def _rows_from_ids(graph: PedigreeGraph, selection: object) -> np.ndarray:
    """Resolve an id selection to graph rows, preserving selection order.

    Checks run as shape, integer form, membership, then duplicates, so the
    first error a caller sees is about a single entry before it is about a pair.
    """
    try:
        ids = _coerce_selection(_IDS, selection)
    except PedigreeValidationError as err:
        if err.code != "value_out_of_range":
            raise
        position = err.fields["position"]
        assert isinstance(position, int)
        raise _unknown_id(err.fields["value"], position, 1) from None
    rows = graph._id_index.resolve(ids, np.int32)
    unresolved = rows < 0
    if unresolved.any():
        position = int(np.argmax(unresolved))
        raise _unknown_id(int(ids[position]), position, int(np.count_nonzero(unresolved)))
    _check_duplicate_rows(rows, graph.n_individuals, "duplicate_view_id", "id", ids)
    return rows


def _rows_from_rows(graph: PedigreeGraph, selection: object) -> np.ndarray:
    """Validate a row selection against the graph's row range, preserving order.

    Checks run as shape, integer form, range, then duplicates, matching the id
    path's single-entry-before-pair order.
    """
    n_individuals = graph.n_individuals
    rows = _coerce_row_selection(
        _ROWS,
        selection,
        n_individuals,
        lambda value, position: _row_out_of_range(value, position, n_individuals),
    )
    _check_duplicate_rows(rows, n_individuals, "duplicate_view_row", "row", rows)
    return rows


def _build_view(graph: PedigreeGraph, *, ids: object | None, rows: object | None) -> PedigreeView:
    """Validate exactly one selection keyword and return the view it names.

    Args:
        graph: The graph whose rows are being selected.
        ids: Ids to select, in view order.
        rows: Graph rows to select, in view order.

    Returns:
        The view over that selection.

    Raises:
        TypeError: When both *ids* and *rows* are given, or neither.
        PedigreeValidationError: ``invalid_shape`` or ``invalid_integer_value``
            for a malformed selection; ``unknown_view_id`` then
            ``duplicate_view_id`` for *ids*; ``view_row_out_of_range`` then
            ``duplicate_view_row`` for *rows*. A value too large for int64
            reads as unknown / out of range, never as a fifth code.
    """
    if (ids is None) == (rows is None):
        raise TypeError("view() takes exactly one of ids= or rows=")
    graph_rows = _rows_from_ids(graph, ids) if rows is None else _rows_from_rows(graph, rows)
    return PedigreeView(graph, _own(graph_rows, np.int32))


class PedigreeView:
    """An ordered selection of one graph's rows: the receiver for view-space queries.

    Build one with :meth:`pedigree_graph.PedigreeGraph.view`, the only caller of
    this constructor and the place every selection is validated. A view keeps its
    graph alive, so relationships stay resolvable through the full pedigree
    however few rows the view names.

    Args:
        graph: The graph the selection indexes.
        graph_rows: Validated, owned, read-only int32 graph rows in selection
            order.
    """

    __slots__ = ("_coordinate_token", "_graph", "_graph_rows", "_ids")

    def __init__(self, graph: PedigreeGraph, graph_rows: np.ndarray) -> None:
        self._graph = graph
        self._graph_rows = graph_rows
        self._ids = _own(graph.ids[graph_rows], np.int64)
        self._coordinate_token = CoordinateToken()

    @property
    def ids(self) -> np.ndarray:
        """Read-only int64 id per view row, in selection order."""
        return self._ids

    @property
    def graph_rows(self) -> np.ndarray:
        """Read-only int32 graph row per view row, in selection order."""
        return self._graph_rows

    @property
    def n_individuals(self) -> int:
        """Number of individuals the view selects."""
        return len(self._graph_rows)

    def __len__(self) -> int:
        return self.n_individuals

    def __repr__(self) -> str:
        return f"PedigreeView(n_individuals={self.n_individuals})"

    def _graph_to_view(self) -> np.ndarray:
        """Return the int32 view row of every graph row, ``-1`` where unselected."""
        table = np.full(self._graph.n_individuals, -1, dtype=np.int32)
        table[self._graph_rows] = np.arange(len(self._graph_rows), dtype=np.int32)
        return table

    def relationship_pairs(
        self,
        *,
        max_degree: int | None = None,
        categories: Iterable[str] | None = None,
    ) -> RelationshipPairs:
        """Return every relationship pair of the selected categories, in view rows.

        Pairs are classified through the full graph, so cousins whose parents
        and grandparents are unselected are still cousins here.  Only pairs
        with both endpoints in this view are reported, and each row is the
        member's position in the view (``0 <= row < len(view)``).  Selectors,
        closest-category precedence, and roles are those of
        :meth:`pedigree_graph.PedigreeGraph.relationship_pairs`; symmetric
        blocks store ``first < second`` in view rows, every block is sorted
        by the canonical unordered view-row key, and every block carries this
        view's own coordinate token.  A view of fewer than two rows returns
        all-empty blocks with the requested flags set.

        Args:
            max_degree: Select every category at or below this degree (0-5).
                Exclusive with *categories*.
            categories: Registry codes to select, any order.  Exclusive with
                *max_degree*.

        Returns:
            A :class:`~pedigree_graph.relationships.RelationshipPairs` over all
            23 codes, in view rows.

        Raises:
            TypeError: Both selectors, neither, or a bare ``str`` for
                *categories*.
            PedigreeValidationError: ``max_degree_out_of_range`` or
                ``unknown_relationship_category``.
        """
        return view_relationship_pairs(self, max_degree=max_degree, categories=categories)

    def relationship_counts(
        self,
        *,
        max_degree: int | None = None,
        categories: Iterable[str] | None = None,
    ) -> RelationshipCountResult:
        """Return the exact number of view-space pairs in each selected category.

        Same selectors as :meth:`relationship_pairs`; each count is the length
        of that call's block.

        Returns:
            A :class:`~pedigree_graph.relationships.RelationshipCountResult`
            over all 23 codes, ``None`` for unselected categories.
        """
        return RelationshipCountResult.from_pairs(self.relationship_pairs(max_degree=max_degree, categories=categories))

    @overload
    def pair_kinship(self, first: RelationshipPairs, /) -> Mapping[str, np.ndarray]: ...
    @overload
    def pair_kinship(self, first: RelationshipPairBlock, /) -> np.ndarray: ...
    @overload
    def pair_kinship(self, first: object, second: object, /) -> np.ndarray: ...
    def pair_kinship(self, first: object, second: object | None = None, /) -> np.ndarray | Mapping[str, np.ndarray]:
        """Return the pedigree-expected kinship of each requested pair, in view rows.

        The same three call forms, values, and guarantees as
        :meth:`pedigree_graph.PedigreeGraph.pair_kinship`.  Rows are view rows
        (``0 <= row < len(view)``) and a block or collection must carry this
        view's own token; the recurrence itself runs through the full graph,
        so unselected ancestors still count, and it starts from and extends the
        owning graph's retained memo, so views and graph share one closure.

        Returns:
            A read-only float32 array aligned to the input pairs, or for a
            collection an immutable mapping over all 23 codes.

        Raises:
            TypeError: As the graph method.
            PedigreeValidationError: As the graph method; ``n_individuals`` in
                ``pair_row_out_of_range`` is the view's size.
            ResourceError: As the graph method.
        """
        return view_pair_kinship(self, first, second)
