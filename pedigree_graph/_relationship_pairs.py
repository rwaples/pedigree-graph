"""Relationship pairs through the Rust row-streaming engine (ADR 0006 and 0010, as amended).

``PedigreeGraph.relationship_pairs`` and ``PedigreeView.relationship_pairs``
end here.  The selector is parsed at the public boundary; this module hands
the graph's own ``BuiltPedigree`` to ``_native.relationship_pairs`` at the
selection's top degree, on the package pool, and wraps the int32 arrays the
core moved out, without copying them, as the owned read-only blocks of a
:class:`~pedigree_graph.relationships.RelationshipPairs`.  A view crosses as
the int32 view row of every graph row, ``-1`` where unselected; the core
keeps pairs with both rows selected, relabels, re-canonicalises symmetric
blocks, and sorts by the view-space key.

``execution`` selects resource use only: ``"speed"`` buffers every task's
pairs then copies once, ``"memory"`` counts first and fills exact blocks in
place.  Both return element-for-element identical results.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pedigree_graph import _native
from pedigree_graph._input import _own_native
from pedigree_graph._registry import RELATIONSHIPS
from pedigree_graph._threads import thread_budget
from pedigree_graph.relationships import RelationshipPairBlock, RelationshipPairs

if TYPE_CHECKING:
    from pedigree_graph._core import PedigreeGraph
    from pedigree_graph._selection import RelationshipSelection
    from pedigree_graph._view import CoordinateToken, PedigreeView

EXECUTIONS = ("speed", "memory")


def check_execution(execution: str) -> str:
    """Return *execution* once it is one of :data:`EXECUTIONS`; raise ``ValueError`` otherwise."""
    if execution not in EXECUTIONS:
        raise ValueError(f"execution must be one of {EXECUTIONS}, got {execution!r}")
    return execution


def relationship_pairs(graph: PedigreeGraph, selection: RelationshipSelection, execution: str) -> RelationshipPairs:
    """The :class:`RelationshipPairs` of *graph* for a parsed *selection*, in graph rows."""
    return _build_result(_native_blocks(graph, selection, None, execution), selection, graph._coordinate_token)


def view_relationship_pairs(view: PedigreeView, selection: RelationshipSelection, execution: str) -> RelationshipPairs:
    """The :class:`RelationshipPairs` of *view* for a parsed *selection*, in view rows.

    A view of fewer than two rows has no pairs and skips the engine, after
    committing the thread budget like every other call.
    """
    if len(view) < 2:
        thread_budget()
        return _build_result({}, selection, view._coordinate_token)
    blocks = _native_blocks(view._graph, selection, view._graph_to_view(), execution)
    return _build_result(blocks, selection, view._coordinate_token)


def _native_blocks(
    graph: PedigreeGraph, selection: RelationshipSelection, view_rows: np.ndarray | None, execution: str
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """The requested blocks from the core, or nothing when the selection is empty.

    The thread budget is committed on every call, an empty selection
    included, like every public operation of the package.
    """
    threads = thread_budget()
    if selection.top_degree is None:
        return {}
    return _native.relationship_pairs(
        graph._built,
        max_degree=selection.top_degree,
        requested=list(selection.ordered),
        threads=threads,
        execution=execution,
        view_rows=view_rows,
    )


def _build_result(
    blocks: dict[str, tuple[np.ndarray, np.ndarray]], selection: RelationshipSelection, token: CoordinateToken
) -> RelationshipPairs:
    """Wrap every registry code as an owned block carrying *token*; codes absent from *blocks* are empty."""
    empty = np.array([], dtype=np.int32)
    result = {}
    for code, category in RELATIONSHIPS.items():
        first, second = blocks.get(code, (empty, empty))
        result[code] = RelationshipPairBlock(
            category,
            _own_native(first, np.int32),
            _own_native(second, np.int32),
            code in selection.codes,
            token,
        )
    return RelationshipPairs(result)
