"""Exact closest-category pair counts through the Rust row-streaming engine.

``PedigreeGraph.relationship_counts`` and ``PedigreeView.relationship_counts``
end here.  The engine (ADR 0010, as amended) classifies every pair one row at
a time, folds precedence, and counts; no pair list is ever built, so peak
memory is O(N).  The graph crosses the boundary as the ``BuiltPedigree`` its
constructor produced, whose columns the core borrows; nothing native persists
between calls.  A view is a boolean mask over graph rows: classification still
runs through the full graph, and a pair counts when both its rows are selected
(ADR 0006).
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import numpy as np

from pedigree_graph import _native
from pedigree_graph._registry import RELATIONSHIPS
from pedigree_graph._threads import thread_budget
from pedigree_graph.relationships import RelationshipCountResult

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pedigree_graph._core import PedigreeGraph
    from pedigree_graph._selection import RelationshipSelection
    from pedigree_graph._view import PedigreeView


def relationship_counts(graph: PedigreeGraph, selection: RelationshipSelection) -> RelationshipCountResult:
    """Count the pairs of every requested category over the whole graph."""
    return _count(graph, None, selection)


def view_relationship_counts(view: PedigreeView, selection: RelationshipSelection) -> RelationshipCountResult:
    """Count the pairs of every requested category with both rows in *view*."""
    selected = np.zeros(view._graph.n_individuals, dtype=np.bool_)
    selected[view._graph_rows] = True
    return _count(view._graph, selected, selection)


def _count(
    graph: PedigreeGraph, selected: np.ndarray | None, selection: RelationshipSelection
) -> RelationshipCountResult:
    """Run the engine at the degree of the highest requested code and keep the requested codes.

    A code's closest-category count depends only on the codes before it in
    registry order, so computing the rest of that degree changes nothing.
    """
    requested = selection.codes
    values: dict[str, int | None] = dict.fromkeys(RELATIONSHIPS, None)
    top = selection.top_degree
    if top is not None:
        threads = thread_budget()
        logger.info("relationship_counts: max_degree=%d, threads=%d", top, threads)
        start = time.perf_counter()
        counted = _native.relationship_counts(
            graph._built,
            max_degree=top,
            threads=threads,
            selected=selected,
        )
        logger.info("relationship_counts total: %.3fs", time.perf_counter() - start)
        values.update({code: counted[code] for code in requested})
    return RelationshipCountResult(values, requested, requested)
