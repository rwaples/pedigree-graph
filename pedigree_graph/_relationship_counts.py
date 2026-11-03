"""Exact closest-category pair counts through the Rust row-streaming engine.

``PedigreeGraph.relationship_counts`` and ``PedigreeView.relationship_counts``
end here.  The engine (ADR 0010, as amended) classifies every pair one row at
a time, folds precedence, and counts; no pair list is ever built, so peak
memory is O(N).  The graph crosses the boundary as the ``BuiltPedigree`` its
constructor produced, whose columns the core borrows; nothing native persists
between calls.  A view crosses as its ``graph_rows``: classification still
runs through every row the engine walks, and a pair counts when both its rows
are selected (ADR 0006).  The core (``relationships::query``, issue #52)
decides the walk's depth, answers a call with nothing to walk, and chooses
whether the view runs on its compact pedigree.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from pedigree_graph import _native, _progress
from pedigree_graph._registry import RELATIONSHIPS
from pedigree_graph._threads import thread_budget
from pedigree_graph.relationships import RelationshipCountResult

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import numpy as np

    from pedigree_graph._core import PedigreeGraph
    from pedigree_graph._progress import NativeCallback, ProgressArg
    from pedigree_graph._selection import RelationshipSelection
    from pedigree_graph._view import PedigreeView


def relationship_counts(
    graph: PedigreeGraph, selection: RelationshipSelection, progress: ProgressArg = None
) -> RelationshipCountResult:
    """Count the pairs of every requested category over the whole graph."""
    return _count(graph, None, selection, _progress.resolve(progress, logger, "relationship_counts"))


def view_relationship_counts(
    view: PedigreeView, selection: RelationshipSelection, progress: ProgressArg = None
) -> RelationshipCountResult:
    """Count the pairs of every requested category with both rows in *view*."""
    watch = _progress.resolve(progress, logger, "relationship_counts")
    return _count(view._graph, view._graph_rows, selection, watch)


def _count(
    graph: PedigreeGraph,
    graph_rows: np.ndarray | None,
    selection: RelationshipSelection,
    watch: NativeCallback | None,
) -> RelationshipCountResult:
    """Count to the highest requested code's degree and keep the requested codes.

    A code's closest-category count depends only on the codes before it in
    registry order, so computing the rest of that degree changes nothing.
    The thread budget is committed on every call, an empty selection
    included, like every public operation of the package.
    """
    requested = selection.codes
    threads = thread_budget()
    # Log arguments are computed even when INFO is off; a call with nothing
    # to walk costs microseconds, so skip them unless they are shown.
    verbose = logger.isEnabledFor(logging.INFO)
    if verbose:
        logger.info(
            "relationship_counts: max_degree=%s, %d categories, threads=%d%s",
            selection.top_degree,
            len(requested),
            threads,
            ", view" if graph_rows is not None else "",
        )
    start = time.perf_counter()
    counted = _native.relationship_counts(
        graph._built,
        requested=list(selection.ordered),
        threads=threads,
        graph_rows=graph_rows,
        progress=watch,
        tick=_progress.TICK_S,
    )
    if verbose:
        logger.info("relationship_counts total: %.3fs", time.perf_counter() - start)
    values: dict[str, int | None] = dict.fromkeys(RELATIONSHIPS, None)
    values.update({code: counted[code] for code in requested})
    return RelationshipCountResult(values, requested, requested)
