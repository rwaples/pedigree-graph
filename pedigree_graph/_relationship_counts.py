"""Exact closest-category pair counts through the Rust row-streaming engine.

``PedigreeGraph.relationship_counts`` and ``PedigreeView.relationship_counts``
end here.  The engine (ADR 0010, as amended) classifies every pair one row at
a time, folds precedence, and counts; no pair list is ever built, so peak
memory is O(N).  The graph's own columns cross the boundary borrowed, and
nothing native persists between calls.  A view is a boolean mask over graph
rows: classification still runs through the full graph, and a pair counts
when both its rows are selected (ADR 0006).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pedigree_graph import _native
from pedigree_graph._pair_extractor import _requested_codes
from pedigree_graph._registry import RELATIONSHIPS
from pedigree_graph._threads import thread_budget
from pedigree_graph.relationships import RelationshipCountResult

if TYPE_CHECKING:
    from collections.abc import Iterable

    from pedigree_graph._core import PedigreeGraph
    from pedigree_graph._view import PedigreeView


def relationship_counts(
    graph: PedigreeGraph,
    *,
    max_degree: int | None = None,
    categories: Iterable[str] | None = None,
) -> RelationshipCountResult:
    """Count the pairs of every requested category over the whole graph."""
    return _count(graph, None, _requested_codes(max_degree, categories))


def view_relationship_counts(
    view: PedigreeView,
    *,
    max_degree: int | None = None,
    categories: Iterable[str] | None = None,
) -> RelationshipCountResult:
    """Count the pairs of every requested category with both rows in *view*."""
    requested = _requested_codes(max_degree, categories)
    selected = np.zeros(view._graph.n_individuals, dtype=np.bool_)
    selected[view._graph_rows] = True
    return _count(view._graph, selected, requested)


def _count(graph: PedigreeGraph, selected: np.ndarray | None, requested: frozenset[str]) -> RelationshipCountResult:
    """Run the engine at the degree of the highest requested code and keep the requested codes.

    A code's closest-category count depends only on the codes before it in
    registry order, so computing the rest of that degree changes nothing.
    """
    values: dict[str, int | None] = dict.fromkeys(RELATIONSHIPS, None)
    if requested:
        top = max(RELATIONSHIPS[code].degree for code in requested)
        counted = _native.relationship_counts(
            graph.mother_rows,
            graph.father_rows,
            graph.twin_rows,
            graph.mother_ids,
            graph.father_ids,
            max_degree=top,
            threads=thread_budget(),
            selected=selected,
        )
        values.update(
            {code: count for code, count in zip(RELATIONSHIPS, counted.tolist(), strict=True) if code in requested}
        )
    return RelationshipCountResult(values, requested, requested, frozenset(), frozenset())
