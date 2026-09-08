"""Lineage counts and connected components over the whole graph (ADR 0006).

The bodies of :meth:`PedigreeGraph.distinct_ancestor_counts`,
:meth:`PedigreeGraph.descendant_path_counts`, and
:meth:`PedigreeGraph.connected_component_ids`.  Each is computed once, stored
on the graph, and handed back read-only.  The numba and scipy primitives live
in :mod:`pedigree_graph._lineage_kernel`; this module owns the coordinate
mapping, the memo, and the component labelling policy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from pedigree_graph._lineage_kernel import _compute_n_ancestors, _compute_n_descendants
from pedigree_graph._topology import readonly

if TYPE_CHECKING:
    from pedigree_graph._core import PedigreeGraph


def distinct_ancestor_counts(pg: PedigreeGraph) -> np.ndarray:
    """Distinct strict ancestors of every row, int32, read-only, memoised.

    An ancestor reachable through several paths is counted once.  The
    boolean transitive closure is order-free, so the graph arrays are used
    directly.
    """
    cached = pg._distinct_ancestor_counts
    if cached is None:
        cached = readonly(
            _compute_n_ancestors(pg.mother_rows, pg.father_rows, pg.n_individuals).astype(np.int32, copy=False)
        )
        pg._distinct_ancestor_counts = cached
    return cached


def descendant_path_counts(pg: PedigreeGraph) -> np.ndarray:
    """Descendant *paths* from every row, int64, read-only, memoised.

    ``counts[v]`` is the number of walks down the pedigree from ``v``: the
    number of children plus the path counts of the children.  Equal to the
    distinct descendant count in a pedigree without marriage loops; larger
    where a descendant reaches ``v`` through more than one child.  The sweep
    needs parents before children, so it runs in the private topological
    order when the input rows are not already ordered that way.
    """
    cached = pg._descendant_path_counts
    if cached is None:
        if pg._rows_are_topological:
            counts = _compute_n_descendants(pg.mother_rows, pg.father_rows, pg.n_individuals)
        else:
            m_idx, f_idx, _ = pg._topological_parents
            counts = pg._topology.per_row_to_graph(_compute_n_descendants(m_idx, f_idx, pg.n_individuals))
        cached = readonly(np.ascontiguousarray(counts, dtype=np.int64))
        pg._descendant_path_counts = cached
    return cached


def connected_component_ids(pg: PedigreeGraph) -> np.ndarray:
    """Smallest original ID in each row's parent-edge component, int64, read-only.

    Two rows share a value exactly when a chain of *represented* parent-child
    edges joins them.  A parent that is missing or external to the graph
    contributes no edge, so two rows naming the same external parent stay
    apart, and an MZ co-twin link is not an edge either.  The value is the
    minimum of :attr:`PedigreeGraph.ids` over the component, which is a
    property of the pedigree rather than of the row order.
    """
    cached = pg._connected_component_ids
    if cached is not None:
        return cached
    n = pg.n_individuals
    if n == 0:
        cached = readonly(np.empty(0, dtype=np.int64))
        pg._connected_component_ids = cached
        return cached
    child = np.concatenate([np.flatnonzero(pg.mother_rows >= 0), np.flatnonzero(pg.father_rows >= 0)])
    parent = np.concatenate([pg.mother_rows[pg.mother_rows >= 0], pg.father_rows[pg.father_rows >= 0]])
    edges = sp.csr_matrix(
        (np.ones(child.shape[0], dtype=np.int8), (child.astype(np.int64), parent.astype(np.int64))),
        shape=(n, n),
    )
    n_components, labels = connected_components(edges, directed=False)
    smallest = np.full(n_components, np.iinfo(np.int64).max, dtype=np.int64)
    np.minimum.at(smallest, labels, pg.ids)
    cached = readonly(smallest[labels])
    pg._connected_component_ids = cached
    return cached
