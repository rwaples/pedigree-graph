"""Experimental APIs — free to break, change, or be removed in any minor release.

The current contents:

- :func:`count_pairs_bfs` — BFS / boolean-matmul / numba relationship pair
  counter (implemented in :mod:`pedigree_graph._bfs_engine`). Counts-only
  API. Differs from the matrix engine (:meth:`PedigreeGraph.count_pairs`)
  on inbred pedigrees: BFS counts *distinct* shared ancestors at depth ≥ 2
  while the matrix engine counts *paths* (multiplicity). Identical on
  non-inbred pedigrees; the codes that may diverge are exactly
  :func:`pedigree_graph._registry.bfs_divergent_codes`.

This submodule is **not** re-exported from the top-level package — callers
must explicitly ``from pedigree_graph.experimental import count_pairs_bfs``.
First call emits a :class:`FutureWarning` to signal the breaking-API
contract.
"""

from __future__ import annotations

__all__ = ["count_pairs_bfs"]

from pedigree_graph._bfs_engine import count_pairs_bfs
