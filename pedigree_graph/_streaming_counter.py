"""Exact close-relative counts from parent arrays and sibling-group sums.

The scalar path computes only the six codes in ``estimate_exact_codes()``.
It needs no adjacency powers or relationship-pair lists. The public wrapper
owns the graph's result cache; the counting helper does not write it.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import numpy as np

from pedigree_graph._registry import RELATIONSHIPS, estimate_exact_codes
from pedigree_graph._threads import thread_budget
from pedigree_graph.relationships import RelationshipCountResult

if TYPE_CHECKING:
    from pedigree_graph._core import PedigreeGraph

logger = logging.getLogger(__name__)


def close_relative_counts(graph: PedigreeGraph) -> RelationshipCountResult:
    """Return the graph's cached exact counts for twins, parents and siblings."""
    thread_budget()
    if graph._close_relative_counts_cache is None:
        start = time.perf_counter()
        counts = _count_close_relatives(graph)
        requested = estimate_exact_codes()
        values = {code: counts[code] if code in requested else None for code in RELATIONSHIPS}
        graph._close_relative_counts_cache = RelationshipCountResult(values, requested, requested)
        logger.info("close_relative_counts total: %.3fs", time.perf_counter() - start)
    return graph._close_relative_counts_cache


def _count_close_relatives(pg: PedigreeGraph) -> dict[str, int]:
    """Count the six close categories without changing the graph's caches."""
    counts = dict.fromkeys(estimate_exact_codes(), 0)
    # Construction validates reciprocal MZ links, two rows per pair.
    counts["MZ"] = int(np.count_nonzero(pg.twin_rows >= 0)) // 2
    counts["MO"] = int(np.count_nonzero(pg.mother_rows >= 0))
    counts["FO"] = int(np.count_nonzero(pg.father_rows >= 0))

    # Group siblings by original parent IDs, including unrepresented parents.
    # As in the pair engine, MZ individuals are excluded from sibling groups.
    sm = pg.mother_ids
    sf = pg.father_ids
    nontwin = pg.twin_rows < 0
    nt = ((sm >= 0) | (sf >= 0)) & nontwin
    nt_m = sm[nt]
    nt_f = sf[nt]
    both = (nt_m >= 0) & (nt_f >= 0)
    if both.any():
        bk_m = nt_m[both]
        bk_f = nt_f[both]
        max_p = int(max(bk_m.max(), bk_f.max())) + 1
        family_key = bk_m.astype(np.int64) * max_p + bk_f.astype(np.int64)
        _, sizes = np.unique(family_key, return_counts=True)
        sizes = sizes.astype(np.int64)
        counts["FS"] = int(((sizes * (sizes - 1)) // 2).sum())

    # Group counts, never a bincount indexed by a potentially sparse parent ID.
    for code, parents in (("MHS", nt_m), ("PHS", nt_f)):
        _, sizes = np.unique(parents[parents >= 0], return_counts=True)
        sizes = sizes.astype(np.int64)
        counts[code] = int(((sizes * (sizes - 1)) // 2).sum()) - counts["FS"]

    counts["MHS"] -= _half_sibs_that_are_parent_offspring(pg.father_rows, sm, nontwin)
    counts["PHS"] -= _half_sibs_that_are_parent_offspring(pg.mother_rows, sf, nontwin)
    return counts


def _half_sibs_that_are_parent_offspring(
    other_parent: np.ndarray,
    shared_parent_id: np.ndarray,
    nontwin: np.ndarray,
) -> int:
    """Count half-sib pairs the precedence fold files as parent-offspring.

    A child and its father who have the same mother are MHS by the sibling
    formula and FO by the lineal one; the fold keeps FO.  *other_parent* is the
    parent row array on the lineal side (``father`` for MHS), *shared_parent_id*
    the original-ID array of the shared side (``_orig_mother``), matching the
    sibling formula's own grouping.  Such a pair can never be a full sib (the
    child's other parent would have to be the parent itself, a cycle).
    """
    child = np.flatnonzero((other_parent >= 0) & nontwin)
    parent = other_parent[child]
    shared = shared_parent_id[child]
    return int(np.count_nonzero(nontwin[parent] & (shared >= 0) & (shared_parent_id[parent] == shared)))
