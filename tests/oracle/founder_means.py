"""The 0.9.3 NumPy per-cohort founder-contribution means, the oracle for the Rust sweep.

``_per_gen_founder_means`` as ``pedigree_graph._ne_founders`` held it through
0.9.3, with only its imports rewritten; ``tests/test_native_generations.py``
holds ``_native.founder_contribution_means`` to it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pedigree_graph._cohorts import ObservedCohorts
from pedigree_graph._ne_common import _checked_founder_matrix
from pedigree_graph._ne_founders import FounderContributionMeans, _founder_columns, _founder_idx

if TYPE_CHECKING:
    from pedigree_graph._core import PedigreeGraph


def _per_gen_founder_means(
    pg: PedigreeGraph,
    founder_idx: np.ndarray | None = None,
    cohorts: ObservedCohorts | None = None,
) -> FounderContributionMeans:
    """Per-cohort mean founder-genome contribution via adjoint propagation.

    Returns ``(m_g, founder_idx)`` where
    ``m_g[b, f_local] = mean_{i ∈ cohort b} c[i, founder_idx[f_local]]``,
    with ``c[i, f]`` the expected genome fraction of i inherited from
    founder genome f under the Mendelian recursion (a founder row
    contributes 1 to its own genome; every other row takes the mean of its
    two parents' rows).

    Computed by iterating the adjoint of the forward recursion.  For each
    target cohort, propagate the cohort uniform vector ``1_{cohort} / N_b``
    backward through child→parent edges one **structural depth** at a time,
    deepest rows first: at depth ``d``, scatter ``0.5 · u[child]`` from each
    child at that depth into its mother and father.  Depth, not the
    generation label, orders the sweep, so labels that merge depths or put
    a parent and child in one cohort only change the grouping, never the
    ancestry.  What remains on the founder rows is summed per genome.

    Time: O(N · k · depth).  Memory: O(N + n_genomes · k).

    Args:
        pg: Pedigree graph.
        founder_idx: Optional precomputed :func:`_founder_idx`.
        cohorts: Optional precomputed grouping; defaults to the graph's.

    Returns:
        ``(m_g, founder_idx)`` — ``m_g`` shape ``(k, n_genomes)`` float64.
    """
    if founder_idx is None:
        founder_idx = _founder_idx(pg)
    if cohorts is None:
        cohorts = ObservedCohorts.for_graph(pg, "ne_long_term_contributions")
    n_founders = int(founder_idx.shape[0])
    n = pg.n_individuals
    m_g = _checked_founder_matrix(cohorts.k, n_founders, "founder_means", np.float64, np.nan)
    if n_founders == 0 or cohorts.k == 0:
        return FounderContributionMeans(m_g, founder_idx)

    mother = np.asarray(pg.mother_rows)
    father = np.asarray(pg.father_rows)
    depth = np.asarray(pg.depth)
    columns = _founder_columns(pg, founder_idx)
    founders = np.flatnonzero(columns >= 0)
    founder_columns = columns[founders]
    d_max = int(depth.max())
    by_depth = [np.flatnonzero(depth == d) for d in range(d_max + 1)]

    for b, in_b in enumerate(cohorts.members()):
        u = np.zeros(n, dtype=np.float64)
        u[in_b] = 1.0 / in_b.shape[0]
        for d in range(int(depth[in_b].max()), 0, -1):
            child = by_depth[d]
            uc = 0.5 * u[child]
            m = mother[child]
            mask = m >= 0
            if mask.any():
                np.add.at(u, m[mask], uc[mask])  # perf: numba candidate
            f = father[child]
            mask = f >= 0
            if mask.any():
                np.add.at(u, f[mask], uc[mask])  # perf: numba candidate
            u[child] = 0.0
        m_g[b] = np.bincount(founder_columns, weights=u[founders], minlength=n_founders)

    return FounderContributionMeans(m_g, founder_idx)
