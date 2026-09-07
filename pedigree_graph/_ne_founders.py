"""Founder-contribution Ne estimator and its primitives (PGQ-006).

Owns the represented-founder index, the adjoint per-cohort mean-contribution
propagation, and the Wray & Thompson 1990 long-term contribution estimator
(:func:`ne_long_term_contributions`) built on them.  ``_founder_idx`` and
``_founder_columns`` are also consumed by the Caballero-Toro engine.

A **represented founder** is a row with no represented mother or father,
whatever its generation label and whether its parents are missing or
external to the graph.  Contribution columns represent **represented
founder genomes** (ADR 0008): parentless MZ co-twins are two founder rows
sharing one column, so their descendants inherit one lineage.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from pedigree_graph._cohorts import ObservedCohorts
from pedigree_graph._ne_common import _checked_founder_matrix
from pedigree_graph._ne_results import NeLTCResult

if TYPE_CHECKING:
    from pedigree_graph._core import PedigreeGraph


class FounderContributionMeans(NamedTuple):
    """Per-cohort mean founder-genome contributions plus the founder index.

    ``m_g[b, f_local]`` is the mean over cohort-``b`` individuals of their
    expected genome fraction from founder genome ``founder_idx[f_local]``;
    ``founder_idx`` maps each column back to the canonical graph row of that
    genome.  Returned by :func:`_per_gen_founder_means` and consumed by
    :func:`_ltc_from`.
    """

    m_g: np.ndarray
    founder_idx: np.ndarray


def _founder_rows(pg: PedigreeGraph) -> np.ndarray:
    """Graph rows with no represented mother and no represented father."""
    return np.flatnonzero((np.asarray(pg.mother) < 0) & (np.asarray(pg.father) < 0)).astype(np.intp)


def _genome_of(pg: PedigreeGraph) -> np.ndarray:
    """Canonical genome-node row per graph row: itself, or the lower-indexed co-twin."""
    rows = np.arange(pg.n, dtype=np.intp)
    twin = np.asarray(pg.twin, dtype=np.intp)
    return np.where((twin >= 0) & (twin < rows), twin, rows)


def _founder_idx(pg: PedigreeGraph) -> np.ndarray:
    """Canonical rows of the represented founder genomes, ascending intp.

    One entry per genome: a parentless MZ pair contributes its lower row.
    """
    founders = _founder_rows(pg)
    return np.unique(_genome_of(pg)[founders]).astype(np.intp)


def _founder_columns(pg: PedigreeGraph, founder_idx: np.ndarray) -> np.ndarray:
    """Column of ``founder_idx`` each represented founder row seeds; ``-1`` elsewhere."""
    columns = np.full(pg.n, -1, dtype=np.int64)
    founders = _founder_rows(pg)
    if founders.shape[0]:
        columns[founders] = np.searchsorted(founder_idx, _genome_of(pg)[founders])
    return columns


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
    n = pg.n
    m_g = _checked_founder_matrix(cohorts.k, n_founders, "founder_means", np.float64, np.nan)
    if n_founders == 0 or cohorts.k == 0:
        return FounderContributionMeans(m_g, founder_idx)

    mother = np.asarray(pg.mother)
    father = np.asarray(pg.father)
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


def _ltc_from(cohorts: ObservedCohorts, means: FounderContributionMeans, tol: float) -> NeLTCResult:
    m_g, founder_idx = means
    if founder_idx.shape[0] == 0 or cohorts.k == 0:
        return NeLTCResult(
            ne=None,
            asymptote_reached=False,
            n_iterations=0,
            max_delta_final=float("nan"),
            sum_c_squared=0.0,
            final_generation=None,
        )

    asymptote_reached = False
    n_iterations = 0
    max_delta_final = float("nan")
    final = 0
    for b in range(1, cohorts.k):
        delta = float(np.max(np.abs(m_g[b] - m_g[b - 1])))
        n_iterations += 1
        max_delta_final = delta
        final = b
        if delta < tol:
            asymptote_reached = True
            break

    sum_c_sq = float((m_g[final] ** 2).sum())
    ne = 1.0 / (2.0 * sum_c_sq) if asymptote_reached and sum_c_sq > 0 else None
    return NeLTCResult(
        ne=ne,
        asymptote_reached=asymptote_reached,
        n_iterations=n_iterations,
        max_delta_final=max_delta_final,
        sum_c_squared=sum_c_sq,
        final_generation=int(cohorts.generations[final]),
    )


def ne_long_term_contributions(pg: PedigreeGraph, *, tol: float = 1e-6) -> NeLTCResult:
    """Wray & Thompson 1990 long-term contribution Ne (Ne_LTC).

    Per-cohort mean founder-genome contribution ``c_b[f] =
    mean_{i ∈ cohort b} c[i, f]`` over the observed generation labels.
    Compare each adjacent observed pair in turn; stop at the first cohort
    where ``max_f |c_b[f] − c_{b−1}[f]| < tol``, or after the last observed
    cohort.  Ne is computed at the stopping cohort as
    ``1 / (2 · Σ_f c_b[f]²)``; ``final_generation`` names that cohort.

    When the asymptote is not reached before the last cohort, ``ne`` is
    ``None`` and ``asymptote_reached`` is ``False``.
    """
    cohorts = ObservedCohorts.for_graph(pg, "ne_long_term_contributions")
    return _ltc_from(cohorts, _per_gen_founder_means(pg, cohorts=cohorts), tol)
