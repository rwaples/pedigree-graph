"""Founder-contribution Ne estimator and its primitives (PGQ-006).

Owns the represented-founder index, the adjoint per-cohort mean-contribution
propagation, and the evaluator of the founder-contribution effective
sizes (:func:`~pedigree_graph.effective_size.ne_long_term_contributions`, whose public function
runs its registry row in ``_ne_estimate``) built on them.

A **represented founder** is a row with no represented mother or father,
whatever its generation label and whether its parents are missing or
external to the graph.  Contribution columns represent **represented
founder genomes** (ADR 0008): parentless MZ co-twins are two founder rows
sharing one column, so their descendants inherit one lineage.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from pedigree_graph import _native
from pedigree_graph._cohorts import ObservedCohorts
from pedigree_graph._input import _own_native
from pedigree_graph._ne_common import _checked_founder_matrix, _genome_of
from pedigree_graph._ne_results import NeLTCResult

if TYPE_CHECKING:
    from pedigree_graph._core import PedigreeGraph

_ASYMPTOTE_TOL = 1e-6
"""Movement below which ``NeLTCResult.asymptote_reached`` is reported; descriptive only."""


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
    return np.flatnonzero((np.asarray(pg.mother_rows) < 0) & (np.asarray(pg.father_rows) < 0)).astype(np.intp)


def _founder_idx(pg: PedigreeGraph) -> np.ndarray:
    """Canonical rows of the represented founder genomes, ascending intp.

    One entry per genome: a parentless MZ pair contributes its lower row.
    """
    founders = _founder_rows(pg)
    return np.unique(_genome_of(pg)[founders]).astype(np.intp)


def _founder_columns(pg: PedigreeGraph, founder_idx: np.ndarray) -> np.ndarray:
    """Column of ``founder_idx`` each represented founder row seeds; ``-1`` elsewhere."""
    columns = np.full(pg.n_individuals, -1, dtype=np.int64)
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

    Computed by iterating the adjoint of the forward recursion, in the Rust
    core.  For each target cohort, propagate the cohort uniform vector
    ``1_{cohort} / N_b`` backward through child→parent edges one
    **structural depth** at a time, deepest rows first: at depth ``d``,
    scatter ``0.5 · u[child]`` from each child at that depth into its mother
    and father.  Depth, not the generation label, orders the sweep, so labels
    that merge depths or put a parent and child in one cohort only change the
    grouping, never the ancestry.  What remains on the founder rows is summed
    per genome.

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
    if n_founders == 0 or cohorts.k == 0:
        return FounderContributionMeans(
            _checked_founder_matrix(cohorts.k, n_founders, "founder_means", np.float64, np.nan), founder_idx
        )

    flat = _native.founder_contribution_means(
        pg._built, pg.depth, cohorts.dense, cohorts.k, _founder_columns(pg, founder_idx), n_founders
    )
    m_g = _own_native(flat, np.float64).reshape(cohorts.k, n_founders)
    return FounderContributionMeans(m_g, founder_idx)


def _ltc_from(cohorts: ObservedCohorts, means: FounderContributionMeans) -> NeLTCResult:
    m_g, founder_idx = means
    if founder_idx.shape[0] == 0 or cohorts.k == 0:
        return NeLTCResult(
            ne=None,
            n_effective_founders=None,
            sum_c_squared=0.0,
            max_delta_final=float("nan"),
            asymptote_reached=False,
            n_cohorts=0,
            final_generation=None,
            no_estimate_code="no_founders",
        )

    last = cohorts.k - 1
    sum_c_sq = float((m_g[last] ** 2).sum())
    max_delta_final = float(np.max(np.abs(m_g[last] - m_g[last - 1]))) if last else float("nan")
    # Closed parentage makes each row's contributions sum to 1, so the
    # cohort mean's do too and Σc² >= 1/n_founders > 0.
    n_ef = 1.0 / sum_c_sq
    return NeLTCResult(
        ne=2.0 * n_ef,
        n_effective_founders=n_ef,
        sum_c_squared=sum_c_sq,
        max_delta_final=max_delta_final,
        asymptote_reached=max_delta_final < _ASYMPTOTE_TOL,
        n_cohorts=cohorts.k,
        final_generation=int(cohorts.generations[last]),
    )
