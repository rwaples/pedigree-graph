"""Rate-of-inbreeding / coancestry Ne estimators (PGQ-006).

The regression-based estimators built on F̄ and θ̄ series, plus the
Gutiérrez 2008 individual-ΔF estimator:

* :func:`ne_inbreeding`         — regression of ``ln(1 − F̄_t)`` on t.
* :func:`ne_coancestry`         — regression of ``ln(1 − θ̄_t)`` on t.
* :func:`ne_individual_delta_f` — Gutiérrez individual ΔF_i via EqG.

Each public estimator resolves its prerequisites from the graph and hands
them to a private evaluator (``_inbreeding_from`` and friends) that works
on an :class:`~pedigree_graph._cohorts.ObservedCohorts` grouping.  The
orchestrator and the 0.7.1 wrappers call the same evaluators, so a direct
call and an orchestrated call cannot disagree.

Also owns :func:`_summary_from_matrix`, the cached-matrix route to
:meth:`PedigreeGraph.mean_kinship_by_generation`, and its 0.7.1 array
adapter :func:`_per_gen_mean_kinship`.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import numpy as np

from pedigree_graph._cohorts import ObservedCohorts
from pedigree_graph._kinship_kernel import (
    _compute_eqg,
    _compute_generation_kinship_summary,
    _densify_labels,
    _finalize_summary,
    _scatter_summary,
)
from pedigree_graph._ne_common import (
    _harmonic_mean,
    _scalar_ne_from_log_regression,
    _transition_ne,
)
from pedigree_graph._ne_results import (
    NeCoancestryResult,
    NeInbreedingResult,
    NeIndividualDeltaFResult,
)

if TYPE_CHECKING:
    import scipy.sparse as sp

    from pedigree_graph._core import PedigreeGraph
    from pedigree_graph.summaries import GenerationKinshipSummary


logger = logging.getLogger(__name__)


def _summary_from_matrix(
    K: sp.csc_matrix,
    labels: np.ndarray,
    twin_idx: np.ndarray,
) -> GenerationKinshipSummary:
    """Generation kinship summary walked from a complete kinship matrix.

    Same grouping and MZ rule as the streamed DP path
    (:func:`~pedigree_graph._kinship_dp._compute_generation_kinship_summary`),
    so the two are interchangeable oracles: upper-triangle entries whose rows
    share an observed label, minus ``(i, twin[i])`` pairs, summed per label
    and divided by :func:`~pedigree_graph._kinship_dp._finalize_summary`.

    Args:
        K: full-symmetric sparse kinship (φ-scale) from
            :meth:`PedigreeGraph.kinship_matrix`.
        labels: per-row cohort label, ``-1`` unknown.
        twin_idx: per-row twin partner row index, ``-1`` for non-twins.
    """
    dense, observed, n_unlabelled = _densify_labels(labels)
    twin = np.ascontiguousarray(twin_idx, dtype=np.int32)
    k = int(observed.shape[0])
    coo = K.tocoo()
    rows, cols, vals = coo.row, coo.col, coo.data
    pair_mask = (rows < cols) & (dense[rows] == dense[cols]) & (dense[rows] < k)
    pair_mask &= ~((twin[rows] >= 0) & (twin[rows] == cols))
    sum_theta = np.bincount(
        dense[rows[pair_mask]].astype(np.intp),
        weights=vals[pair_mask].astype(np.float64),
        minlength=k + 1,
    )
    return _finalize_summary(sum_theta, dense, twin, observed, n_unlabelled)


def _generation_kinship_summary(pg: PedigreeGraph) -> GenerationKinshipSummary:
    """Memoised body of :meth:`PedigreeGraph.mean_kinship_by_generation`.

    Walks the complete kinship matrix when the graph already caches it, else
    streams the retiring DP; both group by the supplied labels, or by
    structural depth when none were supplied.  Stored on the graph, so every
    later call returns the same frozen object.
    """
    cached = pg._generation_kinship_summary
    if cached is not None:
        return cached
    labels = pg.generation_labels
    if labels is None:
        labels = pg.depth
    t0 = time.perf_counter()
    K = pg._complete_kinship_cache
    if K is not None:
        summary = _summary_from_matrix(K, np.asarray(labels), np.asarray(pg.twin))
    else:
        summary = _compute_generation_kinship_summary(
            pg.n,
            pg.mother,
            pg.father,
            pg.twin,
            pg.depth,
            0.0,
            labels=labels,
        )
    pg._generation_kinship_summary = summary
    logger.info(
        "mean_kinship_by_generation: n=%d, groups=%d, unlabelled=%d, %.2fs",
        pg.n,
        len(summary),
        summary.unlabelled_individual_count,
        time.perf_counter() - t0,
    )
    return summary


def _per_gen_mean_kinship(
    K: sp.csc_matrix,
    generation: np.ndarray,
    twin_idx: np.ndarray,
) -> np.ndarray:
    """0.8.0-DELETE: :func:`_summary_from_matrix` in the 0.7.1 array form.

    Float64 array of length ``max(generation) + 1``; NaN where a label is
    absent or its cohort has no eligible pair.
    """
    return _scatter_summary(_summary_from_matrix(K, generation, twin_idx), generation)


def _cohort_means(values: np.ndarray, cohorts: ObservedCohorts) -> np.ndarray:
    """Mean of ``values`` within each observed cohort, NaN for an empty one."""
    out = np.full(cohorts.k, np.nan, dtype=np.float64)
    for b, rows in enumerate(cohorts.members()):
        if rows.shape[0]:
            out[b] = float(values[rows].mean())
    return out


def _inbreeding_from(cohorts: ObservedCohorts, F: np.ndarray) -> NeInbreedingResult:
    mean_f = _cohort_means(F, cohorts)
    ne_scalar, slope, n_used = _scalar_ne_from_log_regression(mean_f, cohorts.generations)
    return NeInbreedingResult(
        ne=ne_scalar,
        generations=cohorts.generations,
        mean_f_per_gen=mean_f,
        transition_from=cohorts.transition_from(),
        transition_to=cohorts.transition_to(),
        ne_per_gen=_transition_ne(mean_f, cohorts.generations),
        slope=slope,
        n_generations_used=n_used,
    )


def ne_inbreeding(pg: PedigreeGraph) -> NeInbreedingResult:
    """Inbreeding-rate Ne (Ne_I).

    Computes per-cohort mean F over the observed generation labels.
    Each adjacent observed-cohort transition reports the gap-corrected
    ``Ne = 1 / (2·ΔF)`` of :func:`~pedigree_graph._ne_common._transition_ne`;
    the aggregate Ne comes from the regression slope of ``ln(1 − F̄)`` on the
    label offset, first observed cohort excluded.
    """
    cohorts = ObservedCohorts.for_graph(pg, "ne_inbreeding")
    return _inbreeding_from(cohorts, pg._inbreeding_values())


def _coancestry_from(cohorts: ObservedCohorts, summary: GenerationKinshipSummary) -> NeCoancestryResult:
    if not np.array_equal(summary.generations, cohorts.generations):
        raise ValueError("generation kinship summary does not describe the estimator's observed cohorts")
    mean_theta = np.asarray(summary.mean_kinship, dtype=np.float64)
    ne_scalar, slope, n_used = _scalar_ne_from_log_regression(mean_theta, cohorts.generations)
    return NeCoancestryResult(
        ne=ne_scalar,
        generations=cohorts.generations,
        mean_theta_per_gen=mean_theta,
        transition_from=cohorts.transition_from(),
        transition_to=cohorts.transition_to(),
        ne_per_gen=_transition_ne(mean_theta, cohorts.generations),
        slope=slope,
        n_generations_used=n_used,
    )


def ne_coancestry(pg: PedigreeGraph) -> NeCoancestryResult:
    """Coancestry-rate Ne (Ne_C).

    Same regression form as Ne_I but on the per-cohort mean kinship θ over
    within-cohort unordered pairs (excluding the diagonal and MZ twin
    pairs) that :meth:`PedigreeGraph.mean_kinship_by_generation` reports.
    The summary is streamed from the DP without materializing K, or walked
    from a complete kinship matrix the graph already caches.
    """
    cohorts = ObservedCohorts.for_graph(pg, "ne_coancestry")
    return _coancestry_from(cohorts, _generation_kinship_summary(pg))


def _individual_delta_f_from(cohorts: ObservedCohorts, F: np.ndarray, eqg: np.ndarray) -> NeIndividualDeltaFResult:
    valid = (eqg > 1.0) & (F < 1.0)
    delta_f = np.full(F.shape[0], np.nan, dtype=np.float64)
    if valid.any():
        delta_f[valid] = 1.0 - np.power(1.0 - F[valid], 1.0 / (eqg[valid] - 1.0))

    k = cohorts.k
    ne_per_gen = np.full(k, np.nan, dtype=np.float64)
    mean_eqg_per_gen = np.full(k, np.nan, dtype=np.float64)
    n_used_per_gen = np.zeros(k, dtype=np.int64)
    for b, rows in enumerate(cohorts.members()):
        in_b = rows[valid[rows]]
        n_used_per_gen[b] = int(in_b.shape[0])
        if in_b.shape[0] == 0:
            continue
        mean_df = float(delta_f[in_b].mean())
        mean_eqg_per_gen[b] = float(eqg[in_b].mean())
        if mean_df > 0:
            ne_per_gen[b] = 1.0 / (2.0 * mean_df)

    return NeIndividualDeltaFResult(
        ne=_harmonic_mean(ne_per_gen) if np.isfinite(ne_per_gen).any() else None,
        generations=cohorts.generations,
        ne_per_gen=ne_per_gen,
        mean_eqg_per_gen=mean_eqg_per_gen,
        n_used_per_gen=n_used_per_gen,
    )


def ne_individual_delta_f(pg: PedigreeGraph) -> NeIndividualDeltaFResult:
    """Gutiérrez 2008 individual ΔF Ne (Ne_iΔF).

    For each individual ``i`` with ``EqG_i > 1`` and ``F_i < 1``:

        ``ΔF_i = 1 − (1 − F_i)^(1/(EqG_i − 1))``.

    Per-cohort ``Ne_g = 1/(2 · mean_{i ∈ cohort g} ΔF_i)``; aggregate is
    the harmonic mean across observed cohorts.  EqG already counts complete
    generations per individual, so labels only group the result.
    """
    cohorts = ObservedCohorts.for_graph(pg, "ne_individual_delta_f")
    F = pg._inbreeding_values()
    eqg = _compute_eqg(np.asarray(pg.mother), np.asarray(pg.father), pg.n)
    return _individual_delta_f_from(cohorts, F, eqg)
