"""Rate-of-inbreeding / coancestry Ne estimators (PGQ-006).

The regression-based estimators built on F̄ and θ̄ series, plus the
Gutiérrez 2008 individual-ΔF estimator:

* :func:`ne_inbreeding`         — regression of ``ln(1 − F̄_t)`` on t.
* :func:`ne_coancestry`         — regression of ``ln(1 − θ̄_t)`` on t.
* :func:`ne_individual_delta_f` — Gutiérrez individual ΔF_i via EqG.

Also owns :func:`_summary_from_matrix`, the cached-matrix route to
:meth:`PedigreeGraph.mean_kinship_by_generation`, and its 0.7.1 array
adapter :func:`_per_gen_mean_kinship`.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import numpy as np

from pedigree_graph._kinship_kernel import (
    _compute_eqg,
    _compute_generation_kinship_summary,
    _densify_labels,
    _finalize_summary,
    _scatter_summary,
)
from pedigree_graph._ne_common import (
    _harmonic_mean,
    _require_complete_generation_labels,
    _scalar_ne_from_log_regression,
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


def ne_inbreeding(pg: PedigreeGraph) -> NeInbreedingResult:
    """Inbreeding-rate Ne (Ne_I).

    Computes per-cohort mean F (founders = 0).  Per-transition Ne_t =
    ``1 / (2·ΔF_t)`` with ``ΔF_t = (F̄_t − F̄_{t−1}) / (1 − F̄_{t−1})``.
    Aggregate Ne from the regression slope of ``ln(1 − F̄_t)`` on t for
    t ≥ 1 (founders excluded).
    """
    _require_complete_generation_labels(pg, "ne_inbreeding")
    F = pg._inbreeding_values()
    gen = np.asarray(pg.generation)
    g_max = int(gen.max())
    mean_f = np.zeros(g_max + 1, dtype=np.float64)
    for g in range(g_max + 1):
        mask = gen == g
        if mask.any():
            mean_f[g] = float(F[mask].mean())

    ne_per_gen = np.full(g_max + 1, np.nan, dtype=np.float64)
    for g in range(1, g_max + 1):
        f_prev = mean_f[g - 1]
        if f_prev >= 1.0:
            continue
        df = (mean_f[g] - f_prev) / (1.0 - f_prev)
        if df > 0:
            ne_per_gen[g] = 1.0 / (2.0 * df)

    ne_scalar, slope, n_used = _scalar_ne_from_log_regression(mean_f)

    return NeInbreedingResult(
        ne=ne_scalar,
        ne_per_gen=ne_per_gen,
        mean_f_per_gen=mean_f,
        slope=slope,
        n_generations_used=n_used,
    )


def ne_coancestry(
    pg: PedigreeGraph,
    K: sp.csc_matrix | None = None,
    theta_per_gen: np.ndarray | None = None,
) -> NeCoancestryResult:
    """Coancestry-rate Ne (Ne_C).

    Same regression form as Ne_I but on per-cohort mean kinship θ over
    within-cohort unordered pairs (excluding the diagonal and MZ twin
    pairs).

    The estimator accepts θ̄_g pre-computed (streamed from the DP
    without materializing K) — preferred path at large N where K's CSC
    would OOM.  If neither θ̄_g nor K is supplied, the K-free streaming
    path is used by default.

    Args:
        pg: Pedigree graph.
        K: optional pre-built sparse kinship matrix.  Used only when
            ``theta_per_gen`` is None.
        theta_per_gen: optional pre-computed per-generation mean
            kinship.  When supplied, K is ignored.
    """
    _require_complete_generation_labels(pg, "ne_coancestry")
    gen = np.asarray(pg.generation)
    g_max = int(gen.max())
    if theta_per_gen is not None:
        mean_theta = np.asarray(theta_per_gen, dtype=np.float64)
    elif K is not None:
        twin = np.asarray(pg.twin)
        mean_theta = _per_gen_mean_kinship(K, gen, twin)
    else:
        mean_theta = pg.per_gen_mean_kinship()

    ne_per_gen = np.full(g_max + 1, np.nan, dtype=np.float64)
    for g in range(1, g_max + 1):
        theta_prev = mean_theta[g - 1]
        if not np.isfinite(theta_prev) or theta_prev >= 1.0:
            continue
        if not np.isfinite(mean_theta[g]):
            continue
        d_theta = (mean_theta[g] - theta_prev) / (1.0 - theta_prev)
        if d_theta > 0:
            ne_per_gen[g] = 1.0 / (2.0 * d_theta)

    ne_scalar, slope, n_used = _scalar_ne_from_log_regression(mean_theta)

    return NeCoancestryResult(
        ne=ne_scalar,
        ne_per_gen=ne_per_gen,
        mean_theta_per_gen=mean_theta,
        slope=slope,
        n_generations_used=n_used,
    )


def ne_individual_delta_f(pg: PedigreeGraph) -> NeIndividualDeltaFResult:
    """Gutiérrez 2008 individual ΔF Ne (Ne_iΔF).

    For each individual ``i`` with ``EqG_i > 1`` and ``F_i < 1``:

        ``ΔF_i = 1 − (1 − F_i)^(1/(EqG_i − 1))``.

    Per-cohort ``Ne_g = 1/(2 · mean_{i ∈ gen g} ΔF_i)``; aggregate is
    the harmonic mean across cohorts.
    """
    _require_complete_generation_labels(pg, "ne_individual_delta_f")
    F = pg._inbreeding_values()
    eqg = _compute_eqg(np.asarray(pg.mother), np.asarray(pg.father), pg.n)
    gen = np.asarray(pg.generation)
    g_max = int(gen.max())

    valid = (eqg > 1.0) & (F < 1.0)
    delta_f = np.full(pg.n, np.nan, dtype=np.float64)
    if valid.any():
        delta_f[valid] = 1.0 - np.power(1.0 - F[valid], 1.0 / (eqg[valid] - 1.0))

    ne_per_gen = np.full(g_max + 1, np.nan, dtype=np.float64)
    mean_eqg_per_gen = np.full(g_max + 1, np.nan, dtype=np.float64)
    n_used_per_gen = np.zeros(g_max + 1, dtype=np.int64)
    for g in range(g_max + 1):
        in_g = (gen == g) & valid
        n_used_per_gen[g] = int(in_g.sum())
        if n_used_per_gen[g] == 0:
            continue
        mean_df = float(delta_f[in_g].mean())
        mean_eqg_per_gen[g] = float(eqg[in_g].mean())
        if mean_df > 0:
            ne_per_gen[g] = 1.0 / (2.0 * mean_df)

    return NeIndividualDeltaFResult(
        ne=_harmonic_mean(ne_per_gen) if np.isfinite(ne_per_gen).any() else None,
        ne_per_gen=ne_per_gen,
        mean_eqg_per_gen=mean_eqg_per_gen,
        n_used_per_gen=n_used_per_gen,
    )
