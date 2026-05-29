"""Rate-of-inbreeding / coancestry Ne estimators (PGQ-006).

The regression-based estimators built on F̄ and θ̄ series, plus the
Gutiérrez 2008 individual-ΔF estimator:

* :func:`ne_inbreeding`         — regression of ``ln(1 − F̄_t)`` on t.
* :func:`ne_coancestry`         — regression of ``ln(1 − θ̄_t)`` on t.
* :func:`ne_individual_delta_f` — Gutiérrez individual ΔF_i via EqG.

Also owns :func:`_per_gen_mean_kinship` (per-cohort mean θ over
within-cohort pairs), shared with :meth:`PedigreeGraph.per_gen_mean_kinship`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pedigree_graph._kinship_kernel import _compute_eqg, _finalize_from_sum_theta
from pedigree_graph._ne_common import _harmonic_mean, _regress_log_one_minus
from pedigree_graph._ne_results import (
    NeCoancestryResult,
    NeInbreedingResult,
    NeIndividualDeltaFResult,
)

if TYPE_CHECKING:
    import scipy.sparse as sp

    from pedigree_graph._core import PedigreeGraph


def _per_gen_mean_kinship(
    K: sp.csc_matrix,
    generation: np.ndarray,
    twin_idx: np.ndarray,
) -> np.ndarray:
    """Mean θ over unordered within-cohort pairs, per generation.

    Excludes the diagonal and MZ twin pairs.  Returns ``np.nan`` for
    cohorts with fewer than 2 non-twin members or where every pair is
    a twin pair.

    Args:
        K: full-symmetric sparse kinship (φ-scale) from
            :meth:`PedigreeGraph.kinship_matrix`.
        generation: per-individual generation index (founders = 0).
        twin_idx: per-individual twin partner row index, ``-1`` for
            non-twins.
    """
    g_max = int(generation.max())

    # COO traversal — restrict to upper triangle, same generation,
    # non-twin pairs.  Sum per-gen θ via bincount, then divide through
    # the shared finalizer that knows the within-cohort pair-count math.
    coo = K.tocoo()
    rows, cols, vals = coo.row, coo.col, coo.data
    pair_mask = (rows < cols) & (generation[rows] == generation[cols])
    pair_mask &= ~((twin_idx[rows] >= 0) & (twin_idx[rows] == cols))
    pair_gens = generation[rows[pair_mask]].astype(np.intp)
    sum_theta = np.bincount(
        pair_gens,
        weights=vals[pair_mask].astype(np.float64),
        minlength=g_max + 1,
    )
    return _finalize_from_sum_theta(sum_theta, generation, twin_idx)


def ne_inbreeding(pg: PedigreeGraph) -> NeInbreedingResult:
    """Inbreeding-rate Ne (Ne_I).

    Computes per-cohort mean F (founders = 0).  Per-transition Ne_t =
    ``1 / (2·ΔF_t)`` with ``ΔF_t = (F̄_t − F̄_{t−1}) / (1 − F̄_{t−1})``.
    Aggregate Ne from the regression slope of ``ln(1 − F̄_t)`` on t for
    t ≥ 1 (founders excluded).
    """
    F = pg.compute_inbreeding()
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

    t = np.arange(1, g_max + 1, dtype=np.float64)
    slope, _ = _regress_log_one_minus(mean_f[1:], t)
    if np.isfinite(slope) and slope < 0:
        ne_scalar: float | None = -1.0 / (2.0 * slope)
    else:
        ne_scalar = None

    return NeInbreedingResult(
        ne=ne_scalar,
        ne_per_gen=ne_per_gen,
        mean_f_per_gen=mean_f,
        slope=slope,
        n_generations_used=int(np.isfinite(np.log1p(-mean_f[1:])).sum()),
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

    t = np.arange(1, g_max + 1, dtype=np.float64)
    slope, _ = _regress_log_one_minus(mean_theta[1:], t)
    if np.isfinite(slope) and slope < 0:
        ne_scalar: float | None = -1.0 / (2.0 * slope)
    else:
        ne_scalar = None

    return NeCoancestryResult(
        ne=ne_scalar,
        ne_per_gen=ne_per_gen,
        mean_theta_per_gen=mean_theta,
        slope=slope,
        n_generations_used=int(np.isfinite(np.log1p(-mean_theta[1:])).sum()),
    )


def ne_individual_delta_f(pg: PedigreeGraph) -> NeIndividualDeltaFResult:
    """Gutiérrez 2008 individual ΔF Ne (Ne_iΔF).

    For each individual ``i`` with ``EqG_i > 1`` and ``F_i < 1``:

        ``ΔF_i = 1 − (1 − F_i)^(1/(EqG_i − 1))``.

    Per-cohort ``Ne_g = 1/(2 · mean_{i ∈ gen g} ΔF_i)``; aggregate is
    the harmonic mean across cohorts.
    """
    F = pg.compute_inbreeding()
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
