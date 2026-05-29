"""Founder-contribution Ne estimator and its primitives (PGQ-006).

Owns the founder-index helper and the adjoint per-generation
mean-contribution propagation, plus the Wray & Thompson 1990 long-term
contribution estimator (:func:`ne_long_term_contributions`) built on
them.  ``_founder_idx`` is also consumed by the Caballero-Toro engine.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pedigree_graph._ne_results import NeLTCResult

if TYPE_CHECKING:
    from pedigree_graph._core import PedigreeGraph


def _founder_idx(pg: PedigreeGraph) -> np.ndarray:
    """Indices of founders (gen-0 individuals)."""
    return np.where(np.asarray(pg.generation) == 0)[0].astype(np.intp)


def _per_gen_founder_means(
    pg: PedigreeGraph,
    founder_idx: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-generation mean founder contribution via adjoint propagation.

    Returns ``(m_g, founder_idx)`` where
    ``m_g[g, f_local] = mean_{i ∈ gen g} c[i, founder_idx[f_local]]``,
    with ``c[i, f]`` the expected genome fraction of i inherited from
    founder f under the Mendelian recursion (founders contribute 1 to
    themselves; non-founders take the mean of their two parents' rows).

    Computed by iterating the adjoint of the forward recursion.  For
    each target generation g, propagate the cohort uniform vector
    ``1_{gen g} / N_g`` backward through child→parent edges.  At
    iteration ``t``, scatter ``0.5 · u[child]`` from each ``child ∈
    gen t+1`` into its mother and father (which may live in any earlier
    generation under the ``gen[i] = max(gen_parents) + 1`` definition).
    By the time iteration ``t`` reads ``u[gen == t+1]``, every later
    generation has already contributed into it — so skip-gen parents are
    handled correctly without bookkeeping.

    Time: O(N · g_max²).  Memory: O(N + n_founders · g_max).

    Args:
        pg: Pedigree graph.
        founder_idx: Optional precomputed founder index array.

    Returns:
        ``(m_g, founder_idx)`` — ``m_g`` shape ``(g_max + 1, n_founders)``
        float64; ``founder_idx`` shape ``(n_founders,)`` intp.
    """
    if founder_idx is None:
        founder_idx = _founder_idx(pg)
    n_founders = len(founder_idx)
    n = pg.n
    gen = np.asarray(pg.generation)
    mother = np.asarray(pg.mother)
    father = np.asarray(pg.father)
    g_max = int(gen.max()) if n > 0 else 0

    m_g = np.full((g_max + 1, n_founders), np.nan, dtype=np.float64)
    if n_founders == 0:
        return m_g, founder_idx

    # Precompute per-generation member indices once — the inner sweep
    # reads the same cohorts O(g_max) times across the outer loop.
    cohorts = [np.flatnonzero(gen == g) for g in range(g_max + 1)]

    # Gen 0 mirrors the forward convention `c[gen==0].mean(axis=0)` —
    # founders sit on the identity diagonal, so column means are 1/N_0.
    n0 = len(cohorts[0])
    if n0 > 0:
        m_g[0] = 1.0 / n0

    for g in range(1, g_max + 1):
        in_g = cohorts[g]
        if len(in_g) == 0:
            continue
        u = np.zeros(n, dtype=np.float64)
        u[in_g] = 1.0 / len(in_g)
        for t in range(g - 1, -1, -1):
            child = cohorts[t + 1]
            if len(child) == 0:
                continue
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
        m_g[g] = u[founder_idx]

    return m_g, founder_idx


def ne_long_term_contributions(
    pg: PedigreeGraph,
    mean_contributions: tuple[np.ndarray, np.ndarray] | None = None,
    tol: float = 1e-6,
) -> NeLTCResult:
    """Wray & Thompson 1990 long-term contribution Ne (Ne_LTC).

    Per-generation mean founder contribution ``c_g[f] =
    mean_{i ∈ gen g} c[i, f]``.  Iterate g = 1, 2, …; stop at the first
    g where ``max_f |c_g[f] − c_{g-1}[f]| < tol``, or after the last
    available generation.  Ne is computed at the stopping g as
    ``1 / (2 · Σ_f c_g[f]²)``.

    When the asymptote is not reached before the last generation, ``ne``
    is ``None`` and ``asymptote_reached`` is ``False``.
    """
    if mean_contributions is None:
        m_g, founder_idx = _per_gen_founder_means(pg)
    else:
        m_g, founder_idx = mean_contributions
    n_founders = len(founder_idx)
    if n_founders == 0:
        return NeLTCResult(
            ne=None,
            asymptote_reached=False,
            n_iterations=0,
            max_delta_final=float("nan"),
            sum_c_squared=0.0,
        )

    gen = np.asarray(pg.generation)
    g_max = int(gen.max())
    if g_max == 0:
        return NeLTCResult(
            ne=None,
            asymptote_reached=False,
            n_iterations=0,
            max_delta_final=float("nan"),
            sum_c_squared=float((m_g[0] ** 2).sum()),
        )

    asymptote_reached = False
    n_iterations = 0
    max_delta_final = float("nan")
    for g in range(1, g_max + 1):
        if not np.isfinite(m_g[g]).all() or not np.isfinite(m_g[g - 1]).all():
            continue
        delta = float(np.max(np.abs(m_g[g] - m_g[g - 1])))
        n_iterations = g
        max_delta_final = delta
        if delta < tol:
            asymptote_reached = True
            break

    final_c = m_g[n_iterations]
    sum_c_sq = float((final_c**2).sum())
    if asymptote_reached and sum_c_sq > 0:
        ne: float | None = 1.0 / (2.0 * sum_c_sq)
    else:
        ne = None

    return NeLTCResult(
        ne=ne,
        asymptote_reached=asymptote_reached,
        n_iterations=n_iterations,
        max_delta_final=max_delta_final,
        sum_c_squared=sum_c_sq,
    )
