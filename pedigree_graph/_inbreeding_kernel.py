"""Meuwissen-Luo F-only inbreeding kernel (PGQ-008).

The sparse ancestor-walk that computes per-individual inbreeding
coefficients without materializing kinship.  Reuses the topological
check from ``_kinship_depth`` and the global arena from
``_kinship_allocator``.
"""

from __future__ import annotations

import numba
import numpy as np


@numba.njit(cache=True)
def _grow_touched_scratch(
    touched: np.ndarray,
    next_in_chain: np.ndarray,
    count: int,
):
    """Double the capacity of the (touched, next_in_chain) scratch pair.

    Mirrors :func:`_grow_global` for the ML F kernel's per-individual
    ancestor-list and linked-list arrays.  Returns ``(touched, next, cap)``.
    """
    old_cap = touched.shape[0]
    new_cap = old_cap * 2
    new_touched = np.empty(new_cap, dtype=np.int32)
    new_next = np.full(new_cap, -1, dtype=np.int32)
    for q in range(count):
        new_touched[q] = touched[q]
        new_next[q] = next_in_chain[q]
    return new_touched, new_next, new_cap


@numba.njit(cache=True)
def _compute_F_meuwissen_luo(
    m_idx: np.ndarray,
    f_idx: np.ndarray,
    depth: np.ndarray,
    n: int,
) -> np.ndarray:
    """Inbreeding coefficient F per individual via Meuwissen & Luo (1992).

    Uses the LDL' decomposition ``A = T D T'`` of the numerator
    relationship matrix.  For each individual i in topological order:

        D[i] = 0.5 - 0.25*(F[s] + F[d])  (both parents known)
             = 0.75 - 0.25*F[p]          (one parent known)
             = 1                          (founder)

        F[i] = (sum over j in ANC_i of t_ij^2 * D[j]) - 1

    where ``t_ii = 1`` and ``t_ij = 0.5 * sum over progeny k of j (with k
    in ANC_i) of t_ik`` — equivalently, the path-sum from i to ancestor j
    obtained by FORWARD propagation: start with t[i]=1, halve along each
    parent edge, sum across multiple paths.

    MZ-naive: twins are treated as full sibs.  K-derived F (matrix path)
    is MZ-aware via twin off-diagonals; the two paths agree except for
    individuals whose only inbreeding path runs through both members of an
    MZ twin pair as ancestors.

    Topological-order required: m_idx[i] < i and f_idx[i] < i for all i
    (validated upstream via :func:`_check_topological`).
    """
    # Persistent scratch.  Sparse-touched, reset via the touched list.
    t = np.zeros(n, dtype=np.float64)
    in_frontier = np.zeros(n, dtype=np.bool_)
    F = np.zeros(n, dtype=np.float64)
    D = np.zeros(n, dtype=np.float64)

    max_depth_global = int(depth.max()) if n > 0 else 0
    bound = 2
    for _ in range(max_depth_global + 1):
        bound *= 2
    bound -= 2  # 2^(max_depth+1) - 2  (full binary ancestor set bound)
    if bound < 64:
        bound = 64
    K_max = bound + 64
    if K_max > n:
        K_max = n
    if K_max < 4:
        K_max = 4

    touched = np.empty(K_max, dtype=np.int32)
    next_in_chain = np.full(K_max, -1, dtype=np.int32)
    head_depth = np.full(max_depth_global + 1, -1, dtype=np.int32)

    for i in range(n):
        s = m_idx[i]
        d = f_idx[i]

        # Compute D[i] from already-known F[parents].  D is needed even
        # when F[i]=0 because future descendants reference D[i].
        if s < 0 and d < 0:
            F[i] = 0.0
            D[i] = 1.0
            continue
        if s < 0:
            F[i] = 0.0
            D[i] = 0.75 - 0.25 * F[d]
            continue
        if d < 0:
            F[i] = 0.0
            D[i] = 0.75 - 0.25 * F[s]
            continue
        D[i] = 0.5 - 0.25 * (F[s] + F[d])

        touched_count = 0
        t[i] = 1.0
        in_frontier[i] = True
        di = int(depth[i])
        next_in_chain[touched_count] = head_depth[di]
        head_depth[di] = touched_count
        touched[touched_count] = i
        touched_count += 1

        for k in range(di, -1, -1):
            pos = head_depth[k]
            while pos >= 0:
                a = touched[pos]
                t_a = t[a]
                p_m = m_idx[a]
                p_f = f_idx[a]
                if p_m >= 0:
                    if not in_frontier[p_m]:
                        in_frontier[p_m] = True
                        t[p_m] = 0.0
                        if touched_count >= K_max:
                            touched, next_in_chain, K_max = _grow_touched_scratch(touched, next_in_chain, touched_count)
                        dp = int(depth[p_m])
                        next_in_chain[touched_count] = head_depth[dp]
                        head_depth[dp] = touched_count
                        touched[touched_count] = p_m
                        touched_count += 1
                    t[p_m] += 0.5 * t_a
                if p_f >= 0:
                    # Mirrors the mother-edge branch above; kept inline for numba.
                    if not in_frontier[p_f]:
                        in_frontier[p_f] = True
                        t[p_f] = 0.0
                        if touched_count >= K_max:
                            touched, next_in_chain, K_max = _grow_touched_scratch(touched, next_in_chain, touched_count)
                        dp = int(depth[p_f])
                        next_in_chain[touched_count] = head_depth[dp]
                        head_depth[dp] = touched_count
                        touched[touched_count] = p_f
                        touched_count += 1
                    t[p_f] += 0.5 * t_a
                pos = next_in_chain[pos]
            head_depth[k] = -1

        # F[i] = sum_j t_ij^2 * D[j] - 1   over j in ANC_i (= touched).
        F_sum = 0.0
        for q in range(touched_count):
            j = touched[q]
            tj = t[j]
            F_sum += tj * tj * D[j]
        F[i] = F_sum - 1.0

        for q in range(touched_count):
            j = touched[q]
            t[j] = 0.0
            in_frontier[j] = False
            next_in_chain[q] = -1

    return F
