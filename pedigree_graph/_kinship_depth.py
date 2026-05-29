"""Pedigree depth / equivalent-generation / topology numba utilities (PGQ-008).

Leaf kernels over the parent-index arrays: generation depth, last
direct-child depth, equivalent complete generations (EqG), and the
topological-order check.  Depends only on numba/numpy.
"""

from __future__ import annotations

import numba
import numpy as np


@numba.njit(cache=True)
def _compute_depth(m_idx: np.ndarray, f_idx: np.ndarray, n: int) -> np.ndarray:
    """Generation depth: founders=0, offspring = max(parent_depth)+1.

    Iterates a fixed-point sweep until all individuals are assigned.
    """
    depth = np.full(n, -1, dtype=np.int32)
    for i in range(n):
        if m_idx[i] < 0 and f_idx[i] < 0:
            depth[i] = 0
    changed = True
    while changed:
        changed = False
        for j in range(n):
            if depth[j] >= 0:
                continue
            m, f = m_idx[j], f_idx[j]
            md = depth[m] if m >= 0 else 0
            fd = depth[f] if f >= 0 else 0
            if md >= 0 and fd >= 0:
                depth[j] = (md if md > fd else fd) + 1
                changed = True
    # Disconnected founders default to depth 0.
    for j in range(n):
        if depth[j] < 0:
            depth[j] = 0
    return depth


@numba.njit(cache=True)
def _compute_last_direct_child_depth(
    m_idx: np.ndarray,
    f_idx: np.ndarray,
    depth: np.ndarray,
    n: int,
) -> np.ndarray:
    """Max depth at which row[k] is still READ by a merge walk.

    Returns, for each k, ``max(depth[k], max(depth[j] : k ∈ {m_idx[j], f_idx[j]}))``.
    After this depth the kernel makes no further reads against ``row_start[k]``
    storage, so the slot may be retired (freed for reuse).  Twin partners are
    intentionally excluded — the MZ twin pass writes to twin rows but never
    reads them via the merge walk.

    Rows with no direct children retain ``depth[k]``, meaning they are
    eligible for retirement immediately after their own processing finishes.
    """
    out = np.empty(n, dtype=np.int32)
    for k in range(n):
        out[k] = depth[k]
    for j in range(n):
        d_j = depth[j]
        m = m_idx[j]
        if m >= 0 and d_j > out[m]:
            out[m] = d_j
        f = f_idx[j]
        if f >= 0 and d_j > out[f]:
            out[f] = d_j
    return out


@numba.njit(cache=True)
def _compute_eqg(m_idx: np.ndarray, f_idx: np.ndarray, n: int) -> np.ndarray:
    """Maignel 1996 equivalent complete generations.

    ``EqG_i = Σ over ancestors a of (1/2)^k`` where k is the meiotic
    distance from i to a.  Founders return 0; an individual with two
    known founder parents returns 1; with two known parents who each
    have two known founder parents returns 2; etc.

    Recursion: ``EqG_i = (n_known_parents/2) + 0.5 · Σ EqG_p`` over
    known parents.  Implemented as a generation-ordered sweep keyed on
    :func:`_compute_depth`.
    """
    eqg = np.zeros(n, dtype=np.float64)
    depth = _compute_depth(m_idx, f_idx, n)
    max_depth = depth.max()
    for d in range(1, max_depth + 1):
        for i in range(n):
            if depth[i] != d:
                continue
            m, f = m_idx[i], f_idx[i]
            v = 0.0
            if m >= 0:
                v += 0.5 + 0.5 * eqg[m]
            if f >= 0:
                v += 0.5 + 0.5 * eqg[f]
            eqg[i] = v
    return eqg


@numba.njit(cache=True)
def _check_topological(m_idx: np.ndarray, f_idx: np.ndarray, n: int) -> bool:
    """Return True iff every parent index strictly precedes its child.

    Used to validate the topological-order invariant required by
    :func:`_compute_F_meuwissen_luo`'s outer ``for i in range(n)`` loop.
    Missing parents (-1) are treated as satisfying the constraint.
    """
    for i in range(n):
        m = m_idx[i]
        f = f_idx[i]
        if m >= 0 and m >= i:
            return False
        if f >= 0 and f >= i:
            return False
    return True
