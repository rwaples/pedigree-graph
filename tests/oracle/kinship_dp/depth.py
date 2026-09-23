"""Retirement depth for the oracle DP: ``_compute_last_direct_child_depth`` of 0.9.1."""

from __future__ import annotations

import numba
import numpy as np


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
