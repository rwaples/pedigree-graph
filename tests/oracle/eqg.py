"""Equivalent complete generations (Maignel 1996), the one numba kernel left here.

Oracle: ``pedigree_graph._kinship_depth`` as 0.9.3 shipped it, verbatim.

Structural depth, the topological order and the kinship DP live in the Rust
core (``pedigree_graph._native``); the DP's Python form is the test oracle
under ``tests/oracle/kinship_dp``.  Depends only on numba/numpy.
"""

from __future__ import annotations

import numba
import numpy as np


@numba.njit(cache=True)
def _compute_eqg(m_idx: np.ndarray, f_idx: np.ndarray, depth: np.ndarray, n: int) -> np.ndarray:
    """Maignel 1996 equivalent complete generations.

    ``EqG_i = Σ over ancestors a of (1/2)^k`` where k is the meiotic
    distance from i to a.  Founders return 0; an individual with two
    known founder parents returns 1; with two known parents who each
    have two known founder parents returns 2; etc.

    Recursion: ``EqG_i = (n_known_parents/2) + 0.5 · Σ EqG_p`` over
    known parents.  Implemented as a generation-ordered sweep keyed on the
    graph's structural ``depth``.
    """
    eqg = np.zeros(n, dtype=np.float64)
    if n == 0:
        return eqg
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
