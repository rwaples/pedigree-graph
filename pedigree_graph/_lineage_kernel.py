"""Topological lineage primitives: per-row ancestor and descendant counts.

These are graph properties of the pedigree DAG, independent of kinship
arithmetic.  Two semantics live here:

* ``_compute_n_descendants`` returns **path counts**: ``n_desc[v]`` is the
  number of walks (v, w) down the DAG, i.e. (number of children) +
  (sum of descendants of children).  In non-inbred pedigrees this equals
  the number of unique descendants; in inbred pedigrees it over-counts
  by the inbreeding rate (a descendant reachable from v through multiple
  child-paths is counted once per path).  This matches the convention
  used for GP / Av / 1C pair counts.
* ``_compute_n_ancestors`` returns **distinct counts**: ``n_anc[v]`` is
  the number of unique strict ancestors of v.  An ancestor reachable
  through multiple paths (loops introduced by inbreeding) is counted
  once.  Matches the L-row retirement semantic used historically by
  pedsum.
"""

from __future__ import annotations

import numba
import numpy as np
import scipy.sparse as sp


@numba.njit(cache=True)
def _compute_n_descendants(
    m_idx: np.ndarray,
    f_idx: np.ndarray,
    n: int,
) -> np.ndarray:
    """Per-row descendant *path* count.

    Topological order is required (parents precede children in row
    index — enforced by ``PedigreeGraph.__init__``).  We iterate row
    indices in reverse, pushing each row's contribution up to its
    parents.  When we visit i, every descendant of i has higher row
    index and was therefore visited first, so ``n_desc[i]`` is final.
    """
    n_desc = np.zeros(n, dtype=np.int64)
    for i in range(n - 1, -1, -1):
        contrib = 1 + n_desc[i]
        m = m_idx[i]
        if m >= 0:
            n_desc[m] += contrib
        f = f_idx[i]
        if f >= 0:
            n_desc[f] += contrib
    return n_desc.astype(np.int32)


def _compute_n_ancestors(
    m_idx: np.ndarray,
    f_idx: np.ndarray,
    n: int,
) -> np.ndarray:
    """Per-row distinct ancestor count via sparse boolean transitive closure.

    Builds the direct parent matrix ``A[i, j] = 1`` iff ``j`` is a parent of
    ``i``, then accumulates ``A | A^2 | A^3 | …`` iteratively until the
    nnz count stops growing.  Each iteration adds one more generation of
    ancestors, so it terminates after at most ``G_max`` steps where
    ``G_max`` is the deepest path in the pedigree.

    ``n_anc[i]`` is the count of distinct strict ancestors of i — an
    ancestor reachable through multiple paths (loops introduced by
    inbreeding) is counted once.

    Memory peaks at the size of the final closure (roughly
    ``sum_i len(ancestors(i))``).  For G_max ≲ 10 this scales to
    ~1M-row pedigrees on commodity hardware; deeper pedigrees may need
    a retirement-style DP to bound peak memory.
    """
    if n == 0:
        return np.zeros(0, dtype=np.int32)

    m_mask = m_idx >= 0
    f_mask = f_idx >= 0
    if not m_mask.any() and not f_mask.any():
        return np.zeros(n, dtype=np.int32)

    rows = np.concatenate(
        [np.where(m_mask)[0], np.where(f_mask)[0]],
    ).astype(np.int32)
    cols = np.concatenate(
        [m_idx[m_mask], f_idx[f_mask]],
    ).astype(np.int32)

    a = sp.csr_matrix(
        (np.ones(len(rows), dtype=np.int8), (rows, cols)),
        shape=(n, n),
    )

    closure = a.copy()
    prev_nnz = -1
    while closure.nnz != prev_nnz:
        prev_nnz = closure.nnz
        step = closure @ a
        # Combine and re-binarize.  Adding the int8 matrices may push
        # counts above 1; clamp back to {0, 1} before the next iteration.
        combined = closure + step
        combined.data = (combined.data > 0).astype(np.int8)
        combined.eliminate_zeros()
        closure = combined

    return np.diff(closure.indptr).astype(np.int32)
