"""Parallel numba kernel for BFS-engine cousin-style pair enumeration.

Used by :func:`pedigree_graph.experimental.count_pairs_bfs` to enumerate
``(i, j)`` pairs that share a depth-``a`` ancestor of one row and a
depth-``b`` ancestor of another, given CSR views of the per-anchor
descendant lists.

This is the first ``parallel=True`` numba kernel in the package; the
kinship DP in :mod:`_kinship_kernel` is serial.
"""

__all__ = ["_enumerate_pairs_kernel"]

import numpy as np
from numba import njit, prange


@njit(parallel=True, cache=True, boundscheck=False)
def _enumerate_pairs_kernel(
    indptr_a: np.ndarray,
    indices_a: np.ndarray,
    indptr_b: np.ndarray,
    indices_b: np.ndarray,
    n: int,
    symmetric: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Two-pass parallel pair enumeration over per-anchor descendant lists.

    Pass 1: count output pairs per anchor X. Pass 2: write pairs into the
    pre-allocated flat ``(out_i, out_j)`` arrays at the cumulative offset for
    that X. ``prange`` gives true thread-parallelism (no GIL); each iteration
    writes only to its own contiguous output slice.

    Args:
        indptr_a: CSR indptr for the depth-a transposed ancestor matrix
            (rows = anchor X, cols = descendants at depth a).
        indices_a: CSR indices for the depth-a transposed ancestor matrix.
        indptr_b: CSR indptr for the depth-b transposed ancestor matrix.
            Ignored when ``symmetric`` is True.
        indices_b: CSR indices for the depth-b transposed ancestor matrix.
            Ignored when ``symmetric`` is True.
        n: Number of anchors (rows of the transposed matrices).
        symmetric: If True, the depth-a and depth-b CSR views are the
            same and we emit only the upper triangle ``(i < j)``;
            otherwise we emit the full cross product.
    """
    sizes = np.empty(n, dtype=np.int64)
    for X in prange(n):
        len_a = indptr_a[X + 1] - indptr_a[X]
        if symmetric:
            sizes[X] = len_a * (len_a - 1) // 2 if len_a >= 2 else 0
        else:
            len_b = indptr_b[X + 1] - indptr_b[X]
            sizes[X] = len_a * len_b

    offsets = np.empty(n + 1, dtype=np.int64)
    offsets[0] = 0
    for X in range(n):
        offsets[X + 1] = offsets[X] + sizes[X]
    total = offsets[n]

    out_i = np.empty(total, dtype=np.int64)
    out_j = np.empty(total, dtype=np.int64)

    for X in prange(n):
        offset = offsets[X]
        a_start = indptr_a[X]
        a_end = indptr_a[X + 1]
        if symmetric:
            len_a = a_end - a_start
            pos = 0
            for ii in range(len_a):
                src_i = indices_a[a_start + ii]
                for jj in range(ii + 1, len_a):
                    out_i[offset + pos] = src_i
                    out_j[offset + pos] = indices_a[a_start + jj]
                    pos += 1
        else:
            b_start = indptr_b[X]
            b_end = indptr_b[X + 1]
            pos = 0
            for ii in range(a_start, a_end):
                src_i = indices_a[ii]
                for jj in range(b_start, b_end):
                    out_i[offset + pos] = src_i
                    out_j[offset + pos] = indices_b[jj]
                    pos += 1

    return out_i, out_j
