"""CSC assembly for the pedigree kinship matrix (PGQ-008).

Leaf module: the int32 overflow-checked indptr builder and the
full-symmetric CSC assembler that turns the DP's per-row storage into
``(indptr, indices, data)`` arrays.
"""

from __future__ import annotations

import numba
import numpy as np


@numba.njit(cache=True)
def _checked_int32_indptr_from_counts(col_counts: np.ndarray) -> np.ndarray:
    """Build an int32 CSC indptr, raising before prefix sums overflow."""
    n = col_counts.shape[0]
    total = np.int64(0)
    max_int32 = np.int64(2147483647)
    for j in range(n):
        total += np.int64(col_counts[j])
        if total > max_int32:
            raise OverflowError(
                "kinship matrix nnz exceeded int32 range -- rebuild with "
                "int64 CSC indices (open a pedigree-graph issue if you hit this)"
            )

    indptr = np.zeros(n + 1, dtype=np.int32)
    total = np.int64(0)
    for j in range(n):
        total += np.int64(col_counts[j])
        indptr[j + 1] = np.int32(total)
    return indptr


@numba.njit(cache=True)
def _assemble_csc(
    cols: np.ndarray,
    vals: np.ndarray,
    row_start: np.ndarray,
    row_count: np.ndarray,
):
    """Assemble the full-symmetric kinship CSC arrays from per-row DP storage.

    The DP stores each individual's kept columns (sorted ascending) in
    ``cols[row_start[i] : row_start[i] + row_count[i]]`` with the matching
    kinship in ``vals``.  The kinship matrix is symmetric, so row ``j``
    equals column ``j``: iterating the stored rows and emitting each
    entry ``(i, j)`` into CSC column ``j`` at row ``i`` reconstructs the
    full symmetric matrix.

    Indices/indptr are int32 to halve K's footprint vs int64 (≈ 1.65 GB
    saved at N=100K); counts and the prefix-sum run in int64 so the
    overflow guard (:func:`_checked_int32_indptr_from_counts`) fires
    before any int32 prefix write can wrap.  At G_ped=6 we see
    nnz ≈ 690 × N, so overflow only kicks in beyond N ≈ 3M.
    """
    n = row_start.shape[0]

    # First pass: count entries per column.
    col_counts = np.zeros(n, dtype=np.int64)
    for i in range(n):
        rs = row_start[i]
        rc = row_count[i]
        for p in range(rc):
            j = cols[rs + p]
            col_counts[j] += np.int64(1)

    indptr = _checked_int32_indptr_from_counts(col_counts)
    nnz = indptr[n]

    indices = np.empty(nnz, dtype=np.int32)
    values = np.empty(nnz, dtype=np.float32)

    # Second pass: fill.  Emit each stored entry (i, j) into CSC column j
    # at row i, advancing a per-column write cursor.  Because the outer
    # loop visits i monotonically, rows within each column come out
    # ascending — exactly CSC's required per-column ordering, so no
    # post-sort is needed.
    col_write = np.zeros(n, dtype=np.int32)
    for i in range(n):
        rs = row_start[i]
        rc = row_count[i]
        for p in range(rc):
            j = cols[rs + p]
            pos = indptr[j] + col_write[j]
            indices[pos] = i
            values[pos] = vals[rs + p]
            col_write[j] += 1

    return indptr, indices, values
