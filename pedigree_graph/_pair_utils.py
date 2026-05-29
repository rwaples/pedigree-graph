"""Stateless pair-array utilities shared by the relationship engines.

Pure functions over index arrays and sparse matrices — no ``PedigreeGraph``
state.  The matrix pair extractor and the streaming counter both build on
these; centralising them keeps the canonical ``(lo, hi)`` ordering and the
graph-space → caller-space coordinate conversion (see PGQ-001) in one place
rather than re-implemented per engine.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import scipy.sparse as sp

__all__ = [
    "dedup_pairs",
    "extract_from_sparse",
    "pairs_from_groups",
    "remap_pairs_to_caller",
]


def dedup_pairs(a_i: np.ndarray, a_j: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Canonicalize (lo, hi) and deduplicate pair arrays via int64 keys."""
    if len(a_i) == 0:
        return np.array([], dtype=np.intp), np.array([], dtype=np.intp)
    lo = np.minimum(a_i, a_j).astype(np.intp)
    hi = np.maximum(a_i, a_j).astype(np.intp)
    max_id = int(hi.max()) + 1
    keys = lo.astype(np.int64) * max_id + hi.astype(np.int64)
    _, unique_idx = np.unique(keys, return_index=True)
    return lo[unique_idx], hi[unique_idx]


def extract_from_sparse(
    M: sp.spmatrix,
    subtract: list[tuple[np.ndarray, np.ndarray]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract nonzero pairs from sparse matrix, dedup, and subtract closer pairs.

    Mutates *M* in place (zeroes diagonal). Callers should not reuse *M*.
    All subtract pairs are batched into a single ``np.isin`` call.
    """
    M.setdiag(0)
    M.eliminate_zeros()
    if M.nnz == 0:
        return np.array([], dtype=np.intp), np.array([], dtype=np.intp)

    a_i, a_j = M.nonzero()
    lo, hi = dedup_pairs(a_i, a_j)

    if subtract and len(lo) > 0:
        # Collect all subtract pairs into one key set, then filter once
        rm_lo_parts: list[np.ndarray] = []
        rm_hi_parts: list[np.ndarray] = []
        for rm_pair in subtract:
            if len(rm_pair[0]) > 0:
                r1, r2 = rm_pair
                rm_lo_parts.append(np.minimum(r1, r2))
                rm_hi_parts.append(np.maximum(r1, r2))
        if rm_lo_parts:
            all_rm_lo = np.concatenate(rm_lo_parts)
            all_rm_hi = np.concatenate(rm_hi_parts)
            max_id = int(max(lo.max(), hi.max(), all_rm_lo.max(), all_rm_hi.max())) + 1
            rm_keys = all_rm_lo.astype(np.int64) * max_id + all_rm_hi.astype(np.int64)
            cand_keys = lo.astype(np.int64) * max_id + hi.astype(np.int64)
            keep = ~np.isin(cand_keys, rm_keys)
            lo, hi = lo[keep], hi[keep]
    return lo, hi


def pairs_from_groups(indices: np.ndarray, group_key: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Generate all (i < j) pairs of indices within each group.

    Uses batch-by-size triu_indices for vectorized pair generation.
    """
    sort_idx = np.argsort(group_key, kind="mergesort")
    sorted_keys = group_key[sort_idx]
    sorted_indices = indices[sort_idx]

    _, starts, counts = np.unique(sorted_keys, return_index=True, return_counts=True)

    multi = counts >= 2
    starts = starts[multi]
    counts = counts[multi]

    if len(starts) == 0:
        return np.array([], dtype=np.intp), np.array([], dtype=np.intp)

    pair_i_parts = []
    pair_j_parts = []
    for size in np.unique(counts):
        gs = starts[counts == size]
        ii, jj = np.triu_indices(size, k=1)
        all_i = (gs[:, np.newaxis] + ii[np.newaxis, :]).ravel()
        all_j = (gs[:, np.newaxis] + jj[np.newaxis, :]).ravel()
        pair_i_parts.append(sorted_indices[all_i])
        pair_j_parts.append(sorted_indices[all_j])

    p1 = np.concatenate(pair_i_parts)
    p2 = np.concatenate(pair_j_parts)

    lo = np.minimum(p1, p2)
    hi = np.maximum(p1, p2)
    return lo.astype(np.intp), hi.astype(np.intp)


def remap_pairs_to_caller(
    pairs: dict[str, tuple[np.ndarray, np.ndarray]],
    remap: np.ndarray,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Convert pair indices from graph-space to caller-space.

    *remap* is the graph-row → caller-row table (``pg._subsample_remap``).
    The remap can permute rows, so each pair is re-canonicalized to
    preserve the ``lo < hi`` invariant that downstream pair-key encoders
    rely on.  Mutates and returns *pairs*.  See PGQ-001.
    """
    for k, (idx1, idx2) in pairs.items():
        if len(idx1) > 0:
            r1 = remap[idx1].astype(np.intp)
            r2 = remap[idx2].astype(np.intp)
            pairs[k] = (np.minimum(r1, r2), np.maximum(r1, r2))
    return pairs
