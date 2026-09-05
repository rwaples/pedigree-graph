"""Stateless pair-array utilities shared by the relationship engines.

Pure functions over index arrays and sparse matrices — no ``PedigreeGraph``
state.  The matrix pair extractor and the BFS engine both build on these, so
the canonical unordered key ``min * n + max`` (ADR 0006 pair contracts 3 and
6), the oriented read of an asymmetric product matrix, and the graph-space →
caller-space conversion of the 0.7.1 adapter live in one place.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import scipy.sparse as sp

__all__ = [
    "canonical_keys",
    "dedup_pairs",
    "oriented_pairs_from_sparse",
    "pairs_from_groups",
    "remap_pairs_to_caller",
    "sort_by_canonical_key",
    "subtract_pairs",
]

_PairArrays = tuple[np.ndarray, np.ndarray]


def canonical_keys(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    """Return the canonical unordered int64 key ``min(a, b) * n + max(a, b)`` per pair.

    Args:
        a: First member of each pair, any orientation.
        b: Second member of each pair, any orientation.
        n: Key base, at least ``max(a, b) + 1``; ``n`` is bounded by the int32
            row capacity so the product fits int64.

    Returns:
        One int64 key per pair, equal for the two orientations of a pair.
    """
    return np.minimum(a, b).astype(np.int64) * n + np.maximum(a, b).astype(np.int64)


def sort_by_canonical_key(a: np.ndarray, b: np.ndarray, n: int) -> _PairArrays:
    """Return ``(a, b)`` reordered by :func:`canonical_keys`, orientation kept."""
    if len(a) == 0:
        return a, b
    order = np.argsort(canonical_keys(a, b, n), kind="stable")
    return a[order], b[order]


def subtract_pairs(keep: _PairArrays, remove: list[_PairArrays]) -> _PairArrays:
    """Drop from *keep* every unordered pair that occurs in any of *remove*.

    Membership is decided on the canonical unordered key ``min * m + max``, so
    the inputs may be in any orientation and *keep* comes back in the
    orientation it arrived in (ADR 0006 pair contract 3).

    Args:
        keep: ``(a, b)`` candidate pair arrays.
        remove: Pair arrays whose unordered pairs are dropped from *keep*.

    Returns:
        The surviving ``(a, b)`` rows of *keep*, order preserved.
    """
    a, b = keep
    parts = [pair for pair in remove if len(pair[0]) > 0]
    if len(a) == 0 or not parts:
        return keep
    rm_a = np.concatenate([pair[0] for pair in parts])
    rm_b = np.concatenate([pair[1] for pair in parts])
    m = int(max(a.max(), b.max(), rm_a.max(), rm_b.max())) + 1
    # Membership only: sorting the raw remove keys plus searchsorted beats
    # np.unique + np.isin at scale, and duplicate remove keys are harmless.
    rm_keys = np.sort(canonical_keys(rm_a, rm_b, m))
    keys = canonical_keys(a, b, m)
    pos = np.searchsorted(rm_keys, keys)
    hit = pos < rm_keys.size
    hit[hit] = rm_keys[pos[hit]] == keys[hit]
    return a[~hit], b[~hit]


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


def oriented_pairs_from_sparse(
    M: sp.spmatrix,
    *,
    row_is_first: bool,
    subtract: list[_PairArrays] | None = None,
) -> _PairArrays:
    """Read an asymmetric relationship product as oriented ``(first, second)`` pairs.

    Each nonzero ``M[r, c]`` is one pair; *row_is_first* says which side of
    the product holds the ``first`` role.  A pair valid in both orientations
    (both ``M[a, b]`` and ``M[b, a]`` nonzero, through different paths of an
    inbred pedigree) is kept once with the lower row as ``first`` (ADR 0006
    pair contract 5).  Mutates *M* in place (zeroes the diagonal).

    Args:
        M: Square sparse product whose nonzeros are the candidate pairs.
        row_is_first: ``True`` when the row index carries the ``first`` role.
        subtract: Closer-category pairs to drop, in any orientation.

    Returns:
        Oriented intp ``(first, second)`` arrays, one entry per unordered
        pair, sorted by canonical key.
    """
    M.setdiag(0)  # ty: ignore[unresolved-attribute]
    M.eliminate_zeros()  # ty: ignore[unresolved-attribute]
    if M.nnz == 0:  # ty: ignore[unresolved-attribute]
        return np.array([], dtype=np.intp), np.array([], dtype=np.intp)
    rows, cols = M.nonzero()  # ty: ignore[unresolved-attribute]
    first, second = (rows, cols) if row_is_first else (cols, rows)
    first = first.astype(np.intp)
    second = second.astype(np.intp)
    keys = canonical_keys(first, second, M.shape[0])
    order = np.lexsort((first, keys))
    sorted_keys = keys[order]
    unique = np.ones(order.size, dtype=bool)
    unique[1:] = sorted_keys[1:] != sorted_keys[:-1]
    kept = order[unique]
    first, second = first[kept], second[kept]
    if subtract:
        first, second = subtract_pairs((first, second), subtract)
    return first, second


def pairs_from_groups(indices: np.ndarray, group_key: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Generate all (i < j) pairs of indices within each group.

    Uses batch-by-size triu_indices for vectorized pair generation.
    """
    if len(indices) == 0:
        return np.array([], dtype=np.intp), np.array([], dtype=np.intp)

    sort_idx = np.argsort(group_key, kind="mergesort")
    sorted_keys = group_key[sort_idx]
    sorted_indices = indices[sort_idx]

    # sorted_keys is already sorted; diff-based run detection avoids
    # np.unique re-sorting/hashing the array it was just handed.
    starts = np.concatenate(([0], np.flatnonzero(sorted_keys[1:] != sorted_keys[:-1]) + 1))
    counts = np.diff(np.append(starts, len(sorted_keys)))

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


# 0.8.0-DELETE: only the from_subsample adapter remaps; views replace it (ADR 0006).
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
