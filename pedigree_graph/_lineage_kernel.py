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


@numba.njit(cache=True)
def _compute_n_descendants(
    m_idx: np.ndarray,
    f_idx: np.ndarray,
    n: int,
) -> np.ndarray:
    """Per-row descendant *path* count.

    Topological order is required (parents precede children in row
    index).  Public rows may arrive in any acyclic order, so
    ``PedigreeGraph.descendant_path_counts`` runs this on the parent
    arrays remapped by :mod:`pedigree_graph._topology` and maps the
    result back.  We iterate row indices in reverse, pushing each row's
    contribution up to its parents.  When we visit i, every descendant
    of i has a higher row index and was therefore visited first, so
    ``n_desc[i]`` is final.

    Returns an ``int64`` array — path counts can exceed unique-descendant
    counts on inbred / loop-heavy pedigrees, so the int32 downcast is
    deferred to the public method which checks for overflow.
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
    return n_desc


@numba.njit(cache=True, inline="always")
def _ancestor_slot_capacity(required: np.int32) -> np.int32:
    """Smallest power-of-two capacity that can hold ``required`` entries."""
    capacity = np.int32(1)
    while capacity < required:
        capacity *= np.int32(2)
    return capacity


@numba.njit(cache=True, inline="always")
def _ancestor_slot_bucket(capacity: np.int32) -> np.int32:
    """Zero-based free-list bucket for a power-of-two capacity."""
    bucket = np.int32(0)
    while capacity > 1:
        capacity //= np.int32(2)
        bucket += np.int32(1)
    return bucket


@numba.njit(cache=True)
def _ancestor_union_length(
    pool: np.ndarray,
    first_start: np.int64,
    first_length: np.int32,
    second_start: np.int64,
    second_length: np.int32,
) -> np.int32:
    """Length of the sorted unique union of two pool slices."""
    i = np.int32(0)
    j = np.int32(0)
    length = np.int32(0)
    while i < first_length and j < second_length:
        first = pool[first_start + i]
        second = pool[second_start + j]
        if first < second:
            i += np.int32(1)
        elif second < first:
            j += np.int32(1)
        else:
            i += np.int32(1)
            j += np.int32(1)
        length += np.int32(1)
    return length + (first_length - i) + (second_length - j)


@numba.njit(cache=True)
def _write_ancestor_union(
    pool: np.ndarray,
    destination: np.int64,
    first_start: np.int64,
    first_length: np.int32,
    second_start: np.int64,
    second_length: np.int32,
) -> np.int32:
    """Write the sorted unique union of two pool slices."""
    i = np.int32(0)
    j = np.int32(0)
    written = np.int32(0)
    while i < first_length and j < second_length:
        first = pool[first_start + i]
        second = pool[second_start + j]
        if first < second:
            value = first
            i += np.int32(1)
        elif second < first:
            value = second
            j += np.int32(1)
        else:
            value = first
            i += np.int32(1)
            j += np.int32(1)
        pool[destination + written] = value
        written += np.int32(1)

    while i < first_length:
        pool[destination + written] = pool[first_start + i]
        i += np.int32(1)
        written += np.int32(1)
    while j < second_length:
        pool[destination + written] = pool[second_start + j]
        j += np.int32(1)
        written += np.int32(1)
    return written


@numba.njit(cache=True)
def _grow_ancestor_pool(pool: np.ndarray, required: np.int64) -> np.ndarray:
    """Grow the ancestor-set pool geometrically to at least ``required``."""
    new_capacity = pool.shape[0] * 2
    while new_capacity < required:
        new_capacity *= 2
    grown = np.empty(new_capacity, dtype=np.int32)
    grown[: pool.shape[0]] = pool
    return grown


@numba.njit(cache=True)
def _acquire_ancestor_slot(
    pool: np.ndarray,
    capacity: np.int32,
    next_alloc: np.int64,
    free_head: np.ndarray,
    free_next: np.ndarray,
    row_start: np.ndarray,
) -> tuple[np.ndarray, np.int64, np.int64, bool]:
    """Pop an exact-capacity slot, or append one to the global pool."""
    bucket = _ancestor_slot_bucket(capacity)
    node = free_head[bucket]
    if node >= 0:
        free_head[bucket] = free_next[node]
        return pool, next_alloc, row_start[node], True

    required = next_alloc + np.int64(capacity)
    if required > pool.shape[0]:
        pool = _grow_ancestor_pool(pool, required)
    start = next_alloc
    return pool, required, start, False


@numba.njit(cache=True, inline="always")
def _release_ancestor_slot(
    row: np.int32,
    capacity: np.int32,
    free_head: np.ndarray,
    free_next: np.ndarray,
) -> None:
    """Push a row's slot onto its exact-capacity free list."""
    bucket = _ancestor_slot_bucket(capacity)
    free_next[row] = free_head[bucket]
    free_head[bucket] = row


@numba.njit(cache=True)
def _compute_n_ancestors_profiled(
    m_idx: np.ndarray,
    f_idx: np.ndarray,
    n: int,
) -> tuple[np.ndarray, np.int64, np.int64, np.int64, np.int64, np.int64]:
    """Distinct ancestor counts via a retiring sorted-set DP.

    Input rows must be topological. Each live row owns a sorted closed ancestor
    set: its strict ancestors followed by itself. A row's slot is returned to
    an exact-capacity free list after its last represented child is processed.

    Returns counts followed by peak live slot capacity, pool high-water entries,
    final pool allocation, parent-set entries scanned, and reused-slot count.
    """
    counts = np.zeros(n, dtype=np.int32)
    if n == 0:
        zero = np.int64(0)
        return counts, zero, zero, zero, zero, zero

    n_children = np.zeros(n, dtype=np.int32)
    for i in range(n):
        mother = m_idx[i]
        if mother >= 0:
            n_children[mother] += np.int32(1)
        father = f_idx[i]
        if father >= 0:
            n_children[father] += np.int32(1)
    remaining_children = n_children.copy()

    row_start = np.full(n, -1, dtype=np.int64)
    row_length = np.zeros(n, dtype=np.int32)
    row_capacity = np.zeros(n, dtype=np.int32)
    free_next = np.full(n, -1, dtype=np.int32)

    n_buckets = 1
    largest_capacity = 1
    while largest_capacity < n:
        largest_capacity *= 2
        n_buckets += 1
    free_head = np.full(n_buckets, -1, dtype=np.int32)

    pool = np.empty(max(n, 16), dtype=np.int32)
    next_alloc = np.int64(0)
    live_capacity = np.int64(0)
    peak_live_capacity = np.int64(0)
    entries_scanned = np.int64(0)
    reused_slots = np.int64(0)

    for i_raw in range(n):
        i = np.int32(i_raw)
        mother = m_idx[i]
        father = f_idx[i]

        mother_start = np.int64(0)
        mother_length = np.int32(0)
        if mother >= 0:
            mother_start = row_start[mother]
            mother_length = row_length[mother]
        father_start = np.int64(0)
        father_length = np.int32(0)
        if father >= 0:
            father_start = row_start[father]
            father_length = row_length[father]

        union_length = _ancestor_union_length(
            pool,
            mother_start,
            mother_length,
            father_start,
            father_length,
        )
        counts[i] = union_length
        entries_scanned += np.int64(mother_length) + np.int64(father_length)

        if n_children[i] > 0:
            capacity = _ancestor_slot_capacity(union_length + np.int32(1))
            pool, next_alloc, start, reused = _acquire_ancestor_slot(
                pool,
                capacity,
                next_alloc,
                free_head,
                free_next,
                row_start,
            )
            if reused:
                reused_slots += np.int64(1)
            written = _write_ancestor_union(
                pool,
                start,
                mother_start,
                mother_length,
                father_start,
                father_length,
            )
            pool[start + written] = i
            row_start[i] = start
            row_length[i] = written + np.int32(1)
            row_capacity[i] = capacity
            entries_scanned += np.int64(mother_length) + np.int64(father_length)
            live_capacity += np.int64(capacity)
            if live_capacity > peak_live_capacity:
                peak_live_capacity = live_capacity

        if mother >= 0:
            remaining_children[mother] -= np.int32(1)
            if remaining_children[mother] == 0:
                _release_ancestor_slot(mother, row_capacity[mother], free_head, free_next)
                live_capacity -= np.int64(row_capacity[mother])
        if father >= 0:
            remaining_children[father] -= np.int32(1)
            if remaining_children[father] == 0:
                _release_ancestor_slot(father, row_capacity[father], free_head, free_next)
                live_capacity -= np.int64(row_capacity[father])

    return (
        counts,
        peak_live_capacity,
        next_alloc,
        np.int64(pool.shape[0]),
        entries_scanned,
        reused_slots,
    )


@numba.njit(cache=True)
def _compute_n_ancestors(
    m_idx: np.ndarray,
    f_idx: np.ndarray,
    n: int,
) -> np.ndarray:
    """Return distinct ancestor counts from the retiring sorted-set DP.

    Input rows must be topological. Storage for each sorted closed ancestor set
    is reused after the row's last represented child has been processed.
    """
    counts, _, _, _, _, _ = _compute_n_ancestors_profiled(m_idx, f_idx, n)
    return counts
