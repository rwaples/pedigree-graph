"""Slab allocator and free-list for the kinship DP row buffer (PGQ-008).

Leaf module: the growable global arena (``_grow_global``), the
size-bucketed free list, the per-row slot/entry append helpers, and the
depth-based row retirement + per-row column sort.  Consumed by the DP
kernel (``_kinship_dp``) and the Meuwissen-Luo walk (``_inbreeding_kernel``).
"""

from __future__ import annotations

from typing import NamedTuple

import numba
import numpy as np

INIT_CAP_PER_ROW = 16


def _suggest_init_cap_per_row(g_ped: int) -> int:
    """Heuristic: bigger initial row capacity when the pedigree is deep.

    The DP per-row buffer doubles every time a row exhausts its slot,
    abandoning the old slot.  At G_ped=6 with random mating, observed row
    sizes hit ~690, so a row starting at 16 entries goes through 6
    doublings — accumulating ``16 + 32 + 64 + ... + 512 = 1008`` dead
    entries per row before reaching its final size of 1024.

    Picking ``init_cap ≈ 2 ** (G_ped + 4)`` (capped at 4096) lets typical
    rows fit in the initial allocation, eliminating most per-row
    relocations and the dead-space buildup.  Tiny pedigrees fall back to
    the 16 floor.
    """
    return max(16, min(4096, 1 << (g_ped + 4)))


@numba.njit(cache=True)
def _grow_global(
    cols: np.ndarray,
    vals: np.ndarray,
    min_size: int,
    grow_stats: np.ndarray,
):
    """Return larger arrays copying existing contents, growing geometrically.

    ``grow_stats`` is a length-3 int64 array used for benchmarking the
    doubling cascade.  Slot 0: call count.  Slot 1: total entries
    copied (summed ``old_len``).  Slot 2: peak allocated entries
    (``max(new_len)``).  Writes are three int64 stores per call, kept
    in nopython mode; production callers can pass a freshly-zeroed
    placeholder and ignore it.
    """
    old_len = cols.shape[0]
    new_len = old_len * 2
    while new_len < min_size:
        new_len *= 2
    new_cols = np.full(new_len, -1, dtype=np.int32)
    new_cols[:old_len] = cols
    new_vals = np.zeros(new_len, dtype=np.float32)
    new_vals[:old_len] = vals
    grow_stats[0] += np.int64(1)
    grow_stats[1] += np.int64(old_len)
    if np.int64(new_len) > grow_stats[2]:
        grow_stats[2] = np.int64(new_len)
    return new_cols, new_vals


class _FreelistBuffers(NamedTuple):
    """Bundle of slab-allocator support state passed through the DP hot loop.

    The kinship DP threads four arrays/scalars through every
    ``_append_entry``, ``_acquire_slot``, and ``_freelist_push`` call —
    the same four every time.  Bundling them into a NamedTuple
    eliminates 4 args at each call site in ``_dp_kinship`` without
    changing what numba sees (NamedTuple fields lower to fixed-offset
    loads under ``inline="always"`` callees).

    Fields:
        freelist_starts: (n_buckets, max_per_bucket) int64 LIFO of slot
            offsets, one bucket per row-size (powers of two of
            init_cap_per_row).
        freelist_tops: per-bucket stack head (int32).
        fl_init_cap: power-of-two anchor used for bucket sizing
            (``init_cap_per_row``); ``0`` under ``retire=False`` to make
            push/pop silent no-ops.
        grow_stats: 3-element int64 array
            ``[grow_call_count, total_entries_copied, peak_allocated]``;
            bench-only telemetry.
    """

    freelist_starts: np.ndarray
    freelist_tops: np.ndarray
    fl_init_cap: np.int32
    grow_stats: np.ndarray


def _freelist_alloc(init_cap_per_row: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Pre-allocate the size-bucketed free-list backing arrays.

    Returns ``(starts, tops)`` where ``starts`` is int64
    ``(n_buckets, n)`` of slot start offsets and ``tops`` is int32
    ``(n_buckets,)`` of per-bucket stack heights (top of stack = next
    write index).  ``max_per_bucket = n`` is the loose worst case
    (every row freed at one cap level); tightening this is a follow-up.

    For the no-retirement path callers may pass ``init_cap_per_row=0``
    or ``n=0`` — both yield 1×1 placeholder arrays that quietly absorb
    push/pop attempts.
    """
    cap = max(1, int(init_cap_per_row))
    n_eff = max(1, int(n))
    # ceil(log2(n / cap)) + 1, with a floor of 1 bucket for tiny pedigrees.
    n_buckets = 1
    v = cap
    while v < n_eff:
        v *= 2
        n_buckets += 1
    starts = np.zeros((n_buckets, n_eff), dtype=np.int64)
    tops = np.zeros(n_buckets, dtype=np.int32)
    return starts, tops


@numba.njit(cache=True)
def _freelist_bucket(cap: np.int32, init_cap_per_row: np.int32) -> np.int32:
    """Return ``floor(log2(cap / init_cap_per_row))`` for power-of-two caps.

    Caps grow only via doubling from ``init_cap_per_row``, so this is an
    integer log2.  Returns 0 for cap == init_cap and clips to 0 below.
    """
    if cap <= init_cap_per_row:
        return np.int32(0)
    b = np.int32(0)
    v = np.int32(init_cap_per_row)
    while v < cap:
        v *= np.int32(2)
        b += np.int32(1)
    return b


@numba.njit(cache=True)
def _freelist_push(
    freelist_starts: np.ndarray,
    freelist_tops: np.ndarray,
    start: np.int64,
    cap: np.int32,
    init_cap_per_row: np.int32,
) -> None:
    """Push ``(start, cap)`` onto its bucket's LIFO stack.

    Silently no-ops when the free list is a placeholder (no buckets, or
    cap out of range, or bucket full) — callers under ``retire=False``
    pass placeholder arrays and rely on this to be inert.
    """
    if init_cap_per_row <= 0 or cap <= 0:
        return
    b = _freelist_bucket(cap, init_cap_per_row)
    if b >= freelist_starts.shape[0]:
        return
    top = freelist_tops[b]
    if top >= freelist_starts.shape[1]:
        # Bucket sizing assumes worst case, so this branch should not
        # fire under documented inputs.  Silent no-op for safety.
        return
    freelist_starts[b, top] = start
    freelist_tops[b] = top + np.int32(1)


@numba.njit(cache=True)
def _freelist_pop(
    freelist_starts: np.ndarray,
    freelist_tops: np.ndarray,
    needed_cap: np.int32,
    init_cap_per_row: np.int32,
) -> np.int64:
    """Pop a slot matching ``needed_cap`` from its bucket; LIFO.

    Returns the slot's start offset, or ``-1`` when the matching bucket
    is empty (callers fall back to ``next_alloc``).  Placeholder
    free-lists always return ``-1``.
    """
    if init_cap_per_row <= 0 or needed_cap <= 0:
        return np.int64(-1)
    b = _freelist_bucket(needed_cap, init_cap_per_row)
    if b >= freelist_starts.shape[0]:
        return np.int64(-1)
    top = freelist_tops[b]
    if top == 0:
        return np.int64(-1)
    new_top = top - np.int32(1)
    start = freelist_starts[b, new_top]
    freelist_tops[b] = new_top
    return start


@numba.njit(cache=True, inline="always")
def _acquire_slot(
    cap: np.int32,
    cols: np.ndarray,
    vals: np.ndarray,
    next_alloc: np.int64,
    freelist_starts: np.ndarray,
    freelist_tops: np.ndarray,
    fl_init_cap: np.int32,
    grow_stats: np.ndarray,
):
    """Acquire a slot of capacity ``cap`` for a row.

    Tries the size-bucketed free list first; falls back to bumping
    ``next_alloc`` and growing the global buffer if needed.  Shared
    by ``_append_entry``'s lazy-allocate and relocation branches —
    both want the same "allocate-or-pop" pattern.

    Returns ``(cols, vals, next_alloc, dest)``: ``cols``/``vals`` may
    have been reallocated by ``_grow_global``; ``dest`` is the offset
    of the acquired slot.
    """
    reused = _freelist_pop(freelist_starts, freelist_tops, cap, fl_init_cap)
    if reused >= 0:
        return cols, vals, next_alloc, np.int64(reused)
    needed = next_alloc + np.int64(cap)
    if needed > cols.shape[0]:
        cols, vals = _grow_global(cols, vals, needed, grow_stats)
    dest = next_alloc
    next_alloc = next_alloc + np.int64(cap)
    return cols, vals, next_alloc, dest


@numba.njit(cache=True, inline="always")
def _append_entry(
    cols: np.ndarray,
    vals: np.ndarray,
    row_start: np.ndarray,
    row_count: np.ndarray,
    row_cap: np.ndarray,
    next_alloc: np.int64,
    row_idx: np.int32,
    col_idx: np.int32,
    val: np.float32,
    buffers,
):
    """Append (col_idx, val) to row_idx.

    Three row states:

    * **live**: ``row_start[i] >= 0`` and ``row_cap[i] > 0`` — append into
      the existing slot, relocating with doubled capacity if full.
    * **never-allocated** (lazy alloc only): ``row_start[i] < 0`` and
      ``row_cap[i] > 0``. ``row_cap[i]`` holds the desired starting
      capacity; the first append here pops a free-list slot of that
      cap or bumps ``next_alloc``.
    * **retired**: ``row_start[i] < 0`` and ``row_cap[i] == 0`` —
      silently drop the write.

    If a live row's capacity is exhausted, relocate to a new segment
    with doubled capacity.  Under retirement, the freed old slot is
    pushed onto the free list and the new larger slot is taken from
    the free list if a matching bucket has a vacant entry — otherwise
    ``next_alloc`` is bumped.  Returns ``(cols, vals, next_alloc)``;
    ``cols``/``vals`` may be reallocated by ``_grow_global``.

    ``buffers`` is a :class:`_FreelistBuffers` carrying the four
    slab-allocator arrays/scalars; under ``retire=False`` it holds
    placeholder arrays and ``fl_init_cap=0``, which makes push/pop
    silent no-ops.  The eager init pre-sets
    ``row_start[i] = i * init_cap`` for all rows, so the
    never-allocated branch never fires under retire=False.
    """
    if row_start[row_idx] < 0:
        if row_cap[row_idx] == 0:
            # Retired — silently drop the write.
            return cols, vals, next_alloc
        # Never-allocated: lazy-allocate a slot of cap = row_cap[row_idx].
        cols, vals, next_alloc, dest = _acquire_slot(
            row_cap[row_idx],
            cols,
            vals,
            next_alloc,
            buffers.freelist_starts,
            buffers.freelist_tops,
            buffers.fl_init_cap,
            buffers.grow_stats,
        )
        row_start[row_idx] = dest
        # row_count stays 0; row_cap unchanged; fall through to append.

    if row_count[row_idx] >= row_cap[row_idx]:
        # Row is full — relocate with doubled capacity.
        new_cap = np.int32(row_cap[row_idx] * np.int32(2))
        # Push the about-to-be-abandoned slot.  No-op under retire=False.
        _freelist_push(
            buffers.freelist_starts,
            buffers.freelist_tops,
            row_start[row_idx],
            row_cap[row_idx],
            buffers.fl_init_cap,
        )
        cols, vals, next_alloc, dest = _acquire_slot(
            new_cap,
            cols,
            vals,
            next_alloc,
            buffers.freelist_starts,
            buffers.freelist_tops,
            buffers.fl_init_cap,
            buffers.grow_stats,
        )
        # Copy existing data to the new slot.
        src = row_start[row_idx]
        cnt = row_count[row_idx]
        for k in range(cnt):
            cols[dest + k] = cols[src + k]
            vals[dest + k] = vals[src + k]
        row_start[row_idx] = dest
        row_cap[row_idx] = new_cap
    # Append.
    pos = row_start[row_idx] + row_count[row_idx]
    cols[pos] = col_idx
    vals[pos] = val
    row_count[row_idx] += 1
    return cols, vals, next_alloc


@numba.njit(cache=True)
def _retire_rows_at_depth(
    d: np.int32,
    last_dcd: np.ndarray,
    row_start: np.ndarray,
    row_count: np.ndarray,
    row_cap: np.ndarray,
    freelist_starts: np.ndarray,
    freelist_tops: np.ndarray,
    fl_init_cap: np.int32,
) -> None:
    """Free every row whose last-direct-child depth equals ``d``.

    Pushes the slot onto the size-bucketed free list and marks the row
    retired via ``row_start = -1`` so subsequent symmetric appends from
    descendant chains land in ``_append_entry``'s sentinel branch and
    dissolve. Under lazy allocation a row may reach retirement having
    never been allocated (``row_start = -1``); in that case there is
    no slot to push, but the row still transitions into the retired
    sentinel state so any later writes silently drop.
    """
    for rk in range(last_dcd.shape[0]):
        if last_dcd[rk] == d:
            if row_start[rk] >= 0 and row_cap[rk] > 0:
                _freelist_push(
                    freelist_starts,
                    freelist_tops,
                    row_start[rk],
                    row_cap[rk],
                    fl_init_cap,
                )
            row_start[rk] = np.int64(-1)
            row_count[rk] = np.int32(0)
            row_cap[rk] = np.int32(0)


@numba.njit(cache=True)
def _sort_row_inplace(cols: np.ndarray, vals: np.ndarray, start: int, count: int):
    """Insertion sort on a single row's slice (for the rare twin fixup)."""
    for i in range(start + 1, start + count):
        kc = cols[i]
        kv = vals[i]
        j = i - 1
        while j >= start and cols[j] > kc:
            cols[j + 1] = cols[j]
            vals[j + 1] = vals[j]
            j -= 1
        cols[j + 1] = kc
        vals[j + 1] = kv
