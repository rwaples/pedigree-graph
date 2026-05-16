"""Shared numba kinship kernel for :class:`~simace.core.pedigree_graph.PedigreeGraph`.

Holds the gen-by-gen DP that computes pedigree kinship (kinship2-style
recursion) together with the CSC assembly pass.  Exposes
:func:`_build_kinship_csc` — the single internal entry point used by
:meth:`PedigreeGraph.kinship_matrix`.

The kernel is moved (not rewritten) from
``fitACE/fitace/pafgrs/kinship_fast.py``.  It is intentionally agnostic
to pandas: all inputs are numpy arrays remapped to 0..n-1 row indices.
"""

__all__ = [
    "_assemble_csc",
    "_build_kinship_csc",
    "_checked_int32_indptr_from_counts",
    "_check_topological",
    "_compute_F_meuwissen_luo",
    "_compute_depth",
    "_compute_eqg",
    "_compute_last_direct_child_depth",
    "_compute_theta_per_gen",
    "_dp_kinship",
    "_per_gen_mean_kinship_from_dp",
]

from typing import NamedTuple

import numba
import numpy as np

# ---------------------------------------------------------------------------
# Depth + DP kernels (numba)
# ---------------------------------------------------------------------------


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


# -- DP storage: flat global buffer with per-row (start, count, cap) tracking.
#
# Each row begins with capacity ``INIT_CAP_PER_ROW``.  When a row overflows,
# its data is copied to the end of the global buffer and its capacity
# doubles.  The global buffer itself grows geometrically when
# ``next_alloc`` exceeds its length.
#
# Because numba doesn't allow in-place array resize, we use a
# "capacity probe" helper that returns fresh arrays if growth is needed.


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


# ---------------------------------------------------------------------------
# Free-list for retired/relocated row slots (size-bucketed LIFO stacks)
# ---------------------------------------------------------------------------
#
# When retirement is enabled, end-of-depth retirement pushes freed
# ``(start, cap)`` slots onto a size-bucketed stack.  Subsequent
# relocations and overflow allocations pop from the matching bucket
# before advancing ``next_alloc``.  Bucket index = log2(cap / init_cap);
# caps are bounded above by ``n`` (a row can't hold more entries than
# individuals), so ``n_buckets = ceil(log2(n / init_cap_per_row)) + 1``
# covers every reachable size.  No overflow bucket is needed.


def _freelist_alloc(
    init_cap_per_row: int, n: int
) -> tuple[np.ndarray, np.ndarray]:
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
            row_cap[row_idx], cols, vals, next_alloc,
            buffers.freelist_starts, buffers.freelist_tops,
            buffers.fl_init_cap, buffers.grow_stats,
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
            new_cap, cols, vals, next_alloc,
            buffers.freelist_starts, buffers.freelist_tops,
            buffers.fl_init_cap, buffers.grow_stats,
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
                    freelist_starts, freelist_tops,
                    row_start[rk], row_cap[rk], fl_init_cap,
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


@numba.njit(cache=True)
def _dp_kinship(
    n: int,
    m_idx: np.ndarray,
    f_idx: np.ndarray,
    tw_idx: np.ndarray,
    depth: np.ndarray,
    threshold: float,
    init_cap_per_row: int,
    retire: bool,
    lazy: bool,
    debug_asserts: bool,
    grow_stats: np.ndarray,
    initial_buffer_override: np.int64,
):
    """Build per-row sorted kinship arrays via gen-by-gen DP.

    When ``retire`` is True, rows are freed at end-of-depth once their
    ``last_direct_child_depth`` is reached, and θ̄ accumulates inline
    during the merge walk so the caller can finalize per-generation
    means without rescanning the row storage.  Subsequent symmetric
    writes targeting a retired row dissolve via the sentinel branch in
    ``_append_entry``.

    When ``lazy`` is True, rows defer allocation to the first write
    through ``_append_entry`` — see that function's three-state
    contract for the row-state machine.  This lets ``cols.shape[0]``
    track the working set rather than the static ``N × init_cap``
    floor.  Founders with no direct child at any later depth
    (``last_dcd[i] == depth[i]``) skip allocation entirely.
    ``lazy=True`` only makes sense paired with ``retire=True`` (the
    free list that lazy alloc draws from is populated by retirement);
    the invalid combination ``(retire=False, lazy=True)`` is rejected.

    The eager path (``lazy=False``) pre-allocates the row buffer up
    front and is used by the CSC assembly path through ``_run_dp``.

    When ``debug_asserts`` is True, each child-row step verifies its
    direct parents have not been retired — the regression gate for
    premature-retirement bugs.

    Returns:
        cols: int32[total_cap], flat col storage (per-row contiguous).
        vals: float32[total_cap], matching values (narrowed from float64
            for ~50 % memory reduction at large N; kinship values lie in
            [0, 1] so float32's 7-digit mantissa keeps relative error well
            below downstream slope-fit tolerances).
        row_start: int64[n], where each row begins in cols/vals.  int64
            because the flat buffer can exceed 2**31 entries at large N
            (e.g. N ≈ 525K with init_cap_per_row=4096).  Retired rows
            have ``row_start = -1``.
        row_count: int32[n], entries per row.
        sum_theta: float64[g_max + 1], inline within-cohort kinship sum
            per generation (only populated when ``retire=True``; under
            ``retire=False`` this is a length-1 placeholder that the
            caller discards).

    Rows are stored symmetrically (row i contains entries for cols j
    and vice versa).  Within each row, entries are sorted by column
    index (ascending).
    """
    # Reject (retire=False, lazy=True): the never-allocated → live
    # transition is fed by retirement pushing slots onto the free list;
    # without retirement, never-allocated rows would all bump
    # next_alloc on first write with no recycling, defeating the win
    # and corrupting the CSC contract (rows the caller expects to
    # iterate via row_start would remain at -1).
    if lazy and not retire:
        raise ValueError("_dp_kinship: lazy=True requires retire=True")

    # Global buffer — geometric growth.  ``init_cap_per_row`` may be
    # tuned upward to skip the doubling cascade when the caller knows
    # typical kinship row sizes (e.g. ``2 ** (G_ped + 4)`` ≈ 1024 at
    # G_ped=6); see :func:`_suggest_init_cap_per_row`.
    init_cap = np.int32(init_cap_per_row)
    max_depth = np.int32(depth.max()) if n > 0 else np.int32(0)

    # ``last_dcd`` is needed by lazy founder-skipping and by
    # end-of-depth retirement; compute once and share.  Under
    # (retire=False, lazy=False) it is unused but cheap (O(n)).
    last_dcd = _compute_last_direct_child_depth(m_idx, f_idx, depth, n)

    # Per-row state and initial flat buffer.
    if lazy:
        # Conservative starting size — _grow_global expands geometrically.
        # max(1<<16, init_cap*1024) entries ≈ 0.5 MB at init_cap=512.
        # ``initial_buffer_override > 0`` lets bench callers swap the
        # heuristic without recompiling the kernel.
        if initial_buffer_override > 0:
            initial_buffer = initial_buffer_override
        else:
            init_cap_i64 = np.int64(init_cap)
            baseline = np.int64(1 << 16)
            scaled = init_cap_i64 * np.int64(1024)
            initial_buffer = baseline if baseline > scaled else scaled
        cols = np.full(initial_buffer, -1, dtype=np.int32)
        vals = np.zeros(initial_buffer, dtype=np.float32)
        row_start = np.full(n, -1, dtype=np.int64)
        row_count = np.zeros(n, dtype=np.int32)
        row_cap = np.full(n, init_cap, dtype=np.int32)
        next_alloc = np.int64(0)
    else:
        total_cap = np.int64(n) * init_cap
        cols = np.full(total_cap, -1, dtype=np.int32)
        vals = np.zeros(total_cap, dtype=np.float32)
        row_start = np.zeros(n, dtype=np.int64)
        row_count = np.zeros(n, dtype=np.int32)
        row_cap = np.full(n, init_cap, dtype=np.int32)
        # Each row starts at position i * init_cap.
        for i in range(n):
            row_start[i] = np.int64(i) * np.int64(init_cap)
        next_alloc = np.int64(n) * init_cap

    # Retirement state.  Placeholders under retire=False satisfy numba's
    # type unifier; push/pop are no-ops because fl_init_cap = 0.
    if retire:
        sum_theta = np.zeros(max_depth + np.int32(1), dtype=np.float64)
        # Bucket sizing: caps are bounded above by n, so
        # n_buckets = ceil(log2(n / init_cap)) + 1.
        n_buckets = np.int32(1)
        v_b = np.int64(init_cap)
        n_int64 = np.int64(n)
        while v_b < n_int64:
            v_b *= np.int64(2)
            n_buckets += np.int32(1)
        max_per_bucket = n if n > 0 else 1
        freelist_starts = np.zeros((n_buckets, max_per_bucket), dtype=np.int64)
        freelist_tops = np.zeros(n_buckets, dtype=np.int32)
        fl_init_cap = init_cap
    else:
        sum_theta = np.zeros(1, dtype=np.float64)
        freelist_starts = np.zeros((1, 1), dtype=np.int64)
        freelist_tops = np.zeros(1, dtype=np.int32)
        fl_init_cap = np.int32(0)

    # Bundle the slab-allocator state once.  Every `_append_entry` call
    # in the DP loop (and the helpers it inlines) reads through these
    # four fields; threading them as a NamedTuple keeps the hot-loop
    # call sites short while still flattening to fixed-offset loads
    # under numba's `inline="always"`.
    buffers = _FreelistBuffers(
        freelist_starts, freelist_tops, fl_init_cap, grow_stats,
    )

    # Diagonal self-kinship for founders only (0.5 with no inbreeding).
    # For non-founders, the diagonal is appended AFTER the merge walk —
    # doing it upfront would break the sorted-row invariant because the
    # merge walk appends cols < j (ancestors have lower indices under
    # depth-first ID assignment), so position 0 must be the smallest col.
    # Unified founder init.  Under lazy alloc, never-needed founders
    # (no descendant ever reads their row) skip allocation entirely;
    # everyone else routes through _append_entry, which lazy-allocates
    # the slot via the never-allocated branch.  Under eager alloc, all
    # founders' slots already exist at ``row_start[i] = i * init_cap``,
    # so _append_entry takes the live-row path and writes directly.
    for i in range(n):
        if m_idx[i] < 0 and f_idx[i] < 0:
            if lazy and last_dcd[i] == depth[i]:
                continue  # never-needed: leave at never-allocated state
            cols, vals, next_alloc = _append_entry(
                cols, vals, row_start, row_count, row_cap, next_alloc,
                np.int32(i), np.int32(i), np.float32(0.5),
                buffers,
            )

    # Founders with no children at any later depth retire immediately;
    # their stored diagonal is never read by a merge walk.  Under lazy
    # alloc these were already skipped in init; the retire pass just
    # transitions them from never-allocated to retired-sentinel so any
    # stray descendant write would dissolve.
    if retire:
        _retire_rows_at_depth(
            np.int32(0), last_dcd, row_start, row_count, row_cap,
            freelist_starts, freelist_tops, fl_init_cap,
        )

    # DP: process in depth order.
    for d in range(1, max_depth + 1):
        for j in range(n):
            if depth[j] != d:
                continue
            m = m_idx[j]
            f = f_idx[j]
            if m < 0 and f < 0:
                continue  # disconnected founder; self-kinship already 0.5

            # Trips if ``last_direct_child_depth`` underestimates row
            # liveness — i.e. a row was retired while still needed.
            if debug_asserts:
                if m >= 0 and row_start[m] < 0:
                    raise AssertionError(
                        "mother row retired before child processed"
                    )
                if f >= 0 and row_start[f] < 0:
                    raise AssertionError(
                        "father row retired before child processed"
                    )

            # --- Self-kinship (inbreeding correction) ---
            km_f = np.float32(0.0)
            if m >= 0 and f >= 0:
                # Look up kinship(m, f) by scanning m's row for column f.
                ms = row_start[m]
                mc = row_count[m]
                # Binary search for f in cols[ms:ms+mc].
                lo = 0
                hi = mc
                while lo < hi:
                    mid = (lo + hi) // 2
                    if cols[ms + mid] < f:
                        lo = mid + 1
                    else:
                        hi = mid
                if lo < mc and cols[ms + lo] == f:
                    km_f = vals[ms + lo]
            # Note: diagonal (j, self_kin) is appended AFTER the merge
            # walk (see below).  Do not pre-populate row j here.

            # --- Merge walk through rel(m) ∪ rel(f) ---
            ms = row_start[m] if m >= 0 else np.int64(0)
            mc = row_count[m] if m >= 0 else np.int32(0)
            fs = row_start[f] if f >= 0 else np.int64(0)
            fc = row_count[f] if f >= 0 else np.int32(0)

            pm = 0
            pf = 0
            while pm < mc or pf < fc:
                k = np.int32(-1)
                mv = np.float32(0.0)
                fv = np.float32(0.0)
                if pm < mc and (pf == fc or cols[ms + pm] <= cols[fs + pf]):
                    if pf < fc and cols[fs + pf] == cols[ms + pm]:
                        k = cols[ms + pm]
                        mv = vals[ms + pm]
                        fv = vals[fs + pf]
                        pm += 1
                        pf += 1
                    else:
                        k = cols[ms + pm]
                        mv = vals[ms + pm]
                        pm += 1
                else:
                    k = cols[fs + pf]
                    fv = vals[fs + pf]
                    pf += 1
                if k == j:
                    continue
                val = np.float32((mv + fv) / 2.0)
                if val <= threshold:
                    continue
                # ``k < j`` counts each unordered within-cohort non-twin
                # pair exactly once; folding the sum into the merge walk
                # lets retirement free row[k] before any rescan.
                if retire and depth[k] == d and k != tw_idx[j] and k < j:
                    sum_theta[d] += np.float64(val)
                # Append (k, val) to row j.  Merge walk yields columns in
                # ascending order, so row j stays sorted.
                cols, vals, next_alloc = _append_entry(
                    cols, vals, row_start, row_count, row_cap, next_alloc,
                    np.int32(j), k, val, buffers,
                )
                # Symmetric fill: append (j, val) to row k.  Since j is
                # processed in depth order and higher j means later
                # processing (same gen IDs are contiguous), the appends
                # to row k come in ascending j order → row k stays
                # sorted.
                cols, vals, next_alloc = _append_entry(
                    cols, vals, row_start, row_count, row_cap, next_alloc,
                    k, np.int32(j), val, buffers,
                )

            # --- Append diagonal (j, self_kin) to row j AFTER merge walk.
            # All merge-walk entries have cols < j (ancestors), so j is
            # the largest column — row j stays sorted.
            self_kin = np.float32((1.0 + km_f) / 2.0)
            cols, vals, next_alloc = _append_entry(
                cols, vals, row_start, row_count, row_cap, next_alloc,
                np.int32(j), np.int32(j), self_kin, buffers,
            )

        # MZ twin pass for this generation.
        for j in range(n):
            if depth[j] != d:
                continue
            tw = tw_idx[j]
            if tw < 0 or tw == j:
                continue
            # kinship(j, tw) = self-kinship(j) — look up the diagonal via
            # binary search (position 0 is NOT the diagonal; merge-walk
            # appends ancestor entries first, diagonal ends up sorted
            # according to its column index = j).
            rs_j0 = row_start[j]
            rc_j0 = row_count[j]
            self_k = np.float32(0.5)  # fallback if not found (shouldn't happen)
            lo_j = 0
            hi_j = rc_j0
            while lo_j < hi_j:
                mid = (lo_j + hi_j) // 2
                if cols[rs_j0 + mid] < j:
                    lo_j = mid + 1
                else:
                    hi_j = mid
            if lo_j < rc_j0 and cols[rs_j0 + lo_j] == j:
                self_k = vals[rs_j0 + lo_j]
            # Find insert position for tw in row j.
            rs_j = row_start[j]
            rc_j = row_count[j]
            lo = 0
            hi = rc_j
            while lo < hi:
                mid = (lo + hi) // 2
                if cols[rs_j + mid] < tw:
                    lo = mid + 1
                else:
                    hi = mid
            if lo < rc_j and cols[rs_j + lo] == tw:
                # Already present (shouldn't happen for fresh twins, but
                # be defensive).  Overwrite value.
                vals[rs_j + lo] = self_k
            else:
                # Need to insert in-place; falls back to append then
                # sort.  Only happens for twins so rare; cheap.
                cols, vals, next_alloc = _append_entry(
                    cols, vals, row_start, row_count, row_cap, next_alloc,
                    np.int32(j), np.int32(tw), self_k, buffers,
                )
                # Re-sort row j (bubble the new entry into place).  Small
                # per-row cost, rare.
                _sort_row_inplace(cols, vals, row_start[j], row_count[j])
            # Similarly for row tw.
            rs_t = row_start[tw]
            rc_t = row_count[tw]
            lo = 0
            hi = rc_t
            while lo < hi:
                mid = (lo + hi) // 2
                if cols[rs_t + mid] < j:
                    lo = mid + 1
                else:
                    hi = mid
            if lo < rc_t and cols[rs_t + lo] == j:
                vals[rs_t + lo] = self_k
            else:
                cols, vals, next_alloc = _append_entry(
                    cols, vals, row_start, row_count, row_cap, next_alloc,
                    np.int32(tw), np.int32(j), self_k, buffers,
                )
                _sort_row_inplace(cols, vals, row_start[tw], row_count[tw])

        # Runs AFTER the MZ twin pass so twin writes land before
        # retirement.  ``_grow_global`` preserves freed offsets safely
        # because retirement only releases rows that have completed all
        # writes at this depth — any pending grow happened earlier.
        if retire:
            _retire_rows_at_depth(
                np.int32(d), last_dcd, row_start, row_count, row_cap,
                freelist_starts, freelist_tops, fl_init_cap,
            )

    return cols, vals, row_start, row_count, sum_theta


# ---------------------------------------------------------------------------
# CSC assembly (numba)
# ---------------------------------------------------------------------------


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
    phen_pos: np.ndarray,
    to_grm: bool,
):
    """Assemble CSC arrays from per-row DP storage, optionally slicing.

    If ``phen_pos`` is non-empty, only rows/cols in ``phen_pos`` are kept
    (the result is a principal submatrix indexed by phen_pos).  The
    row/column indices in the output are 0..len(phen_pos)-1.

    If ``to_grm`` is True, off-diagonal values are scaled by 2 (kinship
    → GRM); diagonal stays as (1 + F_i)/2 * 2 = 1 + F_i (which is the
    standard GRM diagonal).
    """
    n_phen = phen_pos.shape[0]
    # Build full_to_phen[i] = k if row i of the full matrix maps to
    # row k in the output, else -1.
    n_full = row_start.shape[0]
    full_to_phen = np.full(n_full, -1, dtype=np.int32)
    for k in range(n_phen):
        full_to_phen[phen_pos[k]] = np.int32(k)

    # First pass: count entries per column in the sliced matrix.
    # The kinship rows are sorted by column, so we iterate and count
    # only those (i, j) with both i and j in phen_pos.
    #
    # Indices/indptr use int32 throughout to halve K's footprint vs int64
    # (≈ 1.65 GB saved at N=100K).  Count and prefix-sum in int64 so the
    # overflow guard runs before int32 prefix writes can wrap.
    # At G_ped=6 we see nnz ≈ 690 × N, so overflow only kicks in beyond
    # N≈3M.
    col_counts = np.zeros(n_phen, dtype=np.int64)
    for i_full in range(n_full):
        i_phen = full_to_phen[i_full]
        if i_phen < 0:
            continue
        rs = row_start[i_full]
        rc = row_count[i_full]
        for p in range(rc):
            j_full = cols[rs + p]
            j_phen = full_to_phen[j_full]
            if j_phen >= 0:
                col_counts[j_phen] += np.int64(1)

    indptr = _checked_int32_indptr_from_counts(col_counts)
    nnz = indptr[n_phen]

    indices = np.empty(nnz, dtype=np.int32)
    values = np.empty(nnz, dtype=np.float32)

    # Second pass: fill.  For CSC with column j holding rows from
    # (phen → full) mapping, we need to iterate the *column* of the
    # full matrix — but we only have rows.  Luckily, the kinship matrix
    # is symmetric, so column j == row j.  Iterate the rows and emit
    # transposed entries.
    #
    # To keep column-major order preserved per column, maintain a
    # running pointer per column.
    col_write = np.zeros(n_phen, dtype=np.int32)
    for i_full in range(n_full):
        i_phen = full_to_phen[i_full]
        if i_phen < 0:
            continue
        rs = row_start[i_full]
        rc = row_count[i_full]
        for p in range(rc):
            j_full = cols[rs + p]
            j_phen = full_to_phen[j_full]
            if j_phen < 0:
                continue
            # Entry at (i_phen, j_phen).  CSC stores this in column j_phen
            # at row index i_phen.
            pos = indptr[j_phen] + col_write[j_phen]
            indices[pos] = i_phen
            v = vals[rs + p]
            if to_grm:
                # GRM = 2·K (off-diag and diagonal both scale; the
                # diagonal becomes 2 · (1 + F_i)/2 = 1 + F_i).
                v *= np.float32(2.0)
            values[pos] = v
            col_write[j_phen] += 1

    # Rows within each column need to be sorted; fast path: the row
    # order emerged from the outer i_full iteration, which is
    # monotonically increasing.  Since phen_pos is usually in ascending
    # order, i_phen ends up ascending too.  Check this; if not, we'd
    # need a post-sort.  (For general phen_pos orders, add a sort pass.)
    return indptr, indices, values


# ---------------------------------------------------------------------------
# Top-level driver (scipy-free by design — caller wraps into csc_matrix)
# ---------------------------------------------------------------------------


def _validate_dp_args(
    n: int,
    m_idx: np.ndarray,
    f_idx: np.ndarray,
    tw_idx: np.ndarray,
    generation: np.ndarray | None,
    init_cap_per_row: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Coerce arrays + resolve ``depth`` and ``init_cap_per_row`` defaults.

    Shared between :func:`_run_dp` and :func:`_run_dp_retiring` so the
    two entry points reach the kernel with identical, contiguous inputs.
    """
    m_idx = np.ascontiguousarray(m_idx, dtype=np.int32)
    f_idx = np.ascontiguousarray(f_idx, dtype=np.int32)
    tw_idx = np.ascontiguousarray(tw_idx, dtype=np.int32)
    if generation is None:
        depth = _compute_depth(m_idx, f_idx, n)
    else:
        depth = np.ascontiguousarray(generation, dtype=np.int32)
    if init_cap_per_row is None:
        g_ped = int(depth.max()) if n > 0 else 0
        init_cap_per_row = _suggest_init_cap_per_row(g_ped)
    return m_idx, f_idx, tw_idx, depth, int(init_cap_per_row)


class KinshipDPConfig(NamedTuple):
    """Three behavior flags passed through to ``_dp_kinship``.

    Bundled at the Python layer so the two thin shims and any future
    test caller can pick a named preset (``CSC_ASSEMBLY``,
    ``RETIRING_DEFAULT``) instead of threading three positional
    booleans.  Numba never sees this NamedTuple — ``_run_dp_core``
    unpacks it into plain ``bool`` args at the ``_dp_kinship`` call
    boundary, so dispatch fragmentation is impossible by construction.

    Fields:
        retire: free DP rows in place at end-of-depth + accumulate
            inline θ̄.  Required for ``per_gen_mean_kinship``; turned
            off for CSC assembly which needs the full row storage.
        lazy: defer row-slot allocation to the first write.  Only
            valid with ``retire=True``; the never-allocated → live
            transition relies on freelist slots which retirement
            populates.
        debug_asserts: enable retire-correctness asserts inside
            ``_dp_kinship``.  Test/parity use only.
    """

    retire: bool
    lazy: bool
    debug_asserts: bool


# Named presets — the two shapes the production shims actually use.
# Tests/parity callers build their own when they want a non-default
# combination (e.g. lazy=False with retire=True).
_DP_CONFIG_CSC = KinshipDPConfig(retire=False, lazy=False, debug_asserts=False)


class DPResult(NamedTuple):
    """Full output bundle from :func:`_run_dp_core`.

    Either the CSC-assembly path (``retire=False``) or the retiring
    streaming path (``retire=True``) populates a different subset of
    these fields.  ``cols``/``vals``/``row_start``/``row_count`` carry
    the full DP row storage when retirement is off; under retirement
    those buffers have been progressively freed in place and only
    ``sum_theta`` is meaningful.  ``depth`` and ``tw_idx`` are the
    contiguous-coerced versions of the kernel's inputs — returned so
    downstream callers can avoid re-casting.
    """

    cols: np.ndarray
    vals: np.ndarray
    row_start: np.ndarray
    row_count: np.ndarray
    sum_theta: np.ndarray
    depth: np.ndarray
    tw_idx: np.ndarray


def _run_dp_core(
    n: int,
    m_idx: np.ndarray,
    f_idx: np.ndarray,
    tw_idx: np.ndarray,
    generation: np.ndarray | None,
    min_kinship: float,
    init_cap_per_row: int | None,
    *,
    config: KinshipDPConfig,
    grow_stats: np.ndarray | None = None,
    initial_buffer_override: int | None = None,
) -> DPResult:
    """Validate args + run :func:`_dp_kinship`; bundle the full output.

    Single entry point for both the CSC-assembly path
    (``config = KinshipDPConfig(retire=False, lazy=False, ...)``) and the
    retiring streaming θ̄ path
    (``config.retire=True``, ``config.lazy`` follows the caller).  Thin
    Python shims (:func:`_run_dp` and :func:`_run_dp_retiring`)
    preserve the historical signatures and pick the fields each
    downstream caller needs from the returned :class:`DPResult`.

    The :class:`KinshipDPConfig` lives entirely at the Python layer —
    ``_run_dp_core`` unpacks it into plain ``bool`` args at the
    ``_dp_kinship`` call boundary, so the ``@njit`` kernel still sees
    three plain bools and can't fragment its dispatch.

    ``grow_stats`` and ``initial_buffer_override`` are bench-only
    knobs — production callers leave them at ``None``.
    """
    m_idx, f_idx, tw_idx, depth, init_cap_per_row = _validate_dp_args(
        n, m_idx, f_idx, tw_idx, generation, init_cap_per_row,
    )
    if grow_stats is None:
        grow_stats = np.zeros(3, dtype=np.int64)
    override = np.int64(initial_buffer_override or 0)
    cols, vals, row_start, row_count, sum_theta = _dp_kinship(
        n, m_idx, f_idx, tw_idx, depth,
        float(min_kinship), init_cap_per_row,
        bool(config.retire), bool(config.lazy), bool(config.debug_asserts),
        grow_stats, override,
    )
    return DPResult(
        cols=cols, vals=vals, row_start=row_start, row_count=row_count,
        sum_theta=sum_theta, depth=depth, tw_idx=tw_idx,
    )


def _run_dp(
    n: int,
    m_idx: np.ndarray,
    f_idx: np.ndarray,
    tw_idx: np.ndarray,
    generation: np.ndarray | None,
    min_kinship: float,
    init_cap_per_row: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Thin shim: CSC-assembly DP (retire=False, lazy=False).

    Used by :func:`_build_kinship_csc` for CSC assembly and by the
    legacy parity path in :func:`_compute_theta_per_gen`.  Returns the
    DP row storage along with the resolved ``depth`` and contiguous
    ``tw_idx`` so downstream code can avoid re-casting.
    """
    r = _run_dp_core(
        n, m_idx, f_idx, tw_idx, generation, min_kinship, init_cap_per_row,
        config=_DP_CONFIG_CSC,
    )
    return r.cols, r.vals, r.row_start, r.row_count, r.depth, r.tw_idx


def _run_dp_retiring(
    n: int,
    m_idx: np.ndarray,
    f_idx: np.ndarray,
    tw_idx: np.ndarray,
    generation: np.ndarray | None,
    min_kinship: float,
    init_cap_per_row: int | None,
    debug_asserts: bool = False,
    *,
    _lazy: bool = True,
    _grow_stats: np.ndarray | None = None,
    _initial_buffer_override: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Thin shim: retiring DP with end-of-depth retirement + inline θ̄.

    Returns ``(sum_theta, depth, tw_idx)``.  The DP row storage is
    progressively retired during the run and goes out of scope on
    return; only the per-generation θ̄ sum survives.

    ``_lazy`` is a test-only escape hatch: production callers must
    leave it at ``True`` (lazy row-slot allocation).  The parity tests
    pass ``_lazy=False`` to exercise the eager-allocated retire path
    as a comparator.

    ``_grow_stats`` is a benchmark-only escape hatch.  Pass a
    pre-allocated length-3 int64 array to capture
    ``[_grow_global_call_count, total_entries_copied, peak_allocated]``;
    leave ``None`` for production to use a fresh zeroed placeholder.

    ``_initial_buffer_override`` is a benchmark-only escape hatch for
    sweeping initial-buffer heuristics.  Pass ``None`` (production) or
    ``0`` to use the current heuristic
    ``max(1<<16, init_cap_per_row*1024)``; positive int swaps it.
    """
    r = _run_dp_core(
        n, m_idx, f_idx, tw_idx, generation, min_kinship, init_cap_per_row,
        config=KinshipDPConfig(retire=True, lazy=_lazy, debug_asserts=debug_asserts),
        grow_stats=_grow_stats, initial_buffer_override=_initial_buffer_override,
    )
    return r.sum_theta, r.depth, r.tw_idx


def _build_kinship_csc(
    n: int,
    m_idx: np.ndarray,
    f_idx: np.ndarray,
    tw_idx: np.ndarray,
    generation: np.ndarray | None,
    min_kinship: float,
    init_cap_per_row: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute pedigree kinship and return full-symmetric CSC arrays.

    Args:
        n: number of individuals.
        m_idx: 0..n-1 remapped mother row indices; -1 for missing/founder.
        f_idx: 0..n-1 remapped father row indices; -1 for missing/founder.
        tw_idx: 0..n-1 remapped MZ twin partner row indices; -1 for non-twin.
        generation: per-individual generation depth (founders = 0).
            If None, derived via a fixed-point sweep over the parent graph.
        min_kinship: off-diagonal entries with ``value <= min_kinship``
            are dropped during the DP (kernel-side pruning).  Diagonal
            always kept.

    Returns:
        ``(indptr, indices, data)`` suitable for
        ``scipy.sparse.csc_matrix((data, indices, indptr), shape=(n, n))``.
        Storage is full-symmetric (both triangles); indices within each
        column are sorted ascendingly (under the common case of the
        kernel iterating ``phen_pos = arange(n)``).
    """
    cols, vals, row_start, row_count, depth, tw_idx = _run_dp(
        n, m_idx, f_idx, tw_idx, generation, min_kinship, init_cap_per_row,
    )
    phen_pos = np.arange(n, dtype=np.int32)
    indptr, indices, values = _assemble_csc(
        cols,
        vals,
        row_start,
        row_count,
        phen_pos,
        False,
    )
    return indptr, indices, values


# ---------------------------------------------------------------------------
# Streaming per-generation mean kinship (no CSC build)
# ---------------------------------------------------------------------------


@numba.njit(cache=True)
def _stream_sum_theta_per_gen(
    cols: np.ndarray,
    vals: np.ndarray,
    row_start: np.ndarray,
    row_count: np.ndarray,
    generation: np.ndarray,
    twin_idx: np.ndarray,
    g_max: np.int32,
) -> np.ndarray:
    """Sum kinship over within-cohort upper-triangle non-twin pairs.

    Walks the DP row storage directly (already ascending-col-sorted per
    row), counting each unordered pair once at row index < col index.
    Float32 ``vals`` are widened to float64 on accumulation to keep the
    sum well above per-entry ulp at large N.
    """
    sum_theta = np.zeros(g_max + 1, dtype=np.float64)
    n = row_start.shape[0]
    for i in range(n):
        g_i = generation[i]
        tw_i = twin_idx[i]
        rs = row_start[i]
        rc = row_count[i]
        for p in range(rc):
            j = cols[rs + p]
            if j <= i or j == tw_i or generation[j] != g_i:
                continue
            sum_theta[g_i] += np.float64(vals[rs + p])
    return sum_theta


def _finalize_from_sum_theta(
    sum_theta: np.ndarray,
    generation: np.ndarray,
    twin_idx: np.ndarray,
) -> np.ndarray:
    """Divide per-gen θ̄ sums by within-cohort non-twin pair counts.

    Shared by :func:`_per_gen_mean_kinship_from_dp` (post-hoc walk) and
    :func:`_compute_theta_per_gen` (inline accumulator from the
    retiring DP).  Excludes the diagonal and MZ twin pairs; returns
    NaN for cohorts with fewer than 2 non-twin members.
    """
    gen = np.ascontiguousarray(generation, dtype=np.int32)
    twin = np.ascontiguousarray(twin_idx, dtype=np.int32)
    g_max = int(gen.max()) if gen.size else 0
    out = np.full(g_max + 1, np.nan, dtype=np.float64)

    # Single-pass per-gen denominators: n_g via bincount, twin pairs via
    # bincount of the partner-with-smaller-index mask.
    n_per_g = np.bincount(gen, minlength=g_max + 1).astype(np.int64)
    idx = np.arange(len(gen), dtype=np.int32)
    twin_pair = (twin >= 0) & (twin > idx)
    twin_per_g = (
        np.bincount(gen[twin_pair], minlength=g_max + 1).astype(np.int64)
        if twin_pair.any()
        else np.zeros(g_max + 1, dtype=np.int64)
    )
    total_pairs = n_per_g * (n_per_g - 1) // 2 - twin_per_g
    eligible = (n_per_g >= 2) & (total_pairs > 0)
    out[eligible] = sum_theta[eligible] / total_pairs[eligible]
    return out


def _per_gen_mean_kinship_from_dp(
    cols: np.ndarray,
    vals: np.ndarray,
    row_start: np.ndarray,
    row_count: np.ndarray,
    generation: np.ndarray,
    twin_idx: np.ndarray,
) -> np.ndarray:
    """Mean θ per generation, streamed from DP row storage.

    Same semantics as
    :func:`pedigree_graph._effective_size._per_gen_mean_kinship` but
    bypasses materializing the full kinship CSC.  Excludes diagonal and
    MZ twin pairs; returns ``np.nan`` for cohorts with fewer than 2
    non-twin members.

    Args:
        cols, vals, row_start, row_count: DP output from
            :func:`_dp_kinship`.
        generation: per-individual generation index (founders = 0).
        twin_idx: per-individual twin partner row index, ``-1`` for
            non-twins.

    Returns:
        Float64 array of length ``g_max + 1`` with mean θ̄_g per
        generation, NaN for cohorts with no eligible pairs.
    """
    gen = np.ascontiguousarray(generation, dtype=np.int32)
    twin = np.ascontiguousarray(twin_idx, dtype=np.int32)
    g_max = int(gen.max()) if gen.size else 0
    sum_theta = _stream_sum_theta_per_gen(
        cols, vals, row_start, row_count, gen, twin, np.int32(g_max),
    )
    return _finalize_from_sum_theta(sum_theta, gen, twin)


def _compute_theta_per_gen(
    n: int,
    m_idx: np.ndarray,
    f_idx: np.ndarray,
    tw_idx: np.ndarray,
    generation: np.ndarray | None,
    min_kinship: float,
    init_cap_per_row: int | None = None,
    _debug_no_retire: bool = False,
    _debug_asserts: bool = False,
) -> np.ndarray:
    """Per-generation mean kinship θ̄_g without materializing K.

    The retiring DP frees rows in place at end-of-depth and
    accumulates θ̄ inline during the merge walk; no CSC matrix and no
    full N × G row buffer ever co-exist.

    ``_debug_no_retire=True`` falls back to a two-pass path (full DP
    then post-hoc walk) for parity testing.  ``_debug_asserts=True``
    enables retire-correctness asserts inside the kernel; no effect
    under ``_debug_no_retire=True``.

    Args:
        n, m_idx, f_idx, tw_idx, generation, min_kinship,
        init_cap_per_row: same semantics as :func:`_build_kinship_csc`.

    Returns:
        Float64 array of length ``g_max + 1`` with mean θ̄_g per
        generation, NaN for cohorts with fewer than 2 non-twin members.
    """
    if _debug_no_retire:
        cols, vals, row_start, row_count, depth, tw_idx_c = _run_dp(
            n, m_idx, f_idx, tw_idx, generation, min_kinship, init_cap_per_row,
        )
        return _per_gen_mean_kinship_from_dp(
            cols, vals, row_start, row_count, depth, tw_idx_c,
        )
    sum_theta, depth, tw_idx_c = _run_dp_retiring(
        n, m_idx, f_idx, tw_idx, generation, min_kinship, init_cap_per_row,
        debug_asserts=_debug_asserts,
    )
    return _finalize_from_sum_theta(sum_theta, depth, tw_idx_c)


# ---------------------------------------------------------------------------
# Meuwissen & Luo (1992) F-only ancestor walk
# ---------------------------------------------------------------------------


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


@numba.njit(cache=True)
def _grow_touched_scratch(
    touched: np.ndarray,
    next_in_chain: np.ndarray,
    count: int,
):
    """Double the capacity of the (touched, next_in_chain) scratch pair.

    Mirrors :func:`_grow_global` for the ML F kernel's per-individual
    ancestor-list and linked-list arrays.  Returns ``(touched, next, cap)``.
    """
    old_cap = touched.shape[0]
    new_cap = old_cap * 2
    new_touched = np.empty(new_cap, dtype=np.int32)
    new_next = np.full(new_cap, -1, dtype=np.int32)
    for q in range(count):
        new_touched[q] = touched[q]
        new_next[q] = next_in_chain[q]
    return new_touched, new_next, new_cap


@numba.njit(cache=True)
def _compute_F_meuwissen_luo(
    m_idx: np.ndarray,
    f_idx: np.ndarray,
    depth: np.ndarray,
    n: int,
) -> np.ndarray:
    """Inbreeding coefficient F per individual via Meuwissen & Luo (1992).

    Uses the LDL' decomposition ``A = T D T'`` of the numerator
    relationship matrix.  For each individual i in topological order:

        D[i] = 0.5 - 0.25*(F[s] + F[d])  (both parents known)
             = 0.75 - 0.25*F[p]          (one parent known)
             = 1                          (founder)

        F[i] = (sum over j in ANC_i of t_ij^2 * D[j]) - 1

    where ``t_ii = 1`` and ``t_ij = 0.5 * sum over progeny k of j (with k
    in ANC_i) of t_ik`` — equivalently, the path-sum from i to ancestor j
    obtained by FORWARD propagation: start with t[i]=1, halve along each
    parent edge, sum across multiple paths.

    MZ-naive: twins are treated as full sibs.  K-derived F (matrix path)
    is MZ-aware via twin off-diagonals; the two paths agree except for
    individuals whose only inbreeding path runs through both members of an
    MZ twin pair as ancestors.

    Topological-order required: m_idx[i] < i and f_idx[i] < i for all i
    (validated upstream via :func:`_check_topological`).
    """
    # Persistent scratch.  Sparse-touched, reset via the touched list.
    t = np.zeros(n, dtype=np.float64)
    in_frontier = np.zeros(n, dtype=np.bool_)
    F = np.zeros(n, dtype=np.float64)
    D = np.zeros(n, dtype=np.float64)

    max_depth_global = int(depth.max()) if n > 0 else 0
    bound = 2
    for _ in range(max_depth_global + 1):
        bound *= 2
    bound -= 2  # 2^(max_depth+1) - 2  (full binary ancestor set bound)
    if bound < 64:
        bound = 64
    K_max = bound + 64
    if K_max > n:
        K_max = n
    if K_max < 4:
        K_max = 4

    touched = np.empty(K_max, dtype=np.int32)
    next_in_chain = np.full(K_max, -1, dtype=np.int32)
    head_depth = np.full(max_depth_global + 1, -1, dtype=np.int32)

    for i in range(n):
        s = m_idx[i]
        d = f_idx[i]

        # Compute D[i] from already-known F[parents].  D is needed even
        # when F[i]=0 because future descendants reference D[i].
        if s < 0 and d < 0:
            F[i] = 0.0
            D[i] = 1.0
            continue
        if s < 0:
            F[i] = 0.0
            D[i] = 0.75 - 0.25 * F[d]
            continue
        if d < 0:
            F[i] = 0.0
            D[i] = 0.75 - 0.25 * F[s]
            continue
        D[i] = 0.5 - 0.25 * (F[s] + F[d])

        touched_count = 0
        t[i] = 1.0
        in_frontier[i] = True
        di = int(depth[i])
        next_in_chain[touched_count] = head_depth[di]
        head_depth[di] = touched_count
        touched[touched_count] = i
        touched_count += 1

        for k in range(di, -1, -1):
            pos = head_depth[k]
            while pos >= 0:
                a = touched[pos]
                t_a = t[a]
                p_m = m_idx[a]
                p_f = f_idx[a]
                if p_m >= 0:
                    if not in_frontier[p_m]:
                        in_frontier[p_m] = True
                        t[p_m] = 0.0
                        if touched_count >= K_max:
                            touched, next_in_chain, K_max = _grow_touched_scratch(
                                touched, next_in_chain, touched_count
                            )
                        dp = int(depth[p_m])
                        next_in_chain[touched_count] = head_depth[dp]
                        head_depth[dp] = touched_count
                        touched[touched_count] = p_m
                        touched_count += 1
                    t[p_m] += 0.5 * t_a
                if p_f >= 0:
                    # Mirrors the mother-edge branch above; kept inline for numba.
                    if not in_frontier[p_f]:
                        in_frontier[p_f] = True
                        t[p_f] = 0.0
                        if touched_count >= K_max:
                            touched, next_in_chain, K_max = _grow_touched_scratch(
                                touched, next_in_chain, touched_count
                            )
                        dp = int(depth[p_f])
                        next_in_chain[touched_count] = head_depth[dp]
                        head_depth[dp] = touched_count
                        touched[touched_count] = p_f
                        touched_count += 1
                    t[p_f] += 0.5 * t_a
                pos = next_in_chain[pos]
            head_depth[k] = -1

        # F[i] = sum_j t_ij^2 * D[j] - 1   over j in ANC_i (= touched).
        F_sum = 0.0
        for q in range(touched_count):
            j = touched[q]
            tj = t[j]
            F_sum += tj * tj * D[j]
        F[i] = F_sum - 1.0

        for q in range(touched_count):
            j = touched[q]
            t[j] = 0.0
            in_frontier[j] = False
            next_in_chain[q] = -1

    return F
