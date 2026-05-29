"""Kinship DP kernel, its driver, and per-generation theta streaming (PGQ-008).

The hot ``_dp_kinship`` recursion plus the orchestration that drives it:
argument validation, the ``KinshipDPConfig`` / ``DPResult`` records, the
``_run_dp_core`` entry point, the full-CSC build, and the streamed
per-generation mean-kinship (theta) path.  Builds on the depth utilities
(``_kinship_depth``), the slab allocator (``_kinship_allocator``), and the
CSC assembler (``_kinship_csc``).
"""

from __future__ import annotations

from typing import NamedTuple

import numba
import numpy as np

from pedigree_graph._kinship_allocator import (
    _append_entry,
    _FreelistBuffers,
    _retire_rows_at_depth,
    _sort_row_inplace,
    _suggest_init_cap_per_row,
)
from pedigree_graph._kinship_csc import _assemble_csc
from pedigree_graph._kinship_depth import _compute_depth, _compute_last_direct_child_depth


class KinshipDPConfig(NamedTuple):
    """Three behavior flags passed through to ``_dp_kinship``.

    Bundled at the Python layer so :func:`_run_dp_core`'s callers can
    pick the CSC-assembly preset (:data:`_DP_CONFIG_CSC`) or build a
    retiring/debug config inline instead of threading three positional
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


_DP_CONFIG_CSC = KinshipDPConfig(retire=False, lazy=False, debug_asserts=False)


def _validate_dp_args(
    n: int,
    m_idx: np.ndarray,
    f_idx: np.ndarray,
    tw_idx: np.ndarray,
    generation: np.ndarray | None,
    init_cap_per_row: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Coerce arrays + resolve ``depth`` and ``init_cap_per_row`` defaults.

    Called by :func:`_run_dp_core` so every DP dispatch reaches the
    kernel with identical, contiguous inputs regardless of caller.
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
    front and is used by the CSC assembly path through
    ``_run_dp_core(config=_DP_CONFIG_CSC)``.

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

    buffers = _FreelistBuffers(
        freelist_starts,
        freelist_tops,
        fl_init_cap,
        grow_stats,
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
                cols,
                vals,
                row_start,
                row_count,
                row_cap,
                next_alloc,
                np.int32(i),
                np.int32(i),
                np.float32(0.5),
                buffers,
            )

    # Founders with no children at any later depth retire immediately;
    # their stored diagonal is never read by a merge walk.  Under lazy
    # alloc these were already skipped in init; the retire pass just
    # transitions them from never-allocated to retired-sentinel so any
    # stray descendant write would dissolve.
    if retire:
        _retire_rows_at_depth(
            np.int32(0),
            last_dcd,
            row_start,
            row_count,
            row_cap,
            freelist_starts,
            freelist_tops,
            fl_init_cap,
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
                    raise AssertionError("mother row retired before child processed")
                if f >= 0 and row_start[f] < 0:
                    raise AssertionError("father row retired before child processed")

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
                    cols,
                    vals,
                    row_start,
                    row_count,
                    row_cap,
                    next_alloc,
                    np.int32(j),
                    k,
                    val,
                    buffers,
                )
                # Symmetric fill: append (j, val) to row k.  Since j is
                # processed in depth order and higher j means later
                # processing (same gen IDs are contiguous), the appends
                # to row k come in ascending j order → row k stays
                # sorted.
                cols, vals, next_alloc = _append_entry(
                    cols,
                    vals,
                    row_start,
                    row_count,
                    row_cap,
                    next_alloc,
                    k,
                    np.int32(j),
                    val,
                    buffers,
                )

            # --- Append diagonal (j, self_kin) to row j AFTER merge walk.
            # All merge-walk entries have cols < j (ancestors), so j is
            # the largest column — row j stays sorted.
            self_kin = np.float32((1.0 + km_f) / 2.0)
            cols, vals, next_alloc = _append_entry(
                cols,
                vals,
                row_start,
                row_count,
                row_cap,
                next_alloc,
                np.int32(j),
                np.int32(j),
                self_kin,
                buffers,
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
                    cols,
                    vals,
                    row_start,
                    row_count,
                    row_cap,
                    next_alloc,
                    np.int32(j),
                    np.int32(tw),
                    self_k,
                    buffers,
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
                    cols,
                    vals,
                    row_start,
                    row_count,
                    row_cap,
                    next_alloc,
                    np.int32(tw),
                    np.int32(j),
                    self_k,
                    buffers,
                )
                _sort_row_inplace(cols, vals, row_start[tw], row_count[tw])

        # Runs AFTER the MZ twin pass so twin writes land before
        # retirement.  ``_grow_global`` preserves freed offsets safely
        # because retirement only releases rows that have completed all
        # writes at this depth — any pending grow happened earlier.
        if retire:
            _retire_rows_at_depth(
                np.int32(d),
                last_dcd,
                row_start,
                row_count,
                row_cap,
                freelist_starts,
                freelist_tops,
                fl_init_cap,
            )

    return cols, vals, row_start, row_count, sum_theta


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
    (``config = _DP_CONFIG_CSC``) and the retiring streaming θ̄ path
    (``config.retire=True``, ``config.lazy`` follows the caller).
    Callers pick the fields they need from the returned
    :class:`DPResult` — the CSC path consumes
    ``cols``/``vals``/``row_start``/``row_count``, the retiring path
    consumes ``sum_theta``, both consume ``depth``/``tw_idx``.

    The :class:`KinshipDPConfig` lives entirely at the Python layer —
    ``_run_dp_core`` unpacks it into plain ``bool`` args at the
    ``_dp_kinship`` call boundary, so the ``@njit`` kernel still sees
    three plain bools and can't fragment its dispatch.

    ``grow_stats`` and ``initial_buffer_override`` are bench-only
    knobs — production callers leave them at ``None``.
    """
    m_idx, f_idx, tw_idx, depth, init_cap_per_row = _validate_dp_args(
        n,
        m_idx,
        f_idx,
        tw_idx,
        generation,
        init_cap_per_row,
    )
    if grow_stats is None:
        grow_stats = np.zeros(3, dtype=np.int64)
    override = np.int64(initial_buffer_override or 0)
    cols, vals, row_start, row_count, sum_theta = _dp_kinship(
        n,
        m_idx,
        f_idx,
        tw_idx,
        depth,
        float(min_kinship),
        init_cap_per_row,
        bool(config.retire),
        bool(config.lazy),
        bool(config.debug_asserts),
        grow_stats,
        override,
    )
    return DPResult(
        cols=cols,
        vals=vals,
        row_start=row_start,
        row_count=row_count,
        sum_theta=sum_theta,
        depth=depth,
        tw_idx=tw_idx,
    )


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
        init_cap_per_row: optional per-row initial column capacity for the
            DP row buffer.  ``None`` defers to the kernel default.

    Returns:
        ``(indptr, indices, data)`` suitable for
        ``scipy.sparse.csc_matrix((data, indices, indptr), shape=(n, n))``.
        Storage is full-symmetric (both triangles); indices within each
        column are sorted ascending.
    """
    r = _run_dp_core(
        n,
        m_idx,
        f_idx,
        tw_idx,
        generation,
        min_kinship,
        init_cap_per_row,
        config=_DP_CONFIG_CSC,
    )
    indptr, indices, values = _assemble_csc(r.cols, r.vals, r.row_start, r.row_count)
    return indptr, indices, values


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
        cols: DP-output column indices array from :func:`_dp_kinship`.
        vals: DP-output kinship values array from :func:`_dp_kinship`.
        row_start: DP-output per-row offsets array from :func:`_dp_kinship`.
        row_count: DP-output per-row entry counts array from :func:`_dp_kinship`.
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
        cols,
        vals,
        row_start,
        row_count,
        gen,
        twin,
        np.int32(g_max),
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
        n: same semantics as :func:`_build_kinship_csc`.
        m_idx: same semantics as :func:`_build_kinship_csc`.
        f_idx: same semantics as :func:`_build_kinship_csc`.
        tw_idx: same semantics as :func:`_build_kinship_csc`.
        generation: same semantics as :func:`_build_kinship_csc`.
        min_kinship: same semantics as :func:`_build_kinship_csc`.
        init_cap_per_row: same semantics as :func:`_build_kinship_csc`.

    Returns:
        Float64 array of length ``g_max + 1`` with mean θ̄_g per
        generation, NaN for cohorts with fewer than 2 non-twin members.
    """
    if _debug_no_retire:
        r = _run_dp_core(
            n,
            m_idx,
            f_idx,
            tw_idx,
            generation,
            min_kinship,
            init_cap_per_row,
            config=_DP_CONFIG_CSC,
        )
        return _per_gen_mean_kinship_from_dp(
            r.cols,
            r.vals,
            r.row_start,
            r.row_count,
            r.depth,
            r.tw_idx,
        )
    r = _run_dp_core(
        n,
        m_idx,
        f_idx,
        tw_idx,
        generation,
        min_kinship,
        init_cap_per_row,
        config=KinshipDPConfig(retire=True, lazy=True, debug_asserts=_debug_asserts),
    )
    return _finalize_from_sum_theta(r.sum_theta, r.depth, r.tw_idx)
