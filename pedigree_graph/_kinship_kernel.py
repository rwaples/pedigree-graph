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
    "_check_topological",
    "_compute_F_meuwissen_luo",
    "_compute_depth",
    "_compute_eqg",
    "_compute_theta_per_gen",
    "_dp_kinship",
    "_per_gen_mean_kinship_from_dp",
]

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
def _grow_global(cols: np.ndarray, vals: np.ndarray, min_size: int):
    """Return larger arrays copying existing contents, growing geometrically."""
    old_len = cols.shape[0]
    new_len = old_len * 2
    while new_len < min_size:
        new_len *= 2
    new_cols = np.full(new_len, -1, dtype=np.int32)
    new_cols[:old_len] = cols
    new_vals = np.zeros(new_len, dtype=np.float32)
    new_vals[:old_len] = vals
    return new_cols, new_vals


@numba.njit(cache=True)
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
):
    """Append (col_idx, val) to row_idx.

    If the row's capacity is exhausted, relocate the row to a new segment
    at ``next_alloc`` with doubled capacity.  Returns (cols, vals,
    next_alloc).  The arrays may be reallocated; use the returned values.
    """
    if row_count[row_idx] >= row_cap[row_idx]:
        # Row is full — relocate to end of buffer with doubled capacity.
        new_cap = row_cap[row_idx] * 2
        needed = next_alloc + new_cap
        if needed > cols.shape[0]:
            cols, vals = _grow_global(cols, vals, needed)
        # Copy existing data.
        src = row_start[row_idx]
        cnt = row_count[row_idx]
        for k in range(cnt):
            cols[next_alloc + k] = cols[src + k]
            vals[next_alloc + k] = vals[src + k]
        row_start[row_idx] = np.int64(next_alloc)
        row_cap[row_idx] = np.int32(new_cap)
        next_alloc += new_cap
    # Append.
    pos = row_start[row_idx] + row_count[row_idx]
    cols[pos] = col_idx
    vals[pos] = val
    row_count[row_idx] += 1
    return cols, vals, next_alloc


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
):
    """Build per-row sorted kinship arrays via gen-by-gen DP.

    Returns:
        cols: int32[total_cap], flat col storage (per-row contiguous).
        vals: float32[total_cap], matching values (narrowed from float64
            for ~50 % memory reduction at large N; kinship values lie in
            [0, 1] so float32's 7-digit mantissa keeps relative error well
            below downstream slope-fit tolerances).
        row_start: int64[n], where each row begins in cols/vals.  int64
            because the flat buffer can exceed 2**31 entries at large N
            (e.g. N ≈ 525K with init_cap_per_row=4096).
        row_count: int32[n], entries per row.

    Rows are stored symmetrically (row i contains entries for cols j
    and vice versa).  Within each row, entries are sorted by column
    index (ascending).
    """
    # Global buffer — geometric growth.  ``init_cap_per_row`` may be
    # tuned upward to skip the doubling cascade when the caller knows
    # typical kinship row sizes (e.g. ``2 ** (G_ped + 4)`` ≈ 1024 at
    # G_ped=6); see :func:`_suggest_init_cap_per_row`.
    init_cap = np.int32(init_cap_per_row)
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

    # Diagonal self-kinship for founders only (0.5 with no inbreeding).
    # For non-founders, the diagonal is appended AFTER the merge walk —
    # doing it upfront would break the sorted-row invariant because the
    # merge walk appends cols < j (ancestors have lower indices under
    # depth-first ID assignment), so position 0 must be the smallest col.
    for i in range(n):
        if m_idx[i] < 0 and f_idx[i] < 0:
            cols[row_start[i]] = np.int32(i)
            vals[row_start[i]] = np.float32(0.5)
            row_count[i] = np.int32(1)

    # DP: process in depth order.
    max_depth = np.int32(depth.max())
    for d in range(1, max_depth + 1):
        for j in range(n):
            if depth[j] != d:
                continue
            m = m_idx[j]
            f = f_idx[j]
            if m < 0 and f < 0:
                continue  # disconnected founder; self-kinship already 0.5

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
                )
                _sort_row_inplace(cols, vals, row_start[tw], row_count[tw])

    return cols, vals, row_start, row_count


# ---------------------------------------------------------------------------
# CSC assembly (numba)
# ---------------------------------------------------------------------------


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
    # (≈ 1.65 GB saved at N=100K).  Safe as long as ``nnz < 2**31``; the
    # caller (`_build_kinship_csc`) is responsible for guarding against
    # the int32-overflow regime (G_ped × N very large).  At G_ped=6 we
    # see nnz ≈ 690 × N, so overflow only kicks in beyond N≈3M.
    col_counts = np.zeros(n_phen, dtype=np.int32)
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
                col_counts[j_phen] += 1

    indptr = np.zeros(n_phen + 1, dtype=np.int32)
    for j in range(n_phen):
        indptr[j + 1] = indptr[j] + col_counts[j]
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


def _run_dp(
    n: int,
    m_idx: np.ndarray,
    f_idx: np.ndarray,
    tw_idx: np.ndarray,
    generation: np.ndarray | None,
    min_kinship: float,
    init_cap_per_row: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validate args + run the kinship DP.

    Shared by :func:`_build_kinship_csc` (CSC assembly) and
    :func:`_compute_theta_per_gen` (streaming θ̄).  Returns the DP row
    storage along with the resolved ``depth`` and contiguous ``tw_idx``
    so downstream code can avoid re-casting.
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
    cols, vals, row_start, row_count = _dp_kinship(
        n, m_idx, f_idx, tw_idx, depth,
        float(min_kinship), int(init_cap_per_row),
    )
    return cols, vals, row_start, row_count, depth, tw_idx


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
    # _assemble_csc writes int32 indptr/indices.  If the build exceeded
    # int32 range (only possible at extreme scale, e.g. N ≳ 3M at G_ped=6
    # with full mixing), promote to int64 so scipy/downstream code can
    # still operate.  Negative indptr[-1] is the canonical overflow signal
    # from wrap-around during accumulation.
    if indptr[-1] < 0:
        raise OverflowError(
            "kinship matrix nnz exceeded int32 range — rebuild with int64 "
            "CSC indices (open a pedigree-graph issue if you hit this)"
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

    # Pair counts in closed form (matches _per_gen_mean_kinship).
    out = np.full(g_max + 1, np.nan, dtype=np.float64)
    idx = np.arange(len(gen))
    for g in range(g_max + 1):
        in_g = gen == g
        n_g = int(in_g.sum())
        if n_g < 2:
            continue
        twin_in_g = int(((twin >= 0) & in_g & (twin > idx)).sum())
        total_pairs = n_g * (n_g - 1) // 2 - twin_in_g
        if total_pairs <= 0:
            continue
        out[g] = sum_theta[g] / total_pairs
    return out


def _compute_theta_per_gen(
    n: int,
    m_idx: np.ndarray,
    f_idx: np.ndarray,
    tw_idx: np.ndarray,
    generation: np.ndarray | None,
    min_kinship: float,
    init_cap_per_row: int | None = None,
) -> np.ndarray:
    """Per-generation mean kinship θ̄_g without materializing K.

    Runs the same DP as :func:`_build_kinship_csc` but skips the CSC
    assembly entirely — instead, streams the row storage through
    :func:`_per_gen_mean_kinship_from_dp` and returns the per-gen mean
    directly.  DP buffers go out of scope on return.

    Args:
        n, m_idx, f_idx, tw_idx, generation, min_kinship,
        init_cap_per_row: same semantics as :func:`_build_kinship_csc`.

    Returns:
        Float64 array of length ``g_max + 1`` with mean θ̄_g per
        generation, NaN for cohorts with fewer than 2 non-twin members.
    """
    cols, vals, row_start, row_count, depth, tw_idx = _run_dp(
        n, m_idx, f_idx, tw_idx, generation, min_kinship, init_cap_per_row,
    )
    return _per_gen_mean_kinship_from_dp(
        cols, vals, row_start, row_count, depth, tw_idx,
    )


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
