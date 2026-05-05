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
    "_compute_depth",
    "_dp_kinship",
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


@numba.njit(cache=True)
def _grow_global(cols: np.ndarray, vals: np.ndarray, min_size: int):
    """Return larger arrays copying existing contents, growing geometrically."""
    old_len = cols.shape[0]
    new_len = old_len * 2
    while new_len < min_size:
        new_len *= 2
    new_cols = np.full(new_len, -1, dtype=np.int32)
    new_cols[:old_len] = cols
    new_vals = np.zeros(new_len, dtype=np.float64)
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
    val: np.float64,
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
        row_start[row_idx] = np.int32(next_alloc)
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
):
    """Build per-row sorted kinship arrays via gen-by-gen DP.

    Returns:
        cols: int32[total_cap], flat col storage (per-row contiguous).
        vals: float64[total_cap], matching values.
        row_start: int32[n], where each row begins in cols/vals.
        row_count: int32[n], entries per row.

    Rows are stored symmetrically (row i contains entries for cols j
    and vice versa).  Within each row, entries are sorted by column
    index (ascending).
    """
    # Global buffer — geometric growth.
    total_cap = np.int64(n) * INIT_CAP_PER_ROW
    cols = np.full(total_cap, -1, dtype=np.int32)
    vals = np.zeros(total_cap, dtype=np.float64)
    row_start = np.zeros(n, dtype=np.int32)
    row_count = np.zeros(n, dtype=np.int32)
    row_cap = np.full(n, INIT_CAP_PER_ROW, dtype=np.int32)

    # Each row starts at position i * INIT_CAP_PER_ROW.
    for i in range(n):
        row_start[i] = np.int32(i * INIT_CAP_PER_ROW)
    next_alloc = np.int64(n) * INIT_CAP_PER_ROW

    # Diagonal self-kinship for founders only (0.5 with no inbreeding).
    # For non-founders, the diagonal is appended AFTER the merge walk —
    # doing it upfront would break the sorted-row invariant because the
    # merge walk appends cols < j (ancestors have lower indices under
    # depth-first ID assignment), so position 0 must be the smallest col.
    for i in range(n):
        if m_idx[i] < 0 and f_idx[i] < 0:
            cols[row_start[i]] = np.int32(i)
            vals[row_start[i]] = 0.5
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
            km_f = 0.0
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
            ms = row_start[m] if m >= 0 else np.int32(0)
            mc = row_count[m] if m >= 0 else np.int32(0)
            fs = row_start[f] if f >= 0 else np.int32(0)
            fc = row_count[f] if f >= 0 else np.int32(0)

            pm = 0
            pf = 0
            while pm < mc or pf < fc:
                k = np.int32(-1)
                mv = 0.0
                fv = 0.0
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
                val = (mv + fv) / 2.0
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
            self_kin = (1.0 + km_f) / 2.0
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
            self_k = 0.5  # fallback if not found (shouldn't happen)
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
                col_counts[j_phen] += 1

    indptr = np.zeros(n_phen + 1, dtype=np.int64)
    for j in range(n_phen):
        indptr[j + 1] = indptr[j] + col_counts[j]
    nnz = indptr[n_phen]

    indices = np.empty(nnz, dtype=np.int64)
    values = np.empty(nnz, dtype=np.float64)

    # Second pass: fill.  For CSC with column j holding rows from
    # (phen → full) mapping, we need to iterate the *column* of the
    # full matrix — but we only have rows.  Luckily, the kinship matrix
    # is symmetric, so column j == row j.  Iterate the rows and emit
    # transposed entries.
    #
    # To keep column-major order preserved per column, maintain a
    # running pointer per column.
    col_write = np.zeros(n_phen, dtype=np.int64)
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
                v *= 2.0
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


def _build_kinship_csc(
    n: int,
    m_idx: np.ndarray,
    f_idx: np.ndarray,
    tw_idx: np.ndarray,
    generation: np.ndarray | None,
    min_kinship: float,
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
    m_idx = np.ascontiguousarray(m_idx, dtype=np.int32)
    f_idx = np.ascontiguousarray(f_idx, dtype=np.int32)
    tw_idx = np.ascontiguousarray(tw_idx, dtype=np.int32)

    if generation is None:
        depth = _compute_depth(m_idx, f_idx, n)
    else:
        depth = np.ascontiguousarray(generation, dtype=np.int32)

    cols, vals, row_start, row_count = _dp_kinship(
        n,
        m_idx,
        f_idx,
        tw_idx,
        depth,
        float(min_kinship),
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
