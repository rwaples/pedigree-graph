"""One-depth kinship recurrence, MZ fill, and candidate capture kernels."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any, TypeVar

    _F = TypeVar("_F", bound=Callable[..., Any])

    def njit(*args: Any, **kwargs: Any) -> Callable[[_F], _F]:
        """Type-checking identity for numba's decorator."""

else:
    from numba import njit

from pedigree_graph._kinship_allocator import _append_entry, _sort_row_inplace


@njit(cache=True)
def _capture_candidates_at_depth(
    d: np.int32,
    n: int,
    depth: np.ndarray,
    twin: np.ndarray,
    cols: np.ndarray,
    vals: np.ndarray,
    row_start: np.ndarray,
    row_count: np.ndarray,
    candidate_indptr: np.ndarray,
    candidate_indices: np.ndarray,
    candidate_positions: np.ndarray,
    candidate_data: np.ndarray,
) -> None:
    """Merge complete live rows with candidate rows after one depth finishes."""
    half = np.float32(0.5)
    for j in range(n):
        if depth[j] != d:
            continue
        candidate_pos = candidate_indptr[j]
        candidate_end = candidate_indptr[j + 1]
        row_pos = 0
        row_start_j = row_start[j]
        row_end = row_count[j]
        if row_start_j < 0:
            # A childless founder can skip storage under lazy allocation. Its
            # only nonzero candidates are itself and an MZ co-twin.
            while candidate_pos < candidate_end:
                k = candidate_indices[candidate_pos]
                if k == j or k == twin[j]:
                    candidate_data[candidate_positions[candidate_pos]] = half
                candidate_pos += 1
            continue
        while candidate_pos < candidate_end and row_pos < row_end:
            candidate = candidate_indices[candidate_pos]
            relative = cols[row_start_j + row_pos]
            if relative < candidate:
                row_pos += 1
            elif candidate < relative:
                candidate_pos += 1
            else:
                candidate_data[candidate_positions[candidate_pos]] = vals[row_start_j + row_pos]
                candidate_pos += 1
                row_pos += 1


@njit(cache=True)
def _mz_twin_pass(
    d: np.int32,
    n: int,
    depth: np.ndarray,
    tw_idx: np.ndarray,
    cols: np.ndarray,
    vals: np.ndarray,
    row_start: np.ndarray,
    row_count: np.ndarray,
    row_cap: np.ndarray,
    next_alloc: np.int64,
    buffers,
) -> tuple[np.ndarray, np.ndarray, np.int64]:
    """Write the MZ off-diagonal for every twin pair at depth *d*.

    ``kinship(j, tw) = self-kinship(j)`` (= 0.5 without inbreeding).  Both
    rows are written, so row storage stays symmetric.

    Runs once per depth, **depth 0 included**: founders can be MZ co-twins,
    and their edge has to be in place before any child of either twin merge-
    walks the parent row.  Skipping depth 0 does not merely lose the MZ
    entry -- it propagates a zero through the whole subtree below the pair
    (rwaples/pedigree-graph#5).
    """
    for j in range(n):
        if depth[j] != d:
            continue
        tw = tw_idx[j]
        if tw < 0 or tw == j:
            continue
        if row_start[j] < 0:
            # No storage: a never-allocated or retired row under lazy alloc.
            # Reachable at depth 0 only, where founder init skips a founder
            # whose row no merge walk ever reads.  Nothing would read the
            # edge either, and there is no diagonal to copy — skip.
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
        # Similarly for row tw, when it has storage of its own.
        if row_start[tw] < 0:
            continue
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
    return cols, vals, next_alloc


@njit(cache=True)
def _process_depth(
    d: np.int32,
    n: int,
    m_idx: np.ndarray,
    f_idx: np.ndarray,
    tw_idx: np.ndarray,
    depth: np.ndarray,
    label: np.ndarray,
    threshold: float,
    retire: bool,
    debug_asserts: bool,
    cols: np.ndarray,
    vals: np.ndarray,
    row_start: np.ndarray,
    row_count: np.ndarray,
    row_cap: np.ndarray,
    sum_theta: np.ndarray,
    next_alloc: np.int64,
    buffers,
) -> tuple[np.ndarray, np.ndarray, np.int64]:
    """Compute one complete depth, including its MZ twin pass."""
    for j in range(n):
        if depth[j] != d:
            continue
        m = m_idx[j]
        f = f_idx[j]
        if m < 0 and f < 0:
            continue
        g_j = label[j]

        if debug_asserts:
            if m >= 0 and row_start[m] < 0:
                raise AssertionError("mother row retired before child processed")
            if f >= 0 and row_start[f] < 0:
                raise AssertionError("father row retired before child processed")

        km_f = np.float32(0.0)
        if m >= 0 and f >= 0:
            ms = row_start[m]
            mc = row_count[m]
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
            if retire and label[k] == g_j and k != tw_idx[j] and k < j:
                # Inline accumulation lets retirement avoid a later row scan.
                sum_theta[g_j] += np.float64(val)
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

    return _mz_twin_pass(
        d,
        n,
        depth,
        tw_idx,
        cols,
        vals,
        row_start,
        row_count,
        row_cap,
        next_alloc,
        buffers,
    )
