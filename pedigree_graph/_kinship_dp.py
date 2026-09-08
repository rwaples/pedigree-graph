"""Kinship DP kernel, its driver, and per-generation theta streaming (PGQ-008).

The hot ``_dp_kinship`` recursion plus the orchestration that drives it:
argument validation, the ``KinshipDPConfig`` / ``DPResult`` records, the
``_run_dp_core`` entry point, the full-CSC build, and the streamed
per-generation mean-kinship (theta) path.  Builds on the depth utilities
(``_kinship_depth``), the slab allocator (``_kinship_allocator``), and the
CSC assembler (``_kinship_csc``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import scipy.sparse as sp

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any, TypeVar

    _F = TypeVar("_F", bound=Callable[..., Any])

    def njit(*args: Any, **kwargs: Any) -> Callable[[_F], _F]:
        """Identity decorator under type checking only.

        See ``_kinship_allocator.njit`` for the rationale: numba 0.66's stubs
        type ``njit`` as returning a ``Dispatcher`` whose ParamSpec ``__call__``
        cannot be checked against the numpy scalars threaded through this DP,
        even though numba itself unifies them to a single int64 signature.
        """

else:
    from numba import njit

from pedigree_graph._cohorts import _densify_labels
from pedigree_graph._errors import ResourceError
from pedigree_graph._kinship_allocator import (
    _append_entry,
    _FreelistBuffers,
    _retire_rows_at_depth,
    _suggest_init_cap_per_row,
)
from pedigree_graph._kinship_csc import _assemble_csc
from pedigree_graph._kinship_depth import _compute_last_direct_child_depth
from pedigree_graph._kinship_dp_depth import _capture_candidates_at_depth, _mz_twin_pass, _process_depth
from pedigree_graph._topology import build_topology, owned_readonly
from pedigree_graph.summaries import GenerationKinshipSummary


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
            inline θ̄.  Required for the generation kinship summary; turned
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
    labels: np.ndarray
    order: np.ndarray | None = None
    """Depth-major row permutation applied before the kernel, or ``None`` when
    the caller's rows were already depth-major.  ``cols`` / ``vals`` /
    ``row_start`` / ``row_count`` (and ``depth`` / ``tw_idx`` / ``labels``) are
    in this permuted index space; the CSC caller un-permutes via ``order``."""


_DP_CONFIG_CSC = KinshipDPConfig(retire=False, lazy=False, debug_asserts=False)


def _validate_dp_args(
    n: int,
    m_idx: np.ndarray,
    f_idx: np.ndarray,
    tw_idx: np.ndarray,
    depth: np.ndarray,
    labels: np.ndarray | None,
    init_cap_per_row: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Coerce arrays + resolve the ``labels`` and ``init_cap_per_row`` defaults.

    Called by :func:`_run_dp_core` so every DP dispatch reaches the
    kernel with identical, contiguous inputs regardless of caller.
    ``depth`` drives the traversal and must be structural; ``labels``
    only groups the inline θ̄ accumulator and defaults to ``depth``.
    """
    m_idx = owned_readonly(m_idx, np.int32)
    f_idx = owned_readonly(f_idx, np.int32)
    tw_idx = owned_readonly(tw_idx, np.int32)
    depth = owned_readonly(depth, np.int32)
    labels = depth if labels is None else owned_readonly(labels, np.int32)
    if init_cap_per_row is None:
        g_ped = int(depth.max()) if n > 0 else 0
        init_cap_per_row = _suggest_init_cap_per_row(g_ped)
    return m_idx, f_idx, tw_idx, depth, labels, int(init_cap_per_row)


@njit(cache=True)
def _dp_kinship(
    n: int,
    m_idx: np.ndarray,
    f_idx: np.ndarray,
    tw_idx: np.ndarray,
    depth: np.ndarray,
    label: np.ndarray,
    threshold: float,
    init_cap_per_row: int,
    retire: bool,
    lazy: bool,
    debug_asserts: bool,
    grow_stats: np.ndarray,
    initial_buffer_override: np.int64,
    capture_candidates: bool,
    candidate_indptr: np.ndarray,
    candidate_indices: np.ndarray,
    candidate_positions: np.ndarray,
    candidate_data: np.ndarray,
):
    """Build per-row sorted kinship arrays via gen-by-gen DP.

    ``depth`` orders the traversal; ``label`` only buckets the inline
    θ̄ accumulator, so a caller may group by cohort labels that do not
    match structural depth.

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

    When ``capture_candidates`` is true, each completed depth's live exact
    rows are merge-scanned against topology-space candidate rows after the MZ
    pass and before retirement. Only candidate values reach ``candidate_data``.

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
        sum_theta: float64[label_max + 1], inline within-cohort kinship
            sum per label (only populated when ``retire=True``; under
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
        next_alloc = np.int64(n) * np.int64(init_cap)

    # Retirement state.  Placeholders under retire=False satisfy numba's
    # type unifier; push/pop are no-ops because fl_init_cap = 0.
    if retire:
        max_label = np.int32(label.max()) if n > 0 else np.int32(0)
        sum_theta = np.zeros(max_label + np.int32(1), dtype=np.float64)
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

    # Depth-0 MZ twin pass: founders can be co-twins, and their edge has to
    # be written before any child of either twin merge-walks the parent row.
    # Runs before the depth-0 retirement below for the same reason the
    # per-depth pass runs before its own retirement.
    cols, vals, next_alloc = _mz_twin_pass(
        np.int32(0),
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
    if capture_candidates:
        _capture_candidates_at_depth(
            np.int32(0),
            n,
            depth,
            tw_idx,
            cols,
            vals,
            row_start,
            row_count,
            candidate_indptr,
            candidate_indices,
            candidate_positions,
            candidate_data,
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
        cols, vals, next_alloc = _process_depth(
            np.int32(d),
            n,
            m_idx,
            f_idx,
            tw_idx,
            depth,
            label,
            threshold,
            retire,
            debug_asserts,
            cols,
            vals,
            row_start,
            row_count,
            row_cap,
            sum_theta,
            next_alloc,
            buffers,
        )
        if capture_candidates:
            _capture_candidates_at_depth(
                np.int32(d),
                n,
                depth,
                tw_idx,
                cols,
                vals,
                row_start,
                row_count,
                candidate_indptr,
                candidate_indices,
                candidate_positions,
                candidate_data,
            )

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
    depth: np.ndarray,
    min_kinship: float,
    init_cap_per_row: int | None,
    *,
    labels: np.ndarray | None = None,
    config: KinshipDPConfig,
    grow_stats: np.ndarray | None = None,
    initial_buffer_override: int | None = None,
    candidate_indptr: np.ndarray | None = None,
    candidate_indices: np.ndarray | None = None,
    candidate_positions: np.ndarray | None = None,
    candidate_data: np.ndarray | None = None,
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
    ``_dp_kinship`` call boundary. Candidate capture is a separate internal
    mode whose four arrays must be supplied together.

    ``grow_stats`` and ``initial_buffer_override`` are bench-only
    knobs — production callers leave them at ``None``.
    """
    m_idx, f_idx, tw_idx, depth, labels, init_cap_per_row = _validate_dp_args(
        n,
        m_idx,
        f_idx,
        tw_idx,
        depth,
        labels,
        init_cap_per_row,
    )

    # The DP kernel assumes depth-monotonic row indexing: every relative
    # discovered during a row's merge walk has a smaller index, so the diagonal
    # (column j) is appended last and the row stays sorted.  Topological-but-not-
    # depth-major input violates this: a higher-indexed relative from an earlier
    # depth lands after the diagonal, breaks the row sort, and the binary search
    # that reads phi(mother, father) silently returns 0 — zeroing the inbreeding
    # term in the self-kinship diagonal.  Reorder rows into the package's stable
    # depth-major order (:mod:`pedigree_graph._topology`) and let the CSC caller
    # un-permute via DPResult.order; the streaming theta path is label-indexed
    # and permutation-invariant.
    topo = build_topology(depth)
    order = topo.order
    if order is not None:
        m_idx = topo.to_topological(m_idx)
        f_idx = topo.to_topological(f_idx)
        tw_idx = topo.to_topological(tw_idx)
        permuted_depth = topo.gather(depth)
        labels = permuted_depth if labels is depth else topo.gather(labels)
        depth = permuted_depth

    candidate_args = (candidate_indptr, candidate_indices, candidate_positions, candidate_data)
    provided_candidate_args = sum(value is not None for value in candidate_args)
    if provided_candidate_args not in (0, len(candidate_args)):
        raise ValueError("candidate capture requires indptr, indices, positions, and data together")
    capture_candidates = provided_candidate_args == len(candidate_args)
    if capture_candidates:
        assert candidate_indptr is not None
        assert candidate_indices is not None
        assert candidate_positions is not None
        assert candidate_data is not None
        candidate_indptr = np.ascontiguousarray(candidate_indptr, dtype=np.int32)
        candidate_indices = np.ascontiguousarray(candidate_indices, dtype=np.int32)
        candidate_positions = np.ascontiguousarray(candidate_positions, dtype=np.int32)
        if (
            candidate_data.dtype != np.float32
            or not candidate_data.flags.c_contiguous
            or not candidate_data.flags.writeable
        ):
            raise ValueError("candidate_data must be a writeable contiguous float32 array")
    else:
        # Capture branches do not read placeholders. Keep one stable array type
        # without allocating an otherwise-unused O(n) indptr.
        candidate_indptr = np.zeros(1, dtype=np.int32)
        candidate_indices = np.empty(0, dtype=np.int32)
        candidate_positions = np.empty(0, dtype=np.int32)
        candidate_data = np.empty(0, dtype=np.float32)

    if grow_stats is None:
        grow_stats = np.zeros(3, dtype=np.int64)
    override = np.int64(initial_buffer_override or 0)
    cols, vals, row_start, row_count, sum_theta = _dp_kinship(
        n,
        m_idx,
        f_idx,
        tw_idx,
        depth,
        labels,
        float(min_kinship),
        init_cap_per_row,
        bool(config.retire),
        bool(config.lazy),
        bool(config.debug_asserts),
        grow_stats,
        override,
        capture_candidates,
        candidate_indptr,
        candidate_indices,
        candidate_positions,
        candidate_data,
    )
    return DPResult(
        cols=cols,
        vals=vals,
        row_start=row_start,
        row_count=row_count,
        sum_theta=sum_theta,
        depth=depth,
        tw_idx=tw_idx,
        labels=labels,
        order=order,
    )


def _build_kinship_csc(
    n: int,
    m_idx: np.ndarray,
    f_idx: np.ndarray,
    tw_idx: np.ndarray,
    depth: np.ndarray,
    min_kinship: float,
    init_cap_per_row: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute pedigree kinship and return full-symmetric CSC arrays.

    Args:
        n: number of individuals.
        m_idx: 0..n-1 remapped mother row indices; -1 for missing/founder.
        f_idx: 0..n-1 remapped father row indices; -1 for missing/founder.
        tw_idx: 0..n-1 remapped MZ twin partner row indices; -1 for non-twin.
        depth: per-individual structural depth (founders = 0), from
            :func:`pedigree_graph._topology.build_topology`.  Supplied
            generation labels are never a substitute.
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
        depth,
        min_kinship,
        init_cap_per_row,
        config=_DP_CONFIG_CSC,
    )
    try:
        indptr, indices, values = _assemble_csc(r.cols, r.vals, r.row_start, r.row_count)
    except OverflowError as exc:
        nnz = int(np.sum(r.row_count, dtype=np.int64))
        raise ResourceError(
            "csc_index_overflow",
            "kinship matrix nnz exceeds the int32 CSC index range",
            nnz=nnz,
            maximum=int(np.iinfo(np.int32).max),
        ) from exc
    if r.order is None:
        return indptr, indices, values

    # The kernel ran in depth-major order; map the CSC back to caller order.
    # K_sorted[a, b] = phi(order[a], order[b]); the caller wants K[i, j] = phi(i, j)
    # = K_sorted[inv[i], inv[j]], i.e. row/column gather by the inverse permutation.
    k_sorted = sp.csc_matrix((values, indices, indptr), shape=(n, n))
    inv = np.empty(n, dtype=np.intp)
    inv[r.order] = np.arange(n, dtype=np.intp)
    k_caller = k_sorted[inv][:, inv].tocsc()
    k_caller.sort_indices()
    return k_caller.indptr, k_caller.indices, k_caller.data


def _fill_candidate_kinship_values(
    n: int,
    m_idx: np.ndarray,
    f_idx: np.ndarray,
    tw_idx: np.ndarray,
    depth: np.ndarray,
    candidate_indptr: np.ndarray,
    candidate_indices: np.ndarray,
    candidate_positions: np.ndarray,
    candidate_data: np.ndarray,
) -> None:
    """Fill topology-space candidate positions using the complete retiring DP.

    The candidate CSC stores only its upper triangle: column ``j`` contains
    candidate rows ``k <= j`` in stable depth-major space, while its data holds
    positions in the caller's full-symmetric output CSC. The DP keeps complete
    exact live rows, writes matching candidates, and retires rows normally.
    """
    _run_dp_core(
        n,
        m_idx,
        f_idx,
        tw_idx,
        depth,
        0.0,
        None,
        config=KinshipDPConfig(retire=True, lazy=True, debug_asserts=False),
        candidate_indptr=candidate_indptr,
        candidate_indices=candidate_indices,
        candidate_positions=candidate_positions,
        candidate_data=candidate_data,
    )


@njit(cache=True)
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


def _finalize_summary(
    sum_theta: np.ndarray,
    dense: np.ndarray,
    twin_idx: np.ndarray,
    observed: np.ndarray,
    n_unlabelled: int,
) -> GenerationKinshipSummary:
    """Turn per-bucket θ sums into a :class:`GenerationKinshipSummary`.

    ``sum_theta`` holds, per dense bucket, the kinship summed over unordered
    same-bucket pairs with MZ co-twin pairs left out (the kernel and the
    matrix walk both apply ``j != twin[i]``).  The denominator matches:
    ``n_g (n_g - 1) / 2`` minus the MZ pairs whose two co-twins are both in
    bucket ``g``.  A twin whose partner is unlabelled or in another bucket is
    an ordinary member.  The sentinel bucket (unlabelled rows) is dropped.
    """
    k = int(observed.shape[0])
    dense = np.asarray(dense, dtype=np.int32)
    twin = np.asarray(twin_idx, dtype=np.int32)
    labelled = dense < k
    n_per_g = np.bincount(dense[labelled], minlength=k).astype(np.int64)[:k]
    idx = np.arange(dense.shape[0], dtype=np.int32)
    same_group_twin = (twin > idx) & labelled
    same_group_twin[same_group_twin] &= dense[twin[same_group_twin]] == dense[same_group_twin]
    twin_per_g = np.bincount(dense[same_group_twin], minlength=k).astype(np.int64)[:k]
    pair_counts = n_per_g * (n_per_g - 1) // 2 - twin_per_g
    mean_kinship = np.full(k, np.nan, dtype=np.float64)
    eligible = pair_counts > 0
    mean_kinship[eligible] = np.asarray(sum_theta, dtype=np.float64)[:k][eligible] / pair_counts[eligible]
    return GenerationKinshipSummary(
        generations=observed,
        mean_kinship=mean_kinship,
        pair_counts=pair_counts,
        unlabelled_individual_count=n_unlabelled,
    )


def _summary_from_dp_rows(
    cols: np.ndarray,
    vals: np.ndarray,
    row_start: np.ndarray,
    row_count: np.ndarray,
    labels: np.ndarray,
    twin_idx: np.ndarray,
) -> GenerationKinshipSummary:
    """Generation kinship summary streamed from complete DP row storage.

    Post-hoc counterpart of the retiring DP's inline accumulator: walks the
    rows :func:`_dp_kinship` left behind (``retire=False``) and groups by
    ``labels`` without assembling a CSC.

    Args:
        cols: DP-output column indices array from :func:`_dp_kinship`.
        vals: DP-output kinship values array from :func:`_dp_kinship`.
        row_start: DP-output per-row offsets array from :func:`_dp_kinship`.
        row_count: DP-output per-row entry counts array from :func:`_dp_kinship`.
        labels: per-row cohort label in the same row space, ``-1`` unknown.
        twin_idx: per-row twin partner row index, ``-1`` for non-twins.
    """
    dense, observed, n_unlabelled = _densify_labels(labels)
    twin = np.ascontiguousarray(twin_idx, dtype=np.int32)
    sum_theta = _stream_sum_theta_per_gen(
        cols,
        vals,
        row_start,
        row_count,
        dense,
        twin,
        np.int32(observed.shape[0]),
    )
    return _finalize_summary(sum_theta, dense, twin, observed, n_unlabelled)


def _compute_generation_kinship_summary(
    n: int,
    m_idx: np.ndarray,
    f_idx: np.ndarray,
    tw_idx: np.ndarray,
    depth: np.ndarray,
    min_kinship: float,
    init_cap_per_row: int | None = None,
    _debug_no_retire: bool = False,
    _debug_asserts: bool = False,
    *,
    labels: np.ndarray | None = None,
) -> GenerationKinshipSummary:
    """Generation kinship summary without materializing K.

    The retiring DP frees rows in place at end-of-depth and accumulates θ̄
    inline during the merge walk; no CSC matrix and no full N × G row
    buffer ever co-exist.  ``labels`` are densified first
    (:func:`_densify_labels`), so the kernel's accumulator has one bucket
    per observed label plus one sentinel for unlabelled rows, whatever the
    label values are.

    ``_debug_no_retire=True`` falls back to a two-pass path (full DP then
    post-hoc walk) for parity testing.  ``_debug_asserts=True`` enables
    retire-correctness asserts inside the kernel; no effect under
    ``_debug_no_retire=True``.

    Args:
        n: same semantics as :func:`_build_kinship_csc`.
        m_idx: same semantics as :func:`_build_kinship_csc`.
        f_idx: same semantics as :func:`_build_kinship_csc`.
        tw_idx: same semantics as :func:`_build_kinship_csc`.
        depth: same semantics as :func:`_build_kinship_csc`.
        min_kinship: same semantics as :func:`_build_kinship_csc`.
        init_cap_per_row: same semantics as :func:`_build_kinship_csc`.
        labels: cohort labels the result is grouped by, ``-1`` unknown;
            ``None`` groups by *depth*.  Labels never affect the traversal
            or any kinship value.
    """
    raw = depth if labels is None else labels
    dense, observed, n_unlabelled = _densify_labels(raw)
    config = KinshipDPConfig(
        retire=not _debug_no_retire,
        lazy=not _debug_no_retire,
        debug_asserts=_debug_asserts and not _debug_no_retire,
    )
    r = _run_dp_core(
        n,
        m_idx,
        f_idx,
        tw_idx,
        depth,
        min_kinship,
        init_cap_per_row,
        labels=dense,
        config=config,
    )
    if _debug_no_retire:
        sum_theta = _stream_sum_theta_per_gen(
            r.cols,
            r.vals,
            r.row_start,
            r.row_count,
            r.labels,
            r.tw_idx,
            np.int32(observed.shape[0]),
        )
    else:
        sum_theta = r.sum_theta
    # r.labels / r.tw_idx are in the kernel's row space; the denominators only
    # need same-group membership of twin pairs, which any permutation keeps.
    return _finalize_summary(sum_theta, r.labels, r.tw_idx, observed, n_unlabelled)
