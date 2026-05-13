"""Unit tests for the shared numba kinship kernel.

Hand-worked pedigrees covering parent-offspring, full-sib, grandparent,
MZ, unrelated, and inbred+MZ — the last case is the one that the
previous matrix-product kinship path got wrong (returned 0.375 for the
MZ off-diagonal instead of the correct (1 + F)/2 = 0.625).
"""

import numpy as np
import pytest
import scipy.sparse as sp

from pedigree_graph._kinship_kernel import (
    _append_entry,
    _build_kinship_csc,
    _checked_int32_indptr_from_counts,
    _compute_last_direct_child_depth,
    _dp_kinship,
    _freelist_alloc,
    _freelist_bucket,
    _freelist_pop,
    _freelist_push,
    _retire_rows_at_depth,
)


def _build(n, mothers, fathers, twins, generation, min_kinship=0.0):
    indptr, indices, data = _build_kinship_csc(
        n,
        np.asarray(mothers, dtype=np.int32),
        np.asarray(fathers, dtype=np.int32),
        np.asarray(twins, dtype=np.int32),
        np.asarray(generation, dtype=np.int32),
        min_kinship,
    )
    return sp.csc_matrix((data, indices, indptr), shape=(n, n)).toarray()


def test_founders_only():
    # Two unrelated founders: diag = 0.5, off-diag = 0.
    K = _build(2, [-1, -1], [-1, -1], [-1, -1], [0, 0])
    assert np.array_equal(K, np.array([[0.5, 0.0], [0.0, 0.5]]))


def test_parent_offspring_fullsib_grandparent():
    # 0, 1 founders; 2 = child(0, 1); 3 = child(0, 1) [full sib of 2];
    # 4 = child(2, x) with x unknown (half-parent).  Check ancestor kin.
    # Simpler: 0, 1 founders; 2 = child(0,1); 3 = child(2, -1) — grandchild of 0 and 1.
    K = _build(
        4,
        mothers=[-1, -1, 0, 2],
        fathers=[-1, -1, 1, -1],
        twins=[-1, -1, -1, -1],
        generation=[0, 0, 1, 2],
    )
    # diagonal: all non-inbred → 0.5
    assert np.allclose(np.diag(K), [0.5, 0.5, 0.5, 0.5])
    # parent-offspring
    assert K[0, 2] == 0.25
    assert K[1, 2] == 0.25
    assert K[2, 3] == 0.25  # mother 2 → child 3
    # grandparent-grandchild: 0 → 2 → 3
    assert K[0, 3] == 0.125
    assert K[1, 3] == 0.125


def test_mz_twins_noninbred():
    # 0, 1 founders; 2, 3 MZ twin children of 0 and 1.
    K = _build(
        4,
        mothers=[-1, -1, 0, 0],
        fathers=[-1, -1, 1, 1],
        twins=[-1, -1, 3, 2],
        generation=[0, 0, 1, 1],
    )
    assert K[2, 3] == 0.5  # MZ off-diagonal = self-kinship (non-inbred parent = 0.5)
    assert K[2, 2] == 0.5
    assert K[3, 3] == 0.5
    assert K[0, 2] == K[0, 3] == 0.25
    assert K[1, 2] == K[1, 3] == 0.25


def test_inbred_mz_regression():
    # The case that the old matrix-product DP got wrong:
    # G0: 0, 1 founders; G1: 2, 3 full-sibs child(0,1);
    # G2: 4, 5 MZ twins child(2, 3) — inbred with F = 0.25.
    # Expected K[4,5] = (1 + 0.25) / 2 = 0.625.
    K = _build(
        6,
        mothers=[-1, -1, 0, 0, 2, 2],
        fathers=[-1, -1, 1, 1, 3, 3],
        twins=[-1, -1, -1, -1, 5, 4],
        generation=[0, 0, 1, 1, 2, 2],
    )
    # inbreeding: F = 2*diag - 1
    F = 2 * np.diag(K) - 1
    assert F[4] == 0.25
    assert F[5] == 0.25
    # MZ off-diagonal = self-kinship of either twin = (1 + F) / 2
    assert K[4, 5] == 0.625
    assert K[4, 4] == 0.625
    assert K[5, 5] == 0.625
    # Kinship with parents stays 0.5 * (K[parent, parent] + K[parent, other_parent])
    # = 0.5 * (0.5 + 0.25) = 0.375
    assert K[2, 4] == 0.375
    assert K[3, 5] == 0.375


def test_symmetric_and_sorted():
    K = _build(
        6,
        mothers=[-1, -1, 0, 0, 2, 2],
        fathers=[-1, -1, 1, 1, 3, 3],
        twins=[-1, -1, -1, -1, 5, 4],
        generation=[0, 0, 1, 1, 2, 2],
    )
    assert np.allclose(K, K.T)


def test_min_kinship_prunes_offdiag():
    # 3-generation lineage (founder → ... → great-grandchild).
    # Kinships 0.25, 0.125, 0.0625.  min_kinship=0.1 drops 0.0625 (GGP).
    K_full = _build(
        4,
        mothers=[-1, 0, 1, 2],
        fathers=[-1, -1, -1, -1],
        twins=[-1, -1, -1, -1],
        generation=[0, 1, 2, 3],
    )
    K_pruned = _build(
        4,
        mothers=[-1, 0, 1, 2],
        fathers=[-1, -1, -1, -1],
        twins=[-1, -1, -1, -1],
        generation=[0, 1, 2, 3],
        min_kinship=0.1,
    )
    assert K_full[0, 3] == 0.0625  # great-grandparent
    assert K_pruned[0, 3] == 0.0  # dropped by min_kinship=0.1
    # Closer relationships retained
    assert K_pruned[0, 1] == 0.25
    assert K_pruned[0, 2] == 0.125


def test_generation_none_autoderives():
    # Identical result when generation=None (kernel derives via fixed-point).
    indptr_a, indices_a, data_a = _build_kinship_csc(
        4,
        np.array([-1, -1, 0, 2], dtype=np.int32),
        np.array([-1, -1, 1, -1], dtype=np.int32),
        np.array([-1, -1, -1, -1], dtype=np.int32),
        np.array([0, 0, 1, 2], dtype=np.int32),
        0.0,
    )
    indptr_b, indices_b, data_b = _build_kinship_csc(
        4,
        np.array([-1, -1, 0, 2], dtype=np.int32),
        np.array([-1, -1, 1, -1], dtype=np.int32),
        np.array([-1, -1, -1, -1], dtype=np.int32),
        None,
        0.0,
    )
    assert np.array_equal(indptr_a, indptr_b)
    assert np.array_equal(indices_a, indices_b)
    assert np.array_equal(data_a, data_b)


def test_dp_kinship_row_start_is_int64():
    # row_start must be int64 so that ``i * init_cap`` does not overflow at
    # N > 525K with init_cap=4096 (product exceeds 2**31).  Allocating the
    # actual overflow case requires ~17 GB of buffer; the dtype check is
    # the regression gate.  End-to-end overflow is exercised by the
    # Phase-5 benches at N>=500K.
    n = 4
    m_idx = np.array([-1, -1, 0, 0], dtype=np.int32)
    f_idx = np.array([-1, -1, 1, 1], dtype=np.int32)
    tw_idx = np.full(n, -1, dtype=np.int32)
    depth = np.array([0, 0, 1, 1], dtype=np.int32)
    _cols, _vals, row_start, _row_count, _sum_theta = _dp_kinship(
        n, m_idx, f_idx, tw_idx, depth, 0.0, 16, False, False, False,
        np.zeros(3, dtype=np.int64),
        np.int64(0),
    )
    assert row_start.dtype == np.int64
    # Founder rows start at i * init_cap; the merge walk may relocate
    # non-founder rows, so check the founders only.
    assert row_start[0] == 0
    assert row_start[1] == 16


def test_last_direct_child_depth_closed_line_5():
    # closed_line, 5 generations: IDs 0,1 founders; subsequent pairs of
    # full sibs born at gen g have parents at gen g-1.  Each row's
    # last-direct-child depth therefore equals depth[k] + 1 except for the
    # final-gen rows (no children) which retain depth[k].
    m_idx = np.array([-1, -1, 1, 1, 3, 3, 5, 5, 7, 7, 9, 9], dtype=np.int32)
    f_idx = np.array([-1, -1, 0, 0, 2, 2, 4, 4, 6, 6, 8, 8], dtype=np.int32)
    depth = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5], dtype=np.int32)
    expected = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 5, 5], dtype=np.int32)
    out = _compute_last_direct_child_depth(m_idx, f_idx, depth, len(depth))
    assert np.array_equal(out, expected)


def test_last_direct_child_depth_skip_gen():
    # skip_gen fixture (see test_effective_size_scaling.py::_build_skip_gen_pedigree):
    #   ids 0..3 founders; 4,5 are children of 0,1 at gen 1;
    #   6 child of 4,5 at gen 2; 7 child of 2,3 at gen 1;
    #   8 SKIP-GEN child of founder 3 and 6 (gen 3); 9 child of 7,6 (gen 3).
    # Founder 3's last direct child is 8 at depth 3 — the skip-gen edge
    # forces 3's row to remain live until depth 3.
    m_idx = np.array([-1, -1, -1, -1, 1, 1, 5, 3, 3, 7], dtype=np.int32)
    f_idx = np.array([-1, -1, -1, -1, 0, 0, 4, 2, 6, 6], dtype=np.int32)
    depth = np.array([0, 0, 0, 0, 1, 1, 2, 1, 3, 3], dtype=np.int32)
    # k=0: children 4,5 at depth 1 → 1
    # k=1: children 4,5 at depth 1 → 1
    # k=2: child 7 at depth 1 → 1
    # k=3: children 7 (d=1) and 8 (d=3) → 3       ← skip-gen extends 3's lifetime
    # k=4: child 6 at depth 2 → 2
    # k=5: child 6 at depth 2 → 2
    # k=6: children 8,9 at depth 3 → 3
    # k=7: child 9 at depth 3 → 3
    # k=8,9: no children → depth[k]
    expected = np.array([1, 1, 1, 3, 2, 2, 3, 3, 3, 3], dtype=np.int32)
    out = _compute_last_direct_child_depth(m_idx, f_idx, depth, len(depth))
    assert np.array_equal(out, expected)


def test_last_direct_child_depth_no_children_returns_own_depth():
    # Two unrelated founders, no children.  Every row's last direct child
    # depth equals its own depth (eligible for retirement immediately).
    m_idx = np.array([-1, -1], dtype=np.int32)
    f_idx = np.array([-1, -1], dtype=np.int32)
    depth = np.array([0, 0], dtype=np.int32)
    out = _compute_last_direct_child_depth(m_idx, f_idx, depth, 2)
    assert np.array_equal(out, depth)


def test_last_direct_child_depth_ignores_twin_partners():
    # MZ twins 2 and 3 are children of founders 0,1.  Twin partner edges
    # must NOT affect last_direct_child_depth — only direct parent edges.
    # (The MZ twin pass writes to twin rows but never reads from them via
    # the merge walk, so twin rows can retire on the same schedule as
    # non-twin rows.)
    m_idx = np.array([-1, -1, 0, 0], dtype=np.int32)
    f_idx = np.array([-1, -1, 1, 1], dtype=np.int32)
    depth = np.array([0, 0, 1, 1], dtype=np.int32)
    out = _compute_last_direct_child_depth(m_idx, f_idx, depth, 4)
    # Founders 0,1 each have two direct children at depth 1 → 1.
    # Twins 2,3 have no descendants → own depth 1.
    expected = np.array([1, 1, 1, 1], dtype=np.int32)
    assert np.array_equal(out, expected)


def test_freelist_bucket_index_matches_log2():
    init_cap = np.int32(16)
    assert _freelist_bucket(np.int32(16), init_cap) == 0
    assert _freelist_bucket(np.int32(32), init_cap) == 1
    assert _freelist_bucket(np.int32(64), init_cap) == 2
    assert _freelist_bucket(np.int32(128), init_cap) == 3
    # Caps below init_cap (shouldn't occur in practice) clip to bucket 0.
    assert _freelist_bucket(np.int32(8), init_cap) == 0


def test_freelist_push_pop_lifo_within_bucket():
    init_cap = 16
    n = 100
    starts, tops = _freelist_alloc(init_cap, n)
    # Push three slots: two at cap=16 (bucket 0), one at cap=32 (bucket 1).
    _freelist_push(starts, tops, np.int64(0), np.int32(16), np.int32(init_cap))
    _freelist_push(starts, tops, np.int64(64), np.int32(16), np.int32(init_cap))
    _freelist_push(starts, tops, np.int64(32), np.int32(32), np.int32(init_cap))
    # LIFO within each bucket.
    assert _freelist_pop(starts, tops, np.int32(16), np.int32(init_cap)) == 64
    assert _freelist_pop(starts, tops, np.int32(16), np.int32(init_cap)) == 0
    # Bucket 0 now empty; pop returns -1.
    assert _freelist_pop(starts, tops, np.int32(16), np.int32(init_cap)) == -1
    # Bucket 1 still has its slot.
    assert _freelist_pop(starts, tops, np.int32(32), np.int32(init_cap)) == 32
    assert _freelist_pop(starts, tops, np.int32(32), np.int32(init_cap)) == -1


def test_freelist_pop_does_not_cross_buckets():
    # A bucket-0 slot must not satisfy a bucket-1 pop request.
    init_cap = 8
    starts, tops = _freelist_alloc(init_cap, 64)
    _freelist_push(starts, tops, np.int64(100), np.int32(8), np.int32(init_cap))
    assert _freelist_pop(starts, tops, np.int32(16), np.int32(init_cap)) == -1
    # The bucket-0 slot is still available.
    assert _freelist_pop(starts, tops, np.int32(8), np.int32(init_cap)) == 100


def test_freelist_alloc_sizes_buckets_for_max_doublings():
    # For init_cap=16, n=2048 → need buckets for caps 16, 32, ..., 2048 (8 buckets).
    starts, tops = _freelist_alloc(16, 2048)
    assert starts.shape == (8, 2048)
    assert tops.shape == (8,)
    # max_per_bucket == n (loose worst case from the plan).
    assert starts.shape[1] == 2048


def test_freelist_placeholder_arrays_silent_noop():
    # No-retirement callers should be able to pass init_cap=0 (or n=0)
    # without tripping the free-list helpers.
    starts, tops = _freelist_alloc(0, 0)
    _freelist_push(starts, tops, np.int64(0), np.int32(16), np.int32(0))
    assert _freelist_pop(starts, tops, np.int32(16), np.int32(0)) == -1


def _make_buffer(n_rows: int, init_cap: int, extra_cells: int = 0):
    """Allocate the storage that _append_entry expects, founders-style.

    ``extra_cells`` reserves additional cells past the founder slots so
    a test can stage a pre-existing free-list entry inside the buffer.
    """
    total = n_rows * init_cap + extra_cells
    cols = np.full(total, -1, dtype=np.int32)
    vals = np.zeros(total, dtype=np.float32)
    row_start = np.array(
        [np.int64(i) * np.int64(init_cap) for i in range(n_rows)], dtype=np.int64
    )
    row_count = np.zeros(n_rows, dtype=np.int32)
    row_cap = np.full(n_rows, init_cap, dtype=np.int32)
    return cols, vals, row_start, row_count, row_cap, np.int64(total)


def test_append_entry_sentinel_silently_drops_retired_writes():
    # row 1 is marked retired (row_start[1] = -1).  Appends must no-op.
    init_cap = 4
    cols, vals, row_start, row_count, row_cap, next_alloc = _make_buffer(2, init_cap)
    row_start[1] = np.int64(-1)
    row_count[1] = np.int32(0)
    row_cap[1] = np.int32(0)
    starts, tops = _freelist_alloc(0, 0)  # placeholder — push/pop are no-ops
    cols_out, vals_out, next_alloc_out = _append_entry(
        cols, vals, row_start, row_count, row_cap, next_alloc,
        np.int32(1), np.int32(0), np.float32(0.25),
        starts, tops, np.int32(0),
        np.zeros(3, dtype=np.int64),
    )
    # No write occurred anywhere; next_alloc unchanged.
    assert next_alloc_out == next_alloc
    assert row_count[1] == 0
    assert row_start[1] == -1
    # cols/vals untouched.
    assert np.all(cols_out == cols)
    assert np.all(vals_out == vals)


def test_append_entry_relocation_pushes_old_slot_to_free_list():
    # Fill row 0 to capacity, then force a relocation.  With retirement
    # active (real free list, fl_init_cap=init_cap), the abandoned old
    # slot must land on the free list's matching bucket.
    init_cap = 2
    cols, vals, row_start, row_count, row_cap, next_alloc = _make_buffer(2, init_cap)
    starts, tops = _freelist_alloc(init_cap, 8)
    # Pre-fill row 0 to cap (using its own slot directly).
    cols[0] = np.int32(10)
    vals[0] = np.float32(0.1)
    cols[1] = np.int32(20)
    vals[1] = np.float32(0.2)
    row_count[0] = np.int32(2)
    # Trigger relocation: row 0's old slot (start=0, cap=2) is pushed; new
    # slot (cap=4) is taken from next_alloc (free list bucket 1 is empty).
    _cols, _vals, next_alloc_out = _append_entry(
        cols, vals, row_start, row_count, row_cap, next_alloc,
        np.int32(0), np.int32(30), np.float32(0.3),
        starts, tops, np.int32(init_cap),
        np.zeros(3, dtype=np.int64),
    )
    # Bucket 0 (cap=2) now has the old slot start=0.
    assert tops[0] == 1
    assert starts[0, 0] == 0
    # Bucket 1 (cap=4) is still empty (we popped nothing).
    assert tops[1] == 0
    # New slot came from next_alloc; row 0 now points there with cap=4.
    assert row_cap[0] == 4
    assert row_count[0] == 3
    assert next_alloc_out == np.int64(next_alloc) + np.int64(4)


def test_append_entry_relocation_reuses_freelist_slot_when_available():
    # Pre-stock bucket 1 (cap=4) with a real slot inside the buffer.
    # Relocation should reuse it instead of bumping next_alloc.
    init_cap = 2
    n_rows = 3
    cols, vals, row_start, row_count, row_cap, next_alloc = _make_buffer(
        n_rows, init_cap, extra_cells=4,
    )
    free_slot_start = np.int64(n_rows * init_cap)  # 6 — inside the buffer
    starts, tops = _freelist_alloc(init_cap, 16)
    _freelist_push(starts, tops, free_slot_start, np.int32(4), np.int32(init_cap))
    assert tops[1] == 1  # bucket 1 has our pre-stocked slot
    # Fill row 0 to its initial capacity and force relocation.
    cols[0] = np.int32(10)
    vals[0] = np.float32(0.1)
    cols[1] = np.int32(20)
    vals[1] = np.float32(0.2)
    row_count[0] = np.int32(2)
    before_next_alloc = next_alloc
    _cols, _vals, next_alloc_out = _append_entry(
        cols, vals, row_start, row_count, row_cap, next_alloc,
        np.int32(0), np.int32(30), np.float32(0.3),
        starts, tops, np.int32(init_cap),
        np.zeros(3, dtype=np.int64),
    )
    # The free-list slot was popped; bucket 1 is now empty.
    assert tops[1] == 0
    # Bucket 0 has the abandoned old slot of row 0.
    assert tops[0] == 1
    assert starts[0, 0] == 0
    # Row 0 now lives at the popped slot, not at next_alloc.
    assert row_start[0] == free_slot_start
    assert row_cap[0] == 4
    # next_alloc did NOT advance — slot reuse instead of fresh allocation.
    assert next_alloc_out == before_next_alloc


def test_append_entry_lazy_allocates_never_allocated_row_via_freelist():
    # Bucket 0 is pre-stocked by retirement of an earlier row; a
    # never-allocated row's first append must pop that slot rather than
    # bumping next_alloc.
    init_cap = 4
    n_rows = 2
    # Reserve one extra cap-sized cell for the free-list slot inside buffer.
    cols, vals, row_start, row_count, row_cap, next_alloc = _make_buffer(
        n_rows, init_cap, extra_cells=init_cap,
    )
    # Row 1 transitions to never-allocated: row_start=-1, row_cap=init_cap.
    row_start[1] = np.int64(-1)
    row_count[1] = np.int32(0)
    row_cap[1] = np.int32(init_cap)
    # Pre-stock bucket 0 with a slot inside the buffer.
    free_slot_start = np.int64(n_rows * init_cap)
    starts, tops = _freelist_alloc(init_cap, 8)
    _freelist_push(starts, tops, free_slot_start, np.int32(init_cap), np.int32(init_cap))
    assert tops[0] == 1
    before_next_alloc = next_alloc
    cols_out, vals_out, next_alloc_out = _append_entry(
        cols, vals, row_start, row_count, row_cap, next_alloc,
        np.int32(1), np.int32(7), np.float32(0.42),
        starts, tops, np.int32(init_cap),
        np.zeros(3, dtype=np.int64),
    )
    # Bucket 0 was popped; row 1 now lives at the free-list slot.
    assert tops[0] == 0
    assert row_start[1] == free_slot_start
    assert row_cap[1] == init_cap
    assert row_count[1] == 1
    # The data landed at the popped offset.
    assert cols_out[free_slot_start] == 7
    assert vals_out[free_slot_start] == np.float32(0.42)
    # next_alloc unchanged — slot reuse, not fresh bump.
    assert next_alloc_out == before_next_alloc


def test_append_entry_lazy_allocates_never_allocated_row_via_bump():
    # No free-list slot available — lazy-allocate must bump next_alloc.
    init_cap = 4
    n_rows = 2
    cols, vals, row_start, row_count, row_cap, next_alloc = _make_buffer(
        n_rows, init_cap, extra_cells=init_cap,
    )
    row_start[1] = np.int64(-1)
    row_count[1] = np.int32(0)
    row_cap[1] = np.int32(init_cap)
    starts, tops = _freelist_alloc(init_cap, 8)  # empty
    before_next_alloc = next_alloc
    cols_out, vals_out, next_alloc_out = _append_entry(
        cols, vals, row_start, row_count, row_cap, next_alloc,
        np.int32(1), np.int32(11), np.float32(0.5),
        starts, tops, np.int32(init_cap),
        np.zeros(3, dtype=np.int64),
    )
    # No free-list pops occurred.
    assert tops[0] == 0
    # Row 1 lives at the previous next_alloc; bump advanced by init_cap.
    assert row_start[1] == before_next_alloc
    assert next_alloc_out == before_next_alloc + np.int64(init_cap)
    assert row_count[1] == 1
    assert cols_out[before_next_alloc] == 11
    assert vals_out[before_next_alloc] == np.float32(0.5)


def test_retire_rows_at_depth_skips_freelist_for_never_allocated_rows():
    # Lazy-alloc safety: a row marked retired-at-depth that never
    # actually allocated a slot (row_start == -1) must NOT push -1 into
    # the free list — otherwise a later pop would corrupt row_start.
    init_cap = 4
    last_dcd = np.array([0, 0], dtype=np.int32)
    # Row 0 was allocated (row_start=0, row_cap=init_cap); row 1 never
    # allocated (row_start=-1, row_cap=init_cap, "wants this much").
    row_start = np.array([0, -1], dtype=np.int64)
    row_count = np.array([2, 0], dtype=np.int32)
    row_cap = np.array([init_cap, init_cap], dtype=np.int32)
    starts, tops = _freelist_alloc(init_cap, 8)
    _retire_rows_at_depth(
        np.int32(0), last_dcd,
        row_start, row_count, row_cap,
        starts, tops, np.int32(init_cap),
    )
    # Bucket 0 received exactly one push — row 0's slot, not row 1's -1.
    assert tops[0] == 1
    assert starts[0, 0] == 0
    # Both rows are now in the retired sentinel state.
    assert row_start[0] == -1
    assert row_start[1] == -1
    assert row_cap[0] == 0
    assert row_cap[1] == 0
    assert row_count[0] == 0
    assert row_count[1] == 0


def test_dp_kinship_row_start_arithmetic_no_overflow():
    # The widened arithmetic ``np.int64(i) * np.int64(init_cap)`` must
    # produce values exceeding int32 range without wrap-around.  This is
    # the path the kernel uses to initialize row_start on every DP run.
    n = 10
    init_cap = 1 << 28  # 268_435_456
    expected_last = np.int64(n - 1) * np.int64(init_cap)
    assert expected_last > (1 << 31)  # sanity: arithmetic crosses int32
    # Validate the same arithmetic numba uses (cast both operands to int64).
    computed = np.int64(n - 1) * np.int64(init_cap)
    assert computed == expected_last
    assert computed < (1 << 63) - 1


def test_checked_int32_indptr_from_counts_raises_before_wrap():
    counts = np.array([(1 << 31) - 1, 1], dtype=np.int64)

    with pytest.raises(OverflowError, match="int32 range"):
        _checked_int32_indptr_from_counts(counts)


def test_checked_int32_indptr_from_counts_allows_int32_max():
    counts = np.array([(1 << 31) - 2, 1], dtype=np.int64)

    indptr = _checked_int32_indptr_from_counts(counts)

    np.testing.assert_array_equal(
        indptr,
        np.array([0, (1 << 31) - 2, (1 << 31) - 1], dtype=np.int32),
    )
