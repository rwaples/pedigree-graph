"""Unit tests for the parallel BFS pair-enumeration kernel.

Hand-built CSR inputs verify the two-pass parallel kernel emits the
right pair set for symmetric (single-CSR) and asymmetric (two-CSR)
modes. JIT compile cost is paid on first call.
"""

import numpy as np

from pedigree_graph._bfs_kernel import _enumerate_pairs_kernel


def _pairs_set(out_i: np.ndarray, out_j: np.ndarray) -> set[tuple[int, int]]:
    return {(int(i), int(j)) for i, j in zip(out_i, out_j, strict=True)}


def test_symmetric_emits_upper_triangle():
    # 2 anchors. Anchor 0 has descendants {1, 2, 3}; anchor 1 has {3}.
    # Symmetric mode should emit C(3,2)=3 pairs from anchor 0 and 0 from anchor 1.
    indptr = np.array([0, 3, 4], dtype=np.int64)
    indices = np.array([1, 2, 3, 3], dtype=np.int64)
    out_i, out_j = _enumerate_pairs_kernel(
        indptr, indices, indptr, indices, n=2, symmetric=True,
    )
    assert _pairs_set(out_i, out_j) == {(1, 2), (1, 3), (2, 3)}


def test_symmetric_empty_anchor():
    # Anchor 0 has 0 descendants; anchor 1 has 1 descendant. No pairs.
    indptr = np.array([0, 0, 1], dtype=np.int64)
    indices = np.array([5], dtype=np.int64)
    out_i, out_j = _enumerate_pairs_kernel(
        indptr, indices, indptr, indices, n=2, symmetric=True,
    )
    assert out_i.size == 0
    assert out_j.size == 0


def test_asymmetric_emits_cross_product():
    # 2 anchors. A: anchor 0 has {1, 2}; anchor 1 has {3}.
    #            B: anchor 0 has {10};   anchor 1 has {11, 12}.
    # Asymmetric mode emits |A_X| * |B_X| pairs per anchor.
    indptr_a = np.array([0, 2, 3], dtype=np.int64)
    indices_a = np.array([1, 2, 3], dtype=np.int64)
    indptr_b = np.array([0, 1, 3], dtype=np.int64)
    indices_b = np.array([10, 11, 12], dtype=np.int64)
    out_i, out_j = _enumerate_pairs_kernel(
        indptr_a, indices_a, indptr_b, indices_b, n=2, symmetric=False,
    )
    # Anchor 0: (1, 10), (2, 10). Anchor 1: (3, 11), (3, 12).
    assert _pairs_set(out_i, out_j) == {(1, 10), (2, 10), (3, 11), (3, 12)}


def test_all_empty():
    indptr = np.array([0, 0, 0], dtype=np.int64)
    indices = np.array([], dtype=np.int64)
    out_i, _ = _enumerate_pairs_kernel(
        indptr, indices, indptr, indices, n=2, symmetric=True,
    )
    assert out_i.size == 0
    out_i, _ = _enumerate_pairs_kernel(
        indptr, indices, indptr, indices, n=2, symmetric=False,
    )
    assert out_i.size == 0
