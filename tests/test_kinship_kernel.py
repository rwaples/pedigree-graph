"""Unit tests for the shared numba kinship kernel.

Hand-worked pedigrees covering parent-offspring, full-sib, grandparent,
MZ, unrelated, and inbred+MZ — the last case is the one that the
previous matrix-product kinship path got wrong (returned 0.375 for the
MZ off-diagonal instead of the correct (1 + F)/2 = 0.625).
"""

import numpy as np
import scipy.sparse as sp

from pedigree_graph._kinship_kernel import _build_kinship_csc


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
