"""Hand-derived F tests for the Meuwissen-Luo kernel.

Covers founders, single-parent, classic non-trivial matings, deeper
chains, and parity vs. the matrix kinship path on MZ-free fixtures.
The MZ-naive semantics of ML are exercised against the MZ-aware matrix
on a co-coalescence case where the two paths legitimately disagree.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from pedigree_graph import PedigreeGraph
from pedigree_graph._kinship_kernel import (
    _build_kinship_csc,
    _compute_depth,
    _compute_F_meuwissen_luo,
)


def _F(m, f, n=None):
    m = np.asarray(m, dtype=np.int32)
    f = np.asarray(f, dtype=np.int32)
    if n is None:
        n = len(m)
    depth = _compute_depth(m, f, n)
    return _compute_F_meuwissen_luo(m, f, depth, n)


def _F_via_matrix(m, f, tw, gen, n=None):
    m = np.asarray(m, dtype=np.int32)
    f = np.asarray(f, dtype=np.int32)
    tw = np.asarray(tw, dtype=np.int32)
    gen = np.asarray(gen, dtype=np.int32)
    if n is None:
        n = len(m)
    indptr, indices, data = _build_kinship_csc(n, m, f, tw, gen, 0.0)
    K = sp.csc_matrix((data, indices, indptr), shape=(n, n))
    return 2.0 * K.diagonal() - 1.0


def test_all_founders_F_zero():
    F = _F([-1, -1, -1], [-1, -1, -1])
    assert np.allclose(F, [0.0, 0.0, 0.0])


def test_one_known_parent_F_zero():
    # Mother known, father missing — F stays 0 (the unknown parent breaks
    # any ancestry path).  Covers the -1 indexing footgun.
    F = _F([-1, -1, 0, -1], [-1, -1, -1, 0])
    assert np.allclose(F, [0.0, 0.0, 0.0, 0.0])


def test_full_sib_mating():
    # Founders 0, 1; full-sibs 2, 3 = child(0, 1); offspring 4 = child(2, 3).
    F = _F([-1, -1, 0, 0, 2], [-1, -1, 1, 1, 3])
    assert F[4] == pytest.approx(0.25)


def test_half_sib_mating():
    # Mother 0; fathers 1, 2.  Half-sibs 3 = (0, 1), 4 = (0, 2).
    # Offspring 5 = (3, 4).
    F = _F([-1, -1, -1, 0, 0, 3], [-1, -1, -1, 1, 2, 4])
    assert F[5] == pytest.approx(0.125)


def test_selfing():
    # Founder 0 selfed: parents = (0, 0).
    F = _F([-1, 0], [-1, 0])
    assert F[1] == pytest.approx(0.5)


def test_parent_offspring_mating():
    # 0, 1 founders; 2 = (0, 1); 3 = (0, 2): mother x grandchild via
    # 0's lineage.
    F = _F([-1, -1, 0, 0], [-1, -1, 1, 2])
    assert F[3] == pytest.approx(0.25)


def test_closed_line_5gen():
    # Crow & Kimura full-sib closed-line series:
    #   F_0 = 0, F_1 = 0, F_2 = 1/4, F_3 = 3/8, F_4 = 1/2,
    #   F_5 = 5/8 - 1/32 = 0.59375.
    n = 11
    m = [-1, -1, 0, 0, 2, 2, 4, 4, 6, 6, 8]
    f = [-1, -1, 1, 1, 3, 3, 5, 5, 7, 7, 9]
    F = _F(m, f, n)
    expected = [0.0, 0.0, 0.0, 0.0, 0.25, 0.25, 0.375, 0.375, 0.5, 0.5, 0.59375]
    assert np.allclose(F, expected)


def test_skip_gen_pedigree():
    # Reuses the layout from test_effective_size_scaling._build_skip_gen_pedigree
    # — skip-gen edges that depth = max(parent_depth) + 1 must handle.
    # IDs: 0, 1, 2, 3 founders; 4, 5 = (1, 0); 6 = (5, 4); 7 = (3, 2);
    # 8 = (3, 6) skip-gen; 9 = (7, 6).
    n = 10
    m = [-1, -1, -1, -1, 1, 1, 5, 3, 3, 7]
    f = [-1, -1, -1, -1, 0, 0, 4, 2, 6, 6]
    F_ml = _F(m, f, n)
    F_mat = _F_via_matrix(m, f, [-1] * n, [0, 0, 0, 0, 1, 1, 2, 1, 3, 3], n)
    assert np.allclose(F_ml, F_mat, atol=1e-12)


def test_deeper_chain_15gen():
    # 15-generation parent-offspring chain (mother only); F stays 0
    # throughout, but the depth bookkeeping should not break.
    n = 15
    m = [-1, *list(range(n - 1))]
    f = [-1] * n
    F = _F(m, f, n)
    assert np.allclose(F, np.zeros(n))


def test_parity_with_matrix_path_no_mz(small_pedigree: pd.DataFrame):
    # Strip MZ twin info from the fixture so matrix and ML F must agree
    # exactly (MZ-aware vs MZ-naive disagreement only kicks in for
    # MZ-coalescence cases).
    df = small_pedigree.copy()
    df["twin"] = -1
    pg = PedigreeGraph(df)
    F_ml = pg.compute_inbreeding()
    K = pg.kinship_matrix(min_kinship=0.0)
    F_mat = 2.0 * K.diagonal() - 1.0
    assert np.allclose(F_ml, F_mat, atol=1e-10)


def test_mz_naive_documented_difference():
    # MZ co-coalescence: 2, 3 are MZ twins (parents 0, 1); 4 = (2, 3).
    # Matrix (MZ-aware): K[2, 3] = self-kin = 0.5 → F[4] = 0.5.
    # ML (MZ-naive): treats 2, 3 as full-sibs → F[4] = 0.25.
    df = pd.DataFrame(
        {
            "id": [0, 1, 2, 3, 4],
            "mother": [-1, -1, 0, 0, 2],
            "father": [-1, -1, 1, 1, 3],
            "twin": [-1, -1, 3, 2, -1],
            "sex": [1, 0, 1, 0, 1],
            "generation": [0, 0, 1, 1, 2],
        }
    )
    pg = PedigreeGraph(df)
    F_ml = pg.compute_inbreeding()
    K = pg.kinship_matrix(min_kinship=0.0)
    F_mat = 2.0 * K.diagonal() - 1.0
    assert F_ml[4] == pytest.approx(0.25)
    assert F_mat[4] == pytest.approx(0.5)
    # Magnitude of the documented difference:
    assert F_mat[4] - F_ml[4] == pytest.approx(0.25)
