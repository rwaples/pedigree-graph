"""Hand-derived F tests for the Meuwissen-Luo kernel.

Covers founders, single-parent, classic non-trivial matings, deeper
chains, and parity vs. the matrix kinship path.  The ADR 0008 fixtures pin
the MZ-aware semantics: F from the genome-node walk equals the pairwise
self-kinship identity ``2 * phi(i, i) - 1`` and the matrix diagonal on
every MZ co-coalescence case, including founder twins and inbred twins.
The MZ pair contract itself is a construction guard; ``test_construction.py``
owns that table.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
import scipy.sparse as sp

from pedigree_graph import PedigreeGraph
from pedigree_graph._kinship_kernel import (
    _build_kinship_csc,
    _compute_depth,
    _compute_F_meuwissen_luo,
)
from pedigree_graph._kinship_pairwise import pairwise_kinship


def _F(m, f, n=None, tw=None):
    m = np.asarray(m, dtype=np.int32)
    f = np.asarray(f, dtype=np.int32)
    if n is None:
        n = len(m)
    tw = np.full(n, -1, dtype=np.int32) if tw is None else np.asarray(tw, dtype=np.int32)
    depth = _compute_depth(m, f, n)
    return _compute_F_meuwissen_luo(m, f, tw, depth, n)


def _F_via_pairwise(m, f, tw):
    m = np.asarray(m, dtype=np.int32)
    f = np.asarray(f, dtype=np.int32)
    tw = np.asarray(tw, dtype=np.int32)
    rows = np.arange(len(m), dtype=np.int64)
    return 2.0 * pairwise_kinship(m, f, tw, rows, rows).astype(np.float64) - 1.0


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


def test_parity_with_matrix_path_no_mz(small_pedigree):
    # Strip MZ twin info from the fixture so matrix and ML F must agree
    # exactly (MZ-aware vs MZ-naive disagreement only kicks in for
    # MZ-coalescence cases).
    df = small_pedigree.with_columns(pl.lit(-1).cast(small_pedigree.schema["twin"]).alias("twin"))
    pg = PedigreeGraph.from_frame(df)
    F_ml = pg.inbreeding()
    K = pg.kinship_matrix()
    F_mat = 2.0 * K.diagonal() - 1.0
    assert np.allclose(F_ml, F_mat, atol=1e-10)


def _mz_frame(ids, mother, father, twin):
    m = np.asarray(mother, dtype=np.int32)
    f = np.asarray(father, dtype=np.int32)
    return pl.DataFrame(
        {
            "id": ids,
            "mother": mother,
            "father": father,
            "twin": twin,
            "sex": [0] * len(ids),
            "generation": _compute_depth(m, f, len(ids)).tolist(),
        }
    )


# (name, mother, father, twin, {row: expected F}); ids are 0..n-1, parents precede children.
ADR_0008_FIXTURES = [
    (
        "mz_ancestry_no_loop",
        [-1, -1, -1, -1, 0, 0, -1, 4],
        [-1, -1, -1, -1, 1, 1, -1, 2],
        [-1, -1, -1, -1, 5, 4, -1, -1],
        {7: 0.0},
    ),
    (
        "mz_only_link",
        [-1, -1, -1, -1, 0, 0, 4, 5, 6],
        [-1, -1, -1, -1, 1, 1, 2, 3, 7],
        [-1, -1, -1, -1, 5, 4, -1, -1, -1],
        {8: 1 / 8},
    ),
    (
        "mz_plus_full_sib_loop",
        [-1, -1, -1, -1, 2, 2, 0, 0, 6, 7, 8],
        [-1, -1, -1, -1, 3, 3, 1, 1, 4, 5, 9],
        [-1, -1, -1, -1, -1, -1, 7, 6, -1, -1, -1],
        {10: 3 / 16},
    ),
    (
        "double_mz_grandparents",
        [-1, -1, -1, -1, 0, 0, 2, 2, 4, 5, 8],
        [-1, -1, -1, -1, 1, 1, 3, 3, 6, 7, 9],
        [-1, -1, -1, -1, 5, 4, 7, 6, -1, -1, -1],
        {10: 1 / 4},
    ),
    (
        "founder_mz_twins_only_link",
        [-1, -1, -1, -1, 0, 1, 4],
        [-1, -1, -1, -1, 2, 3, 5],
        [1, 0, -1, -1, -1, -1, -1],
        {6: 1 / 8},
    ),
    (
        "inbred_twins",
        [-1, -1, 0, 0, -1, 2, 2, 5, 6, 7],
        [-1, -1, 1, 1, -1, 3, 3, 4, 4, 8],
        [-1, -1, -1, -1, -1, 6, 5, -1, -1, -1],
        {5: 1 / 4, 6: 1 / 4, 9: 0.25 * (0.625 + 0.5)},
    ),
    (
        "twins_mate_each_other",
        [-1, -1, 0, 0, 2],
        [-1, -1, 1, 1, 3],
        [-1, -1, 3, 2, -1],
        {4: 0.5},
    ),
]


@pytest.mark.parametrize(("name", "m", "f", "tw", "expected"), ADR_0008_FIXTURES, ids=[c[0] for c in ADR_0008_FIXTURES])
def test_mz_aware_fixtures(name, m, f, tw, expected):
    n = len(m)
    F_ml = _F(m, f, n, tw)
    F_pw = _F_via_pairwise(m, f, tw)
    F_mat = _F_via_matrix(m, f, tw, _compute_depth(np.asarray(m, dtype=np.int32), np.asarray(f, dtype=np.int32), n), n)
    for row, value in expected.items():
        assert F_ml[row] == pytest.approx(value), name
    np.testing.assert_allclose(F_ml, F_pw, atol=1e-12)
    np.testing.assert_allclose(F_ml, F_mat, atol=1e-12)


def test_mz_aware_fixtures_through_graph():
    _name, m, f, tw, expected = ADR_0008_FIXTURES[1]
    pg = PedigreeGraph.from_frame(_mz_frame(list(range(len(m))), m, f, tw))
    F = pg.inbreeding()
    K = pg.kinship_matrix()
    for row, value in expected.items():
        assert F[row] == pytest.approx(value)
    np.testing.assert_allclose(F, 2.0 * K.diagonal() - 1.0, atol=1e-12)


def test_parity_with_matrix_path_with_mz(small_pedigree):
    pg = PedigreeGraph.from_frame(small_pedigree)
    F_ml = pg.inbreeding()
    K = pg.kinship_matrix()
    assert np.allclose(F_ml, 2.0 * K.diagonal() - 1.0, atol=1e-10)


def test_absent_co_twin_is_not_an_mz_pair():
    # Co-twin outside the subsample remaps to -1: the row is an ordinary individual.
    full = _mz_frame([0, 1, 2, 3, 4], [-1, -1, 0, 0, 2], [-1, -1, 1, 1, 3], [-1, -1, 3, 2, -1])
    sub = full.filter(pl.col("id") != 3)
    F = PedigreeGraph.from_frame(sub).inbreeding()
    assert F[3] == pytest.approx(0.0)
