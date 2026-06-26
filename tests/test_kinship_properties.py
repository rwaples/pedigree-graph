"""Property-based tests for the kinship matrix and pairwise kinship.

Generalises the example-driven kinship tests across random pedigrees:
symmetry, bounds, the inbreeding-encoding diagonal, founder base cases,
the Mendelian quarter for parent-offspring, the kinship recursion, the
matrix-DP vs pairwise-recurrence consistency, and id-relabel invariance.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import non_inbred_pedigree, pedigree_arrays, random_pedigree, relabel_pedigree
from hypothesis import given, settings
from hypothesis import strategies as st

from pedigree_graph import PedigreeGraph

_SETTINGS = settings(deadline=None, max_examples=50)
_HEAVY = settings(deadline=None, max_examples=30)


@_SETTINGS
@given(pg=random_pedigree())
def test_kinship_matrix_symmetric_and_bounded(pg):
    K = pg.kinship_matrix(0.0)
    dense = K.toarray()
    assert np.allclose(dense, dense.T, atol=1e-9)
    assert np.all(K.data >= -1e-12)
    assert np.all(K.data <= 1.0 + 1e-12)


@_SETTINGS
@given(pg=random_pedigree())
def test_kinship_diagonal_encodes_inbreeding(pg):
    # The builders never set twins, so the matrix diagonal (1+F)/2 matches ML F.
    K = pg.kinship_matrix(0.0)
    F = pg.compute_inbreeding()
    assert np.allclose(np.asarray(K.diagonal()), 0.5 * (1.0 + F), atol=1e-9)


@_SETTINGS
@given(pg=random_pedigree())
def test_founders_unrelated_and_self_half(pg):
    K = pg.kinship_matrix(0.0).toarray()
    founders = np.where((pg.mother == -1) & (pg.father == -1))[0]
    for a in founders:
        assert K[a, a] == pytest.approx(0.5)
    for x in range(len(founders)):
        for y in range(x + 1, len(founders)):
            assert K[founders[x], founders[y]] == pytest.approx(0.0, abs=1e-12)


@_SETTINGS
@given(pg=non_inbred_pedigree())
def test_parent_offspring_quarter_when_non_inbred(pg):
    K = pg.kinship_matrix(0.0).toarray()
    for child in range(pg.n):
        for parent in (int(pg.mother[child]), int(pg.father[child])):
            if parent != -1:
                assert K[child, parent] == pytest.approx(0.25)


@_HEAVY
@given(pg=random_pedigree())
def test_kinship_recursion(pg):
    # phi(i,j) = 1/2 (phi(mother_i,j) + phi(father_i,j)) for i not an ancestor
    # of j (gen[i] >= gen[j], i != j); a missing parent contributes 0.
    K = pg.kinship_matrix(0.0).toarray()
    gen = pg.generation
    n = pg.n
    for i in range(n):
        m, f = int(pg.mother[i]), int(pg.father[i])
        if m == -1 and f == -1:
            continue  # founder: no recursion
        for j in range(n):
            if j == i or gen[j] > gen[i]:
                continue
            km = K[m, j] if m != -1 else 0.0
            kf = K[f, j] if f != -1 else 0.0
            assert K[i, j] == pytest.approx(0.5 * (km + kf), abs=1e-9)


@_HEAVY
@given(arrays=pedigree_arrays())
def test_compute_pair_kinship_matches_matrix(arrays):
    ids, mother, father, sex = arrays
    n = len(ids)
    if n < 2:
        return
    pg = PedigreeGraph.from_arrays(ids=ids, mothers=mother, fathers=father, sex=sex)
    a, b = np.triu_indices(n, k=1)
    # compute_pair_kinship runs the pairwise recurrence here (no matrix cached yet);
    # then the DP matrix is built and the two independent paths are compared.
    pairwise = pg.compute_pair_kinship({"all": (a, b)})["all"]
    K = pg.kinship_matrix(0.0).toarray()
    assert np.allclose(pairwise, K[a, b], atol=1e-9)


@_SETTINGS
@given(arrays=pedigree_arrays(), data=st.data())
def test_id_remap_invariance(arrays, data):
    ids, mother, father, sex = arrays
    pg1 = PedigreeGraph.from_arrays(ids=ids, mothers=mother, fathers=father, sex=sex)
    pg2 = relabel_pedigree(arrays, data)
    assert np.array_equal(pg1.kinship_matrix(0.0).toarray(), pg2.kinship_matrix(0.0).toarray())
    assert pg1.count_pairs(max_degree=3) == pg2.count_pairs(max_degree=3)
