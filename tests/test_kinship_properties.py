"""Property-based tests for the kinship matrix and pairwise kinship.

Generalises the example-driven kinship tests across random pedigrees:
symmetry, bounds, founder base cases, the Mendelian quarter for
parent-offspring, the kinship recursion, the matrix-DP vs
pairwise-recurrence consistency, and id-relabel invariance.  The diagonal's
F identity is in test_inbreeding_properties.py.
"""

from __future__ import annotations

import numpy as np
import pytest
from _support import ENVELOPE_UNIT, EXACT_DEPTH_SUM
from conftest import non_inbred_pedigree, pedigree_arrays, random_pedigree, relabel_pedigree
from hypothesis import given, settings
from hypothesis import strategies as st

from pedigree_graph import PedigreeGraph

_SETTINGS = settings(deadline=None, max_examples=50)
_HEAVY = settings(deadline=None, max_examples=30)


@_SETTINGS
@given(pg=random_pedigree())
def test_kinship_matrix_symmetric_and_bounded(pg):
    # The CSC is mirrored from one stored value per pair, so symmetry is
    # exact; a float32 half-sum of values in [0, 1] stays in [0, 1].
    K = pg.kinship_matrix()
    dense = K.toarray()
    assert np.array_equal(dense, dense.T)
    assert np.all(K.data >= 0)
    assert np.all(K.data <= 1)


@_SETTINGS
@given(pg=random_pedigree())
def test_founders_unrelated_and_self_half(pg):
    K = pg.kinship_matrix().toarray()
    founders = np.where((pg.mother_rows == -1) & (pg.father_rows == -1))[0]
    for a in founders:
        assert K[a, a] == pytest.approx(0.5)
    for x in range(len(founders)):
        for y in range(x + 1, len(founders)):
            assert K[founders[x], founders[y]] == pytest.approx(0.0, abs=1e-12)


@_SETTINGS
@given(pg=non_inbred_pedigree())
def test_parent_offspring_quarter_when_non_inbred(pg):
    K = pg.kinship_matrix().toarray()
    for child in range(pg.n_individuals):
        for parent in (int(pg.mother_rows[child]), int(pg.father_rows[child])):
            if parent != -1:
                assert K[child, parent] == pytest.approx(0.25)


@_HEAVY
@given(pg=random_pedigree())
def test_kinship_recursion(pg):
    # phi(i,j) = 1/2 (phi(mother_i,j) + phi(father_i,j)) for i not an ancestor
    # of j (gen[i] >= gen[j], i != j); a missing parent contributes 0.  The
    # matrix stores the correctly rounded float32 half-sum when it peels i
    # (i deeper, or a depth tie with i the greater row, ADR 0009); the float64
    # sum of two float32 values is exact, so the cast is that one rounding.
    # A tie it peels from j reaches the same rational by the other parent
    # pair: equal while every value is exact (test_kinship_exact.py), inside
    # the two-path envelope beyond.
    K = pg.kinship_matrix().toarray()
    gen = np.asarray(pg.depth, dtype=np.int64)
    n = pg.n_individuals
    for i in range(n):
        m, f = int(pg.mother_rows[i]), int(pg.father_rows[i])
        if m == -1 and f == -1:
            continue  # founder: no recursion
        for j in range(n):
            if j == i or gen[j] > gen[i]:
                continue
            km = np.float64(K[m, j]) if m != -1 else 0.0
            kf = np.float64(K[f, j]) if f != -1 else 0.0
            step = np.float32(0.5 * (km + kf))
            depth_sum = gen[i] + gen[j]
            if gen[i] > gen[j] or i > j or depth_sum <= EXACT_DEPTH_SUM:
                assert K[i, j] == step
            else:
                assert abs(np.float64(K[i, j]) - np.float64(step)) <= 2 * (depth_sum + 1) * ENVELOPE_UNIT


@_HEAVY
@given(arrays=pedigree_arrays())
def test_pair_kinship_matches_matrix(arrays):
    ids, mother, father, sex = arrays
    n = len(ids)
    if n < 2:
        return
    pg = PedigreeGraph.from_arrays(ids=ids, mother_ids=mother, father_ids=father, sex=sex)
    a, b = np.triu_indices(n, k=1)
    # The recurrence and the DP matrix implement one pinned float32 recurrence
    # (ADR 0009), so the two independent paths agree to the bit.
    pairwise = pg.pair_kinship(a, b)
    K = pg.kinship_matrix().toarray()
    assert pairwise.tobytes() == K[a, b].tobytes()


@_SETTINGS
@given(arrays=pedigree_arrays(), data=st.data())
def test_id_remap_invariance(arrays, data):
    ids, mother, father, sex = arrays
    pg1 = PedigreeGraph.from_arrays(ids=ids, mother_ids=mother, father_ids=father, sex=sex)
    pg2 = relabel_pedigree(arrays, data)
    assert np.array_equal(pg1.kinship_matrix().toarray(), pg2.kinship_matrix().toarray())
    assert pg1.relationship_counts(max_degree=3) == pg2.relationship_counts(max_degree=3)
