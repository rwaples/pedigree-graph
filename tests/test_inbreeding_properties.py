"""Property-based tests for inbreeding (Meuwissen-Luo F).

F is non-negative, agrees with the matrix-DP diagonal (no twins), and equals
phi(mother, father) for individuals with both parents known (0 otherwise).
"""

from __future__ import annotations

import numpy as np
from conftest import pedigree_arrays, random_pedigree
from hypothesis import example, given, settings

from pedigree_graph import PedigreeGraph

_SETTINGS = settings(deadline=None, max_examples=50)

# Full-sib mating: child 4 = (sib 2, sib 3) where 2,3 = (0,1); F[4] = phi(2,3) = 0.25.
_FULL_SIB_MATING = (
    np.array([0, 1, 2, 3, 4], dtype=np.int64),
    np.array([-1, -1, 0, 0, 2], dtype=np.int64),
    np.array([-1, -1, 1, 1, 3], dtype=np.int64),
    np.array([0, 1, 0, 1, 0], dtype=np.int8),
)


@_SETTINGS
@given(pg=random_pedigree())
def test_inbreeding_nonneg_and_matches_matrix_diagonal(pg):
    F = pg.inbreeding()
    assert np.all(F >= -1e-12)
    K = pg.kinship_matrix(0.0)
    assert np.allclose(F, 2.0 * np.asarray(K.diagonal()) - 1.0, atol=1e-9)


@_SETTINGS
@example(arrays=_FULL_SIB_MATING)
@given(arrays=pedigree_arrays())
def test_inbreeding_equals_parent_kinship(arrays):
    ids, mother, father, sex = arrays
    pg = PedigreeGraph.from_arrays(ids=ids, mothers=mother, fathers=father, sex=sex)
    F = pg.inbreeding()

    both = (mother != -1) & (father != -1)
    # Founders and one-parent rows have no inbreeding path.
    assert np.all(F[~both] == 0.0)
    if both.any():
        idx = np.where(both)[0]
        phi = pg.compute_pair_kinship({"p": (mother[idx].astype(np.int64), father[idx].astype(np.int64))})["p"]
        assert np.allclose(F[idx], phi, atol=1e-9)
