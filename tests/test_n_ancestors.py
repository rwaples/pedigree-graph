"""Tests for ``PedigreeGraph.compute_n_ancestors`` (distinct count).

Pins the *distinct* semantic (as opposed to path-count): in an inbred
pedigree where an ancestor is reachable through multiple paths, the
count is 1, not the number of paths.
"""

import numpy as np

from pedigree_graph import PedigreeGraph


def _pg(ids, mothers, fathers):
    return PedigreeGraph.from_arrays(
        ids=np.asarray(ids),
        mothers=np.asarray(mothers),
        fathers=np.asarray(fathers),
    )


def test_founders_have_zero_ancestors():
    pg = _pg([0, 1, 2], [-1, -1, 0], [-1, -1, 1])
    np.testing.assert_array_equal(
        pg.compute_n_ancestors(), np.array([0, 0, 2], dtype=np.int32),
    )


def test_half_founder_one_known_parent():
    # 0 founder; 1 has mother=0, father unknown.
    pg = _pg([0, 1], [-1, 0], [-1, -1])
    n_anc = pg.compute_n_ancestors()
    assert n_anc[0] == 0
    assert n_anc[1] == 1  # only mother is a known ancestor


def test_deep_lineage_chain():
    # Same chain as the descendants test, mirrored.  Ancestors of 4
    # should be the full set above it: {3, 6, 2, 5, 0, 1} = 6 unique.
    pg = _pg(
        [0, 1, 2, 5, 3, 6, 4],
        [-1, -1, 0, -1, 2, -1, 3],
        [-1, -1, 1, -1, 5, -1, 6],
    )
    n_anc = pg.compute_n_ancestors()
    ids_to_row = {0: 0, 1: 1, 2: 2, 5: 3, 3: 4, 6: 5, 4: 6}
    expected = {0: 0, 1: 0, 2: 2, 5: 0, 3: 4, 6: 0, 4: 6}
    for k, v in expected.items():
        assert int(n_anc[ids_to_row[k]]) == v, k


def test_inbred_pedigree_counts_distinct_not_paths():
    # 0,1 founders; 2,3 full sibs (children of 0,1); 4 = child of 2 x 3.
    # Distinct ancestors of 4: {2, 3, 0, 1} = 4.
    # (Contrast with descendants: 0 has 4 path descendants, not 3.)
    pg = _pg([0, 1, 2, 3, 4], [-1, -1, 0, 0, 2], [-1, -1, 1, 1, 3])
    n_anc = pg.compute_n_ancestors()
    np.testing.assert_array_equal(n_anc, [0, 0, 2, 2, 4])


def test_multi_component_pedigree():
    pg = _pg([0, 1, 2, 3, 4, 5], [-1, -1, 0, -1, -1, 3], [-1, -1, 1, -1, -1, 4])
    np.testing.assert_array_equal(
        pg.compute_n_ancestors(),
        np.array([0, 0, 2, 0, 0, 2], dtype=np.int32),
    )


def test_returns_int32_and_caches():
    pg = _pg([0, 1, 2], [-1, -1, 0], [-1, -1, 1])
    first = pg.compute_n_ancestors()
    assert first.dtype == np.int32
    second = pg.compute_n_ancestors()
    assert first is second
