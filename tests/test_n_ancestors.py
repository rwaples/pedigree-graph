"""Tests for ``PedigreeGraph.distinct_ancestor_counts``.

Pins the *distinct* semantic (as opposed to path-count): in an inbred
pedigree where an ancestor is reachable through multiple paths, the
count is 1, not the number of paths.
"""

import numpy as np
import pytest
from conftest import parity_columns, parity_fixtures

from pedigree_graph import PedigreeGraph
from pedigree_graph._lineage_kernel import _compute_n_ancestors, _compute_n_ancestors_profiled


def _pg(ids, mothers, fathers) -> PedigreeGraph:
    return PedigreeGraph.from_arrays(
        ids=np.asarray(ids),
        mother_ids=np.asarray(mothers),
        father_ids=np.asarray(fathers),
    )


def _set_oracle(pg: PedigreeGraph) -> np.ndarray:
    """Independent Python-set ancestor counts, returned in graph-space rows."""
    mother, father, _ = pg._topological_parents
    closed: list[set[int]] = []
    counts = np.zeros(pg.n_individuals, dtype=np.int32)
    for i, (m, f) in enumerate(zip(mother, father, strict=True)):
        ancestors: set[int] = set()
        if m >= 0:
            ancestors.update(closed[m])
        if f >= 0:
            ancestors.update(closed[f])
        counts[i] = len(ancestors)
        ancestors.add(i)
        closed.append(ancestors)
    return pg._topology.per_row_to_graph(counts)


def test_founders_have_zero_ancestors():
    pg = _pg([0, 1, 2], [-1, -1, 0], [-1, -1, 1])
    np.testing.assert_array_equal(
        pg.distinct_ancestor_counts(),
        np.array([0, 0, 2], dtype=np.int32),
    )


def test_half_founder_one_known_parent():
    # 0 founder; 1 has mother=0, father unknown.
    pg = _pg([0, 1], [-1, 0], [-1, -1])
    n_anc = pg.distinct_ancestor_counts()
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
    n_anc = pg.distinct_ancestor_counts()
    ids_to_row = {0: 0, 1: 1, 2: 2, 5: 3, 3: 4, 6: 5, 4: 6}
    expected = {0: 0, 1: 0, 2: 2, 5: 0, 3: 4, 6: 0, 4: 6}
    for k, v in expected.items():
        assert int(n_anc[ids_to_row[k]]) == v, k


def test_inbred_pedigree_counts_distinct_not_paths():
    # 0,1 founders; 2,3 full sibs (children of 0,1); 4 = child of 2 x 3.
    # Distinct ancestors of 4: {2, 3, 0, 1} = 4.
    # (Contrast with descendants: 0 has 4 path descendants, not 3.)
    pg = _pg([0, 1, 2, 3, 4], [-1, -1, 0, 0, 2], [-1, -1, 1, 1, 3])
    n_anc = pg.distinct_ancestor_counts()
    np.testing.assert_array_equal(n_anc, [0, 0, 2, 2, 4])


def test_multi_component_pedigree():
    pg = _pg([0, 1, 2, 3, 4, 5], [-1, -1, 0, -1, -1, 3], [-1, -1, 1, -1, -1, 4])
    np.testing.assert_array_equal(
        pg.distinct_ancestor_counts(),
        np.array([0, 0, 2, 0, 0, 2], dtype=np.int32),
    )


def test_returns_int32_and_caches():
    pg = _pg([0, 1, 2], [-1, -1, 0], [-1, -1, 1])
    first = pg.distinct_ancestor_counts()
    assert first.dtype == np.int32
    second = pg.distinct_ancestor_counts()
    assert first is second


@pytest.mark.parametrize(
    ("ids", "mothers", "fathers", "expected"),
    [
        ([], [], [], []),
        ([0, 1, 2], [-1, -1, 0], [-1, -1, 1], [0, 0, 2]),
        ([0, 1], [-1, 0], [-1, -1], [0, 1]),
        ([0, 1, 2, 3, 4], [-1, -1, 0, 0, 2], [-1, -1, 1, 1, 3], [0, 0, 2, 2, 4]),
        ([0, 1, 2, 3], [-1, -1, 0, 2], [-1, -1, 1, 1], [0, 0, 2, 3]),
    ],
)
def test_retiring_dp_edge_cases(ids, mothers, fathers, expected):
    pg = _pg(ids, mothers, fathers)
    np.testing.assert_array_equal(pg.distinct_ancestor_counts(), expected)


def test_retiring_dp_handles_the_same_parent_in_both_roles():
    mother = np.array([-1, 0], dtype=np.int32)
    father = np.array([-1, 0], dtype=np.int32)
    np.testing.assert_array_equal(_compute_n_ancestors(mother, father, 2), [0, 1])


@pytest.mark.parametrize("name", ["random_1k", "deep_inbred_60g"])
def test_retiring_dp_matches_set_oracle_on_parity_fixtures(name):
    pg = PedigreeGraph.from_frame(parity_columns(parity_fixtures(name)[name]))
    np.testing.assert_array_equal(pg.distinct_ancestor_counts(), _set_oracle(pg))


def test_retiring_dp_maps_shuffled_rows_back_to_graph_space():
    columns = {
        "id": np.array([4, 1, 3, 0, 2]),
        "mother": np.array([2, -1, 2, -1, 0]),
        "father": np.array([3, -1, 1, -1, 1]),
    }
    pg = PedigreeGraph.from_frame(columns)
    assert not pg._rows_are_topological
    np.testing.assert_array_equal(pg.distinct_ancestor_counts(), _set_oracle(pg))


def test_retiring_dp_reports_reused_storage():
    pg = _pg(
        list(range(10)),
        [-1, -1, 0, 1, 2, 3, 4, 5, 6, 7],
        [-1] * 10,
    )
    mother, father, _ = pg._topological_parents
    counts, peak_live, highwater, allocated, scanned, reused = _compute_n_ancestors_profiled(
        mother, father, pg.n_individuals
    )
    np.testing.assert_array_equal(counts, _set_oracle(pg))
    assert 0 < peak_live <= highwater <= allocated
    assert scanned > 0
    assert reused > 0
