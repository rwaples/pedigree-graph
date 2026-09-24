"""Lineage counts and component IDs on the public 0.8 surface.

Each hand-computed case says in its name which semantic it pins: distinct
ancestors, descendant *paths*, and parent-edge components labelled by the
smallest original ID.  Descendant counts are path counts on purpose (the
method's docstring says so), so the looped case asserts the over-count.  The
fixture sweeps check distinct ancestors against an independent Python-set
closure and component IDs against an independent scipy labelling.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import FIXTURES, parity_columns, parity_graph
from oracle.relationship_pairs import _Matrices
from scipy.sparse.csgraph import connected_components

from pedigree_graph import PedigreeGraph


def _graph(ids, mother, father, twin=None) -> PedigreeGraph:
    columns = {"id": np.asarray(ids), "mother": np.asarray(mother), "father": np.asarray(father)}
    if twin is not None:
        columns["twin"] = np.asarray(twin)
    return PedigreeGraph.from_frame(columns)


# 0 x 1 -> 2, 3; 2 x 3 -> 4.  Ancestor 0 reaches 4 through both 2 and 3.
_LOOP = ([0, 1, 2, 3, 4], [-1, -1, 0, 0, 2], [-1, -1, 1, 1, 3])


def test_distinct_ancestors_count_a_looped_ancestor_once():
    np.testing.assert_array_equal(_graph(*_LOOP).distinct_ancestor_counts(), [0, 0, 2, 2, 4])


def test_descendant_paths_count_a_looped_descendant_per_path():
    # Row 0: children 2 and 3, and 4 once through each of them.
    np.testing.assert_array_equal(_graph(*_LOOP).descendant_path_counts(), [4, 4, 1, 1, 0])


def test_a_deep_chain_counts_every_generation():
    # 0 x 1 -> 2; 2 x 5 -> 3; 3 x 6 -> 4, on rows in id order [0, 1, 2, 5, 3, 6, 4].
    pg = _graph([0, 1, 2, 5, 3, 6, 4], [-1, -1, 0, -1, 2, -1, 3], [-1, -1, 1, -1, 5, -1, 6])
    np.testing.assert_array_equal(pg.distinct_ancestor_counts(), [0, 0, 2, 0, 4, 0, 6])
    np.testing.assert_array_equal(pg.descendant_path_counts(), [3, 3, 2, 2, 1, 1, 0])


@pytest.mark.parametrize(
    ("ids", "mothers", "fathers", "expected"),
    [
        ([0, 1, 2], [-1, -1, 0], [-1, -1, 1], [0, 0, 2]),
        ([0, 1], [-1, 0], [-1, -1], [0, 1]),  # one known parent
        ([0, 1, 2, 3], [-1, -1, 0, 2], [-1, -1, 1, 1], [0, 0, 2, 3]),  # backcross to a founder
    ],
)
def test_distinct_ancestor_edge_cases(ids, mothers, fathers, expected):
    np.testing.assert_array_equal(_graph(ids, mothers, fathers).distinct_ancestor_counts(), expected)


def test_external_parents_add_no_edge_and_no_ancestor():
    # 10 and 11 both name the absent mother 99; 12 is 10's child by 11.
    pg = _graph([10, 11, 12], [99, 99, 10], [-1, -1, 11])
    np.testing.assert_array_equal(pg.distinct_ancestor_counts(), [0, 0, 2])
    np.testing.assert_array_equal(pg.descendant_path_counts(), [1, 1, 0])
    np.testing.assert_array_equal(pg.connected_component_ids(), [10, 10, 10])


def test_two_half_founders_sharing_an_external_parent_stay_apart():
    pg = _graph([5, 7], [99, 99], [-1, -1])
    np.testing.assert_array_equal(pg.connected_component_ids(), [5, 7])


def test_component_id_is_the_smallest_id_not_the_first_row():
    # Three components; in each the smallest ID sits on a later row.
    ids = [50, 20, 8, 60, 30, 9, 70, 1]
    mother = [-1, 50, 50, -1, 60, 60, -1, 70]
    father = [-1, -1, -1, -1, -1, -1, -1, -1]
    pg = _graph(ids, mother, father)
    np.testing.assert_array_equal(pg.connected_component_ids(), [8, 8, 8, 9, 9, 9, 1, 1])


def test_mz_co_twins_are_joined_only_through_parents():
    # Founder twins with no represented parents: separate components.
    pg = _graph([3, 4], [-1, -1], [-1, -1], twin=[4, 3])
    np.testing.assert_array_equal(pg.connected_component_ids(), [3, 4])
    # Twins that share represented parents are one component with them.
    pg = _graph([0, 1, 2, 3], [-1, -1, 0, 0], [-1, -1, 1, 1], twin=[-1, -1, 3, 2])
    np.testing.assert_array_equal(pg.connected_component_ids(), [0, 0, 0, 0])


def test_a_disconnected_pedigree_has_one_label_per_component():
    pg = _graph([0, 1, 2, 3, 4, 5], [-1, -1, 0, -1, -1, 3], [-1, -1, 1, -1, -1, 4])
    np.testing.assert_array_equal(pg.connected_component_ids(), [0, 0, 0, 3, 3, 3])
    np.testing.assert_array_equal(pg.distinct_ancestor_counts(), [0, 0, 2, 0, 0, 2])
    np.testing.assert_array_equal(pg.descendant_path_counts(), [1, 1, 0, 1, 1, 0])


def test_empty_graph():
    pg = _graph([], [], [])
    assert pg.distinct_ancestor_counts().shape == (0,)
    assert pg.descendant_path_counts().shape == (0,)
    assert pg.connected_component_ids().shape == (0,)


@pytest.mark.parametrize(
    ("method", "dtype"),
    [
        ("distinct_ancestor_counts", np.int32),
        ("descendant_path_counts", np.int64),
        ("connected_component_ids", np.int64),
    ],
)
def test_results_are_read_only_typed_and_memoised(method, dtype):
    pg = _graph(*_LOOP)
    first = getattr(pg, method)()
    assert first.dtype == dtype
    assert not first.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        first[0] = 0
    assert getattr(pg, method)() is first


def _scipy_founder_family_ids(pg: PedigreeGraph) -> np.ndarray:
    """A scipy oracle for ``connected_component_ids``: smallest id per component.

    This was fitACE's own construction, reaching into ``_Am`` and ``_Af``, until
    ``grm_io.founder_family_ids`` became a call to ``connected_component_ids``.
    It is kept because an independent implementation is what makes the
    comparison worth running, not because anyone still writes it this way.
    """
    _, labels = connected_components(_Matrices(pg)._A, directed=False)
    comp_min = np.full(int(labels.max()) + 1, np.iinfo(np.int64).max, dtype=np.int64)
    np.minimum.at(comp_min, labels, pg.ids)
    return comp_min[labels].astype(np.int64)


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_component_ids_match_a_scipy_oracle(name):
    fixture = FIXTURES[name]
    if len(fixture["ids"]) == 0:
        pytest.skip("no components in an empty fixture")
    pg = PedigreeGraph.from_frame(parity_columns(fixture))
    np.testing.assert_array_equal(pg.connected_component_ids(), _scipy_founder_family_ids(pg))


def _set_oracle(pg: PedigreeGraph) -> np.ndarray:
    """Independent Python-set ancestor counts in graph rows, swept parents first."""
    mother, father = np.asarray(pg.mother_rows), np.asarray(pg.father_rows)
    closed: dict[int, set[int]] = {}
    counts = np.zeros(pg.n_individuals, dtype=np.int32)
    for i in np.argsort(pg.depth, kind="stable").tolist():
        ancestors: set[int] = set()
        for p in (int(mother[i]), int(father[i])):
            if p >= 0:
                ancestors.update(closed[p])
        counts[i] = len(ancestors)
        ancestors.add(i)
        closed[i] = ancestors
    return counts


@pytest.mark.parametrize("name", ["random_1k", "deep_inbred_60g"])
def test_distinct_ancestors_match_a_set_oracle(name):
    pg = parity_graph(name)
    np.testing.assert_array_equal(pg.distinct_ancestor_counts(), _set_oracle(pg))


def test_distinct_ancestors_map_shuffled_rows_back_to_graph_space():
    pg = _graph([4, 1, 3, 0, 2], [2, -1, 2, -1, 0], [3, -1, 1, -1, 1])
    assert not pg._built.rows_topological
    np.testing.assert_array_equal(pg.distinct_ancestor_counts(), _set_oracle(pg))
