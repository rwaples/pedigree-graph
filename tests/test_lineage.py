"""Lineage counts and component IDs on the public 0.8 surface (slice 6b).

Each hand-computed case says in its name which semantic it pins: distinct
ancestors, descendant *paths*, and parent-edge components labelled by the
smallest original ID.  The fixture sweep at the end replicates fitACE's
current ``founder_family_ids`` construction inline and checks the public
call gives the same labels.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import parity_columns, parity_fixtures
from scipy.sparse.csgraph import connected_components

from pedigree_graph import PedigreeGraph, ResourceError


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


def test_adapters_return_the_same_values_as_writeable_int32():
    pg = _graph(*_LOOP)
    np.testing.assert_array_equal(pg.compute_n_ancestors(), pg.distinct_ancestor_counts())
    np.testing.assert_array_equal(pg.compute_n_descendants(), pg.descendant_path_counts())
    assert pg.compute_n_ancestors().dtype == np.int32
    assert pg.compute_n_descendants().dtype == np.int32
    assert pg.compute_n_ancestors().flags.writeable
    assert pg.compute_n_descendants().flags.writeable
    assert pg.compute_n_ancestors() is pg.compute_n_ancestors()


def test_adapter_overflow_leaves_the_int64_surface_usable(monkeypatch):
    import pedigree_graph._lineage as lineage

    over = np.iinfo(np.int32).max + 1

    def fake_kernel(m, f, n):
        out = np.zeros(n, dtype=np.int64)
        out[0] = over
        return out

    monkeypatch.setattr(lineage, "_compute_n_descendants", fake_kernel)
    pg = _graph([0, 1, 2], [-1, -1, 0], [-1, -1, 1])
    assert int(pg.descendant_path_counts()[0]) == over
    with pytest.raises(ResourceError) as info:
        pg.compute_n_descendants()
    assert info.value.code == "arithmetic_overflow"


FIXTURES = parity_fixtures("random_1k", "deep_inbred_60g")


def _fitace_founder_family_ids(pg: PedigreeGraph) -> np.ndarray:
    """fitACE's current construction (grm_io.founder_family_ids), inline."""
    pg._ensure_parent_csr()
    _, labels = connected_components(pg._Am + pg._Af, directed=False)
    comp_min = np.full(int(labels.max()) + 1, np.iinfo(np.int64).max, dtype=np.int64)
    np.minimum.at(comp_min, labels, pg.ids)
    return comp_min[labels].astype(np.int64)


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_component_ids_match_the_fitace_construction(name):
    fixture = FIXTURES[name]
    if len(fixture["ids"]) == 0:
        pytest.skip("no components in an empty fixture")
    pg = PedigreeGraph(parity_columns(fixture))
    np.testing.assert_array_equal(pg.connected_component_ids(), _fitace_founder_family_ids(pg))


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_lineage_surfaces_match_the_adapters_on_fixtures(name):
    pg = PedigreeGraph(parity_columns(FIXTURES[name]))
    np.testing.assert_array_equal(pg.distinct_ancestor_counts(), pg.compute_n_ancestors())
    try:
        old = pg.compute_n_descendants()
    except ResourceError:
        assert int(pg.descendant_path_counts().max()) > np.iinfo(np.int32).max
    else:
        np.testing.assert_array_equal(pg.descendant_path_counts(), old)
