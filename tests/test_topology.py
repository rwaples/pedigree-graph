"""Unit tests for the private topological order (slice 1b)."""

from __future__ import annotations

import numpy as np
import pytest

from pedigree_graph._topology import Topology, build_topology, remap_rows, structural_depth

# 0, 1 and the disconnected 2 are founders; 3 = child(0, 1); 4 = child(3, 1).
DEPTH_MAJOR_MOTHER = np.array([-1, -1, -1, 0, 3], dtype=np.int32)
DEPTH_MAJOR_FATHER = np.array([-1, -1, -1, 1, 1], dtype=np.int32)


def _permute(mother, father, perm):
    """Rebuild parent arrays after moving reference row ``perm[k]`` to row ``k``."""
    inv = np.empty(len(perm), dtype=np.int32)
    inv[perm] = np.arange(len(perm), dtype=np.int32)
    out = []
    for arr in (mother, father):
        moved = arr[perm]
        out.append(np.where(moved < 0, np.int32(-1), inv[moved]).astype(np.int32))
    return out


class TestBuildTopology:
    def test_depth_major_input_stores_identity_as_none(self):
        topo = build_topology(structural_depth(DEPTH_MAJOR_MOTHER, DEPTH_MAJOR_FATHER, 5))
        assert topo.order is None
        assert topo.inverse is None
        assert topo.depth.tolist() == [0, 0, 0, 1, 2]
        assert topo.depth.dtype == np.int32

    def test_permuted_input_yields_a_topological_order(self):
        perm = np.array([3, 0, 4, 2, 1])
        mother, father = _permute(DEPTH_MAJOR_MOTHER, DEPTH_MAJOR_FATHER, perm)
        topo = build_topology(structural_depth(mother, father, 5))
        assert topo.order is not None
        assert topo.inverse is not None
        m_topo = topo.to_topological(mother)
        f_topo = topo.to_topological(father)
        for row in range(5):
            assert m_topo[row] < row
            assert f_topo[row] < row

    def test_ties_within_a_depth_keep_input_row_order(self):
        # Three founders and two children; the stable sort must not reshuffle
        # rows that share a depth.
        mother = np.array([-1, -1, -1, 0, 1], dtype=np.int32)
        father = np.array([-1, -1, -1, 2, 2], dtype=np.int32)
        topo = build_topology(structural_depth(mother, father, 5))
        assert topo.order is None

    def test_children_first_input_is_reversed_stably(self):
        perm = np.array([4, 3, 0, 1, 2])
        mother, father = _permute(DEPTH_MAJOR_MOTHER, DEPTH_MAJOR_FATHER, perm)
        topo = build_topology(structural_depth(mother, father, 5))
        assert topo.order is not None
        assert topo.depth[topo.order].tolist() == sorted(topo.depth.tolist())

    def test_empty_graph_is_the_identity(self):
        topo = build_topology(structural_depth(np.zeros(0, np.int32), np.zeros(0, np.int32), 0))
        assert topo.order is None
        assert topo.depth.shape == (0,)


class TestIdentityHelpers:
    @pytest.fixture
    def topo(self):
        return build_topology(structural_depth(DEPTH_MAJOR_MOTHER, DEPTH_MAJOR_FATHER, 5))

    def test_to_topological_returns_the_input_object(self, topo):
        assert topo.to_topological(DEPTH_MAJOR_MOTHER) is DEPTH_MAJOR_MOTHER

    def test_translate_returns_the_input_object(self, topo):
        rows = np.array([0, 3, -1])
        assert topo.translate(rows) is rows

    def test_gather_returns_the_input_object(self, topo):
        values = np.arange(5.0)
        assert topo.gather(values) is values

    def test_per_row_to_graph_returns_the_input_object(self, topo):
        values = np.arange(5.0)
        assert topo.per_row_to_graph(values) is values


class TestPermutedHelpers:
    @pytest.fixture
    def topo(self):
        perm = np.array([3, 0, 4, 2, 1])
        mother, father = _permute(DEPTH_MAJOR_MOTHER, DEPTH_MAJOR_FATHER, perm)
        return build_topology(structural_depth(mother, father, 5)), mother, father

    def test_to_topological_reorders_and_translates(self, topo):
        t, mother, _ = topo
        moved = t.to_topological(mother)
        assert moved.dtype == np.int32
        for position, row in enumerate(t.order):
            expected = mother[row]
            assert moved[position] == (-1 if expected < 0 else t.inverse[expected])

    def test_to_topological_passes_missing_parents_through(self, topo):
        t, mother, _ = topo
        moved = t.to_topological(mother)
        assert (moved < 0).sum() == (mother < 0).sum()

    def test_translate_maps_rows_and_passes_minus_one_through(self, topo):
        t, _, _ = topo
        rows = np.array([0, 1, 2, 3, 4, -1])
        translated = t.translate(rows)
        assert translated[:5].tolist() == t.inverse.tolist()
        assert translated[5] == -1

    def test_gather_and_per_row_to_graph_round_trip(self, topo):
        t, _, _ = topo
        values = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        assert t.per_row_to_graph(t.gather(values)).tolist() == values.tolist()

    def test_per_row_to_graph_places_positional_results_on_graph_rows(self, topo):
        t, _, _ = topo
        positional = np.arange(5)
        graph = t.per_row_to_graph(positional)
        for position, row in enumerate(t.order):
            assert graph[row] == positional[position]

    def test_gather_preserves_dtype(self, topo):
        t, _, _ = topo
        assert t.gather(t.depth).dtype == np.int32


def test_remap_rows_is_what_to_topological_uses():
    perm = np.array([3, 0, 4, 2, 1])
    mother, father = _permute(DEPTH_MAJOR_MOTHER, DEPTH_MAJOR_FATHER, perm)
    t = build_topology(structural_depth(mother, father, 5))
    assert remap_rows(mother, t.order, t.inverse).tolist() == t.to_topological(mother).tolist()


def test_topology_is_frozen():
    topo = build_topology(structural_depth(DEPTH_MAJOR_MOTHER, DEPTH_MAJOR_FATHER, 5))
    with pytest.raises(AttributeError):
        topo.depth = np.zeros(5, np.int32)
    assert isinstance(topo, Topology)
