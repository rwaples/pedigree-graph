"""Differential and boundary tests for the Rust topology kernels (slice 10a)."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pedigree_graph import PedigreeGraph, PedigreeValidationError, _native
from pedigree_graph._topology import build_topology, structural_depth
from tests.oracle import topology as oracle


@st.composite
def parent_arrays(draw, max_rows=40, allow_cycle=False):
    """Parent rows in a random row order; optionally with one planted back edge."""
    n = draw(st.integers(min_value=0, max_value=max_rows))
    mother = np.full(n, -1, dtype=np.int32)
    father = np.full(n, -1, dtype=np.int32)
    for i in range(1, n):
        if draw(st.booleans()):
            mother[i] = draw(st.integers(min_value=0, max_value=i - 1))
        if draw(st.booleans()):
            father[i] = draw(st.integers(min_value=0, max_value=i - 1))
    perm = np.asarray(draw(st.permutations(range(n))), dtype=np.int64)
    inverse = np.empty(n, dtype=np.int64)
    inverse[perm] = np.arange(n)
    moved = []
    for arr in (mother, father):
        m = arr[perm]
        moved.append(np.where(m < 0, -1, inverse[np.maximum(m, 0)]).astype(np.int32))
    mother, father = moved
    if allow_cycle and n >= 2 and draw(st.booleans()):
        a = draw(st.integers(min_value=0, max_value=n - 1))
        b = draw(st.integers(min_value=0, max_value=n - 1).filter(lambda x: x != a))
        mother[a] = b
        father[b] = a
    ids = np.asarray(draw(st.permutations(range(n))), dtype=np.int64) * 7 + 3
    return ids, mother, father


def _witness(ids, mother, father):
    try:
        _native.validate_acyclic(ids, mother, father)
    except PedigreeValidationError as err:
        return (err.code, err.fields["ids"])
    return None


@settings(max_examples=300, deadline=None)
@given(parent_arrays(allow_cycle=True))
def test_cycle_witness_matches_the_oracle(arrays):
    ids, mother, father = arrays
    expected = oracle.cycle_witness(ids, mother, father)
    assert _witness(ids, mother, father) == (None if expected is None else ("cycle", expected))


@settings(max_examples=300, deadline=None)
@given(parent_arrays())
def test_acyclic_kernels_match_the_oracle(arrays):
    _ids, mother, father = arrays
    assert _native.is_topological(mother, father) == oracle.is_topological(mother, father)
    depth = _native.structural_depth(mother, father)
    assert depth.dtype == np.int32
    assert np.array_equal(depth, oracle.structural_depth(mother, father))
    native = _native.depth_major_order(depth)
    expected = oracle.depth_major_order(depth)
    if expected is None:
        assert native is None
    else:
        assert native is not None
        assert native[0].dtype == np.int64
        assert np.array_equal(native[0], expected[0])
        assert np.array_equal(native[1], expected[1])


class TestBoundary:
    def test_cycle_crosses_as_a_validation_error_with_fields(self):
        ids = np.array([10, 20, 30], dtype=np.int64)
        mother = np.array([1, 2, 0], dtype=np.int32)
        father = np.array([-1, -1, -1], dtype=np.int32)
        with pytest.raises(PedigreeValidationError, match="cycle through ids") as info:
            _native.validate_acyclic(ids, mother, father)
        assert info.value.code == "cycle"
        assert info.value.fields["ids"] == (10, 20, 30)
        assert isinstance(info.value, ValueError)

    def test_construction_reports_the_same_witness_as_the_kernel(self):
        frame = {"id": [30, 10, 20], "mother": [20, 30, 10], "father": [-1, -1, -1]}
        with pytest.raises(PedigreeValidationError) as info:
            PedigreeGraph.from_frame(frame)
        assert info.value.fields["ids"] == (10, 30, 20)

    def test_unequal_parent_lengths_are_rejected(self):
        with pytest.raises(ValueError, match="same length"):
            _native.is_topological(np.zeros(3, np.int32), np.zeros(2, np.int32))
        with pytest.raises(ValueError, match="same length"):
            _native.validate_acyclic(np.zeros(2, np.int64), np.zeros(3, np.int32), np.zeros(3, np.int32))

    def test_wrong_dtype_is_rejected(self):
        with pytest.raises(TypeError):
            _native.structural_depth(np.zeros(3, np.int64), np.zeros(3, np.int64))

    def test_empty_pedigree(self):
        empty = np.zeros(0, np.int32)
        assert _native.is_topological(empty, empty)
        assert _native.structural_depth(empty, empty).shape == (0,)
        assert _native.depth_major_order(empty) is None
        _native.validate_acyclic(np.zeros(0, np.int64), empty, empty)

    def test_core_version_is_the_distribution_version(self):
        import importlib.metadata

        assert _native.core_version() == importlib.metadata.version("pedigree-graph")


def test_facade_returns_read_only_depth_and_intp_maps():
    mother = np.array([3, -1, 0, -1], dtype=np.int32)
    father = np.array([1, -1, 1, -1], dtype=np.int32)
    depth = structural_depth(mother, father)
    assert not depth.flags.writeable
    topo = build_topology(depth)
    assert topo.order is not None
    assert topo.order.dtype == np.intp
    assert topo.inverse.dtype == np.intp
