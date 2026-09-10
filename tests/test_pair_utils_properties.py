"""Property-based tests for the pure pair-array utilities in _pair_utils.

These operate on plain index arrays (no pedigree), so they are fast and exercise
within-group enumeration, the oriented read of an asymmetric product matrix
(canonicalisation and dual-valid deduplication included), and the graph-to-view
projection.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import scipy.sparse as sp
from hypothesis import given, settings
from hypothesis import strategies as st

from pedigree_graph._pair_utils import (
    oriented_pairs_from_sparse,
    pairs_from_groups,
    project_pairs,
)

_SETTINGS = settings(deadline=None, max_examples=100)


@_SETTINGS
@given(data=st.data())
def test_pairs_from_groups_enumerates_combinations(data):
    k = data.draw(st.integers(min_value=0, max_value=20))
    indices = np.array(
        data.draw(st.lists(st.integers(0, 10_000), min_size=k, max_size=k, unique=True)),
        dtype=np.intp,
    )
    groups = np.array(data.draw(st.lists(st.integers(0, 5), min_size=k, max_size=k)), dtype=np.intp)
    lo, hi = pairs_from_groups(indices, groups)
    assert np.all(lo <= hi)
    want = set()
    for label in set(groups.tolist()):
        members = sorted(int(indices[t]) for t in range(k) if groups[t] == label)
        want.update(combinations(members, 2))
    got = set(zip(lo.tolist(), hi.tolist(), strict=True))
    assert got == want
    assert len(lo) == len(want)  # exactly sum C(k_g, 2), no duplicates


@_SETTINGS
@given(data=st.data())
def test_oriented_pairs_from_sparse_drops_diagonal_and_dedups_by_lower_row(data):
    n = data.draw(st.integers(min_value=1, max_value=12))
    dense = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i != j and data.draw(st.booleans()):
                dense[i, j] = data.draw(st.integers(1, 3))
        dense[i, i] = data.draw(st.integers(0, 3))  # diagonal must be dropped
    row_is_first = data.draw(st.booleans())
    first, second = oriented_pairs_from_sparse(sp.csr_matrix(dense), row_is_first=row_is_first)
    got = list(zip(first.tolist(), second.tolist(), strict=True))
    want = set()
    for i in range(n):
        for j in range(n):
            if i != j and dense[i, j] > 0:
                a, b = (i, j) if row_is_first else (j, i)
                if dense[j, i] > 0:
                    a, b = min(i, j), max(i, j)
                want.add((a, b))
    assert set(got) == want
    assert len(got) == len(want)
    keys = [min(a, b) * n + max(a, b) for a, b in got]
    assert keys == sorted(keys)


@_SETTINGS
@given(data=st.data())
def test_project_pairs_keeps_exactly_the_both_selected_pairs_in_order(data):
    n = data.draw(st.integers(min_value=1, max_value=15))
    selected = data.draw(st.lists(st.integers(0, n - 1), unique=True, max_size=n))
    graph_to_view = np.full(n, -1, dtype=np.int32)
    graph_to_view[selected] = np.arange(len(selected), dtype=np.int32)
    m = data.draw(st.integers(min_value=0, max_value=15))
    first = np.array(data.draw(st.lists(st.integers(0, n - 1), min_size=m, max_size=m)), dtype=np.intp)
    second = np.array(data.draw(st.lists(st.integers(0, n - 1), min_size=m, max_size=m)), dtype=np.intp)
    view_first, view_second = project_pairs(first, second, graph_to_view)
    want = [
        (selected.index(int(a)), selected.index(int(b)))
        for a, b in zip(first.tolist(), second.tolist(), strict=True)
        if a in selected and b in selected
    ]
    assert list(zip(view_first.tolist(), view_second.tolist(), strict=True)) == want
    assert view_first.dtype == np.intp
    assert view_second.dtype == np.intp
