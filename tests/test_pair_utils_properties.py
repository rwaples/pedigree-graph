"""Property-based tests for the pure pair-array utilities in _pair_utils.

These operate on plain index arrays (no pedigree), so they are fast and exercise
canonicalisation, deduplication, within-group enumeration, the int64 pair-key
encoding (incl. large indices), and caller-space remapping.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import scipy.sparse as sp
from hypothesis import given, settings
from hypothesis import strategies as st

from pedigree_graph._pair_utils import (
    dedup_pairs,
    extract_from_sparse,
    pairs_from_groups,
    remap_pairs_to_caller,
)

_SETTINGS = settings(deadline=None, max_examples=100)


@_SETTINGS
@given(data=st.data())
def test_dedup_pairs_canonical_dedup_idempotent(data):
    m = data.draw(st.integers(min_value=0, max_value=30))
    a_i = np.array(data.draw(st.lists(st.integers(0, 1000), min_size=m, max_size=m)), dtype=np.intp)
    a_j = np.array(data.draw(st.lists(st.integers(0, 1000), min_size=m, max_size=m)), dtype=np.intp)
    lo, hi = dedup_pairs(a_i, a_j)
    assert np.all(lo <= hi)
    got = set(zip(lo.tolist(), hi.tolist(), strict=True))
    want = {(min(i, j), max(i, j)) for i, j in zip(a_i.tolist(), a_j.tolist(), strict=True)}
    assert got == want
    # Idempotent: re-deduping canonical pairs is a no-op.
    lo2, hi2 = dedup_pairs(lo, hi)
    assert set(zip(lo2.tolist(), hi2.tolist(), strict=True)) == got


@_SETTINGS
@given(data=st.data())
def test_dedup_pairs_large_indices(data):
    # Stress the lo*max_id+hi int64 key with large (but in-range, < ~3e9) indices;
    # the encoding must stay collision-free below the documented overflow limit.
    n = data.draw(st.integers(min_value=1, max_value=20))
    base = data.draw(st.sampled_from([10**6, 10**8, 2**30]))
    vals = np.array(
        data.draw(st.lists(st.integers(0, base), min_size=2 * n, max_size=2 * n, unique=True)),
        dtype=np.intp,
    )
    a_i, a_j = vals[:n], vals[n:]
    lo, hi = dedup_pairs(a_i, a_j)
    got = set(zip(lo.tolist(), hi.tolist(), strict=True))
    want = {(min(int(i), int(j)), max(int(i), int(j))) for i, j in zip(a_i, a_j, strict=True)}
    assert got == want


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
def test_extract_from_sparse_drops_diagonal(data):
    n = data.draw(st.integers(min_value=1, max_value=12))
    dense = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            if data.draw(st.booleans()):
                w = data.draw(st.integers(1, 3))
                dense[i, j] = dense[j, i] = w
        dense[i, i] = data.draw(st.integers(0, 3))  # diagonal must be dropped
    lo, hi = extract_from_sparse(sp.csr_matrix(dense))
    assert np.all(lo < hi)  # no self-pairs, canonical
    got = set(zip(lo.tolist(), hi.tolist(), strict=True))
    want = {(i, j) for i in range(n) for j in range(i + 1, n) if dense[i, j] > 0}
    assert got == want


@_SETTINGS
@given(data=st.data())
def test_remap_pairs_to_caller_recanonicalizes(data):
    n = data.draw(st.integers(min_value=1, max_value=15))
    remap = np.array(data.draw(st.permutations(range(n))), dtype=np.intp)
    m = data.draw(st.integers(min_value=0, max_value=15))
    gi = np.array(data.draw(st.lists(st.integers(0, n - 1), min_size=m, max_size=m)), dtype=np.intp)
    gj = np.array(data.draw(st.lists(st.integers(0, n - 1), min_size=m, max_size=m)), dtype=np.intp)
    out = remap_pairs_to_caller({"X": (gi, gj)}, remap)
    lo, hi = out["X"]
    assert np.all(lo <= hi)
    want = [(min(int(remap[a]), int(remap[b])), max(int(remap[a]), int(remap[b]))) for a, b in zip(gi, gj, strict=True)]
    assert list(zip(lo.tolist(), hi.tolist(), strict=True)) == want
