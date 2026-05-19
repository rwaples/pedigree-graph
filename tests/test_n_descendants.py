"""Tests for ``PedigreeGraph.compute_n_descendants`` (path-count).

The method's docstring explicitly states that path-count semantics are
intentional: ``n_desc[v]`` counts (v, w) walks down the DAG, not unique
descendants.  The inbred-case test below pins that contract by
asserting the over-count, so a future "fix" to distinct semantics
will fail loudly.
"""

import numpy as np
import pytest

from pedigree_graph import PedigreeGraph


def _pg(ids, mothers, fathers):
    return PedigreeGraph.from_arrays(
        ids=np.asarray(ids),
        mothers=np.asarray(mothers),
        fathers=np.asarray(fathers),
    )


def test_terminals_have_zero_descendants():
    # 0, 1 are founders; 2 is their only child and a terminal.
    pg = _pg([0, 1, 2], [-1, -1, 0], [-1, -1, 1])
    np.testing.assert_array_equal(
        pg.compute_n_descendants(), np.array([1, 1, 0], dtype=np.int32),
    )


def test_deep_lineage_chain():
    # Each individual has one child, in a chain 0 -> 2 -> 3 -> 4.
    # 0 mother of 2; 1 father of 2 (founder); 2 mother of 3; 2's partner is 2
    # itself? No — use distinct founders at each step.
    #   0,1 founders -> 2
    #   2,5 founders for next gen -> 3
    #   3,6 founders -> 4
    pg = _pg(
        [0, 1, 2, 5, 3, 6, 4],
        [-1, -1, 0, -1, 2, -1, 3],
        [-1, -1, 1, -1, 5, -1, 6],
    )
    n_desc = pg.compute_n_descendants()
    # 0's descendants: 2, 3, 4 -> path count 3
    # 1's descendants: 2, 3, 4 -> 3
    # 2: 3, 4 -> 2
    # 5: 3, 4 -> 2
    # 3: 4 -> 1
    # 6: 4 -> 1
    # 4: terminal -> 0
    expected = {0: 3, 1: 3, 2: 2, 5: 2, 3: 1, 6: 1, 4: 0}
    # Row index of id k equals its position in the ids list above.
    ids_to_row = {0: 0, 1: 1, 2: 2, 5: 3, 3: 4, 6: 5, 4: 6}
    for k, v in expected.items():
        assert int(n_desc[ids_to_row[k]]) == v, k


def test_multi_component_pedigree():
    # Two disjoint trios: (0,1)->2 and (3,4)->5.  Counts should not bleed
    # across components.
    pg = _pg([0, 1, 2, 3, 4, 5], [-1, -1, 0, -1, -1, 3], [-1, -1, 1, -1, -1, 4])
    np.testing.assert_array_equal(
        pg.compute_n_descendants(),
        np.array([1, 1, 0, 1, 1, 0], dtype=np.int32),
    )


def test_inbred_pedigree_overcounts_paths():
    # 0,1 founders; 2,3 full sibs (children of 0,1); 4 = child of 2 x 3.
    # Distinct descendants of 0 are {2, 3, 4} = 3.
    # Path count: walks (0,2), (0,3), (0,2,4) via 2, (0,3,4) via 3 = 4.
    pg = _pg([0, 1, 2, 3, 4], [-1, -1, 0, 0, 2], [-1, -1, 1, 1, 3])
    n_desc = pg.compute_n_descendants()
    assert n_desc[0] == 4, "path count 4 (not the 3 distinct descendants)"
    assert n_desc[1] == 4
    assert n_desc[2] == 1
    assert n_desc[3] == 1
    assert n_desc[4] == 0


def test_returns_int32_and_caches():
    pg = _pg([0, 1, 2], [-1, -1, 0], [-1, -1, 1])
    first = pg.compute_n_descendants()
    assert first.dtype == np.int32
    second = pg.compute_n_descendants()
    assert first is second  # identity check on cache


def test_overflow_raises(monkeypatch):
    """A path count exceeding int32 max must raise ``OverflowError``,
    not silently wrap on the downcast.

    Real pedigrees would need billions of rows or pathological loops to
    blow int32, so we inject an overflowing int64 array through the
    kernel boundary.
    """
    pg = _pg([0, 1, 2], [-1, -1, 0], [-1, -1, 1])
    over = np.iinfo(np.int32).max + 1

    def fake_kernel(m, f, n):
        out = np.zeros(n, dtype=np.int64)
        out[0] = over
        return out

    import pedigree_graph._core as core
    monkeypatch.setattr(core, "_compute_n_descendants", fake_kernel)
    with pytest.raises(OverflowError, match="int32 max"):
        pg.compute_n_descendants()


def test_no_overflow_at_int32_max(monkeypatch):
    """Exactly ``int32_max`` is the boundary and must NOT raise."""
    pg = _pg([0, 1, 2], [-1, -1, 0], [-1, -1, 1])
    boundary = np.iinfo(np.int32).max

    def fake_kernel(m, f, n):
        out = np.zeros(n, dtype=np.int64)
        out[0] = boundary
        return out

    import pedigree_graph._core as core
    monkeypatch.setattr(core, "_compute_n_descendants", fake_kernel)
    result = pg.compute_n_descendants()
    assert result.dtype == np.int32
    assert int(result[0]) == boundary
