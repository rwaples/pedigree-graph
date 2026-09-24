"""Smoke test for the host-side depth-major remap in ``tests/oracle/remap.py``.

The remap is test scaffolding: every Numba oracle sweeps through it, so the
oracle-versus-native differentials exercise it on every fixture.  This pins
its two shapes directly: identity for depth-major input, and a stable
depth-major order with a round-tripping inverse otherwise.
"""

from __future__ import annotations

import numpy as np
from oracle.remap import build_topology, remap_rows

from pedigree_graph._topology import structural_depth

# 0, 1 and the disconnected 2 are founders; 3 = child(0, 1); 4 = child(3, 1).
DEPTH_MAJOR_MOTHER = np.array([-1, -1, -1, 0, 3], dtype=np.int32)
DEPTH_MAJOR_FATHER = np.array([-1, -1, -1, 1, 1], dtype=np.int32)


def test_identity_and_permuted_inputs():
    identity = build_topology(structural_depth(DEPTH_MAJOR_MOTHER, DEPTH_MAJOR_FATHER))
    assert identity.order is None
    assert identity.depth.tolist() == [0, 0, 0, 1, 2]
    values = np.arange(5.0)
    assert identity.gather(values) is values
    assert identity.to_topological(DEPTH_MAJOR_MOTHER) is DEPTH_MAJOR_MOTHER

    # Move reference row perm[k] to row k and translate the parent rows.
    perm = np.array([3, 0, 4, 2, 1])
    inv = np.empty(5, dtype=np.int32)
    inv[perm] = np.arange(5, dtype=np.int32)
    mother, father = (
        np.where(a[perm] < 0, -1, inv[a[perm]]).astype(np.int32) for a in (DEPTH_MAJOR_MOTHER, DEPTH_MAJOR_FATHER)
    )
    topo = build_topology(structural_depth(mother, father))
    assert topo.order is not None
    assert topo.depth[topo.order].tolist() == sorted(topo.depth.tolist())
    for parents in (topo.to_topological(mother), topo.to_topological(father)):
        assert all(p < row for row, p in enumerate(parents))
    assert remap_rows(mother, topo.order, topo.inverse).tolist() == topo.to_topological(mother).tolist()
    values = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
    assert topo.per_row_to_graph(topo.gather(values)).tolist() == values.tolist()
