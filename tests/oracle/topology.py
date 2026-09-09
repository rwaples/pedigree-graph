"""Oracles for ``pedigree_graph._native``'s topology kernels.

``structural_depth`` is the fixed-point sweep that was the numba kernel through
0.8.0; ``cycle_witness`` is the Kahn peel plus the id-ordered walk that was
``pedigree_graph._input._check_cycle`` through 0.8.0.
"""

from __future__ import annotations

from collections import deque

import numpy as np


def is_topological(mother_rows: np.ndarray, father_rows: np.ndarray) -> bool:
    """True iff every represented parent row strictly precedes its child row."""
    for i, (m, f) in enumerate(zip(mother_rows.tolist(), father_rows.tolist(), strict=True)):
        if (m >= 0 and m >= i) or (f >= 0 and f >= i):
            return False
    return True


def structural_depth(mother_rows: np.ndarray, father_rows: np.ndarray) -> np.ndarray:
    """Founders 0, otherwise ``max(parent depths) + 1`` with an absent parent counting 0."""
    n = len(mother_rows)
    depth = np.full(n, -1, dtype=np.int32)
    for i in range(n):
        if mother_rows[i] < 0 and father_rows[i] < 0:
            depth[i] = 0
    changed = True
    while changed:
        changed = False
        for j in range(n):
            if depth[j] >= 0:
                continue
            m, f = mother_rows[j], father_rows[j]
            md = depth[m] if m >= 0 else 0
            fd = depth[f] if f >= 0 else 0
            if md >= 0 and fd >= 0:
                depth[j] = max(md, fd) + 1
                changed = True
    depth[depth < 0] = 0
    return depth


def depth_major_order(depth: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Stable sort of rows by depth as ``(order, inverse)``, or ``None`` when already sorted."""
    n = len(depth)
    order = np.argsort(depth, kind="stable").astype(np.int64)
    identity = np.arange(n, dtype=np.int64)
    if np.array_equal(order, identity):
        return None
    inverse = np.empty(n, dtype=np.int64)
    inverse[order] = identity
    return order, inverse


def _remaining_after_kahn(mother_rows: np.ndarray, father_rows: np.ndarray) -> np.ndarray:
    n = len(mother_rows)
    children = np.concatenate([np.arange(n, dtype=np.int64), np.arange(n, dtype=np.int64)])
    parents = np.concatenate([mother_rows, father_rows]).astype(np.int64)
    represented = parents >= 0
    parents = parents[represented]
    children = children[represented]
    indegree = np.bincount(children, minlength=n)
    order = np.argsort(parents, kind="stable")
    parents = parents[order]
    children = children[order]
    starts = np.searchsorted(parents, np.arange(n), side="left")
    ends = np.searchsorted(parents, np.arange(n), side="right")
    peeled = np.zeros(n, dtype=bool)
    queue = deque(int(row) for row in np.flatnonzero(indegree == 0))
    while queue:
        row = queue.popleft()
        peeled[row] = True
        for child in children[starts[row] : ends[row]]:
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(int(child))
    return ~peeled


def cycle_witness(ids: np.ndarray, mother_rows: np.ndarray, father_rows: np.ndarray) -> tuple[int, ...] | None:
    """The ids of one cycle, smallest id first, or ``None`` when the edges are acyclic.

    The witness is chosen by id, not by row, so the same graph reports the
    same cycle whatever order its rows arrive in.
    """
    remaining = _remaining_after_kahn(mother_rows, father_rows)
    if not remaining.any():
        return None
    candidates = np.flatnonzero(remaining)
    row = int(candidates[np.argmin(ids[candidates])])
    walk: list[int] = []
    visited: dict[int, int] = {}
    while row not in visited:
        visited[row] = len(walk)
        walk.append(row)
        parents = [int(p) for p in (mother_rows[row], father_rows[row]) if p >= 0 and remaining[p]]
        row = min(parents, key=lambda p: int(ids[p]))
    cycle = walk[visited[row] :]
    start = int(np.argmin([ids[r] for r in cycle]))
    return tuple(int(ids[r]) for r in cycle[start:] + cycle[:start])
