"""Exact kinship from the pedigree definition, the specification oracle for kinship and F.

Test code only (ADR 0007).  The other kinship oracles (``pair_kinship``,
``inbreeding``, ``kinship_dp``) are earlier production code moved verbatim, so
a recurrence defect they share with the Rust kernel stays invisible to parity.
This one shares no code and no algorithm with them: Henderson's tabular
recurrence for the numerator relationship matrix, written from the definition
and read only from the id, mother, father and twin columns, never from graph
attributes.

* ``A[i, i] = 1 + A[m, f] / 2`` and ``A[i, j] = (A[j, m] + A[j, f]) / 2`` for
  ``j`` earlier in a topological order, where a missing parent, or a parent id
  that is no row (an external id), contributes 0.
* An MZ co-twin processed after its twin copies the twin's row, and the pair
  takes the self value: ``A[t', j] = A[t, j]``, ``A[t, t'] = A[t, t]``.
* Kinship is ``A / 2``; inbreeding is ``A[i, i] - 1``.

Every entry is dyadic, ``A[i, j] * 2**(d_i + d_j + 1)`` an integer, so the
arithmetic is exact in integers scaled by ``2**shift``; each halving asserts
that it divides evenly, which checks that claim as it goes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ExactKinship:
    """Kinship ``numerator[a][b] / 2**shift`` and structural depth, indexed by input row."""

    numerator: list[list[int]]
    shift: int
    depth: list[int]

    def scaled(self, values: np.ndarray) -> list[int]:
        """*values* times ``2**shift`` as exact integers; each must be a multiple of ``2**-shift``."""
        scale = 2.0**self.shift
        out = []
        for value in np.asarray(values, dtype=np.float64).tolist():
            product = value * scale
            assert product.is_integer(), f"{value!r} is finer than 2**-{self.shift}"
            out.append(int(product))
        return out


def exact_kinship(ids: np.ndarray, mother: np.ndarray, father: np.ndarray, twin: np.ndarray) -> ExactKinship:
    """Exact kinship of every pair of input rows.

    Args:
        ids: Individual ids, one per row, any order.
        mother: Mother id per row, ``-1`` when missing; an id that is no row is external.
        father: Father id per row, as ``mother``.
        twin: MZ co-twin id per row, ``-1`` when none.

    Returns:
        The kinship numerators over ``2**shift``, and each row's depth, the
        longest represented-parent chain above it.
    """
    n = len(ids)
    row_of = {int(v): row for row, v in enumerate(ids)}
    parents = [tuple(row_of.get(int(p), -1) for p in (mother[row], father[row])) for row in range(n)]
    co_twin = [row_of[int(t)] if t != -1 else -1 for t in twin]

    depth: list[int | None] = [None] * n
    order: list[int] = []
    for start in range(n):
        stack = [(start, False)]
        while stack:
            row, expanded = stack.pop()
            if depth[row] is not None:
                continue
            if expanded:
                depth[row] = max((depth[p] + 1 for p in parents[row] if p != -1), default=0)
                order.append(row)
                continue
            stack.append((row, True))
            stack.extend((p, False) for p in parents[row] if p != -1 and depth[p] is None)

    # 2 d_i + 1 bits hold A; 30 more let a float32 rounding of the deepest
    # kinship be scaled to an integer too.
    shift = 2 * max(depth, default=0) + 32
    one = 1 << shift

    def half(value: int) -> int:
        assert value % 2 == 0, "a relationship entry is not dyadic at this scale"
        return value // 2

    a = [[0] * n for _ in range(n)]
    done: list[int] = []
    processed = [False] * n
    for row in order:
        t = co_twin[row]
        if t != -1 and processed[t]:
            for j in done:
                a[row][j] = a[j][row] = a[t][j]
            a[row][row] = a[t][t]
        else:
            m, f = parents[row]
            for j in done:
                value = (a[j][m] if m != -1 else 0) + (a[j][f] if f != -1 else 0)
                a[row][j] = a[j][row] = half(value)
            a[row][row] = one + (half(a[m][f]) if m != -1 and f != -1 else 0)
        done.append(row)
        processed[row] = True

    # A over 2**shift is kinship over 2**(shift + 1).
    return ExactKinship(numerator=a, shift=shift + 1, depth=[int(d) for d in depth])
