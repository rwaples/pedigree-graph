"""The readable pairwise-kinship recurrence, kept as the differential oracle for the Rust kernel.

Test code only (ADR 0007): ``pedigree_graph`` never imports this, and it is
not a fallback.  It is ``_pairwise_kinship_py`` moved here verbatim when
slice 13 put ``pair_kinship`` on the Rust core, and it states the ADR 0009
value definition in graph space with an explicit depth array, which is the
contract the native kernel is held to bit for bit.
"""

from __future__ import annotations

from functools import cache

import numpy as np


def pair_kinship(
    mother: np.ndarray,
    father: np.ndarray,
    twin: np.ndarray,
    depth: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
) -> np.ndarray:
    """Pedigree-expected kinship per requested pair, as a readable recursive oracle.

    The ADR 0009 recurrence written with ``functools.cache`` recursion over
    graph-space arrays and an explicit depth, so a reader can check it against
    the ADR line by line: ``phi(a, a) = (1 + phi(m, f)) / 2`` with a missing
    parent contributing 0 and an MZ co-twin taking the self formula; otherwise
    the endpoint of greater depth, ties to the greater row, is peeled to its
    parents.  Every step is the correctly rounded float32 of a half-sum of two
    float32 operands.  Small pedigrees only: Python recursion grows with
    pedigree depth.

    Args:
        mother: Mother row per graph row, ``-1`` when absent.
        father: Father row per graph row, ``-1`` when absent.
        twin: MZ co-twin row per graph row, ``-1`` when absent.
        depth: Structural depth per graph row.
        first: First endpoint of each requested pair, as graph rows.
        second: Second endpoint of each requested pair, as graph rows.

    Returns:
        float32 array of length ``len(first)``, positionally aligned to the
        inputs.
    """
    mother = np.asarray(mother, dtype=np.int64)
    father = np.asarray(father, dtype=np.int64)
    twin = np.asarray(twin, dtype=np.int64)
    depth = np.asarray(depth, dtype=np.int64)
    half = np.float32(0.5)
    one = np.float32(1.0)
    zero = np.float32(0.0)

    @cache
    def _phi(a: int, b: int) -> np.float32:
        if a > b:
            a, b = b, a
        if depth[a] > depth[b]:
            other, peeled = b, a
        else:
            other, peeled = a, b
        m = int(mother[peeled])
        f = int(father[peeled])
        if other == peeled or twin[other] == peeled or twin[peeled] == other:
            if m < 0 or f < 0:
                return half
            return half * (one + _phi(m, f))
        left = _phi(m, other) if m >= 0 else zero
        right = _phi(f, other) if f >= 0 else zero
        return half * (left + right)

    out = np.empty(len(first), dtype=np.float32)
    for k in range(len(first)):
        out[k] = _phi(int(first[k]), int(second[k]))
    return out
