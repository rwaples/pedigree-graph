"""Pair-array helpers for the relationship tests: canonical keys, the block invariants, sibling sets.

Test code only (ADR 0007).  The classification oracle is
:mod:`oracle.relationship_paths`; these helpers only read and check blocks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pedigree_graph import PedigreeGraph, RelationshipPairs

_PairArrays = tuple[np.ndarray, np.ndarray]


def canonical_keys(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    """Return the canonical unordered int64 key ``min(a, b) * n + max(a, b)`` per pair.

    Args:
        a: First member of each pair, any orientation.
        b: Second member of each pair, any orientation.
        n: Key base, at least ``max(a, b) + 1``; ``n`` is bounded by the int32
            row capacity so the product fits int64.

    Returns:
        One int64 key per pair, equal for the two orientations of a pair.
    """
    return np.minimum(a, b).astype(np.int64) * n + np.maximum(a, b).astype(np.int64)


def sort_by_canonical_key(a: np.ndarray, b: np.ndarray, n: int) -> _PairArrays:
    """Return ``(a, b)`` reordered by :func:`canonical_keys`, orientation kept."""
    if len(a) == 0:
        return a, b
    order = np.argsort(canonical_keys(a, b, n), kind="stable")
    return a[order], b[order]


def check_exclusive(pairs: RelationshipPairs) -> None:
    """Assert the ADR 0006 pair invariants on *pairs*.

    Every block is sorted by canonical unordered key with no repeated pair
    (so no asymmetric block holds both orientations), and no unordered pair
    appears in two blocks.

    Args:
        pairs: The result to check.

    Raises:
        AssertionError: Naming the offending pair and its block(s).
    """
    n = 1 + max(
        (int(max(block.first_rows.max(), block.second_rows.max())) for block in pairs.values() if len(block)), default=0
    )
    seen: dict[int, str] = {}
    for code, block in pairs.items():
        keys = canonical_keys(block.first_rows, block.second_rows, n)
        assert np.all(keys[1:] > keys[:-1]), f"{code}: block is not strictly sorted by canonical key"
        for key, first, second in zip(
            keys.tolist(), block.first_rows.tolist(), block.second_rows.tolist(), strict=True
        ):
            other = seen.setdefault(key, code)
            assert other == code, f"pair ({first}, {second}) appears in both {other} and {code}"


def sibling_pairs(graph: PedigreeGraph) -> tuple[_PairArrays, _PairArrays, _PairArrays]:
    """``(full_sib, maternal_hs, paternal_hs)`` graph-row pairs, ``lo < hi``, unfolded.

    Siblings share an original parent id, external ids included: full sibs
    both known ids, maternal or paternal half sibs only the mother's or the
    father's.  MZ co-twins are siblings too (issue #29).
    """
    mother = np.asarray(graph.mother_ids).tolist()
    father = np.asarray(graph.father_ids).tolist()
    found: tuple[list, list, list] = ([], [], [])
    for parent_ids, half in ((mother, found[1]), (father, found[2])):
        groups: dict[int, list[int]] = {}
        for row, parent in enumerate(parent_ids):
            if parent >= 0:
                groups.setdefault(parent, []).append(row)
        for rows in groups.values():
            for i, a in enumerate(rows):
                for b in rows[i + 1 :]:
                    if mother[a] >= 0 and mother[a] == mother[b] and father[a] >= 0 and father[a] == father[b]:
                        if parent_ids is mother:
                            found[0].append((a, b))
                    else:
                        half.append((a, b))
    for block in found:
        block.sort()

    def arrays(pairs: list) -> _PairArrays:
        block = np.array(pairs, dtype=np.intp).reshape(-1, 2)
        return block[:, 0], block[:, 1]

    return arrays(found[0]), arrays(found[1]), arrays(found[2])
