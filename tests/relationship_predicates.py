"""Engine-independent orientation predicates for relationship pairs.

Each predicate walks ``mother_rows`` / ``father_rows`` directly, so a test can
check a block's role orientation without trusting the matrix products that
produced it.  Sibling links follow the engine's rule of sharing an original
parent *id*, which also covers parents external to the pedigree.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pedigree_graph import RELATIONSHIPS

if TYPE_CHECKING:
    from pedigree_graph import PedigreeGraph


class AncestorWalk:
    """Exact-meiosis ancestor sets of every row, built once per graph."""

    def __init__(self, graph: PedigreeGraph, max_up: int = 5) -> None:
        self.mother_rows = np.asarray(graph.mother_rows)
        self.father_rows = np.asarray(graph.father_rows)
        self.mother_ids = np.asarray(graph.mother_ids)
        self.father_ids = np.asarray(graph.father_ids)
        n = graph.n_individuals
        levels: list[list[set[int]]] = [[{row} for row in range(n)]]
        for _ in range(max_up):
            previous = levels[-1]
            levels.append([{parent for x in previous[row] for parent in self._parents(x)} for row in range(n)])
        self.levels = levels

    def _parents(self, row: int) -> list[int]:
        return [int(p) for p in (self.mother_rows[row], self.father_rows[row]) if p >= 0]

    def ancestors(self, row: int, up: int) -> set[int]:
        """Rows exactly *up* meioses above *row* (``{row}`` for ``up == 0``)."""
        return self.levels[up][row]

    def share_parent(self, a: int, b: int) -> bool:
        """Whether *a* and *b* have a common known parent id."""
        return bool(
            (self.mother_ids[a] >= 0 and self.mother_ids[a] == self.mother_ids[b])
            or (self.father_ids[a] >= 0 and self.father_ids[a] == self.father_ids[b])
        )

    def oriented_pair_is_valid(self, code: str, first: int, second: int) -> bool:
        """Whether ``(first, second)`` satisfies *code*'s role orientation."""
        category = RELATIONSHIPS[code]
        if code == "MO":
            return int(self.mother_rows[first]) == second
        if code == "FO":
            return int(self.father_rows[first]) == second
        if category.down == 0:
            return second in self.ancestors(first, category.up)
        if category.down == 1:
            return any(x != second and self.share_parent(x, second) for x in self.ancestors(first, category.up - 1))
        return bool(self.ancestors(first, category.up) & self.ancestors(second, category.down))

    def dual_valid(self, code: str, first: int, second: int) -> bool:
        """Whether both orientations of the pair satisfy *code*."""
        return self.oriented_pair_is_valid(code, first, second) and self.oriented_pair_is_valid(code, second, first)
