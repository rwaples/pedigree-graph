"""Native relationship reductions over graph-space rows."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np

from pedigree_graph import _native
from pedigree_graph._input import _own_native
from pedigree_graph._threads import thread_budget

if TYPE_CHECKING:
    from collections.abc import Mapping

    from pedigree_graph._core import PedigreeGraph


@dataclass(frozen=True, slots=True, eq=False)
class RelationshipBurden:
    """Closest-category counts and per-person relatives at degrees 1 through 5.

    ``per_person[row, degree - 1]`` counts distinct relatives at that degree.
    MZ pairs appear in ``category_counts`` and ``same_depth_pairs`` but not in
    the per-person degree columns. ``same_depth_pairs[d]`` counts related pairs
    whose two graph rows both have structural depth ``d``.
    """

    category_counts: Mapping[str, int]
    per_person: np.ndarray
    same_depth_pairs: np.ndarray


def relationship_burden(graph: PedigreeGraph) -> RelationshipBurden:
    """Compute burden with one Rust traversal and O(N) output storage."""
    counts, per_person, same_depth = _native.relationship_burden(graph._built, graph.depth, threads=thread_budget())
    rows = _own_native(per_person, np.uint32).reshape((graph.n_individuals, 5))
    depth = _own_native(same_depth, np.uint64)
    return RelationshipBurden(MappingProxyType(counts), rows, depth)
