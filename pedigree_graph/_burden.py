"""Native relationship reductions over graph-space rows."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np

from pedigree_graph import _native, _progress
from pedigree_graph._input import _own_native
from pedigree_graph._threads import thread_budget

if TYPE_CHECKING:
    from collections.abc import Mapping

    from pedigree_graph._core import PedigreeGraph
    from pedigree_graph._progress import ProgressArg

# Progress lines only: burden logs no start or total line (ADR 0017).
logger = logging.getLogger(__name__)
#: Per-person degree columns, the core's width shared with the R binding.
DEGREES = _native.BURDEN_DEGREES


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


def relationship_burden(graph: PedigreeGraph, progress: ProgressArg = None) -> RelationshipBurden:
    """Compute burden with one Rust traversal and O(N) output storage."""
    watch = _progress.resolve(progress, logger, "relationship_burden")
    counts, per_person, same_depth = _native.relationship_burden(
        graph._built, graph.depth, threads=thread_budget(), progress=watch, tick=_progress.TICK_S
    )
    rows = _own_native(per_person, np.uint32).reshape((graph.n_individuals, DEGREES))
    depth = _own_native(same_depth, np.uint64)
    return RelationshipBurden(MappingProxyType(counts), rows, depth)
