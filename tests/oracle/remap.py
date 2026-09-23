"""The 0.9.3 host-side depth-major remap, kept for the oracles that sweep in it.

Through 0.9.3 ``pedigree_graph._topology`` moved the parent arrays into the
stable depth-major order before a Numba kernel ran and scattered the result
back.  The Rust core sorts and indexes graph rows itself, so the package no
longer needs this; the verbatim Numba oracles (``inbreeding``, ``lineage``,
``kinship_dp``) still take topological parents, and this is how the tests
give them those.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pedigree_graph import _native
from pedigree_graph._topology import readonly


def remap_rows(rows: np.ndarray, order: np.ndarray, inverse: np.ndarray) -> np.ndarray:
    """Reorder a row-reference array by *order* and translate its values via *inverse*.

    Args:
        rows: per-row references to other rows (mother, father, twin), ``-1``
            where absent.
        order: new→old permutation (topological position → graph row).
        inverse: old→new permutation (graph row → topological position).

    Returns:
        Contiguous int32 array in the permuted row space whose stored
        references point at permuted rows.  ``-1`` passes through.
    """
    moved = rows[order]
    return readonly(np.ascontiguousarray(np.where(moved < 0, np.int32(-1), inverse[moved].astype(np.int32))))


@dataclass(frozen=True, slots=True)
class Topology:
    """Structural depth plus the graph ↔ topological row maps.

    Attributes:
        depth: int32 structural depth per graph row (founders 0).
        order: intp topological position → graph row, or ``None`` when the
            graph rows are already depth-major.
        inverse: intp graph row → topological position, ``None`` alongside
            ``order``.
    """

    depth: np.ndarray
    order: np.ndarray | None
    inverse: np.ndarray | None

    def to_topological(self, rows: np.ndarray) -> np.ndarray:
        """Move a row-reference array (mother/father/twin) into topological space."""
        if self.order is None or self.inverse is None:
            return rows
        return remap_rows(rows, self.order, self.inverse)

    def gather(self, values: np.ndarray) -> np.ndarray:
        """Reorder per-row values (depth, labels, F) into topological space."""
        if self.order is None:
            return values
        return readonly(np.ascontiguousarray(values[self.order]))

    def per_row_to_graph(self, values: np.ndarray) -> np.ndarray:
        """Scatter a per-row kernel output back onto graph rows."""
        if self.order is None:
            return values
        out = np.empty_like(values)
        out[self.order] = values
        return out


def build_topology(depth: np.ndarray) -> Topology:
    """Derive the stable depth-major permutation from *depth*.

    Args:
        depth: structural depth per graph row, from :func:`structural_depth`.

    Returns:
        The :class:`Topology`; ``order``/``inverse`` are ``None`` when the
        graph rows are already depth-major.
    """
    permutation = _native.depth_major_order(np.ascontiguousarray(depth, dtype=np.int32))
    if permutation is None:
        return Topology(depth=depth, order=None, inverse=None)
    order, inverse = permutation
    return Topology(depth=depth, order=order.astype(np.intp, copy=False), inverse=inverse.astype(np.intp, copy=False))
