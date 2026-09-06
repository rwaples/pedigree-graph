"""The private topological row order shared by every order-dependent kernel.

Public graph coordinates are input rows, in any acyclic order.  Several
kernels need parents to precede children in the index space they sweep:
the Meuwissen-Luo inbreeding walk, the descendant path-count reverse
sweep, the pairwise kinship peel, the kinship DP, and the Caballero-Toro
forward sweep.  This module builds one order for all of them --- stable
depth-major --- and the maps that move arrays and row references between
graph space and that order.

Stable depth-major is a topological order because a child's structural
depth strictly exceeds both parents'.  Ties within a depth keep input row
order, so no original ID ever influences the result.  It is also exactly
the order :func:`pedigree_graph._kinship_dp._run_dp_core` sorts into, and
in it "greater depth, then greater row" is simply "greater row", which is
what lets the pairwise kernel peel by row alone and still match the matrix
to the bit (ADR 0009).

When the input rows are already depth-major the permutation is the
identity; ``order`` and ``inverse`` are then ``None`` and every routing
helper returns its argument untouched.

Depth and the permutation are built separately.  ``PedigreeGraph.depth`` is
public and several callers want only it; paying for the depth-major sort there
would tax every graph whether or not an order-dependent kernel is ever called.
"""

from __future__ import annotations

__all__ = ["Topology", "build_topology", "owned_readonly", "readonly", "remap_rows", "structural_depth"]

from dataclasses import dataclass

import numpy as np

from pedigree_graph._kinship_depth import _compute_depth


def readonly(values: np.ndarray) -> np.ndarray:
    """Mark a derived array read-only before it reaches a numba kernel.

    Numba types a writeable and a read-only array as different signatures, so a
    kernel fed the graph's owned (read-only) arrays on depth-major input and a
    freshly permuted (writeable) copy otherwise would be compiled twice.  Every
    array these kernels read is immutable in fact; saying so keeps one signature.
    """
    values.setflags(write=False)
    return values


def owned_readonly(values: np.ndarray, dtype: type) -> np.ndarray:
    """Coerce to a contiguous read-only array without touching the caller's flags."""
    out = np.ascontiguousarray(values, dtype=dtype)
    if out is values and out.flags.writeable:
        out = out.copy()
    return readonly(out)


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

    def translate(self, rows: np.ndarray) -> np.ndarray:
        """Translate graph row values (pair endpoints) to topological positions."""
        if self.inverse is None:
            return rows
        return np.where(rows < 0, -1, self.inverse[rows])

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


def structural_depth(mother_rows: np.ndarray, father_rows: np.ndarray, n: int) -> np.ndarray:
    """Depth from the parent edges alone: founders 0, child ``max(parents) + 1``.

    Args:
        mother_rows: int32 mother row per graph row, ``-1`` when absent.
        father_rows: int32 father row per graph row, ``-1`` when absent.
        n: number of rows.

    Returns:
        Read-only int32 depth per graph row.
    """
    return readonly(_compute_depth(mother_rows, father_rows, n))


def build_topology(depth: np.ndarray) -> Topology:
    """Derive the stable depth-major permutation from *depth*.

    Args:
        depth: structural depth per graph row, from :func:`structural_depth`.

    Returns:
        The :class:`Topology`; ``order``/``inverse`` are ``None`` when the
        graph rows are already depth-major.
    """
    n = depth.shape[0]
    order = np.argsort(depth, kind="stable").astype(np.intp)
    identity = np.arange(n, dtype=np.intp)
    if np.array_equal(order, identity):
        return Topology(depth=depth, order=None, inverse=None)
    inverse = np.empty(n, dtype=np.intp)
    inverse[order] = identity
    return Topology(depth=depth, order=order, inverse=inverse)
