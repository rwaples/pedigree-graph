"""Structural depth and the read-only coercions the graph's derived arrays share.

Every order-dependent kernel (the pairwise kinship peel, the kinship DP, the
inbreeding walk, the lineage and generation sweeps) lives in the Rust core,
which sorts rows into stable depth-major order itself and indexes graph rows
directly; nothing on the Python side remaps parents or scatters results back.
What stays here is the depth those kernels take, and the helpers that freeze
arrays before they are handed out.
"""

from __future__ import annotations

__all__ = ["owned_readonly", "readonly", "structural_depth"]

import numpy as np

from pedigree_graph import _native


def readonly(values: np.ndarray) -> np.ndarray:
    """Mark *values* read-only in place and return it."""
    values.setflags(write=False)
    return values


def owned_readonly(values: np.ndarray, dtype: type) -> np.ndarray:
    """Coerce to a contiguous read-only array without touching the caller's flags."""
    out = np.ascontiguousarray(values, dtype=dtype)
    if out is values and out.flags.writeable:
        out = out.copy()
    return readonly(out)


def structural_depth(mother_rows: np.ndarray, father_rows: np.ndarray) -> np.ndarray:
    """Depth from the parent edges alone: founders 0, child ``max(parents) + 1``.

    Computed by the Rust core; the parent edges must already have passed the
    construction-time cycle check.

    Args:
        mother_rows: contiguous int32 mother row per graph row, ``-1`` when absent.
        father_rows: contiguous int32 father row per graph row, ``-1`` when absent.

    Returns:
        Read-only int32 depth per graph row.
    """
    return readonly(_native.structural_depth(mother_rows, father_rows))
