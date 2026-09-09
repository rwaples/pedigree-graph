"""Shared numba kinship kernel for :class:`~simace.core.pedigree_graph.PedigreeGraph`.

Compatibility facade (PGQ-008): the kernel was split into focused
modules and the symbols are re-exported here so existing import paths
(``from pedigree_graph._kinship_kernel import …``) stay stable.

Module map:

* ``_kinship_depth``     — EqG / last-direct-child depth.
* ``_kinship_allocator`` — slab arena, free list, row append/retire/sort.
* ``_kinship_csc``       — int32 indptr builder + full-symmetric CSC assembly.
* ``_kinship_dp``        — the DP recursion, its driver, and theta streaming.
* ``_inbreeding_kernel`` — Meuwissen-Luo F-only ancestor walk.
* ``_kinship_kernel``    — this facade.
"""

from __future__ import annotations

__all__ = [
    "_assemble_csc",
    "_build_kinship_csc",
    "_checked_int32_indptr_from_counts",
    "_compute_F_meuwissen_luo",
    "_compute_eqg",
    "_compute_generation_kinship_summary",
    "_compute_last_direct_child_depth",
    "_dp_kinship",
    "_summary_from_dp_rows",
]

from pedigree_graph._inbreeding_kernel import _compute_F_meuwissen_luo
from pedigree_graph._kinship_allocator import _append_entry as _append_entry
from pedigree_graph._kinship_allocator import _freelist_alloc as _freelist_alloc
from pedigree_graph._kinship_allocator import _freelist_bucket as _freelist_bucket
from pedigree_graph._kinship_allocator import _freelist_pop as _freelist_pop
from pedigree_graph._kinship_allocator import _freelist_push as _freelist_push

# Internal helpers re-exported for backward compatibility: the test suite
# and _ne_rates import these from this module (PGQ-008).  The ``as``
# aliases mark them as intentional re-exports.
from pedigree_graph._kinship_allocator import _FreelistBuffers as _FreelistBuffers
from pedigree_graph._kinship_allocator import _retire_rows_at_depth as _retire_rows_at_depth
from pedigree_graph._kinship_csc import _assemble_csc, _checked_int32_indptr_from_counts
from pedigree_graph._kinship_depth import (
    _compute_eqg,
    _compute_last_direct_child_depth,
)
from pedigree_graph._kinship_dp import KinshipDPConfig as KinshipDPConfig
from pedigree_graph._kinship_dp import (
    _build_kinship_csc,
    _compute_generation_kinship_summary,
    _dp_kinship,
    _summary_from_dp_rows,
)
from pedigree_graph._kinship_dp import _densify_labels as _densify_labels
from pedigree_graph._kinship_dp import _finalize_summary as _finalize_summary
from pedigree_graph._kinship_dp import _run_dp_core as _run_dp_core
