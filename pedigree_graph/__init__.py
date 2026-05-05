"""pedigree-graph: sparse-matrix-based pedigree relationship extraction.

Public API:
    PedigreeGraph     — parent→child DAG with relationship extraction
    REL_REGISTRY      — ordered registry of relationship types
    PAIR_KINSHIP      — kinship coefficient by code (single source of truth)
    RelType           — NamedTuple describing a single relationship class
"""

from pedigree_graph._core import (
    PAIR_KINSHIP,
    REL_REGISTRY,
    PedigreeGraph,
    RelType,
)

__all__ = [
    "PAIR_KINSHIP",
    "REL_REGISTRY",
    "PedigreeGraph",
    "RelType",
]
