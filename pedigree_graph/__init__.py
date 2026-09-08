"""pedigree-graph: sparse-matrix-based pedigree relationship extraction.

Public API:
    PedigreeGraph        — parent→child DAG with relationship extraction,
        pair kinship, and complete / relationship-limited /
        approximate-support kinship matrices
    RELATIONSHIPS        — immutable ordered registry of the 23 relationship
        categories; iteration order is the same-degree precedence for
        closest-category classification
    RelationshipCategory — one category: code, label, degree, nominal kinship,
        the up/down path shape, and the two positional roles
    RelationshipPairs    — what graph.relationship_pairs(max_degree=...) or
        graph.relationship_pairs(categories=...) returns: an immutable mapping
        over all 23 codes
    RelationshipPairBlock — one category's pairs: owned read-only int32
        first_rows / second_rows in the category's role orientation, the
        roles, requested, len, and (first, second) unpacking
    RelationshipCountResult — what graph.relationship_counts(...),
        view.relationship_counts(...), or the memory-bounded
        graph.estimate_relationship_counts(max_degree=...) returns: an
        immutable mapping over all 23 codes to int | None, plus the
        requested / exact / approximate / clamped code sets (clamped:
        requested codes whose inclusion-exclusion residual underflowed and
        was floored at 0; that 0 is not a true absence)
    PedigreeView         — ordered view of a graph's rows, built with
        graph.view(ids=...) or graph.view(rows=...); exposes read-only ids,
        graph_rows, n_individuals, len, relationship_pairs /
        relationship_counts, and pair_kinship in view rows

Errors (ADR 0006 — each carries a stable ``.code`` and immutable ``.fields``):
    PedigreeValidationError, MissingMetadataError (both ValueError),
    ResourceError (RuntimeError)

Threads:
    configure_threads — one package-wide budget, set before the first
        parallel call.  Precedence: configure_threads(n) >
        PEDIGREE_GRAPH_THREADS > 1.  Repeating the committed value is fine;
        changing it after the budget is committed is a RuntimeError.

Public non-root modules:
    pedigree_graph.relationships — the registry and the pair / count result types
    pedigree_graph.summaries     — GenerationKinshipSummary
    pedigree_graph.effective_size — estimators, result classes,
        estimate_effective_sizes, and the cohort utilities
    pedigree_graph.typing        — FrameLike, the structural table protocol
"""

from pedigree_graph._core import PedigreeGraph
from pedigree_graph._errors import (
    MissingMetadataError,
    PedigreeValidationError,
    ResourceError,
)
from pedigree_graph._threads import configure_threads
from pedigree_graph._view import PedigreeView
from pedigree_graph.relationships import (
    RELATIONSHIPS,
    RelationshipCategory,
    RelationshipCountResult,
    RelationshipPairBlock,
    RelationshipPairs,
)

__all__ = [
    "RELATIONSHIPS",
    "MissingMetadataError",
    "PedigreeGraph",
    "PedigreeValidationError",
    "PedigreeView",
    "RelationshipCategory",
    "RelationshipCountResult",
    "RelationshipPairBlock",
    "RelationshipPairs",
    "ResourceError",
    "configure_threads",
]
