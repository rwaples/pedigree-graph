"""pedigree-graph: sparse-matrix-based pedigree relationship extraction.

Public API:
    PedigreeGraph        — parent→child DAG with relationship extraction
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
    RelationshipCountResult — what graph.relationship_counts(...) or
        view.relationship_counts(...) returns: an immutable mapping over all
        23 codes to int | None, plus the requested / exact / approximate /
        clamped code sets
    PedigreeView         — ordered view of a graph's rows, built with
        graph.view(ids=...) or graph.view(rows=...); exposes read-only ids,
        graph_rows, n_individuals, len, and relationship_pairs /
        relationship_counts in view rows

0.7.1 compatibility (removed in 0.8.0) — detached snapshots of the registry,
so mutating them changes nothing the engines read:
    REL_REGISTRY      — ordered registry of relationship types
    PAIR_KINSHIP      — kinship coefficient by code
    RelType           — NamedTuple describing a single relationship class

Errors (ADR 0006 — each carries a stable ``.code`` and immutable ``.fields``):
    PedigreeValidationError, MissingMetadataError (both ValueError),
    ResourceError (RuntimeError)

Threads:
    configure_threads — one package-wide budget, set before the first
        parallel call.  Precedence: configure_threads(n) >
        PEDIGREE_GRAPH_THREADS > 1.  Repeating the committed value is fine;
        changing it after the budget is committed is a RuntimeError.

Effective population size (Ne):
    Result classes: NeCaballeroToroResult, NeCoancestryResult, NeHillResult,
        NeIndividualDeltaFResult, NeInbreedingResult, NeLTCResult,
        NeSexRatioResult, NeVarianceResult
    Estimators: ne_caballero_toro, ne_coancestry, ne_hill_overlapping,
        ne_inbreeding, ne_individual_delta_f, ne_long_term_contributions,
        ne_sex_ratio, ne_variance_family_size
    Convenience: compute_all_ne (runs all eight estimators, sharing
        cached F, streamed θ̄, and founder-contribution summaries where
        applicable)
"""

from pedigree_graph._cohort_utils import (
    CohortWindow,
    eligible_cohort_range,
)
from pedigree_graph._core import (
    FrameLike,
    PedigreeGraph,
)
from pedigree_graph._effective_size import (
    GenerationInterval,
    NeCaballeroToroResult,
    NeCoancestryResult,
    NeHillResult,
    NeInbreedingResult,
    NeIndividualDeltaFResult,
    NeLTCResult,
    NeSexRatioResult,
    NeVarianceResult,
    compute_all_ne,
    ne_caballero_toro,
    ne_coancestry,
    ne_hill_overlapping,
    ne_inbreeding,
    ne_individual_delta_f,
    ne_long_term_contributions,
    ne_sex_ratio,
    ne_variance_family_size,
)
from pedigree_graph._errors import (
    MissingMetadataError,
    PedigreeValidationError,
    ResourceError,
)
from pedigree_graph._registry import (  # 0.8.0-DELETE
    PAIR_KINSHIP,
    REL_REGISTRY,
    RelType,
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
    "PAIR_KINSHIP",  # 0.8.0-DELETE
    "RELATIONSHIPS",
    "REL_REGISTRY",  # 0.8.0-DELETE
    "CohortWindow",
    "FrameLike",
    "GenerationInterval",
    "MissingMetadataError",
    "NeCaballeroToroResult",
    "NeCoancestryResult",
    "NeHillResult",
    "NeInbreedingResult",
    "NeIndividualDeltaFResult",
    "NeLTCResult",
    "NeSexRatioResult",
    "NeVarianceResult",
    "PedigreeGraph",
    "PedigreeValidationError",
    "PedigreeView",
    "RelType",  # 0.8.0-DELETE
    "RelationshipCategory",
    "RelationshipCountResult",
    "RelationshipPairBlock",
    "RelationshipPairs",
    "ResourceError",
    "compute_all_ne",
    "configure_threads",
    "eligible_cohort_range",
    "ne_caballero_toro",
    "ne_coancestry",
    "ne_hill_overlapping",
    "ne_inbreeding",
    "ne_individual_delta_f",
    "ne_long_term_contributions",
    "ne_sex_ratio",
    "ne_variance_family_size",
]
