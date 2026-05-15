"""pedigree-graph: sparse-matrix-based pedigree relationship extraction.

Public API:
    PedigreeGraph     — parent→child DAG with relationship extraction
    REL_REGISTRY      — ordered registry of relationship types
    PAIR_KINSHIP      — kinship coefficient by code (single source of truth)
    RelType           — NamedTuple describing a single relationship class

Effective population size (Ne):
    Result classes: NeCaballeroToroResult, NeCoancestryResult, NeHillResult,
        NeIndividualDeltaFResult, NeInbreedingResult, NeLTCResult,
        NeSexRatioResult, NeVarianceResult
    Estimators: ne_caballero_toro, ne_coancestry, ne_hill_overlapping,
        ne_inbreeding, ne_individual_delta_f, ne_long_term_contributions,
        ne_sex_ratio, ne_variance_family_size
    Convenience: compute_all_ne (runs all eight, sharing K and the
        founder-contribution matrix)
"""

from pedigree_graph._cohort_utils import (
    CohortWindow,
    eligible_cohort_range,
)
from pedigree_graph._core import (
    PAIR_KINSHIP,
    REL_REGISTRY,
    PedigreeGraph,
    RelType,
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

__all__ = [
    "PAIR_KINSHIP",
    "REL_REGISTRY",
    "CohortWindow",
    "GenerationInterval",
    "NeCaballeroToroResult",
    "NeCoancestryResult",
    "NeHillResult",
    "NeInbreedingResult",
    "NeIndividualDeltaFResult",
    "NeLTCResult",
    "NeSexRatioResult",
    "NeVarianceResult",
    "PedigreeGraph",
    "RelType",
    "compute_all_ne",
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
