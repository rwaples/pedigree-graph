"""Pedigree-based effective population size (Ne) estimators (ADR 0006).

The eight estimators share one contract: results are frozen records whose
per-cohort arrays align with the observed generation labels they carry
(``generations``, or ``parent_generations`` for the family-size variance and
``cohort_years`` for Hill), rate transitions describe adjacent observed
cohorts with the gap-corrected rate of
:func:`~pedigree_graph._ne_common._transition_ne`, and every array is an
owned read-only copy.  Founders are represented founders: rows with no
represented parent, whatever their label.

Estimator coverage:

* :func:`ne_inbreeding`              — regression of ``ln(1 − F̄)`` on the label offset.
* :func:`ne_coancestry`              — regression of ``ln(1 − θ̄)`` on the label offset.
* :func:`ne_variance_family_size`    — Caballero 1994 eq. 6 (separate sex,
  sex-of-offspring covariance).
* :func:`ne_sex_ratio`               — Wright ``4 N_m N_f / (N_m + N_f)``.
* :func:`ne_individual_delta_f`      — Gutiérrez 2008 individual ΔF_i via EqG.
* :func:`ne_long_term_contributions` — Wray & Thompson 1990 founder-genome contributions.
* :func:`ne_hill_overlapping`        — Hill 1979 (collapses to Ne_V at L=1).
* :func:`ne_caballero_toro`          — Caballero & Toro 2002 self-coancestry regression.

Every estimator groups by the supplied generation labels, or by structural
depth when the graph carries none; partly known labels raise
:class:`~pedigree_graph.MissingMetadataError` (``missing_generation_labels``).
An empty graph is a valid no-estimate case: every estimator returns its
record with ``ne=None`` and zero-length arrays.
"""

from __future__ import annotations

from pedigree_graph._cohort_utils import CohortWindow, eligible_cohort_range
from pedigree_graph._ne_caballero_toro import ne_caballero_toro
from pedigree_graph._ne_family_size import ne_sex_ratio, ne_variance_family_size
from pedigree_graph._ne_founders import ne_long_term_contributions
from pedigree_graph._ne_hill import ne_hill_overlapping
from pedigree_graph._ne_rates import ne_coancestry, ne_inbreeding, ne_individual_delta_f
from pedigree_graph._ne_results import (
    GenerationInterval,
    NeCaballeroToroResult,
    NeCoancestryResult,
    NeHillResult,
    NeInbreedingResult,
    NeIndividualDeltaFResult,
    NeLTCResult,
    NeSexRatioResult,
    NeVarianceResult,
)

__all__ = [
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
