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

:func:`estimate_effective_sizes` runs a selection of them over one lazily
built prerequisite memo, serially in canonical order, and returns an
:class:`EffectiveSizeResults` with all eight keys; an unselected or refused
estimator maps to an :class:`UnavailableEffectiveSize`.

Metadata dependency matrix.  Every guard raises
:class:`~pedigree_graph.MissingMetadataError` naming the estimator in
``operation``; guards run in the order listed, before any work, and an
empty graph bypasses them all (every estimator then returns its record
with ``ne=None`` and zero-length arrays).

* ``ne_inbreeding``, ``ne_coancestry``, ``ne_individual_delta_f``: complete
  generation labels, or none (``missing_generation_labels`` when partial;
  absent labels group by structural depth).
* ``ne_long_term_contributions``, ``ne_caballero_toro``: generation labels
  as above, then closed represented parentage (``incomplete_parentage``
  when a row has exactly one represented parent).
* ``ne_variance_family_size``, ``ne_sex_ratio``: generation labels as
  above, then complete sex (``missing_sex``, ``status`` ``"absent"`` or
  ``"partial"``); uniform but known sex is valid and warns.
* ``ne_hill_overlapping`` without birth years: as ``ne_variance_family_size``,
  then collapses to it.  With birth years: ignores generation labels;
  complete sex, then a known-age edge for each parent role
  (``insufficient_parent_age_data``).
"""

from __future__ import annotations

from pedigree_graph._cohort_utils import CohortWindow, eligible_cohort_range
from pedigree_graph._ne_caballero_toro import ne_caballero_toro
from pedigree_graph._ne_estimate import (
    ALL_EFFECTIVE_SIZE_ESTIMATORS,
    EffectiveSizeResults,
    UnavailableEffectiveSize,
    estimate_effective_sizes,
)
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
    "ALL_EFFECTIVE_SIZE_ESTIMATORS",
    "CohortWindow",
    "EffectiveSizeResults",
    "GenerationInterval",
    "NeCaballeroToroResult",
    "NeCoancestryResult",
    "NeHillResult",
    "NeInbreedingResult",
    "NeIndividualDeltaFResult",
    "NeLTCResult",
    "NeSexRatioResult",
    "NeVarianceResult",
    "UnavailableEffectiveSize",
    "eligible_cohort_range",
    "estimate_effective_sizes",
    "ne_caballero_toro",
    "ne_coancestry",
    "ne_hill_overlapping",
    "ne_inbreeding",
    "ne_individual_delta_f",
    "ne_long_term_contributions",
    "ne_sex_ratio",
    "ne_variance_family_size",
]
