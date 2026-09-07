"""0.8.0-DELETE: the 0.7.1 effective-size surface (PGQ-006, slice 6c).

The final estimators and result records live in
:mod:`pedigree_graph.effective_size`.  This module keeps the 0.7.1 import
paths (``from pedigree_graph._effective_size import …`` and the package
root) working until slice 7: the eight root ``ne_*`` accept the old injected
prerequisites, run the same private evaluators as the final estimators, and
scatter each result onto the dense ``0 .. max(label)`` records of
:mod:`pedigree_graph._ne_legacy`; :func:`compute_all_ne` keeps its eager
prerequisites and worker pool.

Module map:

* ``_cohorts``          — observed-cohort grouping shared with the kinship summary.
* ``_ne_common``        — shared numeric helpers (harmonic mean, gap rate, log-regression).
* ``_ne_results``       — final result records + serialization.
* ``_ne_legacy``        — 0.7.1 result records and the dense scatter (0.8.0-DELETE).
* ``_ne_family_size``   — family-size table, Ne_V, Ne_sr.
* ``_ne_founders``      — represented founders, contributions, Ne_LTC.
* ``_ne_caballero_toro``— CT accumulators + Ne_CT.
* ``_ne_hill``          — Hill overlapping-generation Ne_H.
* ``_ne_rates``         — Ne_I, Ne_C, Ne_iΔF and the generation kinship summary.
* ``effective_size``    — the public final surface.
* ``_effective_size``   — this facade + :func:`compute_all_ne` orchestration.

Founders are excluded from the ΔF / Δθ regressions; they are included
in the parent set for the gen-0 → gen-1 family-size variance transition.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

import numpy as np

# Internal helpers and typed models re-exported for backward compatibility:
# ``_core`` and the test suite import several of these from this module
# (PGQ-006).  The ``as`` aliases mark them as intentional re-exports.
from pedigree_graph._cohorts import ObservedCohorts
from pedigree_graph._kinship_kernel import _compute_eqg
from pedigree_graph._ne_caballero_toro import CTAccumulators as CTAccumulators  # noqa: TC001
from pedigree_graph._ne_caballero_toro import (
    _caballero_toro_accumulators,
    _caballero_toro_from,
)
from pedigree_graph._ne_family_size import FamilySizeEntry as FamilySizeEntry
from pedigree_graph._ne_family_size import FamilySizeTable as FamilySizeTable
from pedigree_graph._ne_family_size import Sigma2Decomposition as Sigma2Decomposition
from pedigree_graph._ne_family_size import (
    _generation_family_table,
    _sex_column,
    _sex_ratio_from,
    _variance_from,
    _warn_if_uniform_sex,
)
from pedigree_graph._ne_family_size import _sex_specific_family_table as _sex_specific_family_table
from pedigree_graph._ne_family_size import _sigma2_from_quadrants as _sigma2_from_quadrants
from pedigree_graph._ne_founders import (
    FounderContributionMeans,
    _founder_idx,
    _ltc_from,
    _per_gen_founder_means,
)
from pedigree_graph._ne_hill import ne_hill_overlapping as _ne_hill_overlapping
from pedigree_graph._ne_legacy import (
    GenerationInterval,
    NeCaballeroToroResult,
    NeCoancestryResult,
    NeHillResult,
    NeInbreedingResult,
    NeIndividualDeltaFResult,
    NeLTCResult,
    NeSexRatioResult,
    NeVarianceResult,
    legacy_caballero_toro,
    legacy_coancestry,
    legacy_inbreeding,
    legacy_individual_delta_f,
    legacy_ltc,
    legacy_sex_ratio,
    legacy_variance,
)
from pedigree_graph._ne_metadata import _require_closed_parentage, _require_complete_sex
from pedigree_graph._ne_rates import (
    _coancestry_from,
    _generation_kinship_summary,
    _inbreeding_from,
    _individual_delta_f_from,
    _summary_from_matrix,
)
from pedigree_graph._ne_rates import _per_gen_mean_kinship as _per_gen_mean_kinship
from pedigree_graph.summaries import GenerationKinshipSummary

if TYPE_CHECKING:
    import scipy.sparse as sp

    from pedigree_graph._core import PedigreeGraph

__all__ = [
    "GenerationInterval",
    "NeCaballeroToroResult",
    "NeCoancestryResult",
    "NeHillResult",
    "NeInbreedingResult",
    "NeIndividualDeltaFResult",
    "NeLTCResult",
    "NeSexRatioResult",
    "NeVarianceResult",
    "compute_all_ne",
    "ne_caballero_toro",
    "ne_coancestry",
    "ne_hill_overlapping",
    "ne_inbreeding",
    "ne_individual_delta_f",
    "ne_long_term_contributions",
    "ne_sex_ratio",
    "ne_variance_family_size",
]


# ---------------------------------------------------------------------------
# 0.8.0-DELETE: package-root estimators with the 0.7.1 signatures.
#
# Each resolves its prerequisites (accepting the 0.7.1 injected payloads),
# runs the same private evaluator the final estimators in
# ``pedigree_graph.effective_size`` use, and scatters the result onto the
# dense 0.7.1 record layout.
# ---------------------------------------------------------------------------


def ne_inbreeding(pg: PedigreeGraph) -> NeInbreedingResult:
    """0.8.0-DELETE: :func:`pedigree_graph.effective_size.ne_inbreeding` on the dense record."""
    cohorts = ObservedCohorts.for_graph(pg, "ne_inbreeding")
    return legacy_inbreeding(_inbreeding_from(cohorts, pg._inbreeding_values()))


def _summary_from_dense_theta(theta_per_gen: np.ndarray, cohorts: ObservedCohorts) -> GenerationKinshipSummary:
    """0.8.0-DELETE: read a 0.7.1 ``max(label) + 1`` θ̄ array back onto the observed cohorts."""
    theta = np.asarray(theta_per_gen, dtype=np.float64)[cohorts.generations]
    return GenerationKinshipSummary(
        generations=cohorts.generations,
        mean_kinship=theta,
        pair_counts=np.where(np.isfinite(theta), 1, 0).astype(np.int64),
        unlabelled_individual_count=cohorts.unlabelled_individual_count,
    )


def ne_coancestry(
    pg: PedigreeGraph,
    K: sp.csc_matrix | None = None,
    theta_per_gen: np.ndarray | None = None,
) -> NeCoancestryResult:
    """0.8.0-DELETE: :func:`pedigree_graph.effective_size.ne_coancestry` with injected θ̄.

    Args:
        pg: Pedigree graph.
        K: optional pre-built sparse kinship matrix.  Used only when
            ``theta_per_gen`` is None.
        theta_per_gen: optional pre-computed per-generation mean kinship in
            the dense 0.7.1 layout.  When supplied, K is ignored.
    """
    cohorts = ObservedCohorts.for_graph(pg, "ne_coancestry")
    if theta_per_gen is not None:
        summary = _summary_from_dense_theta(theta_per_gen, cohorts)
    elif K is not None:
        summary = _summary_from_matrix(K, np.asarray(pg.generation), np.asarray(pg.twin))
    else:
        summary = _generation_kinship_summary(pg)
    return legacy_coancestry(_coancestry_from(cohorts, summary))


def ne_variance_family_size(pg: PedigreeGraph) -> NeVarianceResult:
    """0.8.0-DELETE: :func:`pedigree_graph.effective_size.ne_variance_family_size` on the dense record."""
    cohorts = ObservedCohorts.for_graph(pg, "ne_variance_family_size")
    _require_complete_sex(pg, "ne_variance_family_size")
    _warn_if_uniform_sex(pg, "ne_variance_family_size")
    return legacy_variance(_variance_from(cohorts, _generation_family_table(pg, cohorts)))


def ne_sex_ratio(pg: PedigreeGraph) -> NeSexRatioResult:
    """0.8.0-DELETE: :func:`pedigree_graph.effective_size.ne_sex_ratio` on the dense record."""
    cohorts = ObservedCohorts.for_graph(pg, "ne_sex_ratio")
    _require_complete_sex(pg, "ne_sex_ratio")
    _warn_if_uniform_sex(pg, "ne_sex_ratio")
    return legacy_sex_ratio(_sex_ratio_from(cohorts, _sex_column(pg)))


def ne_individual_delta_f(pg: PedigreeGraph) -> NeIndividualDeltaFResult:
    """0.8.0-DELETE: :func:`pedigree_graph.effective_size.ne_individual_delta_f` on the dense record."""
    cohorts = ObservedCohorts.for_graph(pg, "ne_individual_delta_f")
    eqg = _compute_eqg(np.asarray(pg.mother), np.asarray(pg.father), pg.n)
    return legacy_individual_delta_f(_individual_delta_f_from(cohorts, pg._inbreeding_values(), eqg))


def ne_long_term_contributions(
    pg: PedigreeGraph,
    mean_contributions: FounderContributionMeans | tuple[np.ndarray, np.ndarray] | None = None,
    tol: float = 1e-6,
) -> NeLTCResult:
    """0.8.0-DELETE: :func:`pedigree_graph.effective_size.ne_long_term_contributions` with injected means."""
    cohorts = ObservedCohorts.for_graph(pg, "ne_long_term_contributions")
    _require_closed_parentage(pg, "ne_long_term_contributions")
    if mean_contributions is None:
        means = _per_gen_founder_means(pg, cohorts=cohorts)
    else:
        means = FounderContributionMeans(*mean_contributions)
    return legacy_ltc(_ltc_from(cohorts, means, tol))


def ne_hill_overlapping(pg: PedigreeGraph, vk_scale: bool = False) -> NeHillResult:
    """0.8.0-DELETE: :func:`pedigree_graph.effective_size.ne_hill_overlapping` with positional ``vk_scale``."""
    return _ne_hill_overlapping(pg, vk_scale=vk_scale)


def ne_caballero_toro(
    pg: PedigreeGraph,
    ct_accumulators: CTAccumulators | None = None,
) -> NeCaballeroToroResult:
    """0.8.0-DELETE: :func:`pedigree_graph.effective_size.ne_caballero_toro` with injected accumulators."""
    cohorts = ObservedCohorts.for_graph(pg, "ne_caballero_toro")
    _require_closed_parentage(pg, "ne_caballero_toro")
    if ct_accumulators is None:
        ct_accumulators = _caballero_toro_accumulators(pg, _founder_idx(pg), pg._inbreeding_values(), cohorts=cohorts)
    return legacy_caballero_toro(_caballero_toro_from(cohorts, ct_accumulators))


# ---------------------------------------------------------------------------
# Convenience entry: build founder structures once, dispatch all 8
# ---------------------------------------------------------------------------


# Union of the eight estimator result dataclasses.  The values of the
# ``compute_all_ne`` dict are always one of these — never an untyped dict.
NeResult = (
    NeInbreedingResult
    | NeCoancestryResult
    | NeVarianceResult
    | NeSexRatioResult
    | NeIndividualDeltaFResult
    | NeLTCResult
    | NeHillResult
    | NeCaballeroToroResult
)


def compute_all_ne(
    pg: PedigreeGraph,
    skip_ne_coancestry: bool = False,
    n_threads: int = 1,
    hill_vk_scale: bool = False,
) -> dict[str, NeResult]:
    """0.8.0-DELETE: run all eight Ne estimators on ``pg`` in the 0.7.1 layout.

    Builds the founder-contribution structures once and reuses them for
    every contribution-dependent estimator.  F is computed lazily via
    Meuwissen-Luo and cached on the graph; per-generation mean kinship
    θ̄_g is streamed from the DP without materializing the full sparse
    kinship matrix (``pg.per_gen_mean_kinship()``).  When ``n_threads``
    is greater than 1, shared mutable graph caches are populated before
    independent estimators are dispatched to worker threads.

    Args:
        pg: Pedigree graph.
        skip_ne_coancestry: when True, skip the coancestry-rate Ne
            estimator (and its DP run) entirely; ``ne_coancestry`` slot
            is populated with NaN per-gen arrays and ``ne=None``.  Use
            on very large pedigrees when only the 7 non-coancestry
            estimators are needed.
        n_threads: maximum number of worker threads for independent
            estimator calls.  ``1`` preserves serial execution.
        hill_vk_scale: forwarded to :func:`ne_hill_overlapping` as
            ``vk_scale``; when True applies Waples 2002 eq. 5 rescaling
            of ``Vk`` to the constant-N reference before computing
            Ne_H.

    Returns a dict keyed on estimator name; each value is the matching
    0.7.1 frozen result record.
    """
    if n_threads < 1:
        raise ValueError("n_threads must be >= 1")

    F = pg._inbreeding_values()
    cohorts = ObservedCohorts.for_graph(pg, "compute_all_ne")
    _require_closed_parentage(pg, "ne_long_term_contributions")
    founder_idx = _founder_idx(pg)
    ltc_means = _per_gen_founder_means(pg, founder_idx=founder_idx, cohorts=cohorts)
    ct_acc = _caballero_toro_accumulators(pg, founder_idx, F, cohorts=cohorts)

    if skip_ne_coancestry:
        g_max = int(np.asarray(pg.generation).max()) if pg.n > 0 else 0
        ne_coancestry_result = NeCoancestryResult.empty(g_max)
        theta_per_gen = None
    else:
        # Stream θ̄_g without materializing K.  pg caches the result so a
        # later direct ne_coancestry call shares the same array.
        theta_per_gen = pg.per_gen_mean_kinship()

    # One dispatch table drives both paths; serial and threaded run the same
    # estimators with the same pre-computed payloads.  ne_coancestry joins the
    # table only when not skipped — otherwise its slot is filled from the NaN
    # sentinel below.  Order is irrelevant: the shared graph caches every
    # estimator reads (F, founder summaries, θ̄) are populated above, before
    # any estimator runs, which is also what makes the threaded path safe.
    # Each value is (estimator, kwargs); every estimator takes pg as its first
    # positional argument, supplied at the call site below.
    tasks = {
        "ne_inbreeding": (ne_inbreeding, {}),
        "ne_variance_family_size": (ne_variance_family_size, {}),
        "ne_sex_ratio": (ne_sex_ratio, {}),
        "ne_individual_delta_f": (ne_individual_delta_f, {}),
        "ne_long_term_contributions": (ne_long_term_contributions, {"mean_contributions": ltc_means}),
        "ne_hill_overlapping": (ne_hill_overlapping, {"vk_scale": hill_vk_scale}),
        "ne_caballero_toro": (ne_caballero_toro, {"ct_accumulators": ct_acc}),
    }
    if not skip_ne_coancestry:
        tasks["ne_coancestry"] = (ne_coancestry, {"theta_per_gen": theta_per_gen})

    if n_threads == 1:
        results: dict[str, NeResult] = {name: func(pg, **kwargs) for name, (func, kwargs) in tasks.items()}  # ty: ignore[invalid-argument-type]
    else:
        results = {}
        with ThreadPoolExecutor(max_workers=min(n_threads, len(tasks))) as executor:
            futures = {name: executor.submit(func, pg, **kwargs) for name, (func, kwargs) in tasks.items()}  # ty: ignore[invalid-argument-type]
            for name, future in futures.items():
                results[name] = future.result()

    if skip_ne_coancestry:
        results["ne_coancestry"] = ne_coancestry_result
    return results


def _result_to_dict(result: NeResult) -> dict[str, Any]:
    """Serialize any frozen Ne result; falls back to ``dataclasses.asdict``."""
    if hasattr(result, "to_dict"):
        return result.to_dict()
    return asdict(result)
