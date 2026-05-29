"""Pedigree-based effective population size (Ne) estimators — facade.

This module is a compatibility facade (PGQ-006): the estimators, result
dataclasses, and helpers now live in focused ``_ne_*`` sibling modules,
and are re-exported here so existing import paths
(``from pedigree_graph._effective_size import …``) stay stable.

Module map:

* ``_ne_common``        — shared numeric helpers (harmonic mean, log-regression).
* ``_ne_results``       — result dataclasses + serialization.
* ``_ne_family_size``   — family-size table, Ne_V, Ne_sr.
* ``_ne_founders``      — founder contributions, Ne_LTC.
* ``_ne_caballero_toro``— CT accumulators + Ne_CT.
* ``_ne_hill``          — Hill overlapping-generation Ne_H.
* ``_ne_rates``         — Ne_I, Ne_C, Ne_iΔF and per-gen mean kinship.
* ``_effective_size``   — this facade + :func:`compute_all_ne` orchestration.

Estimator coverage:

* :func:`ne_inbreeding`              — regression of ``ln(1 − F̄_t)`` on t.
* :func:`ne_coancestry`              — regression of ``ln(1 − θ̄_t)`` on t.
* :func:`ne_variance_family_size`    — Caballero 1994 eq. 6 (separate sex,
  sex-of-offspring covariance).
* :func:`ne_sex_ratio`               — Wright ``4 N_m N_f / (N_m + N_f)``.
* :func:`ne_individual_delta_f`      — Gutiérrez 2008 individual ΔF_i via EqG.
* :func:`ne_long_term_contributions` — Wray & Thompson 1990 founder contributions.
* :func:`ne_hill_overlapping`        — Hill 1979 (collapses to Ne_V at L=1).
* :func:`ne_caballero_toro`          — Caballero & Toro 2002 self-coancestry regression.

Convenience entry: :func:`compute_all_ne` runs all eight estimators in
one call, sharing cached F, streamed θ̄, and founder-contribution
summaries where applicable.

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
from pedigree_graph._ne_caballero_toro import CTAccumulators as CTAccumulators
from pedigree_graph._ne_caballero_toro import (
    _caballero_toro_accumulators,
    ne_caballero_toro,
)
from pedigree_graph._ne_family_size import FamilySizeEntry as FamilySizeEntry
from pedigree_graph._ne_family_size import FamilySizeTable as FamilySizeTable
from pedigree_graph._ne_family_size import Sigma2Decomposition as Sigma2Decomposition
from pedigree_graph._ne_family_size import _sex_specific_family_table as _sex_specific_family_table
from pedigree_graph._ne_family_size import _sigma2_from_quadrants as _sigma2_from_quadrants
from pedigree_graph._ne_family_size import (
    ne_sex_ratio,
    ne_variance_family_size,
)
from pedigree_graph._ne_founders import (
    _founder_idx,
    _per_gen_founder_means,
    ne_long_term_contributions,
)
from pedigree_graph._ne_hill import ne_hill_overlapping
from pedigree_graph._ne_rates import _per_gen_mean_kinship as _per_gen_mean_kinship
from pedigree_graph._ne_rates import (
    ne_coancestry,
    ne_inbreeding,
    ne_individual_delta_f,
)
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

if TYPE_CHECKING:
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
    """Run all eight Ne estimators on ``pg``.

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
    frozen result dataclass.
    """
    if n_threads < 1:
        raise ValueError("n_threads must be >= 1")

    F = pg.compute_inbreeding()
    founder_idx = _founder_idx(pg)
    ltc_means = _per_gen_founder_means(pg, founder_idx=founder_idx)
    ct_acc = _caballero_toro_accumulators(pg, founder_idx, F)

    if skip_ne_coancestry:
        g_max = int(np.asarray(pg.generation).max()) if pg.n > 0 else 0
        ne_coancestry_result = NeCoancestryResult.empty(g_max)
        theta_per_gen = None
    else:
        # Stream θ̄_g without materializing K.  pg caches the result so a
        # later direct ne_coancestry call shares the same array.
        theta_per_gen = pg.per_gen_mean_kinship()

    if n_threads == 1:
        ne_variance_result = ne_variance_family_size(pg)
        if not skip_ne_coancestry:
            ne_coancestry_result = ne_coancestry(pg, theta_per_gen=theta_per_gen)
        return {
            "ne_inbreeding": ne_inbreeding(pg),
            "ne_coancestry": ne_coancestry_result,
            "ne_variance_family_size": ne_variance_result,
            "ne_sex_ratio": ne_sex_ratio(pg),
            "ne_individual_delta_f": ne_individual_delta_f(pg),
            "ne_long_term_contributions": ne_long_term_contributions(pg, mean_contributions=ltc_means),
            "ne_hill_overlapping": ne_hill_overlapping(pg, vk_scale=hill_vk_scale),
            "ne_caballero_toro": ne_caballero_toro(pg, ct_accumulators=ct_acc),
        }

    tasks = {
        "ne_inbreeding": (ne_inbreeding, (pg,), {}),
        "ne_variance_family_size": (ne_variance_family_size, (pg,), {}),
        "ne_sex_ratio": (ne_sex_ratio, (pg,), {}),
        "ne_individual_delta_f": (ne_individual_delta_f, (pg,), {}),
        "ne_long_term_contributions": (ne_long_term_contributions, (pg,), {"mean_contributions": ltc_means}),
        "ne_hill_overlapping": (ne_hill_overlapping, (pg,), {"vk_scale": hill_vk_scale}),
        "ne_caballero_toro": (ne_caballero_toro, (pg,), {"ct_accumulators": ct_acc}),
    }
    if not skip_ne_coancestry:
        tasks["ne_coancestry"] = (ne_coancestry, (pg,), {"theta_per_gen": theta_per_gen})

    results: dict[str, NeResult] = {}
    max_workers = min(n_threads, len(tasks))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {name: executor.submit(func, *args, **kwargs) for name, (func, args, kwargs) in tasks.items()}
        for name, future in futures.items():
            results[name] = future.result()

    if skip_ne_coancestry:
        results["ne_coancestry"] = ne_coancestry_result
    return results


def _result_to_dict(result: Any) -> dict[str, Any]:
    """Serialize any frozen Ne result; falls back to ``dataclasses.asdict``."""
    if hasattr(result, "to_dict"):
        return result.to_dict()
    return asdict(result)
