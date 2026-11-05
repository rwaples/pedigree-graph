"""Hill 1979 separate-sex overlapping-generation Ne (Ne_H) (PGQ-006).

Owns the parental-age diagnostic table and the evaluators behind
:func:`~pedigree_graph.effective_size.ne_hill_overlapping`, whose public
function runs its registry row in ``_ne_estimate``.  They build on the
family-size primitives in ``_ne_family_size`` (the
sex-of-offspring table, its Caballero 1994 eq. 17 drift denominators, and
the Waples 2002 Vk rescaling) and on the cohort-eligibility window in
``_cohort_utils``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from pedigree_graph._cohorts import ObservedCohorts
from pedigree_graph._ne_common import _harmonic_mean
from pedigree_graph._ne_family_size import (
    _family_size_moments,
    _sex_specific_family_table,
    _waples_vk2_expectation,
)
from pedigree_graph._ne_results import NeHillResult

if TYPE_CHECKING:
    from pedigree_graph._cohort_utils import CohortWindow
    from pedigree_graph._core import PedigreeGraph
    from pedigree_graph._ne_family_size import FamilySizeTable
    from pedigree_graph._ne_results import GenerationInterval, NeVarianceResult


def _hill_age_table(pg: PedigreeGraph) -> dict[str, np.ndarray]:
    """Parental-age-at-offspring-birth histograms (descriptive diagnostic).

    NOT used in Ne computation — survival is not observable from
    reproduction alone in real animal pedigrees, so the standard life-
    table interpretation (e.g. ``m_x`` per-capita fecundity) does not
    apply.  Returned as edge counts per parental age, per parent sex.
    """
    assert pg.birth_year is not None

    def _ages(parent_label: Literal["mother", "father"]) -> tuple[np.ndarray, np.ndarray]:
        _, diffs = pg._known_parent_edges_for(parent_label)
        if diffs.size == 0:
            return np.array([], dtype=np.int32), np.array([], dtype=np.int64)
        ages, counts = np.unique(diffs, return_counts=True)
        return ages.astype(np.int32), counts.astype(np.int64)

    ages_m, counts_m = _ages("father")
    ages_f, counts_f = _ages("mother")
    return {
        "ages_m": ages_m,
        "offspring_count_m": counts_m,
        "ages_f": ages_f,
        "offspring_count_f": counts_f,
    }


def _birth_year_family_table(pg: PedigreeGraph) -> FamilySizeTable:
    """The family-size table grouped by birth year (Hill's birth-year branch)."""
    assert pg.birth_year is not None
    cohorts = ObservedCohorts.from_labels(np.asarray(pg.birth_year))
    return _sex_specific_family_table(
        np.asarray(pg.mother_rows), np.asarray(pg.father_rows), np.asarray(pg.sex), cohorts
    )


def _hill_from_variance(variance: NeVarianceResult, vk_scale: bool) -> NeHillResult:
    """The sentinel branch: Hill at ``L = 1`` is the Ne_V passthrough.

    ``vk_scaled`` still records what was asked, so the result describes the
    request whatever branch answered it.
    """
    return NeHillResult(
        ne=variance.ne,
        generation_interval=1.0,
        collapses_to_ne_v=True,
        vk_scaled=vk_scale,
        no_estimate_code=variance.no_estimate_code,
    )


def _hill_from(
    pg: PedigreeGraph,
    gi: GenerationInterval,
    window: CohortWindow,
    table: FamilySizeTable,
    vk_scale: bool,
) -> NeHillResult:
    """The birth-year branch over resolved prerequisites."""
    ne_per_c: list[float] = []
    Ne_m_per_c: list[float] = []
    Ne_f_per_c: list[float] = []
    Vk_m_per_c: list[float] = []
    Vk_f_per_c: list[float] = []
    kbar_m_per_c: list[float] = []
    kbar_f_per_c: list[float] = []
    N1_m_per_c: list[int] = []
    N1_f_per_c: list[int] = []
    cohort_years_kept: list[int] = []

    T = gi.T
    # A windowed cohort dropped for a zero pathway mean, so an empty result
    # names that cause, as the sentinel does, rather than eligibility.
    single_sex_offspring = False
    for c, entry in table.items():
        if c < window.c_min or c > window.c_max:
            continue
        # Need at least 2 of each sex for sample variance (ddof=1) and a
        # meaningful covariance — _family_size_moments returns None below.
        d = _family_size_moments(entry)
        if d is None:
            continue
        kbar_m, kbar_f, n_m, n_f = d.kbar_m, d.kbar_f, d.n_m, d.n_f
        sigma2_m, sigma2_f = d.sigma2_m, d.sigma2_f

        if vk_scale:
            # Waples 2002 eq. 5 rescales σ² to k̄ = 2, after which Hill
            # eq. 10's σ²_s + 2 applies per sex.
            sigma2_m = _waples_vk2_expectation(sigma2_m, kbar_m)
            sigma2_f = _waples_vk2_expectation(sigma2_f, kbar_f)
            denom_m = sigma2_m + 2.0
            denom_f = sigma2_f + 2.0
        else:
            # Caballero 1994 eq. 17, Ne_V's form: the branches agree at T = 1.
            denominators = d.drift_denominators()
            if denominators is None:
                single_sex_offspring = True
                continue
            denom_m, denom_f = denominators
        if denom_m <= 0 or denom_f <= 0:
            continue
        ne_m_c = 4.0 * n_m * T / denom_m
        ne_f_c = 4.0 * n_f * T / denom_f
        if ne_m_c + ne_f_c <= 0:
            continue
        # Wright 1938 sex-ratio combination (paper eq. 3)
        ne_c = 4.0 * ne_m_c * ne_f_c / (ne_m_c + ne_f_c)
        ne_per_c.append(ne_c)
        Ne_m_per_c.append(ne_m_c)
        Ne_f_per_c.append(ne_f_c)
        Vk_m_per_c.append(sigma2_m)
        Vk_f_per_c.append(sigma2_f)
        kbar_m_per_c.append(kbar_m)
        kbar_f_per_c.append(kbar_f)
        N1_m_per_c.append(n_m)
        N1_f_per_c.append(n_f)
        cohort_years_kept.append(int(c))

    by = pg.birth_year
    assert by is not None  # generation_interval guard above is None iff birth_year is None
    n_unknown = int((by < 0).sum())
    known = by[by >= 0]
    n_left = int((known < window.c_min).sum())
    n_right = int((known > window.c_max).sum())
    age_table = _hill_age_table(pg)
    n_pairs = gi.n_edges

    if not ne_per_c:
        return NeHillResult(
            ne=None,
            generation_interval=gi.T,
            collapses_to_ne_v=False,
            T_m=gi.T_m,
            T_f=gi.T_f,
            vk_scaled=vk_scale,
            cohort_window=window,
            n_eligible_cohorts=0,
            n_excluded_left_censored=n_left,
            n_excluded_right_censored=n_right,
            n_unknown_birth_year=n_unknown,
            age_table=age_table,
            n_offspring_pairs=n_pairs,
            no_estimate_code="single_sex_offspring" if single_sex_offspring else "no_eligible_cohorts",
        )

    # Every kept cohort has a positive, finite Ne: both sexes share T and a
    # positive denominator, so ne_m_c and ne_f_c share a sign, and a cohort
    # whose sum is not positive was skipped above.
    ne_arr = np.array(ne_per_c, dtype=np.float64)
    ne_h = _harmonic_mean(ne_arr)
    ne_m_arr = np.array(Ne_m_per_c, dtype=np.float64)
    ne_f_arr = np.array(Ne_f_per_c, dtype=np.float64)
    ne_m_scalar = _harmonic_mean(ne_m_arr) if np.isfinite(ne_m_arr).any() else None
    ne_f_scalar = _harmonic_mean(ne_f_arr) if np.isfinite(ne_f_arr).any() else None

    return NeHillResult(
        ne=ne_h,
        generation_interval=gi.T,
        collapses_to_ne_v=False,
        T_m=gi.T_m,
        T_f=gi.T_f,
        N1_m=float(np.mean(N1_m_per_c)),
        N1_f=float(np.mean(N1_f_per_c)),
        Vk_m=float(np.mean(Vk_m_per_c)),
        Vk_f=float(np.mean(Vk_f_per_c)),
        kbar_m=float(np.mean(kbar_m_per_c)),
        kbar_f=float(np.mean(kbar_f_per_c)),
        Ne_m=ne_m_scalar,
        Ne_f=ne_f_scalar,
        vk_scaled=vk_scale,
        cohort_window=window,
        n_eligible_cohorts=len(ne_per_c),
        n_excluded_left_censored=n_left,
        n_excluded_right_censored=n_right,
        n_unknown_birth_year=n_unknown,
        cohort_years=np.array(cohort_years_kept, dtype=np.int32),
        ne_per_cohort=ne_arr,
        Ne_m_per_cohort=ne_m_arr,
        Ne_f_per_cohort=ne_f_arr,
        Vk_m_per_cohort=np.array(Vk_m_per_c, dtype=np.float64),
        Vk_f_per_cohort=np.array(Vk_f_per_c, dtype=np.float64),
        N1_m_per_cohort=np.array(N1_m_per_c, dtype=np.int64),
        N1_f_per_cohort=np.array(N1_f_per_c, dtype=np.int64),
        age_table=age_table,
        n_offspring_pairs=n_pairs,
    )
