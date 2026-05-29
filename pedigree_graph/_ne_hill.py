"""Hill 1979 separate-sex overlapping-generation Ne (Ne_H) (PGQ-006).

Owns the parental-age diagnostic table and :func:`ne_hill_overlapping`,
which builds on the family-size primitives in ``_ne_family_size`` (the
sex-of-offspring table, the Caballero 1994 eq. 6 variance reassembly, and
the Waples 2002 Vk rescaling) and on the cohort-eligibility window in
``_cohort_utils``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from pedigree_graph._cohort_utils import eligible_cohort_range
from pedigree_graph._ne_common import _harmonic_mean
from pedigree_graph._ne_family_size import (
    _sex_specific_family_table,
    _sigma2_from_quadrants,
    _waples_vk2_expectation,
    ne_variance_family_size,
)
from pedigree_graph._ne_results import NeHillResult

if TYPE_CHECKING:
    from pedigree_graph._core import PedigreeGraph


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


def ne_hill_overlapping(pg: PedigreeGraph, vk_scale: bool = False) -> NeHillResult:
    """Hill 1979 separate-sex overlapping-generation Ne (Ne_H).

    Dispatches on ``pg.birth_year``:

    * **Sentinel branch** (``pg.birth_year is None``, or one sex has no
      qualifying parent-child edges): the discrete-generation form of
      Hill 1979 reduces to Ne_V (Caballero 1994 eq. 6 = Hill 1979 eq. 8
      under symmetric sexes), so ``ne`` is the
      :func:`ne_variance_family_size` passthrough and
      ``collapses_to_ne_v=True``.
    * **Birth-year branch**: per eligible birth-year cohort, computes
      sex-specific Ne via Hill 1979 eq. (4)::

          Ne_s(c) = 4·N1_s(c)·T / (σ²_s(c) + 2)         s ∈ {m, f}

      and combines them with Wright 1938 sex-ratio (paper eq. 3)::

          1/Ne(c) = 1/(4·Ne_m(c)) + 1/(4·Ne_f(c))

      where ``σ²_m = V(k_mm) + V(k_mf) + 2·Cov(k_mm, k_mf)`` is the
      per-male lifetime offspring variance reassembled from sex-of-
      offspring quadrants (Caballero 1994 eq. 6) and ``σ²_f`` is the
      symmetric female form.  ``N1_s(c)`` is the number of newborns of
      sex ``s`` in cohort ``c`` (zero-offspring individuals included);
      ``T = (T_m + T_f) / 2`` from
      :attr:`PedigreeGraph.generation_interval`.

      Scenario-scalar ``ne``, ``Ne_m``, ``Ne_f`` are harmonic means
      across cohorts within
      :func:`~pedigree_graph._cohort_utils.eligible_cohort_range`.

      Under balanced sex (``N1_m ≈ N1_f``) this matches the eq. (10)
      form ``Ne = 8·N·T / (σ²_m + σ²_f + 4)``; the per-sex form is
      preferred because it surfaces ``Ne_m`` and ``Ne_f`` directly and
      handles sex-asymmetric cohorts correctly.

    Args:
        pg: Pedigree graph.
        vk_scale: when ``True``, rescale ``σ²_m`` and ``σ²_f`` per
            cohort via Waples 2002 eq. (5) so the resulting Ne assumes
            a constant-N reference (``k̄ = 2``).  Removes demographic
            non-stationarity from ``Vk`` over populations spanning
            growth or decline.  Default ``False`` (raw sample
            variances).
    """
    # pg.generation_interval is None iff pg.birth_year is None or one sex
    # has no qualifying edges — both cases collapse to Ne_V passthrough.
    gi = pg.generation_interval
    if gi is None:
        v = ne_variance_family_size(pg)
        return NeHillResult(
            ne=v.ne,
            generation_interval=1.0,
            collapses_to_ne_v=True,
        )

    window = eligible_cohort_range(pg)
    table = _sex_specific_family_table(
        np.asarray(pg.mother),
        np.asarray(pg.father),
        np.asarray(pg.sex),
        np.asarray(pg.generation),
        cohort=pg.birth_year,
        cohort_mode="arbitrary",
    )

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
    for c, entry in table.items():
        if c < window.c_min or c > window.c_max:
            continue
        # Need at least 2 of each sex for sample variance (ddof=1) and a
        # meaningful covariance — _sigma2_from_quadrants returns None below.
        decomp = _sigma2_from_quadrants(entry)
        if decomp is None:
            continue
        v_mm, v_mf, v_fm, v_ff, cov_m, cov_f, kbar_m, kbar_f, n_m, n_f = decomp
        sigma2_m = v_mm + v_mf + 2.0 * cov_m  # Hill 1979 σ²_m
        sigma2_f = v_fm + v_ff + 2.0 * cov_f

        if vk_scale:
            sigma2_m = _waples_vk2_expectation(sigma2_m, kbar_m)
            sigma2_f = _waples_vk2_expectation(sigma2_f, kbar_f)

        denom_m = sigma2_m + 2.0
        denom_f = sigma2_f + 2.0
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
        )

    ne_arr = np.array(ne_per_c, dtype=np.float64)
    ne_h = _harmonic_mean(ne_arr) if np.isfinite(ne_arr).any() else None
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
