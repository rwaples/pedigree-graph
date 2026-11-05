"""Family-size Ne estimators and their shared primitives (PGQ-006).

Owns the sex-of-offspring family-size table and its moments, with the
Caballero 1994 eq. 17 per-sex drift denominators, plus the evaluators of the two estimators built
directly on them, :func:`~pedigree_graph.effective_size.ne_variance_family_size` (Ne_V) and
:func:`~pedigree_graph.effective_size.ne_sex_ratio` (Ne_sr), whose public functions run their
registry rows in ``_ne_estimate``.
The Hill overlapping-generation estimator (``_ne_hill``) also consumes
``_sex_specific_family_table`` / ``_sigma2_from_quadrants`` /
``_waples_vk2_expectation`` from here.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from pedigree_graph._ne_common import _harmonic_mean
from pedigree_graph._ne_results import NeSexRatioResult, NeVarianceResult

if TYPE_CHECKING:
    from pedigree_graph._cohorts import ObservedCohorts
    from pedigree_graph._core import PedigreeGraph
    from pedigree_graph._ne_results import VarianceNoEstimate


# ---------------------------------------------------------------------------
# Typed payload models
#
# Internal contracts for the intermediate structures the estimators pass
# around (PGQ-005).  These replace the former stringly typed dicts so a
# missing or misnamed field is a construction-time error rather than a
# silent ``KeyError`` deep inside an estimator.  Not part of the public
# package API; importable for tests and downstream typing.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FamilySizeEntry:
    """One parent cohort's sex-of-offspring family-size quadrants.

    Produced by :func:`_sex_specific_family_table` (one entry per parent
    cohort) and consumed by :func:`_sigma2_from_quadrants`.  The four
    ``k_*`` arrays are per-parent lifetime offspring counts; arrays are
    aligned *within sex* — ``k_mm``/``k_mf`` index the same males as
    ``males_in_parent_gen`` (length ``n_m``), and ``k_fm``/``k_ff`` the
    same females as ``females_in_parent_gen`` (length ``n_f``).

    Attributes:
        males_in_parent_gen: ``(n_m,)`` intp — row indices of males in
            the cohort.
        females_in_parent_gen: ``(n_f,)`` intp — row indices of females
            in the cohort.
        k_mm: ``(n_m,)`` int64 — male offspring per male.
        k_mf: ``(n_m,)`` int64 — female offspring per male.
        k_fm: ``(n_f,)`` int64 — male offspring per female.
        k_ff: ``(n_f,)`` int64 — female offspring per female.
    """

    males_in_parent_gen: np.ndarray
    females_in_parent_gen: np.ndarray
    k_mm: np.ndarray
    k_mf: np.ndarray
    k_fm: np.ndarray
    k_ff: np.ndarray


# Per-parent-cohort table keyed on the observed cohort label (a generation
# label for Ne_V, a birth year for Hill), one entry per observed label.
FamilySizeTable = dict[int, FamilySizeEntry]


class Sigma2Decomposition(NamedTuple):
    """Sex-of-offspring family-size moments for one cohort.

    Returned by :func:`_sigma2_from_quadrants`.  The per-sex lifetime
    offspring variance is ``σ²_m = v_mm + v_mf + 2·cov_m`` (Hill 1979 eq.
    10's definition) and ``σ²_f = v_fm + v_ff + 2·cov_f``; the pathway means
    ``mu_*`` are Caballero 1994 eq. 17's ``μ_sm``, ``μ_sf``.

    Attributes:
        v_mm, v_mf, v_fm, v_ff: sample variances (ddof=1) of the four
            offspring-count quadrants.
        cov_m: ``Cov(k_mm, k_mf)``; ``cov_f``: ``Cov(k_fm, k_ff)``.
        mu_mm, mu_mf, mu_fm, mu_ff: mean sons / daughters per male, and
            per female.
        n_m, n_f: cohort male / female counts (each ``≥ 2``).
    """

    v_mm: float
    v_mf: float
    v_fm: float
    v_ff: float
    cov_m: float
    cov_f: float
    mu_mm: float
    mu_mf: float
    mu_fm: float
    mu_ff: float
    n_m: int
    n_f: int

    @property
    def sigma2_m(self) -> float:
        return self.v_mm + self.v_mf + 2.0 * self.cov_m

    @property
    def sigma2_f(self) -> float:
        return self.v_fm + self.v_ff + 2.0 * self.cov_f

    @property
    def kbar_m(self) -> float:
        return self.mu_mm + self.mu_mf

    @property
    def kbar_f(self) -> float:
        return self.mu_fm + self.mu_ff

    def drift_denominators(self) -> tuple[float, float] | None:
        """Caballero 1994 eq. 17's per-sex denominators ``(D_m, D_f)``.

        ``N_es = 4·N_s / D_s`` with, at ``α_I = α_O = 0`` (eq. 19's
        large-population form)::

            D_s = 1/μ_sm + 1/μ_sf + S²_sm/μ_sm² + 2·S_sm,sf/(μ_sm·μ_sf) + S²_sf/μ_sf²

        The first two terms are Mendelian segregation; the rest is the
        variance of each parent's offspring counts, each pathway scaled by
        its mean.  Where each cohort replaces itself (``μ_mm = μ_ff = 1``,
        ``μ_mf = F/M``) this is Hill 1979 eq. 9, and at ``M = F`` eq. 10's
        ``σ²_s + 2``.  ``None`` when a pathway mean is 0: the cohort's
        parents of one sex had no sons or no daughters.
        """
        if min(self.mu_mm, self.mu_mf, self.mu_fm, self.mu_ff) <= 0:
            return None
        # Same-sex pathway first in both, so swapping the sexes swaps D_m
        # and D_f bit for bit.
        return (
            _sex_denominator(self.mu_mm, self.mu_mf, self.v_mm, self.v_mf, self.cov_m),
            _sex_denominator(self.mu_ff, self.mu_fm, self.v_ff, self.v_fm, self.cov_f),
        )


def _sex_denominator(mu_same: float, mu_other: float, v_same: float, v_other: float, cov: float) -> float:
    """One sex's ``D_s`` of Caballero 1994 eq. 17 at ``α = 0``."""
    return (
        1.0 / mu_same + 1.0 / mu_other + v_same / mu_same**2 + 2.0 * cov / (mu_same * mu_other) + v_other / mu_other**2
    )


# ---------------------------------------------------------------------------
# Family-size table + variance primitives
# ---------------------------------------------------------------------------


def _sex_specific_family_table(
    mother: np.ndarray,
    father: np.ndarray,
    sex: np.ndarray,
    cohorts: ObservedCohorts,
) -> FamilySizeTable:
    """Per-parent-cohort counts of male/female offspring per parent.

    For each observed cohort ``c``, tally the lifetime offspring of every
    parent in cohort ``c``, partitioned by offspring sex.  Offspring may
    live in any later cohort, so the table correctly attributes skip-gen
    children to the parent's own cohort.  Every observed label gets an
    entry, the last one included: a cohort with no offspring simply has
    all-zero counts, and downstream eligibility filtering decides what to
    keep.  Rows outside every cohort (label ``-1``) belong to no entry but
    still count as offspring of their parents.

    Args:
        mother: per-individual mother row index (``-1`` for unknown).
        father: per-individual father row index (``-1`` for unknown).
        sex: per-individual sex labels (``1`` male, ``0`` female).
        cohorts: the grouping, by generation label or by birth year.

    Returns:
        A :data:`FamilySizeTable` keyed on observed label whose values are
        :class:`FamilySizeEntry` records.
    """
    n = int(cohorts.dense.shape[0])
    sex = np.asarray(sex, dtype=np.int8)
    mother = np.asarray(mother)
    father = np.asarray(father)

    father_present = father >= 0
    mother_present = mother >= 0
    male_offspring = sex == 1

    # Lifetime offspring counts by parent row, counted once globally and
    # sliced per cohort afterwards.
    k_mm_all = np.bincount(father[father_present & male_offspring], minlength=n).astype(np.int64)
    k_mf_all = np.bincount(father[father_present & ~male_offspring], minlength=n).astype(np.int64)
    k_fm_all = np.bincount(mother[mother_present & male_offspring], minlength=n).astype(np.int64)
    k_ff_all = np.bincount(mother[mother_present & ~male_offspring], minlength=n).astype(np.int64)

    out: FamilySizeTable = {}
    for label, rows in zip(cohorts.generations, cohorts.members(), strict=True):
        m_arr = rows[sex[rows] == 1]
        f_arr = rows[sex[rows] == 0]
        out[int(label)] = FamilySizeEntry(
            males_in_parent_gen=m_arr,
            females_in_parent_gen=f_arr,
            k_mm=k_mm_all[m_arr],
            k_mf=k_mf_all[m_arr],
            k_fm=k_fm_all[f_arr],
            k_ff=k_ff_all[f_arr],
        )
    return out


_PACKAGE_DIR = str(Path(__file__).parent)


def _warn_if_uniform_sex(pg: PedigreeGraph, caller: str) -> None:
    """RuntimeWarning when ``pg.sex`` is uniform — usually a missing ``sex=``.

    Fired by :func:`~pedigree_graph.effective_size.ne_variance_family_size` and
    :func:`~pedigree_graph.effective_size.ne_sex_ratio` (and Hill's collapse branch),
    which are degenerate on a single-sex pedigree.  Absent sex is refused
    before this point (``missing_sex``), so a uniform column here was
    supplied by the caller.  Estimators still return ``ne=None`` (matching
    a legitimately single-sex pedigree), so the warning is the only
    diagnostic.

    Attributed to the first frame outside the package rather than to a fixed
    ``stacklevel``.  Every estimator that fires this is reachable both
    directly and through ``estimate_effective_sizes``, from different call
    depths, so no one constant can name the caller on both.
    """
    if pg.n_individuals == 0:
        return
    sex = np.asarray(pg.sex)
    if len(np.unique(sex)) < 2:
        warnings.warn(
            f"{caller}: pg.sex is uniform (all {int(sex[0])}); estimator is degenerate and will return ne=None",
            RuntimeWarning,
            skip_file_prefixes=(_PACKAGE_DIR,),
        )


def _sigma2_from_quadrants(entry: FamilySizeEntry) -> Sigma2Decomposition | None:
    """Sex-of-offspring moments of one :class:`FamilySizeEntry`.

    The four quadrant variances (ddof=1), the two within-sex covariances
    and the four pathway means.  Returns ``None`` when the cohort lacks two
    of either sex (Ne is undefined for that transition).
    """
    kmm = entry.k_mm
    kmf = entry.k_mf
    kfm = entry.k_fm
    kff = entry.k_ff
    n_m = len(kmm)
    n_f = len(kfm)
    if n_m < 2 or n_f < 2:
        return None
    return Sigma2Decomposition(
        v_mm=float(kmm.var(ddof=1)),
        v_mf=float(kmf.var(ddof=1)),
        v_fm=float(kfm.var(ddof=1)),
        v_ff=float(kff.var(ddof=1)),
        cov_m=float(np.cov(kmm, kmf, ddof=1)[0, 1]),
        cov_f=float(np.cov(kfm, kff, ddof=1)[0, 1]),
        mu_mm=float(kmm.mean()),
        mu_mf=float(kmf.mean()),
        mu_fm=float(kfm.mean()),
        mu_ff=float(kff.mean()),
        n_m=n_m,
        n_f=n_f,
    )


def _waples_vk2_expectation(vk1: float, kbar1: float, kbar2: float = 2.0) -> float:
    """Waples 2002 eq. 5 — expected Vk under a constant-N reference.

    Rescales an observed lifetime offspring-count variance ``vk1`` from
    its empirical mean ``kbar1`` to the value it would take if the
    population were at constant N (so the long-run lifetime mean is
    ``kbar2 = 2`` per Wright-Fisher / Caswell).  Used to strip
    demographic non-stationarity out of ``Vk`` before applying Hill 1979
    eq. (10) to overlapping-generation pedigrees that span growth or
    decline.

    Formula::

        E(Vk2) = kbar2 · [1 + (kbar2 / kbar1) · (Vk1 / kbar1 − 1)]

    Returns ``vk1`` unchanged when ``kbar1 <= 0`` (degenerate cohort).
    """
    if kbar1 <= 0:
        return vk1
    return float(kbar2 * (1.0 + (kbar2 / kbar1) * (vk1 / kbar1 - 1.0)))


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------


def _variance_from(cohorts: ObservedCohorts, table: FamilySizeTable) -> NeVarianceResult:
    k = cohorts.k
    ne_per_t = np.full(k, np.nan, dtype=np.float64)
    v_mm = np.full(k, np.nan, dtype=np.float64)
    v_mf = np.full(k, np.nan, dtype=np.float64)
    v_fm = np.full(k, np.nan, dtype=np.float64)
    v_ff = np.full(k, np.nan, dtype=np.float64)
    cov_m = np.full(k, np.nan, dtype=np.float64)
    cov_f = np.full(k, np.nan, dtype=np.float64)
    any_estimable = False

    for b, label in enumerate(cohorts.generations):
        d = _sigma2_from_quadrants(table[int(label)])
        if d is None or d.kbar_m <= 0 or d.kbar_f <= 0:
            continue
        any_estimable = True
        v_mm[b], v_mf[b], v_fm[b], v_ff[b] = d.v_mm, d.v_mf, d.v_fm, d.v_ff
        cov_m[b], cov_f[b] = d.cov_m, d.cov_f
        denominators = d.drift_denominators()
        if denominators is None:
            continue
        d_m, d_f = denominators
        # Caballero 1994 eq. 18 over eq. 17's N_es = 4·N_s/D_s.
        ne_per_t[b] = 1.0 / (d_m / (16.0 * d.n_m) + d_f / (16.0 * d.n_f))

    ne = _harmonic_mean(ne_per_t) if np.isfinite(ne_per_t).any() else None
    no_estimate_code: VarianceNoEstimate | None = None
    if ne is None:
        no_estimate_code = "single_sex_offspring" if any_estimable else "too_few_parents"
    return NeVarianceResult(
        ne=ne,
        parent_generations=cohorts.generations,
        ne_per_transition=ne_per_t,
        v_mm=v_mm,
        v_mf=v_mf,
        v_fm=v_fm,
        v_ff=v_ff,
        cov_m=cov_m,
        cov_f=cov_f,
        no_estimate_code=no_estimate_code,
    )


def _sex_column(pg: PedigreeGraph) -> np.ndarray:
    """The validated sex column; an empty graph has none and needs none."""
    sex = pg.sex
    return np.zeros(0, dtype=np.int8) if sex is None else np.asarray(sex)


def _generation_family_table(pg: PedigreeGraph, cohorts: ObservedCohorts) -> FamilySizeTable:
    """The family-size table grouped by generation label (Ne_V and Hill's collapse)."""
    return _sex_specific_family_table(np.asarray(pg.mother_rows), np.asarray(pg.father_rows), _sex_column(pg), cohorts)


def _sex_ratio_from(cohorts: ObservedCohorts, sex: np.ndarray) -> NeSexRatioResult:
    k = cohorts.k
    labelled = cohorts.dense < k
    n_male = np.bincount(cohorts.dense[labelled & (sex == 1)], minlength=k).astype(np.int64)
    n_female = np.bincount(cohorts.dense[labelled & (sex == 0)], minlength=k).astype(np.int64)

    ne_per_gen = np.full(k, np.nan, dtype=np.float64)
    for b in range(k):
        if n_male[b] > 0 and n_female[b] > 0:
            ne_per_gen[b] = 4.0 * n_male[b] * n_female[b] / (n_male[b] + n_female[b])

    ne = _harmonic_mean(ne_per_gen) if np.isfinite(ne_per_gen).any() else None
    return NeSexRatioResult(
        ne=ne,
        generations=cohorts.generations,
        ne_per_gen=ne_per_gen,
        n_male_per_gen=n_male,
        n_female_per_gen=n_female,
        no_estimate_code="no_cohort_with_both_sexes" if ne is None else None,
    )
