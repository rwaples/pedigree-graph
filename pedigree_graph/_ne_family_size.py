"""Family-size Ne estimators and their shared primitives (PGQ-006).

Owns the sex-of-offspring family-size table and the Caballero 1994 eq. 6
variance reassembly, plus the two estimators built directly on them:
:func:`ne_variance_family_size` (Ne_V) and :func:`ne_sex_ratio` (Ne_sr).
The Hill overlapping-generation estimator (``_ne_hill``) also consumes
``_sex_specific_family_table`` / ``_sigma2_from_quadrants`` /
``_waples_vk2_expectation`` from here.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from pedigree_graph._cohorts import ObservedCohorts
from pedigree_graph._ne_common import _harmonic_mean
from pedigree_graph._ne_metadata import _require_complete_sex
from pedigree_graph._ne_results import NeSexRatioResult, NeVarianceResult

if TYPE_CHECKING:
    from pedigree_graph._core import PedigreeGraph


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
    """Caballero 1994 eq. 6 reassembly terms for one cohort.

    Returned by :func:`_sigma2_from_quadrants`.  Callers reassemble the
    per-sex lifetime-offspring variance as ``σ²_m = v_mm + v_mf +
    2·cov_m`` and ``σ²_f = v_fm + v_ff + 2·cov_f``.  Remains
    tuple-unpackable for positional call sites.

    Attributes:
        v_mm, v_mf, v_fm, v_ff: sample variances (ddof=1) of the four
            offspring-count quadrants.
        cov_m: ``Cov(k_mm, k_mf)``; ``cov_f``: ``Cov(k_fm, k_ff)``.
        kbar_m, kbar_f: mean total lifetime offspring per male / female.
        n_m, n_f: cohort male / female counts (each ``≥ 2``).
    """

    v_mm: float
    v_mf: float
    v_fm: float
    v_ff: float
    cov_m: float
    cov_f: float
    kbar_m: float
    kbar_f: float
    n_m: int
    n_f: int


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


def _warn_if_uniform_sex(pg: PedigreeGraph, caller: str) -> None:
    """RuntimeWarning when ``pg.sex`` is uniform — usually a missing ``sex=``.

    Fired by :func:`ne_variance_family_size` and :func:`ne_sex_ratio`,
    which are degenerate on a single-sex pedigree.  Absent sex is refused
    before this point (``missing_sex``), so a uniform column here was
    supplied by the caller.  Estimators still return ``ne=None`` (matching
    a legitimately single-sex pedigree), so the warning is the only
    diagnostic.
    """
    if pg.n_individuals == 0:
        return
    sex = np.asarray(pg.sex)
    if len(np.unique(sex)) < 2:
        warnings.warn(
            f"{caller}: pg.sex is uniform (all {int(sex[0])}); estimator is degenerate and will return ne=None",
            RuntimeWarning,
            stacklevel=3,
        )


def _sigma2_from_quadrants(entry: FamilySizeEntry) -> Sigma2Decomposition | None:
    """Caballero 1994 eq. 6 reassembly from sex-of-offspring quadrants.

    Computes the per-sex lifetime-offspring variance ``σ²_s = V(k_ss) +
    V(k_sf) + 2·Cov(k_ss, k_sf)`` and the per-sex offspring-count mean
    ``k̄_s`` from one :class:`FamilySizeEntry`.  Returns ``None`` when the
    cohort lacks two of either sex (Ne is undefined for that transition).

    Returns:
        A :class:`Sigma2Decomposition` (six variance/covariance terms,
        two means, two cohort sizes), or ``None``.
    """
    kmm = entry.k_mm
    kmf = entry.k_mf
    kfm = entry.k_fm
    kff = entry.k_ff
    n_m = len(kmm)
    n_f = len(kfm)
    if n_m < 2 or n_f < 2:
        return None
    kbar_m = float((kmm + kmf).mean())
    kbar_f = float((kfm + kff).mean())
    v_mm = float(kmm.var(ddof=1))
    v_mf = float(kmf.var(ddof=1))
    v_fm = float(kfm.var(ddof=1))
    v_ff = float(kff.var(ddof=1))
    cov_m = float(np.cov(kmm, kmf, ddof=1)[0, 1])
    cov_f = float(np.cov(kfm, kff, ddof=1)[0, 1])
    return Sigma2Decomposition(
        v_mm=v_mm,
        v_mf=v_mf,
        v_fm=v_fm,
        v_ff=v_ff,
        cov_m=cov_m,
        cov_f=cov_f,
        kbar_m=kbar_m,
        kbar_f=kbar_f,
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

    for b, label in enumerate(cohorts.generations):
        decomp = _sigma2_from_quadrants(table[int(label)])
        if decomp is None:
            continue
        v_mm_p, v_mf_p, v_fm_p, v_ff_p, cov_m_p, cov_f_p, kbar_m, kbar_f, n_m, n_f = decomp
        if kbar_m <= 0 or kbar_f <= 0:
            continue
        v_mm[b] = v_mm_p
        v_mf[b] = v_mf_p
        v_fm[b] = v_fm_p
        v_ff[b] = v_ff_p
        cov_m[b] = cov_m_p
        cov_f[b] = cov_f_p
        var_km_total = v_mm_p + v_mf_p + 2.0 * cov_m_p
        var_kf_total = v_fm_p + v_ff_p + 2.0 * cov_f_p
        df = (var_km_total / kbar_m) / (4.0 * n_m * kbar_m) + (var_kf_total / kbar_f) / (4.0 * n_f * kbar_f)
        if df > 0:
            ne_per_t[b] = 1.0 / (2.0 * df)

    return NeVarianceResult(
        ne=_harmonic_mean(ne_per_t) if np.isfinite(ne_per_t).any() else None,
        parent_generations=cohorts.generations,
        ne_per_transition=ne_per_t,
        v_mm=v_mm,
        v_mf=v_mf,
        v_fm=v_fm,
        v_ff=v_ff,
        cov_m=cov_m,
        cov_f=cov_f,
    )


def _sex_column(pg: PedigreeGraph) -> np.ndarray:
    """The validated sex column; an empty graph has none and needs none."""
    sex = pg.sex
    return np.zeros(0, dtype=np.int8) if sex is None else np.asarray(sex)


def _generation_family_table(pg: PedigreeGraph, cohorts: ObservedCohorts) -> FamilySizeTable:
    """The family-size table grouped by generation label (Ne_V and Hill's collapse)."""
    return _sex_specific_family_table(np.asarray(pg.mother_rows), np.asarray(pg.father_rows), _sex_column(pg), cohorts)


def ne_variance_family_size(pg: PedigreeGraph) -> NeVarianceResult:
    """Variance-of-family-size Ne (Ne_V) — Caballero 1994 eq. 6.

    For each observed parent cohort, decompose lifetime offspring counts per
    parent by offspring sex (``k_mm, k_mf, k_fm, k_ff``).  Skip-gen
    children (offspring in a later cohort than the next) are attributed to
    the parent's own cohort, so a parent's k-totals reflect their full
    reproductive output even when offspring span several cohorts.

    ``V(k_m) = V(k_mm) + V(k_mf) + 2·Cov(k_mm, k_mf)`` is the per-male
    total-offspring variance built from this decomposition.  Discrete-
    generation Ne for the cohort is

        ``ΔF = (V(k_m)/k̄_m) / (4 · N_m · k̄_m) +
                  (V(k_f)/k̄_f) / (4 · N_f · k̄_f)``,

    with ``Ne_p = 1/(2·ΔF)``.  When ``V(k)/k̄ → 1`` (Poisson) and
    ``N_m = N_f``, this reduces to Wright's ``4 N_m N_f / (N_m + N_f)``.
    Aggregate Ne is the harmonic mean across parent cohorts.

    Requires complete generation labels (or none) and complete sex, in
    that order: absent or partly unknown sex raises ``missing_sex``.
    Emits a ``RuntimeWarning`` when ``pg.sex`` is uniformly 0 or 1.  The
    estimator still returns ``ne=None`` (consistent with a legitimate
    single-sex pedigree), so the warning is the only diagnostic.
    """
    cohorts = ObservedCohorts.for_graph(pg, "ne_variance_family_size")
    _require_complete_sex(pg, "ne_variance_family_size")
    _warn_if_uniform_sex(pg, "ne_variance_family_size")
    return _variance_from(cohorts, _generation_family_table(pg, cohorts))


def _sex_ratio_from(cohorts: ObservedCohorts, sex: np.ndarray) -> NeSexRatioResult:
    k = cohorts.k
    labelled = cohorts.dense < k
    n_male = np.bincount(cohorts.dense[labelled & (sex == 1)], minlength=k).astype(np.int64)
    n_female = np.bincount(cohorts.dense[labelled & (sex == 0)], minlength=k).astype(np.int64)

    ne_per_gen = np.full(k, np.nan, dtype=np.float64)
    for b in range(k):
        if n_male[b] > 0 and n_female[b] > 0:
            ne_per_gen[b] = 4.0 * n_male[b] * n_female[b] / (n_male[b] + n_female[b])

    return NeSexRatioResult(
        ne=_harmonic_mean(ne_per_gen) if np.isfinite(ne_per_gen).any() else None,
        generations=cohorts.generations,
        ne_per_gen=ne_per_gen,
        n_male_per_gen=n_male,
        n_female_per_gen=n_female,
    )


def ne_sex_ratio(pg: PedigreeGraph) -> NeSexRatioResult:
    """Wright sex-ratio Ne (Ne_sr).

    ``Ne_g = 4·Nm_g·Nf_g / (Nm_g + Nf_g)`` per observed cohort; aggregate is
    the harmonic mean across cohorts with both sexes present.

    Emits a ``RuntimeWarning`` when ``pg.sex`` is uniformly 0 or 1.  The
    estimator still returns ``ne=None`` in that case (consistent with a
    legitimate single-sex pedigree), so the warning is the only diagnostic.
    """
    cohorts = ObservedCohorts.for_graph(pg, "ne_sex_ratio")
    _require_complete_sex(pg, "ne_sex_ratio")
    _warn_if_uniform_sex(pg, "ne_sex_ratio")
    return _sex_ratio_from(cohorts, _sex_column(pg))
