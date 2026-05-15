"""Pedigree-based effective population size (Ne) estimators.

Each estimator consumes a :class:`~pedigree_graph.PedigreeGraph` (and an
optional precomputed kinship matrix where applicable) and returns a
frozen result dataclass with a per-generation series and a scenario-level
scalar aggregate.

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

Convenience entry: :func:`compute_all_ne` builds the kinship matrix and
the founder-contribution matrix once and dispatches all eight estimators.

Founders are excluded from the ΔF / Δθ regressions; they are included
in the parent set for the gen-0 → gen-1 family-size variance transition.
"""

from __future__ import annotations

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

import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from pedigree_graph._cohort_utils import CohortWindow, eligible_cohort_range
from pedigree_graph._kinship_kernel import _compute_eqg, _finalize_from_sum_theta

if TYPE_CHECKING:
    import scipy.sparse as sp

    from pedigree_graph._core import PedigreeGraph


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _per_gen_mean_kinship(
    K: sp.csc_matrix,
    generation: np.ndarray,
    twin_idx: np.ndarray,
) -> np.ndarray:
    """Mean θ over unordered within-cohort pairs, per generation.

    Excludes the diagonal and MZ twin pairs.  Returns ``np.nan`` for
    cohorts with fewer than 2 non-twin members or where every pair is
    a twin pair.

    Args:
        K: full-symmetric sparse kinship (φ-scale) from
            :meth:`PedigreeGraph.kinship_matrix`.
        generation: per-individual generation index (founders = 0).
        twin_idx: per-individual twin partner row index, ``-1`` for
            non-twins.
    """
    g_max = int(generation.max())

    # COO traversal — restrict to upper triangle, same generation,
    # non-twin pairs.  Sum per-gen θ via bincount, then divide through
    # the shared finalizer that knows the within-cohort pair-count math.
    coo = K.tocoo()
    rows, cols, vals = coo.row, coo.col, coo.data
    pair_mask = (rows < cols) & (generation[rows] == generation[cols])
    pair_mask &= ~((twin_idx[rows] >= 0) & (twin_idx[rows] == cols))
    pair_gens = generation[rows[pair_mask]].astype(np.intp)
    sum_theta = np.bincount(
        pair_gens, weights=vals[pair_mask].astype(np.float64), minlength=g_max + 1,
    )
    return _finalize_from_sum_theta(sum_theta, generation, twin_idx)


def _sex_specific_family_table(
    mother: np.ndarray,
    father: np.ndarray,
    sex: np.ndarray,
    generation: np.ndarray,
    *,
    cohort: np.ndarray | None = None,
    cohort_mode: Literal["compact_generation", "arbitrary"] = "compact_generation",
) -> dict[int, dict[str, np.ndarray]]:
    """Per-parent-cohort counts of male/female offspring per parent.

    For each parent cohort ``c``, tally the lifetime offspring of every
    parent in cohort ``c``, partitioned by offspring sex.  Offspring
    may live in any later cohort, so the table correctly attributes
    skip-gen children to the parent's own cohort.  Under strictly
    layered pedigrees this collapses to the standard
    one-transition-per-cohort decomposition.

    Args:
        mother: per-individual mother row index (``-1`` for unknown).
        father: per-individual father row index (``-1`` for unknown).
        sex: per-individual sex labels (``1`` male, ``0`` female).
        generation: per-individual generation index (used by
            ``compact_generation`` mode).
        cohort: per-individual cohort label.  Required when
            ``cohort_mode == 'arbitrary'``; ignored in
            ``compact_generation`` mode.
        cohort_mode:
            * ``'compact_generation'`` (default, backward compatible):
              cohort labels are ``0 .. g_max - 1`` (max excluded — last
              cohort has no offspring under a discrete-generation
              pedigree).
            * ``'arbitrary'``: cohort labels are taken from
              ``np.unique(cohort[cohort >= 0])``.  ``-1`` sentinels
              are filtered; the max cohort is included (downstream
              eligibility filtering handles right-censoring).

    Returns:
        Dict keyed on parent cohort label.  Each entry is a dict with
        arrays:

        * ``males_in_parent_gen`` — row indices of males in cohort
        * ``females_in_parent_gen`` — row indices of females in cohort
        * ``k_mm`` — per-male count of male offspring
        * ``k_mf`` — per-male count of female offspring
        * ``k_fm`` — per-female count of male offspring
        * ``k_ff`` — per-female count of female offspring
    """
    n = len(generation)
    sex = np.asarray(sex, dtype=np.int8)

    if cohort_mode == "compact_generation":
        cohort_arr = np.asarray(generation)
        g_max = int(cohort_arr.max())
        cohort_labels: list[int] = list(range(g_max))
    elif cohort_mode == "arbitrary":
        if cohort is None:
            raise ValueError("cohort_mode='arbitrary' requires the cohort argument")
        cohort_arr = np.asarray(cohort)
        cohort_labels = [int(c) for c in np.unique(cohort_arr[cohort_arr >= 0])]
    else:
        raise ValueError(
            f"cohort_mode must be 'compact_generation' or 'arbitrary'; got {cohort_mode!r}"
        )

    # Per-cohort parent arrays + global parent → local index maps.
    cohort_males: dict[int, np.ndarray] = {}
    cohort_females: dict[int, np.ndarray] = {}
    parent_to_male_local = np.full(n, -1, dtype=np.int32)
    parent_to_female_local = np.full(n, -1, dtype=np.int32)
    for c in cohort_labels:
        in_c = cohort_arr == c
        m_arr = np.where(in_c & (sex == 1))[0]
        f_arr = np.where(in_c & (sex == 0))[0]
        cohort_males[c] = m_arr
        cohort_females[c] = f_arr
        parent_to_male_local[m_arr] = np.arange(len(m_arr), dtype=np.int32)
        parent_to_female_local[f_arr] = np.arange(len(f_arr), dtype=np.int32)

    k_mm: dict[int, np.ndarray] = {c: np.zeros(len(cohort_males[c]), dtype=np.int64) for c in cohort_labels}
    k_mf: dict[int, np.ndarray] = {c: np.zeros(len(cohort_males[c]), dtype=np.int64) for c in cohort_labels}
    k_fm: dict[int, np.ndarray] = {c: np.zeros(len(cohort_females[c]), dtype=np.int64) for c in cohort_labels}
    k_ff: dict[int, np.ndarray] = {c: np.zeros(len(cohort_females[c]), dtype=np.int64) for c in cohort_labels}

    # Single pass over individuals with at least one parent edge; the
    # parent-presence filter is a safe superset of "non-founder" and
    # works for both compact_generation and arbitrary cohort modes.
    offs_idx = np.where((np.asarray(father) >= 0) | (np.asarray(mother) >= 0))[0]

    for i in offs_idx:
        o_sex = sex[i]
        f = int(father[i])
        m = int(mother[i])
        if f >= 0:
            lf = int(parent_to_male_local[f])
            if lf >= 0:
                fp = int(cohort_arr[f])
                if o_sex == 1:
                    k_mm[fp][lf] += 1
                else:
                    k_mf[fp][lf] += 1
        if m >= 0:
            lm = int(parent_to_female_local[m])
            if lm >= 0:
                mp = int(cohort_arr[m])
                if o_sex == 1:
                    k_fm[mp][lm] += 1
                else:
                    k_ff[mp][lm] += 1

    return {
        c: {
            "males_in_parent_gen": cohort_males[c],
            "females_in_parent_gen": cohort_females[c],
            "k_mm": k_mm[c],
            "k_mf": k_mf[c],
            "k_fm": k_fm[c],
            "k_ff": k_ff[c],
        }
        for c in cohort_labels
    }


def _regress_log_one_minus(values: np.ndarray, t: np.ndarray) -> tuple[float, float]:
    """OLS of ``ln(1 − values)`` on t; return (slope, intercept).

    NaN-skipping, requires ``≥ 2`` finite points; returns ``(nan, nan)``
    otherwise.  Values ``≥ 1`` are dropped (log diverges).
    """
    finite = np.isfinite(values) & (values < 1.0)
    if finite.sum() < 2:
        return float("nan"), float("nan")
    y = np.log(1.0 - values[finite])
    x = t[finite]
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GenerationInterval:
    """Sex-split generation interval (Hill 1979 ``L``).

    ``T_m`` is the mean of ``child.birth_year − sire.birth_year`` over
    all sire-offspring edges where both endpoints have known
    ``birth_year``; ``T_f`` is the symmetric form over dam-offspring
    edges; ``T = (T_m + T_f) / 2``.  ``n_edges`` is the total count
    of qualifying edges (sire + dam) used in the means.

    Skip-generation edges are included unconditionally — Hill's
    pathway means in eq. (9) make no distinction.
    """

    T: float
    T_m: float
    T_f: float
    n_edges: int

    def to_dict(self) -> dict[str, Any]:
        """Serialize as a YAML-ready dict."""
        return {
            "T": float(self.T),
            "T_m": float(self.T_m),
            "T_f": float(self.T_f),
            "n_edges": int(self.n_edges),
        }


@dataclass(frozen=True, slots=True)
class NeInbreedingResult:
    """Inbreeding-rate (Ne_I) result.

    Attributes:
        ne: scalar Ne from regression of ``ln(1 − F̄_t)`` on t (founders excluded).
        ne_per_gen: per-transition Ne (one per gen-transition g − 1 → g, g ≥ 1).
        mean_f_per_gen: per-cohort mean F.
        slope: regression slope (log scale).
        n_generations_used: number of points in the regression.
    """

    ne: float | None
    ne_per_gen: np.ndarray
    mean_f_per_gen: np.ndarray
    slope: float
    n_generations_used: int

    def to_dict(self) -> dict[str, Any]:
        """Serialize as a YAML-ready dict (numpy arrays → list)."""
        return {
            "ne": _optional_float(self.ne),
            "ne_per_gen": [_optional_float(v) for v in self.ne_per_gen],
            "mean_f_per_gen": [float(v) for v in self.mean_f_per_gen],
            "slope": _optional_float(self.slope),
            "n_generations_used": int(self.n_generations_used),
        }


@dataclass(frozen=True, slots=True)
class NeCoancestryResult:
    """Coancestry-rate (Ne_C) result.

    Attributes:
        ne: scalar Ne from regression of ``ln(1 − θ̄_t)`` on t (founders excluded).
        ne_per_gen: per-transition Ne.
        mean_theta_per_gen: per-cohort mean θ over within-cohort pairs.
        slope: regression slope.
        n_generations_used: number of points in the regression.
    """

    ne: float | None
    ne_per_gen: np.ndarray
    mean_theta_per_gen: np.ndarray
    slope: float
    n_generations_used: int

    @classmethod
    def empty(cls, g_max: int) -> NeCoancestryResult:
        """All-NaN result of the right shape; used when Ne_C is skipped."""
        return cls(
            ne=None,
            ne_per_gen=np.full(g_max + 1, np.nan, dtype=np.float64),
            mean_theta_per_gen=np.full(g_max + 1, np.nan, dtype=np.float64),
            slope=float("nan"),
            n_generations_used=0,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize as a YAML-ready dict."""
        return {
            "ne": _optional_float(self.ne),
            "ne_per_gen": [_optional_float(v) for v in self.ne_per_gen],
            "mean_theta_per_gen": [_optional_float(v) for v in self.mean_theta_per_gen],
            "slope": _optional_float(self.slope),
            "n_generations_used": int(self.n_generations_used),
        }


@dataclass(frozen=True, slots=True)
class NeVarianceResult:
    """Variance-of-family-size (Ne_V) result.

    Caballero 1994 eq. 6 with separate sexes.  ``V(k_m) = V(k_mm) +
    V(k_mf) + 2·Cov(k_mm, k_mf)`` is the per-male total-offspring
    variance built from the sex-of-offspring decomposition; symmetrically
    for females.

    Per-transition arrays (``ne_per_transition``, ``v_mm``, …) are
    indexed by **parent generation** ``p ∈ [0, g_max)``: entry ``p``
    summarises the lifetime reproduction of cohort ``p``, which under
    skip-gen pedigrees may include offspring spread across multiple
    descendant generations.  Aggregate Ne is the harmonic mean.
    """

    ne: float | None
    ne_per_transition: np.ndarray
    v_mm: np.ndarray
    v_mf: np.ndarray
    v_fm: np.ndarray
    v_ff: np.ndarray
    cov_m: np.ndarray
    cov_f: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        """Serialize as a YAML-ready dict."""
        return {
            "ne": _optional_float(self.ne),
            "ne_per_transition": [_optional_float(v) for v in self.ne_per_transition],
            "v_mm": [float(v) for v in self.v_mm],
            "v_mf": [float(v) for v in self.v_mf],
            "v_fm": [float(v) for v in self.v_fm],
            "v_ff": [float(v) for v in self.v_ff],
            "cov_m": [float(v) for v in self.cov_m],
            "cov_f": [float(v) for v in self.cov_f],
        }


@dataclass(frozen=True, slots=True)
class NeSexRatioResult:
    """Wright sex-ratio (Ne_sr) result.

    ``Ne_t = 4·Nm_t·Nf_t / (Nm_t + Nf_t)`` per generation; aggregate is
    the harmonic mean across cohorts with at least one of each sex.
    """

    ne: float | None
    ne_per_gen: np.ndarray
    n_male_per_gen: np.ndarray
    n_female_per_gen: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        """Serialize as a YAML-ready dict."""
        return {
            "ne": _optional_float(self.ne),
            "ne_per_gen": [_optional_float(v) for v in self.ne_per_gen],
            "n_male_per_gen": [int(v) for v in self.n_male_per_gen],
            "n_female_per_gen": [int(v) for v in self.n_female_per_gen],
        }


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------


def _harmonic_mean(values: np.ndarray) -> float:
    """Harmonic mean over finite, strictly positive entries; ``nan`` if none."""
    finite = np.isfinite(values) & (values > 0)
    if not finite.any():
        return float("nan")
    return float(finite.sum() / np.sum(1.0 / values[finite]))


def _optional_float(x: float | None) -> float | None:
    """``None`` for missing or non-finite; else ``float(x)``.

    Used by every ``NeXxxResult.to_dict`` to coerce optional scalar Ne /
    diagnostic fields to YAML-safe JSON values.
    """
    if x is None or not np.isfinite(x):
        return None
    return float(x)


def _warn_if_uniform_sex(pg: PedigreeGraph, caller: str) -> None:
    """RuntimeWarning when ``pg.sex`` is uniform — usually a missing ``sex=``.

    Fired by :func:`ne_variance_family_size` and :func:`ne_sex_ratio`
    (both estimators are degenerate on single-sex pedigrees and the
    overwhelmingly common cause is the caller forgetting to pass ``sex=``
    to :meth:`PedigreeGraph.from_arrays`, which silently defaults the
    array to all-female).  Estimators still return ``ne=None`` (matching
    a legitimately single-sex pedigree), so the warning is the only
    diagnostic.
    """
    if pg.n == 0:
        return
    sex = np.asarray(pg.sex)
    if len(np.unique(sex)) < 2:
        warnings.warn(
            f"{caller}: pg.sex is uniform (all {int(sex[0])}); estimator is "
            "degenerate and will return ne=None. Did you forget to pass sex= "
            "to PedigreeGraph.from_arrays?",
            RuntimeWarning,
            stacklevel=3,
        )


def _sigma2_from_quadrants(
    entry: dict[str, np.ndarray],
) -> tuple[float, float, float, float, float, float, float, float, int, int] | None:
    """Caballero 1994 eq. 6 reassembly from sex-of-offspring quadrants.

    Computes the per-sex lifetime-offspring variance ``σ²_s = V(k_ss) +
    V(k_sf) + 2·Cov(k_ss, k_sf)`` and the per-sex offspring-count mean
    ``k̄_s`` from one row of :func:`_sex_specific_family_table` output.
    Returns ``None`` when the cohort lacks two of either sex (Ne is
    undefined for that transition).

    Returns:
        ``(v_mm, v_mf, v_fm, v_ff, cov_m, cov_f, kbar_m, kbar_f, n_m,
        n_f)`` — six variance/covariance terms, two means, two cohort
        sizes.  Callers reassemble ``σ²_m`` and ``σ²_f`` from the
        appropriate sum.
    """
    kmm = entry["k_mm"]
    kmf = entry["k_mf"]
    kfm = entry["k_fm"]
    kff = entry["k_ff"]
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
    return v_mm, v_mf, v_fm, v_ff, cov_m, cov_f, kbar_m, kbar_f, n_m, n_f


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


def ne_inbreeding(pg: PedigreeGraph) -> NeInbreedingResult:
    """Inbreeding-rate Ne (Ne_I).

    Computes per-cohort mean F (founders = 0).  Per-transition Ne_t =
    ``1 / (2·ΔF_t)`` with ``ΔF_t = (F̄_t − F̄_{t−1}) / (1 − F̄_{t−1})``.
    Aggregate Ne from the regression slope of ``ln(1 − F̄_t)`` on t for
    t ≥ 1 (founders excluded).
    """
    F = pg.compute_inbreeding()
    gen = np.asarray(pg.generation)
    g_max = int(gen.max())
    mean_f = np.zeros(g_max + 1, dtype=np.float64)
    for g in range(g_max + 1):
        mask = gen == g
        if mask.any():
            mean_f[g] = float(F[mask].mean())

    ne_per_gen = np.full(g_max + 1, np.nan, dtype=np.float64)
    for g in range(1, g_max + 1):
        f_prev = mean_f[g - 1]
        if f_prev >= 1.0:
            continue
        df = (mean_f[g] - f_prev) / (1.0 - f_prev)
        if df > 0:
            ne_per_gen[g] = 1.0 / (2.0 * df)

    t = np.arange(1, g_max + 1, dtype=np.float64)
    slope, _ = _regress_log_one_minus(mean_f[1:], t)
    if np.isfinite(slope) and slope < 0:
        ne_scalar: float | None = -1.0 / (2.0 * slope)
    else:
        ne_scalar = None

    return NeInbreedingResult(
        ne=ne_scalar,
        ne_per_gen=ne_per_gen,
        mean_f_per_gen=mean_f,
        slope=slope,
        n_generations_used=int(np.isfinite(np.log1p(-mean_f[1:])).sum()),
    )


def ne_coancestry(
    pg: PedigreeGraph,
    K: sp.csc_matrix | None = None,
    theta_per_gen: np.ndarray | None = None,
) -> NeCoancestryResult:
    """Coancestry-rate Ne (Ne_C).

    Same regression form as Ne_I but on per-cohort mean kinship θ over
    within-cohort unordered pairs (excluding the diagonal and MZ twin
    pairs).

    The estimator accepts θ̄_g pre-computed (streamed from the DP
    without materializing K) — preferred path at large N where K's CSC
    would OOM.  If neither θ̄_g nor K is supplied, the K-free streaming
    path is used by default.

    Args:
        pg: Pedigree graph.
        K: optional pre-built sparse kinship matrix.  Used only when
            ``theta_per_gen`` is None.
        theta_per_gen: optional pre-computed per-generation mean
            kinship.  When supplied, K is ignored.
    """
    gen = np.asarray(pg.generation)
    g_max = int(gen.max())
    if theta_per_gen is not None:
        mean_theta = np.asarray(theta_per_gen, dtype=np.float64)
    elif K is not None:
        twin = np.asarray(pg.twin)
        mean_theta = _per_gen_mean_kinship(K, gen, twin)
    else:
        mean_theta = pg.per_gen_mean_kinship()

    ne_per_gen = np.full(g_max + 1, np.nan, dtype=np.float64)
    for g in range(1, g_max + 1):
        theta_prev = mean_theta[g - 1]
        if not np.isfinite(theta_prev) or theta_prev >= 1.0:
            continue
        if not np.isfinite(mean_theta[g]):
            continue
        d_theta = (mean_theta[g] - theta_prev) / (1.0 - theta_prev)
        if d_theta > 0:
            ne_per_gen[g] = 1.0 / (2.0 * d_theta)

    t = np.arange(1, g_max + 1, dtype=np.float64)
    slope, _ = _regress_log_one_minus(mean_theta[1:], t)
    if np.isfinite(slope) and slope < 0:
        ne_scalar: float | None = -1.0 / (2.0 * slope)
    else:
        ne_scalar = None

    return NeCoancestryResult(
        ne=ne_scalar,
        ne_per_gen=ne_per_gen,
        mean_theta_per_gen=mean_theta,
        slope=slope,
        n_generations_used=int(np.isfinite(np.log1p(-mean_theta[1:])).sum()),
    )


def ne_variance_family_size(pg: PedigreeGraph) -> NeVarianceResult:
    """Variance-of-family-size Ne (Ne_V) — Caballero 1994 eq. 6.

    For each parent generation p, decompose lifetime offspring counts per
    parent by offspring sex (``k_mm, k_mf, k_fm, k_ff``).  Skip-gen
    children (offspring at gen > p+1) are attributed to the parent's own
    cohort, so a parent's k-totals reflect their full reproductive
    output even when offspring span multiple generations.

    ``V(k_m) = V(k_mm) + V(k_mf) + 2·Cov(k_mm, k_mf)`` is the per-male
    total-offspring variance built from this decomposition.  Discrete-
    generation Ne for the transition is

        ``ΔF = (V(k_m)/k̄_m) / (4 · N_m · k̄_m) +
                  (V(k_f)/k̄_f) / (4 · N_f · k̄_f)``,

    with ``Ne_p = 1/(2·ΔF)``.  When ``V(k)/k̄ → 1`` (Poisson) and
    ``N_m = N_f``, this reduces to Wright's ``4 N_m N_f / (N_m + N_f)``.
    Aggregate Ne is the harmonic mean across parent generations.

    Emits a ``RuntimeWarning`` when ``pg.sex`` is uniformly 0 or 1 —
    almost always a sign that the caller forgot to pass ``sex=`` to
    :meth:`PedigreeGraph.from_arrays` and is unwittingly running on the
    all-female default.  The estimator still returns ``ne=None``
    (consistent with a legitimate single-sex pedigree), so the warning
    is the only diagnostic.
    """
    _warn_if_uniform_sex(pg, "ne_variance_family_size")

    table = _sex_specific_family_table(
        np.asarray(pg.mother),
        np.asarray(pg.father),
        np.asarray(pg.sex),
        np.asarray(pg.generation),
    )
    g_max = int(np.asarray(pg.generation).max())
    # Indexed by parent generation p ∈ [0, g_max).  Slot p = g_max is
    # absent because g_max individuals have no offspring in the pedigree.
    ne_per_t = np.full(g_max, np.nan, dtype=np.float64)
    v_mm = np.full(g_max, np.nan, dtype=np.float64)
    v_mf = np.full(g_max, np.nan, dtype=np.float64)
    v_fm = np.full(g_max, np.nan, dtype=np.float64)
    v_ff = np.full(g_max, np.nan, dtype=np.float64)
    cov_m = np.full(g_max, np.nan, dtype=np.float64)
    cov_f = np.full(g_max, np.nan, dtype=np.float64)

    for p, entry in table.items():
        decomp = _sigma2_from_quadrants(entry)
        if decomp is None:
            continue
        v_mm_p, v_mf_p, v_fm_p, v_ff_p, cov_m_p, cov_f_p, kbar_m, kbar_f, n_m, n_f = decomp
        if kbar_m <= 0 or kbar_f <= 0:
            continue
        v_mm[p] = v_mm_p
        v_mf[p] = v_mf_p
        v_fm[p] = v_fm_p
        v_ff[p] = v_ff_p
        cov_m[p] = cov_m_p
        cov_f[p] = cov_f_p
        var_km_total = v_mm_p + v_mf_p + 2.0 * cov_m_p
        var_kf_total = v_fm_p + v_ff_p + 2.0 * cov_f_p
        df = (var_km_total / kbar_m) / (4.0 * n_m * kbar_m) + (var_kf_total / kbar_f) / (4.0 * n_f * kbar_f)
        if df > 0:
            ne_per_t[p] = 1.0 / (2.0 * df)

    return NeVarianceResult(
        ne=_harmonic_mean(ne_per_t) if np.isfinite(ne_per_t).any() else None,
        ne_per_transition=ne_per_t,
        v_mm=v_mm,
        v_mf=v_mf,
        v_fm=v_fm,
        v_ff=v_ff,
        cov_m=cov_m,
        cov_f=cov_f,
    )


def ne_sex_ratio(pg: PedigreeGraph) -> NeSexRatioResult:
    """Wright sex-ratio Ne (Ne_sr).

    ``Ne_t = 4·Nm_t·Nf_t / (Nm_t + Nf_t)`` per generation; aggregate is
    the harmonic mean across cohorts with both sexes present.

    Emits a ``RuntimeWarning`` when ``pg.sex`` is uniformly 0 or 1 —
    almost always a sign the caller forgot to pass ``sex=`` to
    :meth:`PedigreeGraph.from_arrays` and is unwittingly running on the
    all-female default.  The estimator still returns ``ne=None`` in
    that case (consistent with a legitimate single-sex pedigree), so
    the warning is the only diagnostic.
    """
    _warn_if_uniform_sex(pg, "ne_sex_ratio")

    gen = np.asarray(pg.generation)
    sex = np.asarray(pg.sex)
    g_max = int(gen.max())
    n_male = np.zeros(g_max + 1, dtype=np.int64)
    n_female = np.zeros(g_max + 1, dtype=np.int64)
    for g in range(g_max + 1):
        in_g = gen == g
        n_male[g] = int(((sex == 1) & in_g).sum())
        n_female[g] = int(((sex == 0) & in_g).sum())

    ne_per_gen = np.full(g_max + 1, np.nan, dtype=np.float64)
    for g in range(g_max + 1):
        if n_male[g] > 0 and n_female[g] > 0:
            ne_per_gen[g] = 4.0 * n_male[g] * n_female[g] / (n_male[g] + n_female[g])

    return NeSexRatioResult(
        ne=_harmonic_mean(ne_per_gen) if np.isfinite(ne_per_gen).any() else None,
        ne_per_gen=ne_per_gen,
        n_male_per_gen=n_male,
        n_female_per_gen=n_female,
    )


# ---------------------------------------------------------------------------
# Internal helpers (shared by step 2 estimators)
# ---------------------------------------------------------------------------


def _founder_idx(pg: PedigreeGraph) -> np.ndarray:
    """Indices of founders (gen-0 individuals)."""
    return np.where(np.asarray(pg.generation) == 0)[0].astype(np.intp)


def _per_gen_founder_means(
    pg: PedigreeGraph,
    founder_idx: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-generation mean founder contribution via adjoint propagation.

    Returns ``(m_g, founder_idx)`` where
    ``m_g[g, f_local] = mean_{i ∈ gen g} c[i, founder_idx[f_local]]``,
    with ``c[i, f]`` the expected genome fraction of i inherited from
    founder f under the Mendelian recursion (founders contribute 1 to
    themselves; non-founders take the mean of their two parents' rows).

    Computed by iterating the adjoint of the forward recursion.  For
    each target generation g, propagate the cohort uniform vector
    ``1_{gen g} / N_g`` backward through child→parent edges.  At
    iteration ``t``, scatter ``0.5 · u[child]`` from each ``child ∈
    gen t+1`` into its mother and father (which may live in any earlier
    generation under the ``gen[i] = max(gen_parents) + 1`` definition).
    By the time iteration ``t`` reads ``u[gen == t+1]``, every later
    generation has already contributed into it — so skip-gen parents are
    handled correctly without bookkeeping.

    Time: O(N · g_max²).  Memory: O(N + n_founders · g_max).

    Args:
        pg: Pedigree graph.
        founder_idx: Optional precomputed founder index array.

    Returns:
        ``(m_g, founder_idx)`` — ``m_g`` shape ``(g_max + 1, n_founders)``
        float64; ``founder_idx`` shape ``(n_founders,)`` intp.
    """
    if founder_idx is None:
        founder_idx = _founder_idx(pg)
    n_founders = len(founder_idx)
    n = pg.n
    gen = np.asarray(pg.generation)
    mother = np.asarray(pg.mother)
    father = np.asarray(pg.father)
    g_max = int(gen.max()) if n > 0 else 0

    m_g = np.full((g_max + 1, n_founders), np.nan, dtype=np.float64)
    if n_founders == 0:
        return m_g, founder_idx

    # Precompute per-generation member indices once — the inner sweep
    # reads the same cohorts O(g_max) times across the outer loop.
    cohorts = [np.flatnonzero(gen == g) for g in range(g_max + 1)]

    # Gen 0 mirrors the forward convention `c[gen==0].mean(axis=0)` —
    # founders sit on the identity diagonal, so column means are 1/N_0.
    n0 = len(cohorts[0])
    if n0 > 0:
        m_g[0] = 1.0 / n0

    for g in range(1, g_max + 1):
        in_g = cohorts[g]
        if len(in_g) == 0:
            continue
        u = np.zeros(n, dtype=np.float64)
        u[in_g] = 1.0 / len(in_g)
        for t in range(g - 1, -1, -1):
            child = cohorts[t + 1]
            if len(child) == 0:
                continue
            uc = 0.5 * u[child]
            m = mother[child]
            mask = m >= 0
            if mask.any():
                np.add.at(u, m[mask], uc[mask])  # perf: numba candidate
            f = father[child]
            mask = f >= 0
            if mask.any():
                np.add.at(u, f[mask], uc[mask])  # perf: numba candidate
            u[child] = 0.0
        m_g[g] = u[founder_idx]

    return m_g, founder_idx


def _caballero_toro_accumulators(
    pg: PedigreeGraph,
    founder_idx: np.ndarray,
    F: np.ndarray,
) -> dict[str, Any]:
    """Streaming forward sweep producing per-(g, f) self-coancestry sums.

    For each generation g and founder f, accumulates the count of
    descendants of f in gen g and the sum of their self-coancestry
    ``(1 + F_i) / 2``.  Avoids materializing the dense
    ``(n × n_founders)`` contribution matrix by maintaining per-individual
    ancestor sets only over the working frontier; sets are retired via a
    per-individual remaining-child counter once the last child has been
    processed.

    "Descendant of f" is graph reachability — equivalent to ``c[i, f] >
    0`` because the forward recursion only adds non-negatives, so a
    non-zero ⇔ at least one ancestor path exists.

    Args:
        pg: Pedigree graph.
        founder_idx: Founder indices (output of :func:`_founder_idx`).
        F: Per-individual inbreeding coefficients (length ``pg.n``).

    Returns:
        Dict with:
        * ``sums``: shape ``(g_max + 1, n_founders)``, float64.
        * ``counts``: shape ``(g_max + 1, n_founders)``, int64.
        * ``peak_ancestor_set_size``: max size of any single ancestor set.
        * ``peak_live_ancestor_sets``: max simultaneously live sets.
        * ``total_ancestor_pair_visits``: Σ over i of len(ancestor_set[i]).
        * ``founder_idx``: copy of input.
    """
    n = pg.n
    n_founders = len(founder_idx)
    gen = np.asarray(pg.generation)
    mother = np.asarray(pg.mother)
    father = np.asarray(pg.father)
    g_max = int(gen.max()) if n > 0 else 0

    sums = np.zeros((g_max + 1, n_founders), dtype=np.float64)
    counts = np.zeros((g_max + 1, n_founders), dtype=np.int64)
    if n_founders == 0:
        return {
            "sums": sums,
            "counts": counts,
            "peak_ancestor_set_size": 0,
            "peak_live_ancestor_sets": 0,
            "total_ancestor_pair_visits": 0,
            "founder_idx": founder_idx,
        }

    self_coancestry = (1.0 + F) / 2.0

    # Per-individual remaining-child counter for ancestor-set retirement.
    n_children = np.zeros(n, dtype=np.int64)
    for parents in (mother, father):
        valid = parents >= 0
        if valid.any():
            np.add.at(n_children, parents[valid], 1)
    n_remaining = n_children.copy()

    # Reverse map: individual index → founder local index (or −1).
    founder_local_of = np.full(n, -1, dtype=np.int64)
    founder_local_of[founder_idx] = np.arange(n_founders, dtype=np.int64)

    ancestor_sets: dict[int, np.ndarray] = {}
    peak_set_size = 0
    peak_live = 0
    total_pair_visits = 0

    cohorts = [np.flatnonzero(gen == g) for g in range(g_max + 1)]

    # perf: numba candidate — Python loop over n individuals with dict ops.
    for g in range(g_max + 1):
        for i_np in cohorts[g]:
            i = int(i_np)
            f_local = int(founder_local_of[i])
            if f_local >= 0:
                anc = np.array([f_local], dtype=np.int32)
            else:
                m = int(mother[i])
                f = int(father[i])
                anc_m = ancestor_sets.get(m) if m >= 0 else None
                anc_f = ancestor_sets.get(f) if f >= 0 else None
                if anc_m is not None and anc_f is not None:
                    anc = np.union1d(anc_m, anc_f).astype(np.int32, copy=False)
                elif anc_m is not None:
                    anc = anc_m
                elif anc_f is not None:
                    anc = anc_f
                else:
                    anc = np.empty(0, dtype=np.int32)

            if anc.size > 0:
                sums[g, anc] += self_coancestry[i]
                counts[g, anc] += 1
                total_pair_visits += int(anc.size)
                if anc.size > peak_set_size:
                    peak_set_size = int(anc.size)

            # Retire parents whose last child has now been processed.
            for p_raw in (mother[i], father[i]):
                p = int(p_raw)
                if p < 0:
                    continue
                n_remaining[p] -= 1
                if n_remaining[p] == 0 and p in ancestor_sets:
                    del ancestor_sets[p]

            # Keep i's ancestor set only if it still has children to feed.
            if n_children[i] > 0 and anc.size > 0:
                ancestor_sets[i] = anc

            if len(ancestor_sets) > peak_live:
                peak_live = len(ancestor_sets)

    return {
        "sums": sums,
        "counts": counts,
        "peak_ancestor_set_size": int(peak_set_size),
        "peak_live_ancestor_sets": int(peak_live),
        "total_ancestor_pair_visits": int(total_pair_visits),
        "founder_idx": founder_idx,
    }


# ---------------------------------------------------------------------------
# Result dataclasses (step 2)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class NeIndividualDeltaFResult:
    """Gutiérrez 2008/2009 individual ΔF (Ne_iΔF) result.

    Per individual i with EqG_i > 1 and F_i < 1:
    ``ΔF_i = 1 − (1 − F_i)^(1/(EqG_i − 1))``.  Per-cohort Ne_g =
    ``1/(2 · mean_g ΔF_i)``; aggregate Ne is the harmonic mean across
    cohorts.
    """

    ne: float | None
    ne_per_gen: np.ndarray
    mean_eqg_per_gen: np.ndarray
    n_used_per_gen: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        """Serialize as a YAML-ready dict."""
        return {
            "ne": _optional_float(self.ne),
            "ne_per_gen": [_optional_float(v) for v in self.ne_per_gen],
            "mean_eqg_per_gen": [_optional_float(v) for v in self.mean_eqg_per_gen],
            "n_used_per_gen": [int(v) for v in self.n_used_per_gen],
        }


@dataclass(frozen=True, slots=True)
class NeLTCResult:
    """Wray & Thompson 1990 long-term contribution (Ne_LTC) result.

    Founder contributions are propagated forward through the pedigree
    until the per-generation mean contribution stabilizes
    (``max |Δc| < 1e-6``) or the last available generation is reached.

    ``Ne = 1 / (2 · Σ_f c_f²)`` over founders at the final iteration.
    When the asymptote is not reached, ``ne`` is ``None`` and
    ``asymptote_reached`` is ``False``.
    """

    ne: float | None
    asymptote_reached: bool
    n_iterations: int
    max_delta_final: float
    sum_c_squared: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize as a YAML-ready dict."""
        return {
            "ne": _optional_float(self.ne),
            "asymptote_reached": bool(self.asymptote_reached),
            "n_iterations": int(self.n_iterations),
            "max_delta_final": _optional_float(self.max_delta_final),
            "sum_c_squared": float(self.sum_c_squared),
        }


@dataclass(frozen=True, slots=True)
class NeHillResult:
    """Hill 1979 separate-sex overlapping-generation Ne (Ne_H).

    Two operating modes:

    * **Sentinel branch** (``collapses_to_ne_v=True``): used when
      ``pg.birth_year is None``.  Hill 1979 with ``L = 1`` reduces
      algebraically to Ne_V (Caballero 1994 eq. 6 / Hill 1979 eq. 8 with
      equal sexes), so ``ne`` is the Ne_V passthrough.  New diagnostic
      fields are all ``None`` / ``0`` defaults.
    * **Birth-year branch** (``collapses_to_ne_v=False``): used when
      ``pg.birth_year`` is set.  Ne is computed per eligible birth-year
      cohort via Hill 1979 eq. (10)::

          Ne(c) = 8·N1(c)·T / (σ²_m(c) + σ²_f(c) + 4)

      where ``σ²_m`` and ``σ²_f`` are the Caballero 1994 eq. 6
      sex-of-offspring variance reassemblies, ``N1(c) = N_m(c) + N_f(c)``
      is the total cohort size, and ``T = (T_m + T_f) / 2`` is the
      sex-averaged generation interval.  Scenario-scalar ``ne`` is the
      harmonic mean over eligible cohorts.

    Diagnostic fields ``T_m``, ``T_f``, ``N1_m``, ``N1_f``, ``Vk_m``,
    ``Vk_f`` are scenario-level means over eligible cohorts and do not
    re-enter the Ne computation.
    """

    ne: float | None
    generation_interval: float
    collapses_to_ne_v: bool
    # Sex-split generation interval (Hill 1979 L)
    T_m: float | None = None
    T_f: float | None = None
    # Scenario-level diagnostic means over eligible cohorts
    N1_m: float | None = None
    N1_f: float | None = None
    Vk_m: float | None = None
    Vk_f: float | None = None
    # Mean lifetime offspring per individual (zeros included), per sex
    kbar_m: float | None = None
    kbar_f: float | None = None
    # Sex-decomposed Ne (Wright 1938 / paper eq. 3 combination):
    #   1/Ne = 1/(4·Ne_m) + 1/(4·Ne_f)
    # Per-cohort Ne_s = 4·N1_s·T/(Vk_s + 2), then harmonic-mean across cohorts.
    Ne_m: float | None = None
    Ne_f: float | None = None
    # Whether Vk_m/Vk_f were rescaled to constant-N reference via Waples 2002 eq. 5
    vk_scaled: bool = False
    # Cohort eligibility
    cohort_window: CohortWindow | None = None
    n_eligible_cohorts: int = 0
    n_excluded_right_censored: int = 0
    n_excluded_left_censored: int = 0
    n_unknown_birth_year: int = 0
    # Per-cohort series (one entry per eligible cohort year) — supports
    # rolling-window analyses and reproduction of paper-style "early N
    # cohorts" vs "recent N cohorts" comparisons.  All arrays share the
    # same index as ``cohort_years``.  None on the sentinel branch.
    cohort_years: np.ndarray | None = None
    ne_per_cohort: np.ndarray | None = None
    Ne_m_per_cohort: np.ndarray | None = None
    Ne_f_per_cohort: np.ndarray | None = None
    Vk_m_per_cohort: np.ndarray | None = None
    Vk_f_per_cohort: np.ndarray | None = None
    N1_m_per_cohort: np.ndarray | None = None
    N1_f_per_cohort: np.ndarray | None = None
    # Per-individual age table — descriptive only, not used in Ne
    age_table: dict[str, np.ndarray] | None = None
    n_offspring_pairs: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize as a YAML-ready dict."""
        out: dict[str, Any] = {
            "ne": _optional_float(self.ne),
            "generation_interval": float(self.generation_interval),
            "collapses_to_ne_v": bool(self.collapses_to_ne_v),
            "T_m": _optional_float(self.T_m),
            "T_f": _optional_float(self.T_f),
            "N1_m": _optional_float(self.N1_m),
            "N1_f": _optional_float(self.N1_f),
            "Vk_m": _optional_float(self.Vk_m),
            "Vk_f": _optional_float(self.Vk_f),
            "kbar_m": _optional_float(self.kbar_m),
            "kbar_f": _optional_float(self.kbar_f),
            "Ne_m": _optional_float(self.Ne_m),
            "Ne_f": _optional_float(self.Ne_f),
            "vk_scaled": bool(self.vk_scaled),
            "cohort_window": None if self.cohort_window is None else self.cohort_window._asdict(),
            "n_eligible_cohorts": int(self.n_eligible_cohorts),
            "n_excluded_right_censored": int(self.n_excluded_right_censored),
            "n_excluded_left_censored": int(self.n_excluded_left_censored),
            "n_unknown_birth_year": int(self.n_unknown_birth_year),
            "n_offspring_pairs": int(self.n_offspring_pairs),
        }
        for name in (
            "cohort_years",
            "ne_per_cohort",
            "Ne_m_per_cohort",
            "Ne_f_per_cohort",
            "Vk_m_per_cohort",
            "Vk_f_per_cohort",
            "N1_m_per_cohort",
            "N1_f_per_cohort",
        ):
            arr = getattr(self, name)
            out[name] = None if arr is None else [float(v) for v in arr]
        if self.age_table is None:
            out["age_table"] = None
        else:
            out["age_table"] = {k: v.tolist() for k, v in self.age_table.items()}
        return out


@dataclass(frozen=True, slots=True)
class NeCaballeroToroResult:
    """Caballero & Toro 2002 self-coancestry rate (Ne_CT) result.

    For each founder f and generation g > 0, computes the mean self-
    coancestry of f's descendants at gen g:
    ``f̄_s,f,g = mean_{i ∈ desc(f,g)} (1 + F_i) / 2``.
    Averages over founders that have descendants at each gen, regresses
    ``ln(1 − f̄_s,g)`` on g, and reports
    ``ne = −1 / (2·slope)``.
    """

    ne: float | None
    ne_per_gen: np.ndarray
    mean_self_coancestry_per_gen: np.ndarray
    n_founders_with_descendants_per_gen: np.ndarray
    slope: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize as a YAML-ready dict."""
        return {
            "ne": _optional_float(self.ne),
            "ne_per_gen": [_optional_float(v) for v in self.ne_per_gen],
            "mean_self_coancestry_per_gen": [
                _optional_float(v) for v in self.mean_self_coancestry_per_gen
            ],
            "n_founders_with_descendants_per_gen": [int(v) for v in self.n_founders_with_descendants_per_gen],
            "slope": _optional_float(self.slope),
        }


# ---------------------------------------------------------------------------
# Estimators (step 2)
# ---------------------------------------------------------------------------


def ne_individual_delta_f(pg: PedigreeGraph) -> NeIndividualDeltaFResult:
    """Gutiérrez 2008 individual ΔF Ne (Ne_iΔF).

    For each individual ``i`` with ``EqG_i > 1`` and ``F_i < 1``:

        ``ΔF_i = 1 − (1 − F_i)^(1/(EqG_i − 1))``.

    Per-cohort ``Ne_g = 1/(2 · mean_{i ∈ gen g} ΔF_i)``; aggregate is
    the harmonic mean across cohorts.
    """
    F = pg.compute_inbreeding()
    eqg = _compute_eqg(np.asarray(pg.mother), np.asarray(pg.father), pg.n)
    gen = np.asarray(pg.generation)
    g_max = int(gen.max())

    valid = (eqg > 1.0) & (F < 1.0)
    delta_f = np.full(pg.n, np.nan, dtype=np.float64)
    if valid.any():
        delta_f[valid] = 1.0 - np.power(1.0 - F[valid], 1.0 / (eqg[valid] - 1.0))

    ne_per_gen = np.full(g_max + 1, np.nan, dtype=np.float64)
    mean_eqg_per_gen = np.full(g_max + 1, np.nan, dtype=np.float64)
    n_used_per_gen = np.zeros(g_max + 1, dtype=np.int64)
    for g in range(g_max + 1):
        in_g = (gen == g) & valid
        n_used_per_gen[g] = int(in_g.sum())
        if n_used_per_gen[g] == 0:
            continue
        mean_df = float(delta_f[in_g].mean())
        mean_eqg_per_gen[g] = float(eqg[in_g].mean())
        if mean_df > 0:
            ne_per_gen[g] = 1.0 / (2.0 * mean_df)

    return NeIndividualDeltaFResult(
        ne=_harmonic_mean(ne_per_gen) if np.isfinite(ne_per_gen).any() else None,
        ne_per_gen=ne_per_gen,
        mean_eqg_per_gen=mean_eqg_per_gen,
        n_used_per_gen=n_used_per_gen,
    )


def ne_long_term_contributions(
    pg: PedigreeGraph,
    mean_contributions: tuple[np.ndarray, np.ndarray] | None = None,
    tol: float = 1e-6,
) -> NeLTCResult:
    """Wray & Thompson 1990 long-term contribution Ne (Ne_LTC).

    Per-generation mean founder contribution ``c_g[f] =
    mean_{i ∈ gen g} c[i, f]``.  Iterate g = 1, 2, …; stop at the first
    g where ``max_f |c_g[f] − c_{g-1}[f]| < tol``, or after the last
    available generation.  Ne is computed at the stopping g as
    ``1 / (2 · Σ_f c_g[f]²)``.

    When the asymptote is not reached before the last generation, ``ne``
    is ``None`` and ``asymptote_reached`` is ``False``.
    """
    if mean_contributions is None:
        m_g, founder_idx = _per_gen_founder_means(pg)
    else:
        m_g, founder_idx = mean_contributions
    n_founders = len(founder_idx)
    if n_founders == 0:
        return NeLTCResult(
            ne=None,
            asymptote_reached=False,
            n_iterations=0,
            max_delta_final=float("nan"),
            sum_c_squared=0.0,
        )

    gen = np.asarray(pg.generation)
    g_max = int(gen.max())
    if g_max == 0:
        return NeLTCResult(
            ne=None,
            asymptote_reached=False,
            n_iterations=0,
            max_delta_final=float("nan"),
            sum_c_squared=float((m_g[0] ** 2).sum()),
        )

    asymptote_reached = False
    n_iterations = 0
    max_delta_final = float("nan")
    for g in range(1, g_max + 1):
        if not np.isfinite(m_g[g]).all() or not np.isfinite(m_g[g - 1]).all():
            continue
        delta = float(np.max(np.abs(m_g[g] - m_g[g - 1])))
        n_iterations = g
        max_delta_final = delta
        if delta < tol:
            asymptote_reached = True
            break

    final_c = m_g[n_iterations]
    sum_c_sq = float((final_c**2).sum())
    if asymptote_reached and sum_c_sq > 0:
        ne: float | None = 1.0 / (2.0 * sum_c_sq)
    else:
        ne = None

    return NeLTCResult(
        ne=ne,
        asymptote_reached=asymptote_reached,
        n_iterations=n_iterations,
        max_delta_final=max_delta_final,
        sum_c_squared=sum_c_sq,
    )


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


def ne_caballero_toro(
    pg: PedigreeGraph,
    ct_accumulators: dict[str, Any] | None = None,
) -> NeCaballeroToroResult:
    """Caballero & Toro 2002 self-coancestry rate Ne (Ne_CT).

    For each founder f and generation g > 0, descendants are detected
    via graph reachability — equivalently, ``c[i, f] > 0`` under the
    Mendelian recursion.  Self-coancestry per descendant is
    ``(1 + F_i) / 2``; averaged within each founder's descendant set,
    then averaged across founders that have descendants at gen g.  Ne
    from the regression slope of ``ln(1 − f̄_s,g)`` on g.
    """
    if ct_accumulators is None:
        founder_idx = _founder_idx(pg)
        F = pg.compute_inbreeding()
        ct_accumulators = _caballero_toro_accumulators(pg, founder_idx, F)

    sums = ct_accumulators["sums"]
    counts = ct_accumulators["counts"]
    g_max = sums.shape[0] - 1

    valid = counts > 0
    # per_founder_mean[g, f] = mean self-coancestry of f's descendants in gen g
    per_founder_mean = np.where(valid, sums / np.maximum(counts, 1), 0.0)
    n_with_desc_per_gen = valid.sum(axis=1).astype(np.int64)
    mean_fs_per_gen = np.full(g_max + 1, np.nan, dtype=np.float64)
    nz = n_with_desc_per_gen > 0
    if nz.any():
        mean_fs_per_gen[nz] = per_founder_mean.sum(axis=1)[nz] / n_with_desc_per_gen[nz]
    # Gen 0 has no "descendants" in the CT regression sense; force NaN/0
    # to match the historical contract (regression starts at g=1).
    mean_fs_per_gen[0] = np.nan
    n_with_desc_per_gen[0] = 0

    ne_per_gen = np.full(g_max + 1, np.nan, dtype=np.float64)
    for g in range(1, g_max + 1):
        # For g=1 we anchor `prev = 0.5` (the natural self-coancestry floor for
        # non-inbred individuals, since fs = (1+F)/2 and founder F = 0).  An
        # earlier version anchored at 0.0, which produced a meaningless
        # ``Ne = 1/(2·0.5) = 1`` artifact at g=1; the actual drift signal is the
        # deviation of fs above 0.5, which only starts accumulating from g=2
        # onward.  At g=1 ``d == 0`` and the ``d > 0`` guard now correctly
        # leaves the entry as NaN.
        prev = mean_fs_per_gen[g - 1] if g >= 2 else 0.5
        if not np.isfinite(prev) or prev >= 1.0 or not np.isfinite(mean_fs_per_gen[g]):
            continue
        d = (mean_fs_per_gen[g] - prev) / (1.0 - prev)
        if d > 0:
            ne_per_gen[g] = 1.0 / (2.0 * d)

    t = np.arange(1, g_max + 1, dtype=np.float64)
    slope, _ = _regress_log_one_minus(mean_fs_per_gen[1:], t)
    if np.isfinite(slope) and slope < 0:
        ne_scalar: float | None = -1.0 / (2.0 * slope)
    else:
        ne_scalar = None

    return NeCaballeroToroResult(
        ne=ne_scalar,
        ne_per_gen=ne_per_gen,
        mean_self_coancestry_per_gen=mean_fs_per_gen,
        n_founders_with_descendants_per_gen=n_with_desc_per_gen,
        slope=slope,
    )


# ---------------------------------------------------------------------------
# Convenience entry: build K and contribution matrix once, dispatch all 8
# ---------------------------------------------------------------------------


def compute_all_ne(
    pg: PedigreeGraph,
    skip_ne_coancestry: bool = False,
    n_threads: int = 1,
    hill_vk_scale: bool = False,
) -> dict[str, Any]:
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

    results: dict[str, Any] = {}
    max_workers = min(n_threads, len(tasks))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            name: executor.submit(func, *args, **kwargs)
            for name, (func, args, kwargs) in tasks.items()
        }
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
