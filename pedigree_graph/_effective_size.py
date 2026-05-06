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

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from pedigree_graph._kinship_kernel import _compute_eqg

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
    out = np.full(g_max + 1, np.nan, dtype=np.float64)

    # COO traversal — restrict to upper triangle, same generation,
    # non-twin pairs.
    coo = K.tocoo()
    rows, cols, vals = coo.row, coo.col, coo.data
    pair_mask = (rows < cols) & (generation[rows] == generation[cols])
    pair_mask &= ~((twin_idx[rows] >= 0) & (twin_idx[rows] == cols))
    rows = rows[pair_mask]
    cols = cols[pair_mask]
    vals = vals[pair_mask]
    pair_gens = generation[rows]

    for g in range(g_max + 1):
        in_g = generation == g
        n_g = int(in_g.sum())
        if n_g < 2:
            continue
        # Twin pairs (count each pair once: partner with smaller index).
        twin_in_g = int(((twin_idx >= 0) & in_g & (twin_idx > np.arange(len(generation)))).sum())
        total_pairs = n_g * (n_g - 1) // 2 - twin_in_g
        if total_pairs <= 0:
            continue
        sum_theta = float(vals[pair_gens == g].sum())
        out[g] = sum_theta / total_pairs
    return out


def _sex_specific_family_table(
    mother: np.ndarray,
    father: np.ndarray,
    sex: np.ndarray,
    generation: np.ndarray,
) -> dict[int, dict[str, np.ndarray]]:
    """Per-transition counts of male/female offspring per parent.

    For each transition ``g − 1 → g`` (g ≥ 1), partition offspring by
    sex and tally offspring counts for each parent in cohort g − 1.

    Returns:
        Dict keyed on ``g`` (transition target generation).  Each entry
        is a dict with arrays:

        * ``males_in_parent_gen`` — row indices of males in g − 1
        * ``females_in_parent_gen`` — row indices of females in g − 1
        * ``k_mm`` — per-male count of male offspring at gen g
        * ``k_mf`` — per-male count of female offspring at gen g
        * ``k_fm`` — per-female count of male offspring at gen g
        * ``k_ff`` — per-female count of female offspring at gen g
    """
    g_max = int(generation.max())
    n = len(generation)
    sex = np.asarray(sex, dtype=np.int8)
    out: dict[int, dict[str, np.ndarray]] = {}
    for g in range(1, g_max + 1):
        parent_mask = generation == g - 1
        offspring_mask = generation == g
        males_in_parent = np.where(parent_mask & (sex == 1))[0]
        females_in_parent = np.where(parent_mask & (sex == 0))[0]
        # Map parent row → local index in males/females arrays.
        parent_to_male_local = np.full(n, -1, dtype=np.int32)
        parent_to_female_local = np.full(n, -1, dtype=np.int32)
        parent_to_male_local[males_in_parent] = np.arange(len(males_in_parent), dtype=np.int32)
        parent_to_female_local[females_in_parent] = np.arange(len(females_in_parent), dtype=np.int32)

        k_mm = np.zeros(len(males_in_parent), dtype=np.int64)
        k_mf = np.zeros(len(males_in_parent), dtype=np.int64)
        k_fm = np.zeros(len(females_in_parent), dtype=np.int64)
        k_ff = np.zeros(len(females_in_parent), dtype=np.int64)

        offs_idx = np.where(offspring_mask)[0]
        for i in offs_idx:
            o_sex = sex[i]
            f = father[i]
            m = mother[i]
            if f >= 0:
                lf = parent_to_male_local[f]
                if lf >= 0:
                    if o_sex == 1:
                        k_mm[lf] += 1
                    else:
                        k_mf[lf] += 1
            if m >= 0:
                lm = parent_to_female_local[m]
                if lm >= 0:
                    if o_sex == 1:
                        k_fm[lm] += 1
                    else:
                        k_ff[lm] += 1

        out[g] = {
            "males_in_parent_gen": males_in_parent,
            "females_in_parent_gen": females_in_parent,
            "k_mm": k_mm,
            "k_mf": k_mf,
            "k_fm": k_fm,
            "k_ff": k_ff,
        }
    return out


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
            "ne": None if self.ne is None or not np.isfinite(self.ne) else float(self.ne),
            "ne_per_gen": [None if not np.isfinite(v) else float(v) for v in self.ne_per_gen],
            "mean_f_per_gen": [float(v) for v in self.mean_f_per_gen],
            "slope": None if not np.isfinite(self.slope) else float(self.slope),
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

    def to_dict(self) -> dict[str, Any]:
        """Serialize as a YAML-ready dict."""
        return {
            "ne": None if self.ne is None or not np.isfinite(self.ne) else float(self.ne),
            "ne_per_gen": [None if not np.isfinite(v) else float(v) for v in self.ne_per_gen],
            "mean_theta_per_gen": [None if not np.isfinite(v) else float(v) for v in self.mean_theta_per_gen],
            "slope": None if not np.isfinite(self.slope) else float(self.slope),
            "n_generations_used": int(self.n_generations_used),
        }


@dataclass(frozen=True, slots=True)
class NeVarianceResult:
    """Variance-of-family-size (Ne_V) result.

    Caballero 1994 eq. 6 with separate sexes.  ``V(k_m) = V(k_mm) +
    V(k_mf) + 2·Cov(k_mm, k_mf)`` is the per-male total-offspring
    variance built from the sex-of-offspring decomposition; symmetrically
    for females.

    Aggregate Ne is the harmonic mean across transitions.
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
            "ne": None if self.ne is None or not np.isfinite(self.ne) else float(self.ne),
            "ne_per_transition": [None if not np.isfinite(v) else float(v) for v in self.ne_per_transition],
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
            "ne": None if self.ne is None or not np.isfinite(self.ne) else float(self.ne),
            "ne_per_gen": [None if not np.isfinite(v) else float(v) for v in self.ne_per_gen],
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


def ne_coancestry(pg: PedigreeGraph, K: sp.csc_matrix | None = None) -> NeCoancestryResult:
    """Coancestry-rate Ne (Ne_C).

    Same regression form as Ne_I but on per-cohort mean kinship θ over
    within-cohort unordered pairs (excluding the diagonal and MZ twin
    pairs).
    """
    if K is None:
        K = pg.kinship_matrix()
    gen = np.asarray(pg.generation)
    twin = np.asarray(pg.twin)
    g_max = int(gen.max())
    mean_theta = _per_gen_mean_kinship(K, gen, twin)

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

    For each transition g−1 → g, decompose offspring counts per parent
    by offspring sex (``k_mm, k_mf, k_fm, k_ff``).  ``V(k_m) = V(k_mm)
    + V(k_mf) + 2·Cov(k_mm, k_mf)`` is the per-male total-offspring
    variance built from this decomposition.  Discrete-generation Ne for
    the transition is

        ``ΔF = (V(k_m)/k̄_m) / (4 · N_m · k̄_m) +
                  (V(k_f)/k̄_f) / (4 · N_f · k̄_f)``,

    with ``Ne_t = 1/(2·ΔF)``.  When ``V(k)/k̄ → 1`` (Poisson) and
    ``N_m = N_f``, this reduces to Wright's ``4 N_m N_f / (N_m + N_f)``.
    Aggregate Ne is the harmonic mean across transitions.
    """
    table = _sex_specific_family_table(
        np.asarray(pg.mother),
        np.asarray(pg.father),
        np.asarray(pg.sex),
        np.asarray(pg.generation),
    )
    g_max = int(np.asarray(pg.generation).max())
    n_trans = g_max  # transitions g=1..g_max
    ne_per_t = np.full(n_trans + 1, np.nan, dtype=np.float64)
    v_mm = np.full(n_trans + 1, np.nan, dtype=np.float64)
    v_mf = np.full(n_trans + 1, np.nan, dtype=np.float64)
    v_fm = np.full(n_trans + 1, np.nan, dtype=np.float64)
    v_ff = np.full(n_trans + 1, np.nan, dtype=np.float64)
    cov_m = np.full(n_trans + 1, np.nan, dtype=np.float64)
    cov_f = np.full(n_trans + 1, np.nan, dtype=np.float64)

    for g, entry in table.items():
        kmm = entry["k_mm"]
        kmf = entry["k_mf"]
        kfm = entry["k_fm"]
        kff = entry["k_ff"]
        n_m = len(kmm)
        n_f = len(kfm)
        if n_m < 2 or n_f < 2:
            continue
        # Per-male totals.
        k_m_total = kmm + kmf
        k_f_total = kfm + kff
        kbar_m = float(k_m_total.mean())
        kbar_f = float(k_f_total.mean())
        if kbar_m <= 0 or kbar_f <= 0:
            continue
        v_mm[g] = float(kmm.var(ddof=1))
        v_mf[g] = float(kmf.var(ddof=1))
        v_fm[g] = float(kfm.var(ddof=1))
        v_ff[g] = float(kff.var(ddof=1))
        cov_m[g] = float(np.cov(kmm, kmf, ddof=1)[0, 1])
        cov_f[g] = float(np.cov(kfm, kff, ddof=1)[0, 1])
        var_km_total = v_mm[g] + v_mf[g] + 2.0 * cov_m[g]
        var_kf_total = v_fm[g] + v_ff[g] + 2.0 * cov_f[g]
        df = (var_km_total / kbar_m) / (4.0 * n_m * kbar_m) + (var_kf_total / kbar_f) / (4.0 * n_f * kbar_f)
        if df > 0:
            ne_per_t[g] = 1.0 / (2.0 * df)

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
    """
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


def _founder_contribution_matrix(
    pg: PedigreeGraph,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-individual genetic contribution from each founder.

    ``c[i, f_local]`` = expected fraction of individual i's genome
    inherited from founder ``founder_idx[f_local]``.  Founders contribute
    1 to themselves and 0 to other founders; non-founders take the mean
    of their two parents' rows.

    Returns:
        ``(c, founder_idx)`` — ``c`` shape ``(n, n_founders)``, dtype
        float64; ``founder_idx`` shape ``(n_founders,)``, dtype intp.
    """
    n = pg.n
    gen = np.asarray(pg.generation)
    mother = np.asarray(pg.mother)
    father = np.asarray(pg.father)

    founder_idx = np.where(gen == 0)[0]
    n_founders = len(founder_idx)

    c = np.zeros((n, n_founders), dtype=np.float64)
    if n_founders == 0:
        return c, founder_idx
    c[founder_idx, np.arange(n_founders)] = 1.0

    g_max = int(gen.max())
    for g in range(1, g_max + 1):
        in_g = np.where(gen == g)[0]
        if len(in_g) == 0:
            continue
        m_idx_g = mother[in_g]
        f_idx_g = father[in_g]
        c_m = np.zeros((len(in_g), n_founders), dtype=np.float64)
        c_f = np.zeros((len(in_g), n_founders), dtype=np.float64)
        has_m = m_idx_g >= 0
        has_f = f_idx_g >= 0
        if has_m.any():
            c_m[has_m] = c[m_idx_g[has_m]]
        if has_f.any():
            c_f[has_f] = c[f_idx_g[has_f]]
        c[in_g] = 0.5 * (c_m + c_f)
    return c, founder_idx


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
            "ne": None if self.ne is None or not np.isfinite(self.ne) else float(self.ne),
            "ne_per_gen": [None if not np.isfinite(v) else float(v) for v in self.ne_per_gen],
            "mean_eqg_per_gen": [None if not np.isfinite(v) else float(v) for v in self.mean_eqg_per_gen],
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
            "ne": None if self.ne is None or not np.isfinite(self.ne) else float(self.ne),
            "asymptote_reached": bool(self.asymptote_reached),
            "n_iterations": int(self.n_iterations),
            "max_delta_final": None if not np.isfinite(self.max_delta_final) else float(self.max_delta_final),
            "sum_c_squared": float(self.sum_c_squared),
        }


@dataclass(frozen=True, slots=True)
class NeHillResult:
    """Hill 1979 age-structured (Ne_H) result, discrete-generation collapse.

    Under strictly discrete, non-overlapping generations the generation
    interval ``L = 1`` and the Hill 1979 formula reduces to Ne_V.  This
    class is a thin wrapper that records that collapse for traceability;
    the scalar ``ne`` is taken from :func:`ne_variance_family_size`.
    """

    ne: float | None
    generation_interval: float
    collapses_to_ne_v: bool

    def to_dict(self) -> dict[str, Any]:
        """Serialize as a YAML-ready dict."""
        return {
            "ne": None if self.ne is None or not np.isfinite(self.ne) else float(self.ne),
            "generation_interval": float(self.generation_interval),
            "collapses_to_ne_v": bool(self.collapses_to_ne_v),
        }


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
            "ne": None if self.ne is None or not np.isfinite(self.ne) else float(self.ne),
            "ne_per_gen": [None if not np.isfinite(v) else float(v) for v in self.ne_per_gen],
            "mean_self_coancestry_per_gen": [
                None if not np.isfinite(v) else float(v) for v in self.mean_self_coancestry_per_gen
            ],
            "n_founders_with_descendants_per_gen": [int(v) for v in self.n_founders_with_descendants_per_gen],
            "slope": None if not np.isfinite(self.slope) else float(self.slope),
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
    contribution: tuple[np.ndarray, np.ndarray] | None = None,
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
    if contribution is None:
        c, founder_idx = _founder_contribution_matrix(pg)
    else:
        c, founder_idx = contribution
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
            sum_c_squared=float((c[founder_idx] ** 2).sum() / n_founders**2 * n_founders),
        )

    c_per_gen = np.full((g_max + 1, n_founders), np.nan, dtype=np.float64)
    for g in range(g_max + 1):
        in_g = gen == g
        if in_g.any():
            c_per_gen[g] = c[in_g].mean(axis=0)

    asymptote_reached = False
    n_iterations = 0
    max_delta_final = float("nan")
    for g in range(1, g_max + 1):
        if not np.isfinite(c_per_gen[g]).all() or not np.isfinite(c_per_gen[g - 1]).all():
            continue
        delta = float(np.max(np.abs(c_per_gen[g] - c_per_gen[g - 1])))
        n_iterations = g
        max_delta_final = delta
        if delta < tol:
            asymptote_reached = True
            break

    final_c = c_per_gen[n_iterations]
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


def ne_hill_overlapping(pg: PedigreeGraph) -> NeHillResult:
    """Hill 1979 age-structured Ne (Ne_H), discrete-generation regression sentinel.

    simACE pedigrees are strictly discrete and non-overlapping (no age
    column, generations replace each other), so the generation interval
    ``L = 1`` and Hill's age-structured variance Ne reduces algebraically
    to :func:`ne_variance_family_size`.  This wrapper records the
    collapse for downstream consumers and explicit traceability.
    """
    v_result = ne_variance_family_size(pg)
    return NeHillResult(
        ne=v_result.ne,
        generation_interval=1.0,
        collapses_to_ne_v=True,
    )


def ne_caballero_toro(
    pg: PedigreeGraph,
    K: sp.csc_matrix | None = None,
    contribution: tuple[np.ndarray, np.ndarray] | None = None,
) -> NeCaballeroToroResult:
    """Caballero & Toro 2002 self-coancestry rate Ne (Ne_CT).

    For each founder f and generation g > 0, descendants are detected
    via ``c[i, f] > 0`` (i.e., founder f contributes a non-zero genetic
    fraction to i).  Self-coancestry per descendant is ``(1 + F_i) / 2``;
    averaged within each founder's descendant set, then averaged across
    founders that have descendants at gen g.  Ne from the regression
    slope of ``ln(1 − f̄_s,g)`` on g.
    """
    if K is None:
        K = pg.kinship_matrix()
    F = pg.compute_inbreeding()
    if contribution is None:
        c, founder_idx = _founder_contribution_matrix(pg)
    else:
        c, founder_idx = contribution
    n_founders = len(founder_idx)
    gen = np.asarray(pg.generation)
    g_max = int(gen.max())

    self_coancestry = (1.0 + F) / 2.0
    mean_fs_per_gen = np.full(g_max + 1, np.nan, dtype=np.float64)
    n_with_desc_per_gen = np.zeros(g_max + 1, dtype=np.int64)

    for g in range(1, g_max + 1):
        in_g = np.where(gen == g)[0]
        if len(in_g) == 0:
            continue
        # Per-founder mean self-coancestry over descendants in gen g.
        per_founder = np.full(n_founders, np.nan, dtype=np.float64)
        for f_local in range(n_founders):
            desc_mask = c[in_g, f_local] > 0.0
            if desc_mask.any():
                per_founder[f_local] = float(self_coancestry[in_g[desc_mask]].mean())
        with_desc = np.isfinite(per_founder)
        n_with_desc_per_gen[g] = int(with_desc.sum())
        if with_desc.any():
            mean_fs_per_gen[g] = float(per_founder[with_desc].mean())

    ne_per_gen = np.full(g_max + 1, np.nan, dtype=np.float64)
    for g in range(1, g_max + 1):
        prev = mean_fs_per_gen[g - 1] if g >= 2 else 0.0
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


def compute_all_ne(pg: PedigreeGraph) -> dict[str, Any]:
    """Run all eight Ne estimators on ``pg``.

    Builds the sparse kinship matrix and the founder-contribution
    matrix once and reuses them for every kinship/contribution-dependent
    estimator.

    Returns a dict keyed on estimator name; each value is the matching
    frozen result dataclass.
    """
    K = pg.kinship_matrix()
    contribution = _founder_contribution_matrix(pg)
    return {
        "ne_inbreeding": ne_inbreeding(pg),
        "ne_coancestry": ne_coancestry(pg, K=K),
        "ne_variance_family_size": ne_variance_family_size(pg),
        "ne_sex_ratio": ne_sex_ratio(pg),
        "ne_individual_delta_f": ne_individual_delta_f(pg),
        "ne_long_term_contributions": ne_long_term_contributions(pg, contribution=contribution),
        "ne_hill_overlapping": ne_hill_overlapping(pg),
        "ne_caballero_toro": ne_caballero_toro(pg, K=K, contribution=contribution),
    }


def _result_to_dict(result: Any) -> dict[str, Any]:
    """Serialize any frozen Ne result; falls back to ``dataclasses.asdict``."""
    if hasattr(result, "to_dict"):
        return result.to_dict()
    return asdict(result)
