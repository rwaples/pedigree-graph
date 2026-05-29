"""Result dataclasses for the Ne estimators (PGQ-006).

Every estimator returns one of these frozen dataclasses, each carrying a
per-generation (or per-cohort) series plus a scenario-level scalar
aggregate and a ``to_dict()`` serializer producing YAML/JSON-safe
values.  :class:`GenerationInterval` is the sex-split Hill 1979 ``L``
returned by :attr:`PedigreeGraph.generation_interval`.

These types are pure data + serialization; the estimators that build
them live in the ``_ne_*`` sibling modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from pedigree_graph._cohort_utils import CohortWindow


def _optional_float(x: float | None) -> float | None:
    """``None`` for missing or non-finite; else ``float(x)``.

    Used by every ``NeXxxResult.to_dict`` to coerce optional scalar Ne /
    diagnostic fields to YAML-safe JSON values.
    """
    if x is None or not np.isfinite(x):
        return None
    return float(x)


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
            "mean_self_coancestry_per_gen": [_optional_float(v) for v in self.mean_self_coancestry_per_gen],
            "n_founders_with_descendants_per_gen": [int(v) for v in self.n_founders_with_descendants_per_gen],
            "slope": _optional_float(self.slope),
        }
