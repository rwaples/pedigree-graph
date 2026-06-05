"""Result dataclasses for the Ne estimators (PGQ-006).

Every estimator returns one of these frozen dataclasses, each carrying a
per-generation (or per-cohort) series plus a scenario-level scalar
aggregate.  Serialization is shared: the records mix in
:class:`_SerializableResult`, whose ``to_dict`` walks the dataclass fields
through :func:`_to_jsonable` (dtype-driven, non-finite floats → ``None``)
instead of each record re-implementing the same coercions.
:class:`GenerationInterval` is the sex-split Hill 1979 ``L`` returned by
:attr:`PedigreeGraph.generation_interval`.

These types are pure data + serialization; the estimators that build
them live in the ``_ne_*`` sibling modules.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from pedigree_graph._cohort_utils import CohortWindow


def _optional_float(x: float | None) -> float | None:
    """``None`` for missing or non-finite; else ``float(x)``.

    Used by :func:`_to_jsonable` to coerce optional scalar Ne / diagnostic
    fields to YAML-safe JSON values.
    """
    if x is None or not np.isfinite(x):
        return None
    return float(x)


def _to_jsonable(value: Any) -> Any:
    """Coerce one result-dataclass field to a YAML/JSON-safe value.

    The result records hold a small, uniform set of field shapes — optional
    scalars (Ne and diagnostics), 1-D numeric series (per-generation or
    per-cohort), integer counts, boolean flags, an optional nested
    :class:`~pedigree_graph._cohort_utils.CohortWindow`, and the Hill age
    table (``dict[str, np.ndarray]``).  Each result used to serialize these
    by hand; centralising the rules here removes ~150 lines of near-identical
    ``to_dict`` boilerplate and keeps the coercions consistent.

    Numeric arrays follow their dtype: integer series become ``list[int]``,
    floating series become ``list[float | None]`` (non-finite → ``None``, so
    the output is always valid YAML/JSON rather than carrying ``nan``).
    Scalars follow their Python type.  An unrecognised shape *raises* rather
    than serializing silently, so a future field with an unusual type is
    caught in review instead of shipped mis-serialized.
    """
    if value is None:
        return None
    # bool before int: bool is an int subclass and must stay a JSON bool.
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, np.ndarray):
        if value.ndim != 1:
            raise TypeError(f"_to_jsonable: expected a 1-D array, got {value.ndim}-D")
        if value.dtype.kind in ("i", "u"):
            return [int(v) for v in value]
        return [_optional_float(v) for v in value]
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return _optional_float(value)
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    asdict = getattr(value, "_asdict", None)  # NamedTuple records, e.g. CohortWindow
    if callable(asdict):
        return asdict()
    raise TypeError(f"_to_jsonable: no serialization rule for {type(value).__name__}")


class _SerializableResult:
    """Mixin: serialize a result dataclass to a YAML-ready dict.

    Walks the dataclass fields and coerces each via :func:`_to_jsonable`.
    Mixed into the result records (which stay ``@dataclass(frozen=True,
    slots=True)``) so they share one serializer instead of nine copies.
    """

    __slots__ = ()

    def to_dict(self) -> dict[str, Any]:
        """Serialize as a YAML-ready dict (numpy arrays → lists)."""
        return {f.name: _to_jsonable(getattr(self, f.name)) for f in fields(self)}


@dataclass(frozen=True, slots=True)
class GenerationInterval(_SerializableResult):
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


@dataclass(frozen=True, slots=True)
class NeInbreedingResult(_SerializableResult):
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


@dataclass(frozen=True, slots=True)
class NeCoancestryResult(_SerializableResult):
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


@dataclass(frozen=True, slots=True)
class NeVarianceResult(_SerializableResult):
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


@dataclass(frozen=True, slots=True)
class NeSexRatioResult(_SerializableResult):
    """Wright sex-ratio (Ne_sr) result.

    ``Ne_t = 4·Nm_t·Nf_t / (Nm_t + Nf_t)`` per generation; aggregate is
    the harmonic mean across cohorts with at least one of each sex.
    """

    ne: float | None
    ne_per_gen: np.ndarray
    n_male_per_gen: np.ndarray
    n_female_per_gen: np.ndarray


@dataclass(frozen=True, slots=True)
class NeIndividualDeltaFResult(_SerializableResult):
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


@dataclass(frozen=True, slots=True)
class NeLTCResult(_SerializableResult):
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


@dataclass(frozen=True, slots=True)
class NeHillResult(_SerializableResult):
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


@dataclass(frozen=True, slots=True)
class NeCaballeroToroResult(_SerializableResult):
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
