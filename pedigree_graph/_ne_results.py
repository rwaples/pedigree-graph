"""Result records for the Ne estimators (PGQ-006, slice 6c).

Every estimator returns one of these frozen dataclasses, each carrying its
observed generation labels next to the per-cohort or per-transition series
it indexes, plus a scenario-level scalar aggregate.  Construction owns every
array: each is copied contiguous and read-only in its declared dtype, and
the lengths are checked against the label array they align with, so a
result cannot be changed through a constructor input and cannot describe
one cohort with two different arrays.

Serialization is shared: the records mix in :class:`_SerializableResult`,
whose ``to_dict`` walks the dataclass fields through :func:`_to_jsonable`
(dtype-driven, non-finite floats → ``None``).  :class:`GenerationInterval`
is the sex-split Hill 1979 ``L`` returned by
:attr:`PedigreeGraph.generation_interval`.

These types are pure data + serialization; the estimators that build them
live in the ``_ne_*`` sibling modules and are exported from
:mod:`pedigree_graph.effective_size`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from types import MappingProxyType
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
    table (a mapping of arrays).  Centralising the rules here keeps the
    coercions consistent across every record.

    Numeric arrays follow their dtype: integer series become ``list[int]``,
    floating series become ``list[float | None]`` (non-finite → ``None``, so
    the output is always valid YAML/JSON rather than carrying ``nan``).
    Scalars follow their Python type.  Mappings become fresh ordinary dicts.
    An unrecognised shape *raises* rather than serializing silently, so a
    future field with an unusual type is caught in review instead of shipped
    mis-serialized.
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
    if isinstance(value, Mapping):
        return {k: _to_jsonable(v) for k, v in value.items()}
    asdict_fn = getattr(value, "_asdict", None)  # NamedTuple records, e.g. CohortWindow
    if callable(asdict_fn):
        return asdict_fn()
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
        return {f.name: _to_jsonable(getattr(self, f.name)) for f in fields(self)}  # ty: ignore[invalid-argument-type]


def _owned_copy(values: object, dtype: type | np.dtype) -> np.ndarray:
    """A fresh contiguous read-only 1-D copy in ``dtype``; never aliases the input."""
    out = np.array(values, dtype=dtype, copy=True, order="C")
    if out.ndim != 1:
        raise ValueError(f"expected a 1-D array, got {out.ndim}-D")
    out.setflags(write=False)
    return out


# Array fields declare the dtype they own and the label array they align
# with.  ``cohort``/``transition`` align with ``generations`` (one entry per
# observed cohort, or per adjacent pair), ``parent`` with
# ``parent_generations``, ``cohort_year`` with Hill's ``cohort_years``.
_AXIS_LABELS: Mapping[str, str] = MappingProxyType(
    {
        "cohort": "generations",
        "transition": "generations",
        "parent": "parent_generations",
        "cohort_year": "cohort_years",
    }
)


def _meta(dtype: type, axis: str, *, optional: bool = False, labels: bool = False) -> dict[str, Any]:
    return {"dtype": dtype, "axis": axis, "optional": optional, "labels": labels}


def _same(a: object, b: object) -> bool:
    """Field equality where NaN equals NaN, for arrays, mappings, and scalars."""
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return (
            isinstance(a, np.ndarray)
            and isinstance(b, np.ndarray)
            and a.dtype == b.dtype
            and np.array_equal(a, b, equal_nan=True)
        )
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b):
        return True
    return bool(a == b)


class _FrozenResult(_SerializableResult):
    """Base for the final records: own every array and check the axes.

    ``__post_init__`` copies each declared array field into a contiguous
    read-only array of its declared dtype, then checks that every series
    has one entry per label of the axis it declares, that label arrays are
    strictly ascending, and that ``transition_from`` / ``transition_to``
    are exactly the adjacent pairs of ``generations``.  Two records are
    equal when every field is, arrays element-wise with NaN equal to NaN.
    """

    __slots__ = ()
    __hash__ = None  # type: ignore[assignment]

    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self):
            return NotImplemented
        return all(_same(getattr(self, f.name), getattr(other, f.name)) for f in fields(self))  # ty: ignore[invalid-argument-type]

    def __post_init__(self) -> None:
        for f in fields(self):  # ty: ignore[invalid-argument-type]
            meta = f.metadata
            if "dtype" not in meta:
                continue
            value = getattr(self, f.name)
            if value is None:
                if meta["optional"]:
                    continue
                raise TypeError(f"{type(self).__name__}.{f.name} must be an array")
            object.__setattr__(self, f.name, _owned_copy(value, meta["dtype"]))
        self._check_axes()

    def _check_axes(self) -> None:
        name = type(self).__name__
        for f in fields(self):  # ty: ignore[invalid-argument-type]
            meta = f.metadata
            if "axis" not in meta:
                continue
            value = getattr(self, f.name)
            if value is None:
                continue
            labels = getattr(self, _AXIS_LABELS[meta["axis"]])
            if labels is None:
                raise ValueError(f"{name}.{f.name} is set but {_AXIS_LABELS[meta['axis']]} is None")
            expected = max(labels.shape[0] - 1, 0) if meta["axis"] == "transition" else labels.shape[0]
            if value.shape[0] != expected:
                raise ValueError(f"{name}.{f.name} has {value.shape[0]} entries for {expected} labels")
            if meta.get("labels") and value.shape[0] > 1 and not np.all(np.diff(value) > 0):
                raise ValueError(f"{name}.{f.name} must be strictly ascending")
        generations: np.ndarray | None = getattr(self, "generations", None)
        transition_from: np.ndarray | None = getattr(self, "transition_from", None)
        transition_to: np.ndarray | None = getattr(self, "transition_to", None)
        if (
            generations is not None
            and transition_from is not None
            and transition_to is not None
            and not (
                np.array_equal(transition_from, generations[:-1]) and np.array_equal(transition_to, generations[1:])
            )
        ):
            raise ValueError(f"{name}: transition labels must be the adjacent pairs of generations")


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


@dataclass(frozen=True, slots=True, eq=False)
class NeInbreedingResult(_FrozenResult):
    """Inbreeding-rate (Ne_I) result.

    Attributes:
        ne: scalar Ne from regression of ``ln(1 − F̄)`` on the label offset,
            first observed cohort excluded.
        generations: observed generation labels, int32, ascending.
        mean_f_per_gen: per-cohort mean F, aligned with ``generations``.
        transition_from: ``generations[:-1]``.
        transition_to: ``generations[1:]``.
        ne_per_gen: Ne of each adjacent observed-cohort transition
            (``transition_from[i] → transition_to[i]``), gap-corrected.
        slope: regression slope (log scale).
        n_generations_used: post-baseline cohorts in the regression.
    """

    ne: float | None
    generations: np.ndarray = field(metadata=_meta(np.int32, "cohort", labels=True))
    mean_f_per_gen: np.ndarray = field(metadata=_meta(np.float64, "cohort"))
    transition_from: np.ndarray = field(metadata=_meta(np.int32, "transition"))
    transition_to: np.ndarray = field(metadata=_meta(np.int32, "transition"))
    ne_per_gen: np.ndarray = field(metadata=_meta(np.float64, "transition"))
    slope: float = float("nan")
    n_generations_used: int = 0


@dataclass(frozen=True, slots=True, eq=False)
class NeCoancestryResult(_FrozenResult):
    """Coancestry-rate (Ne_C) result.

    Attributes:
        ne: scalar Ne from regression of ``ln(1 − θ̄)`` on the label offset,
            first observed cohort excluded.
        generations: observed generation labels, int32, ascending.
        mean_theta_per_gen: per-cohort mean θ over within-cohort pairs.
        transition_from: ``generations[:-1]``.
        transition_to: ``generations[1:]``.
        ne_per_gen: Ne of each adjacent observed-cohort transition.
        slope: regression slope.
        n_generations_used: post-baseline cohorts in the regression.
    """

    ne: float | None
    generations: np.ndarray = field(metadata=_meta(np.int32, "cohort", labels=True))
    mean_theta_per_gen: np.ndarray = field(metadata=_meta(np.float64, "cohort"))
    transition_from: np.ndarray = field(metadata=_meta(np.int32, "transition"))
    transition_to: np.ndarray = field(metadata=_meta(np.int32, "transition"))
    ne_per_gen: np.ndarray = field(metadata=_meta(np.float64, "transition"))
    slope: float = float("nan")
    n_generations_used: int = 0


@dataclass(frozen=True, slots=True, eq=False)
class NeVarianceResult(_FrozenResult):
    """Variance-of-family-size (Ne_V) result.

    Caballero 1994 eq. 6 with separate sexes.  ``V(k_m) = V(k_mm) +
    V(k_mf) + 2·Cov(k_mm, k_mf)`` is the per-male total-offspring
    variance built from the sex-of-offspring decomposition; symmetrically
    for females.

    Every array is indexed by **parent cohort**: entry ``p`` summarises the
    lifetime reproduction of the cohort labelled ``parent_generations[p]``,
    which under skip-gen pedigrees may include offspring spread across
    several later cohorts.  Every observed label is present, including the
    last one; a cohort with no usable reproduction stays NaN.
    ``ne_per_transition`` keeps its historical name but describes that
    lifetime reproduction, not a unique transition.  Aggregate Ne is the
    harmonic mean.
    """

    ne: float | None
    parent_generations: np.ndarray = field(metadata=_meta(np.int32, "parent", labels=True))
    ne_per_transition: np.ndarray = field(metadata=_meta(np.float64, "parent"))
    v_mm: np.ndarray = field(metadata=_meta(np.float64, "parent"))
    v_mf: np.ndarray = field(metadata=_meta(np.float64, "parent"))
    v_fm: np.ndarray = field(metadata=_meta(np.float64, "parent"))
    v_ff: np.ndarray = field(metadata=_meta(np.float64, "parent"))
    cov_m: np.ndarray = field(metadata=_meta(np.float64, "parent"))
    cov_f: np.ndarray = field(metadata=_meta(np.float64, "parent"))


@dataclass(frozen=True, slots=True, eq=False)
class NeSexRatioResult(_FrozenResult):
    """Wright sex-ratio (Ne_sr) result.

    ``Ne_g = 4·Nm_g·Nf_g / (Nm_g + Nf_g)`` per observed cohort; aggregate
    is the harmonic mean across cohorts with at least one of each sex.
    """

    ne: float | None
    generations: np.ndarray = field(metadata=_meta(np.int32, "cohort", labels=True))
    ne_per_gen: np.ndarray = field(metadata=_meta(np.float64, "cohort"))
    n_male_per_gen: np.ndarray = field(metadata=_meta(np.int64, "cohort"))
    n_female_per_gen: np.ndarray = field(metadata=_meta(np.int64, "cohort"))


@dataclass(frozen=True, slots=True, eq=False)
class NeIndividualDeltaFResult(_FrozenResult):
    """Gutiérrez et al. 2008 individual increase in inbreeding (Ne_iΔF) result.

    The individual increase in inbreeding of Gutiérrez et al. 2008 (Genet.
    Sel. Evol. 40:359, eq. 2) is ``ΔF_i = 1 − (1 − F_i)^(1/t_i)``, where
    ``t_i`` is the individual's equivalent complete generations.  A row is
    eligible when ``t_i > 0``, and when ``F_i < 1`` — the second is this
    package's guard, not the paper's.  Their §2.1 averages ΔF_i over a
    reference subpopulation to give ΔF̄ and reports ``Ne = 1/(2·ΔF̄)``.

    ``ne_unrelated_founders`` is this package's own diagnostic rather than
    theirs, and no line of the paper contains it.  Its justification is
    measurement, not theory.  Eq. 1 is ``F_t = 1 − (1 − ΔF)^t``, so eq. 2
    recovers the census size only where the pedigree has accumulated ``t``
    generations of drift.  Founders that are genuinely unrelated and non-inbred
    leave generation 1 at ``F = 0`` exactly, so ``t`` generations of pedigree
    carry ``t − 1`` generations of drift: measured on Wright-Fisher pedigrees,
    mean F tracks the idealised curve at ``g − 1`` rather than ``g`` at every
    generation out to 16, and the resulting upward bias in ``ne`` is
    ``t/(t − 1)`` in the large-N limit, less an O(1/N) term from the convexity
    of ``F ↦ ΔF_i``.  At ``N = 1000``, ``t = 5`` the measured bias is +24.66%
    against a 25.00% prediction, and this field's residual there is −0.16% (20
    Wright-Fisher replicates per cell, ADR 0012).  The paper's own remedy is
    methodological instead: choosing the reference subpopulation by pedigree
    depth (its Table II reports Ne at no pedigree depth restriction, ``t ≥ 4``
    and ``t ≥ 8``) and reading ΔF_i against equivalent generations (its
    Figs. 3-4), and ``reference=`` is how a caller does that here.

    The assumption is true of a simulated pedigree and false of a real one,
    whose founders are merely where record-keeping stopped and are generally
    related: there is no lag to remove and the field introduces a *downward*
    bias instead.  It is a diagnostic for simulated or genuinely
    founder-complete pedigrees, and ``ne`` remains the estimator.

    Attributes:
        ne: ``1/(2·ΔF̄)`` over the eligible rows of the reference
            subpopulation; ``None`` when none are eligible or ``ΔF̄ ≤ 0``.
        generations: observed generation labels, int32, ascending.
        ne_per_gen: ``1/(2·ΔF̄_g)`` taking cohort ``g`` as the reference
            subpopulation, aligned with ``generations``; NaN where that
            cohort has no eligible row or ``ΔF̄_g ≤ 0``.  Equivalent complete
            generations are counted per individual, so no label-gap
            correction applies.
        mean_eqg_per_gen: mean equivalent complete generations over the
            eligible rows of each cohort; NaN where there are none.
        n_used_per_gen: eligible rows per cohort.
        standard_error: the paper's ``σ_Ne = (2/√N)·Ne²·σ_ΔF``, with ``N =
            n_reference`` and σ_ΔF the sample standard deviation (``ddof=1``,
            this package's choice; the paper does not state one and the
            difference is O(1/N)).  ``None`` when ``ne`` is ``None`` or
            ``n_reference < 2``.
        n_reference: eligible rows of the reference subpopulation — the ``N``
            behind both ΔF̄ and σ_ΔF, so the paper's two formulas describe the
            same set of individuals.
        reference_generation: the observed label shared by every eligible
            reference row, or ``None`` when they span several cohorts or none.
        ne_unrelated_founders: ``1/(2·ΔF̄′)`` over the reference rows with
            ``t_i > 1``, averaging ``ΔF′_i = 1 − (1 − F_i)^(1/(t_i − 1))``.
            Package-defined and absent from the paper, for the reasons above.
            Eligibility is stricter than ``ne``'s (``t > 1``, not ``t > 0``),
            so fewer rows than ``n_reference`` can stand behind it, and no
            count is reported for it.  ``None`` when no reference row
            qualifies, when ``ΔF̄′ ≤ 0``, or when the graph is empty.
    """

    ne: float | None
    generations: np.ndarray = field(metadata=_meta(np.int32, "cohort", labels=True))
    ne_per_gen: np.ndarray = field(metadata=_meta(np.float64, "cohort"))
    mean_eqg_per_gen: np.ndarray = field(metadata=_meta(np.float64, "cohort"))
    n_used_per_gen: np.ndarray = field(metadata=_meta(np.int64, "cohort"))
    standard_error: float | None = None
    n_reference: int = 0
    reference_generation: int | None = None
    ne_unrelated_founders: float | None = None


@dataclass(frozen=True, slots=True, eq=False)
class NeLTCResult(_FrozenResult):
    """Founder-genome long-term contribution (Ne_LTC) result.

    Both effective sizes are read from the last observed cohort.  See
    :func:`~pedigree_graph.effective_size.ne_long_term_contributions` for the
    sources and for the assumptions ``ne`` carries.

    A graph with no represented founder or no observed cohort is the
    degenerate case named below, and the only one where ``ne`` and
    ``n_effective_founders`` are ``None``.

    Attributes:
        ne: ``2/Σ_f c_f²``, Wray & Thompson 1990 eq. 31.  Derived, and sound
            only under regular random mating (α = 0) with long-term
            contribution variance at its asymptote.
        n_effective_founders: ``1/Σ_f c_f²``, Caballero & Toro 2000 eq. 19.
            Assumption-free, and the quantity that still stands where either
            of ``ne``'s assumptions is in doubt.
        sum_c_squared: ``Σ_f c_f²`` over founder genomes at that cohort;
            ``0.0`` in the degenerate case.
        max_delta_final: ``max_f |c_last[f] − c_{last−1}[f]|``, the movement
            into the cohort the estimate was read from, and the evidence a
            caller weighs against the asymptotic-variance assumption.
            ``nan`` when only one cohort is observed.
        asymptote_reached: descriptive only — ``max_delta_final`` fell below
            the estimator's fixed tolerance.  It gates nothing.
        n_cohorts: observed cohorts, i.e. how much series stands behind
            ``max_delta_final``; ``0`` in the degenerate case.
        final_generation: label of the last observed cohort; ``None`` in the
            degenerate case.
    """

    ne: float | None
    n_effective_founders: float | None
    sum_c_squared: float
    max_delta_final: float
    asymptote_reached: bool
    n_cohorts: int
    final_generation: int | None


@dataclass(frozen=True, slots=True, eq=False)
class NeHillResult(_FrozenResult):
    """Hill 1979 separate-sex overlapping-generation Ne (Ne_H).

    The result has three states, identified by ``collapses_to_ne_v`` and
    ``n_eligible_cohorts``:

    * **Sentinel** (``collapses_to_ne_v=True``): used when
      ``pg.birth_year is None``.  Hill 1979 with ``L = 1`` reduces
      algebraically to Ne_V (Caballero 1994 eq. 6 / Hill 1979 eq. 8 with
      equal sexes), so ``ne`` is the Ne_V passthrough.  Birth-year diagnostic
      fields retain their ``None`` / ``0`` defaults.
    * **Birth-year empty** (``collapses_to_ne_v=False`` and
      ``n_eligible_cohorts == 0``): birth-year context is present, but no
      cohort has enough observations for an estimate.  ``ne``, cohort
      summary diagnostics, and per-cohort arrays are ``None``.
    * **Birth-year populated** (``collapses_to_ne_v=False`` and
      ``n_eligible_cohorts > 0``): Ne is computed per eligible birth-year
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
    cohort_years: np.ndarray | None = field(
        default=None, metadata=_meta(np.int32, "cohort_year", optional=True, labels=True)
    )
    ne_per_cohort: np.ndarray | None = field(default=None, metadata=_meta(np.float64, "cohort_year", optional=True))
    Ne_m_per_cohort: np.ndarray | None = field(default=None, metadata=_meta(np.float64, "cohort_year", optional=True))
    Ne_f_per_cohort: np.ndarray | None = field(default=None, metadata=_meta(np.float64, "cohort_year", optional=True))
    Vk_m_per_cohort: np.ndarray | None = field(default=None, metadata=_meta(np.float64, "cohort_year", optional=True))
    Vk_f_per_cohort: np.ndarray | None = field(default=None, metadata=_meta(np.float64, "cohort_year", optional=True))
    N1_m_per_cohort: np.ndarray | None = field(default=None, metadata=_meta(np.int64, "cohort_year", optional=True))
    N1_f_per_cohort: np.ndarray | None = field(default=None, metadata=_meta(np.int64, "cohort_year", optional=True))
    # Per-individual age table — descriptive only, not used in Ne
    age_table: Mapping[str, np.ndarray] | None = None
    n_offspring_pairs: int = 0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.age_table is not None:
            frozen = {str(key): _owned_copy(value, np.asarray(value).dtype) for key, value in self.age_table.items()}
            object.__setattr__(self, "age_table", MappingProxyType(frozen))
        self._validate_state()

    def _validate_state(self) -> None:
        context = ("T_m", "T_f", "cohort_window", "age_table")
        summaries = ("N1_m", "N1_f", "Vk_m", "Vk_f", "kbar_m", "kbar_f", "Ne_m", "Ne_f")
        series = (
            "cohort_years",
            "ne_per_cohort",
            "Ne_m_per_cohort",
            "Ne_f_per_cohort",
            "Vk_m_per_cohort",
            "Vk_f_per_cohort",
            "N1_m_per_cohort",
            "N1_f_per_cohort",
        )
        if self.n_eligible_cohorts < 0:
            raise ValueError("NeHillResult.n_eligible_cohorts must be non-negative")
        if self.collapses_to_ne_v:
            nonzero_counts = any(
                getattr(self, name) != 0
                for name in (
                    "n_eligible_cohorts",
                    "n_excluded_right_censored",
                    "n_excluded_left_censored",
                    "n_unknown_birth_year",
                    "n_offspring_pairs",
                )
            )
            if (
                self.generation_interval != 1.0
                or nonzero_counts
                or any(getattr(self, name) is not None for name in context + summaries + series)
            ):
                raise ValueError("NeHillResult sentinel state requires L=1 and no birth-year diagnostics")
            return
        if self.n_eligible_cohorts == 0:
            if (
                any(getattr(self, name) is None for name in context)
                or self.ne is not None
                or any(getattr(self, name) is not None for name in summaries + series)
            ):
                raise ValueError(
                    "NeHillResult birth-year-empty state requires birth-year context and no cohort estimates"
                )
            return
        required = ("ne", *context, *summaries, *series)
        message = "NeHillResult birth-year-populated state requires complete diagnostics for every eligible cohort"
        if any(getattr(self, name) is None for name in required):
            raise ValueError(message)
        cohort_years = self.cohort_years
        if cohort_years is None or len(cohort_years) != self.n_eligible_cohorts:
            raise ValueError(message)


@dataclass(frozen=True, slots=True, eq=False)
class NeCaballeroToroResult(_FrozenResult):
    """Caballero & Toro 2002 self-coancestry rate (Ne_CT) result.

    For each represented founder genome f and observed cohort after the
    first, computes the mean self-coancestry of f's descendants in that
    cohort, ``f̄_s,f,g = mean_{i ∈ desc(f,g)} (1 + F_i) / 2``, averages over
    founder genomes with descendants there, and regresses
    ``ln(1 − f̄_s,g)`` on the label offset, reporting
    ``ne = −1 / (2·slope)``.  The first observed cohort is the baseline: its
    mean is NaN and its founder-descendant count 0, and the first transition
    starts from the conceptual non-inbred self-coancestry ``0.5``.
    """

    ne: float | None
    generations: np.ndarray = field(metadata=_meta(np.int32, "cohort", labels=True))
    mean_self_coancestry_per_gen: np.ndarray = field(metadata=_meta(np.float64, "cohort"))
    n_founders_with_descendants_per_gen: np.ndarray = field(metadata=_meta(np.int64, "cohort"))
    transition_from: np.ndarray = field(metadata=_meta(np.int32, "transition"))
    transition_to: np.ndarray = field(metadata=_meta(np.int32, "transition"))
    ne_per_gen: np.ndarray = field(metadata=_meta(np.float64, "transition"))
    slope: float = float("nan")
