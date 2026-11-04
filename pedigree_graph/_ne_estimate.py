"""``estimate_effective_sizes``: the eight estimators over one prerequisite memo.

The orchestrator runs the selected estimators serially in canonical order.
:data:`_REGISTRY` gives each one a row naming its metadata guards and its
build, so what an estimator requires and what it assembles are read in one
place rather than reconstructed from a dispatch chain.

Each estimator first runs its guards in its own context, so a failure names
that estimator, then pulls what it needs from a per-call memo of named
prerequisites (:class:`_Prerequisites`): the observed cohorts, F, the
equivalent complete generations, the generation kinship summary, the
represented founders, the two family-size tables, Ne_V's family-size
variance record, the founder means, the per-cohort group coancestry, the
generation interval, and the cohort window.  Each is built at most once per
call and only when a selected estimator asks for it, so an unselected
estimator costs nothing and a shared prerequisite is never built twice.
Completed estimator results are memoized beside them in their own
namespace.  Hill's absent-birth-year collapse reads the variance record as
a prerequisite, after running Ne_V's own guards under Hill's name, so Ne_V
alone ever fills the Ne_V result and its public key stays
``not_requested`` when only Hill is selected.

Each public ``ne_X(pg)`` is the same dispatch over a memo of its own: a
one-row call through :data:`_REGISTRY`, so an estimator's guards and its
wiring are written once, here.

There is no worker pool on this path (ADR 0007): the old pool dispatched
formulas only after eagerly building the expensive prerequisites, and
running the kinship and founder prerequisites concurrently would multiply
peak memory.  The package thread budget is committed once, after selection
is validated, and applies through the kernels the prerequisites call.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal

from pedigree_graph._cohort_utils import eligible_cohort_range
from pedigree_graph._cohorts import ObservedCohorts
from pedigree_graph._errors import MissingMetadataError
from pedigree_graph._ne_family_size import (
    _generation_family_table,
    _sex_column,
    _sex_ratio_from,
    _variance_from,
    _warn_if_uniform_sex,
)
from pedigree_graph._ne_founders import _founder_idx, _ltc_from, _per_gen_founder_means
from pedigree_graph._ne_group_coancestry import _group_coancestry_by_cohort, _group_coancestry_from
from pedigree_graph._ne_hill import _birth_year_family_table, _hill_from, _hill_from_variance
from pedigree_graph._ne_metadata import (
    _require_closed_parentage,
    _require_complete_generation_labels,
    _require_complete_sex,
)
from pedigree_graph._ne_rates import (
    _coancestry_from,
    _equivalent_generations,
    _generation_kinship_summary,
    _inbreeding_from,
    _individual_delta_f_from,
    _reference_rows,
)
from pedigree_graph._ne_results import (
    NeCoancestryResult,
    NeGroupCoancestryResult,
    NeHillResult,
    NeInbreedingResult,
    NeIndividualDeltaFResult,
    NeLTCResult,
    NeSexRatioResult,
    NeVarianceResult,
)
from pedigree_graph._threads import thread_budget

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    import numpy as np

    from pedigree_graph._cohort_utils import CohortWindow
    from pedigree_graph._core import PedigreeGraph
    from pedigree_graph._ne_family_size import FamilySizeTable
    from pedigree_graph._ne_founders import FounderContributionMeans
    from pedigree_graph._ne_group_coancestry import GroupCoancestryByCohort
    from pedigree_graph._ne_results import GenerationInterval
    from pedigree_graph.summaries import GenerationKinshipSummary

__all__ = [
    "ALL_EFFECTIVE_SIZE_ESTIMATORS",
    "EffectiveSizeResults",
    "UnavailableEffectiveSize",
    "estimate_effective_sizes",
]

ALL_EFFECTIVE_SIZE_ESTIMATORS: tuple[str, ...] = (
    "ne_inbreeding",
    "ne_coancestry",
    "ne_variance_family_size",
    "ne_sex_ratio",
    "ne_individual_delta_f",
    "ne_long_term_contributions",
    "ne_hill_overlapping",
    "ne_group_coancestry",
)
"""The eight estimator names, in canonical execution and output order."""

EffectiveSizeResult = (
    NeInbreedingResult
    | NeCoancestryResult
    | NeVarianceResult
    | NeSexRatioResult
    | NeIndividualDeltaFResult
    | NeLTCResult
    | NeHillResult
    | NeGroupCoancestryResult
)


@dataclass(frozen=True, slots=True)
class UnavailableEffectiveSize:
    """Why an estimator's key carries no result.

    Attributes:
        reason: ``"not_requested"`` when the estimator was not selected,
            ``"missing_metadata"`` when it was selected and refused the
            pedigree.
        code: The :class:`~pedigree_graph.MissingMetadataError` code, or
            ``None`` when not requested.
        fields: That error's immutable fields; empty when not requested.
    """

    reason: Literal["not_requested", "missing_metadata"]
    code: str | None
    fields: Mapping[str, object]

    def __post_init__(self) -> None:
        if self.reason not in ("not_requested", "missing_metadata"):
            raise ValueError(f"unknown reason {self.reason!r}")
        if (self.code is None) != (self.reason == "not_requested"):
            raise ValueError("code is None exactly when the estimator was not requested")
        object.__setattr__(self, "fields", MappingProxyType(dict(self.fields)))

    @classmethod
    def not_requested(cls) -> UnavailableEffectiveSize:
        return cls("not_requested", None, {})

    @classmethod
    def from_error(cls, error: MissingMetadataError) -> UnavailableEffectiveSize:
        return cls("missing_metadata", error.code, error.fields)

    def to_dict(self) -> dict[str, Any]:
        """``{"reason", "code", "fields"}`` with a fresh ordinary ``fields`` dict."""
        return {"reason": self.reason, "code": self.code, "fields": dict(self.fields)}


class EffectiveSizeResults(Mapping[str, EffectiveSizeResult | UnavailableEffectiveSize]):
    """What :func:`estimate_effective_sizes` returns: all eight keys, canonical order.

    A deeply immutable, tuple-backed mapping.  Every key of
    :data:`ALL_EFFECTIVE_SIZE_ESTIMATORS` is present, in that order; a value is
    the estimator's frozen result or an :class:`UnavailableEffectiveSize`.
    Access and equality follow ordinary ``Mapping`` semantics.
    """

    __slots__ = ("_items",)
    _items: tuple[tuple[str, EffectiveSizeResult | UnavailableEffectiveSize], ...]

    def __init__(self, items: Iterable[tuple[str, EffectiveSizeResult | UnavailableEffectiveSize]]) -> None:
        pairs = tuple(items)
        if tuple(name for name, _ in pairs) != ALL_EFFECTIVE_SIZE_ESTIMATORS:
            raise ValueError("EffectiveSizeResults needs exactly the eight estimator keys in canonical order")
        object.__setattr__(self, "_items", pairs)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError(f"{type(self).__name__} is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError(f"{type(self).__name__} is immutable")

    def __getitem__(self, key: str) -> EffectiveSizeResult | UnavailableEffectiveSize:
        for name, value in self._items:
            if name == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (name for name, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __repr__(self) -> str:
        body = ", ".join(f"{name}={type(value).__name__}" for name, value in self._items)
        return f"{type(self).__name__}({body})"

    def to_dict(self) -> dict[str, Any]:
        """A fresh insertion-ordered dict with every nested result serialized."""
        return {name: value.to_dict() for name, value in self._items}


type _Guard = Callable[[PedigreeGraph, str], None]
"""One metadata guard.  All but the uniform-sex notice raise rather than warn."""

_LABELS: tuple[_Guard, ...] = (_require_complete_generation_labels,)
_LABELS_AND_SEX: tuple[_Guard, ...] = (
    _require_complete_generation_labels,
    _require_complete_sex,
    _warn_if_uniform_sex,
)


def _always(guards: tuple[_Guard, ...]) -> Callable[[PedigreeGraph], tuple[_Guard, ...]]:
    """Guards that do not depend on the graph."""
    return lambda _pg: guards


def _hill_guards(pg: PedigreeGraph) -> tuple[_Guard, ...]:
    """Hill is the one estimator whose guards depend on the graph.

    The birth-year branch groups by birth year and never reads a generation
    label, so partly unknown labels are no obstacle to it.  The collapse
    branch is Ne_V, and runs Ne_V's guards, read from Ne_V's own row so the
    two cannot drift: the branch reads Ne_V's record without guards of its
    own.
    """
    if pg.birth_year is not None:
        return (_require_complete_sex,)
    return _REGISTRY["ne_variance_family_size"].guards(pg)


def _build_hill(memo: _Prerequisites) -> NeHillResult:
    """Hill's two branches over the shared memo.

    The collapse branch reaches Ne_V through the memo rather than recomputing
    it, which is how a selection holding both estimators pays for one variance
    table while Hill alone leaves the public Ne_V key ``not_requested``.
    """
    if memo.pg.n_individuals == 0:
        return NeHillResult(
            ne=None,
            generation_interval=1.0,
            collapses_to_ne_v=True,
            vk_scaled=memo.hill_vk_scale,
            no_estimate_code="too_few_parents",
        )
    if memo.pg.birth_year is not None:
        gi = memo.generation_interval()
        assert gi is not None  # birth years are present, so the prerequisite returns or raises
        return _hill_from(memo.pg, gi, memo.cohort_window(), memo.birth_year_family_table(), memo.hill_vk_scale)
    # Hill's collapse guards are Ne_V's own (`_hill_guards` reads them from
    # Ne_V's row) and have run under Hill's name, so it reads Ne_V's record
    # as a prerequisite: no second uniform-sex notice and no warning filter.
    return _hill_from_variance(memo.family_size_variance(), memo.hill_vk_scale)


@dataclass(frozen=True, slots=True)
class _Estimator:
    """What one estimator requires of a graph, and what it builds from the memo.

    Attributes:
        guards: The guards to run, in order, given the graph.  Ordered
            because a graph can fail several and the first to run names the
            refusal; a function of the graph because Hill's two branches
            require different metadata.
        build: Assembles the result from the per-call memo once the guards
            have passed.  Every prerequisite it reads is built at most once
            per call, for whichever estimator asks first.
    """

    guards: Callable[[PedigreeGraph], tuple[_Guard, ...]]
    build: Callable[[_Prerequisites], EffectiveSizeResult]


_REGISTRY: Mapping[str, _Estimator] = MappingProxyType(
    {
        "ne_inbreeding": _Estimator(
            _always(_LABELS),
            lambda memo: _inbreeding_from(memo.observed_cohorts(), memo.inbreeding()),
        ),
        "ne_coancestry": _Estimator(
            _always(_LABELS),
            lambda memo: _coancestry_from(memo.observed_cohorts(), memo.theta_summary()),
        ),
        "ne_variance_family_size": _Estimator(_always(_LABELS_AND_SEX), lambda memo: memo.family_size_variance()),
        "ne_sex_ratio": _Estimator(
            _always(_LABELS_AND_SEX),
            lambda memo: _sex_ratio_from(memo.observed_cohorts(), _sex_column(memo.pg)),
        ),
        "ne_individual_delta_f": _Estimator(
            _always(_LABELS),
            lambda memo: _individual_delta_f_from(
                memo.observed_cohorts(), memo.inbreeding(), memo.eqg(), memo.delta_f_reference
            ),
        ),
        "ne_long_term_contributions": _Estimator(
            _always((*_LABELS, _require_closed_parentage)),
            lambda memo: _ltc_from(memo.observed_cohorts(), memo.founder_means()),
        ),
        "ne_hill_overlapping": _Estimator(_hill_guards, _build_hill),
        "ne_group_coancestry": _Estimator(
            _always(_LABELS),
            lambda memo: _group_coancestry_from(memo.observed_cohorts(), memo.group_coancestry()),
        ),
    }
)
"""Every estimator's guards and build, keyed by name.

These eight rows are the whole of the dispatch: adding an estimator is one row
here and one name in :data:`ALL_EFFECTIVE_SIZE_ESTIMATORS`.
"""


class _Prerequisites:
    """Per-call memo of named prerequisites, and of the results built over them.

    The two namespaces are separate, so a prerequisite and an estimator may
    share a name without either shadowing the other.
    """

    __slots__ = ("_built", "_results", "delta_f_reference", "hill_vk_scale", "pg")

    def __init__(
        self, pg: PedigreeGraph, *, hill_vk_scale: bool = False, delta_f_reference: np.ndarray | None = None
    ) -> None:
        self.pg = pg
        self.hill_vk_scale = hill_vk_scale
        self.delta_f_reference = delta_f_reference
        self._built: dict[str, Any] = {}
        self._results: dict[str, EffectiveSizeResult] = {}

    def _prerequisite(self, name: str, build: Callable[[], Any]) -> Any:
        if name not in self._built:
            self._built[name] = build()
        return self._built[name]

    def computed(self) -> frozenset[str]:
        """Names built so far, prerequisites and results alike; what the closure tests spy on."""
        return frozenset(self._built) | frozenset(self._results)

    # Prerequisites.  Guards run in the estimator context, never here.

    def observed_cohorts(self) -> ObservedCohorts:
        labels = self.pg.generation_labels
        return self._prerequisite(
            "observed_cohorts", lambda: ObservedCohorts.from_labels(self.pg.depth if labels is None else labels)
        )

    def inbreeding(self) -> np.ndarray:
        return self._prerequisite("inbreeding", self.pg._inbreeding_values)

    def eqg(self) -> np.ndarray:
        return self._prerequisite(
            "eqg",
            lambda: _equivalent_generations(self.pg),
        )

    def theta_summary(self) -> GenerationKinshipSummary:
        return self._prerequisite("theta_summary", lambda: _generation_kinship_summary(self.pg))

    def represented_founders(self) -> np.ndarray:
        return self._prerequisite("represented_founders", lambda: _founder_idx(self.pg))

    def generation_family_table(self) -> FamilySizeTable:
        return self._prerequisite(
            "generation_family_table", lambda: _generation_family_table(self.pg, self.observed_cohorts())
        )

    def birth_year_family_table(self) -> FamilySizeTable:
        return self._prerequisite("birth_year_family_table", lambda: _birth_year_family_table(self.pg))

    def founder_means(self) -> FounderContributionMeans:
        return self._prerequisite(
            "founder_means",
            lambda: _per_gen_founder_means(
                self.pg, founder_idx=self.represented_founders(), cohorts=self.observed_cohorts()
            ),
        )

    def group_coancestry(self) -> GroupCoancestryByCohort:
        return self._prerequisite(
            "group_coancestry", lambda: _group_coancestry_by_cohort(self.pg, self.observed_cohorts())
        )

    def generation_interval(self) -> GenerationInterval | None:
        # The graph's cached_property is the real cache (repeated direct calls
        # compute it once per graph); this entry only records it in computed().
        return self._prerequisite("generation_interval", lambda: self.pg.generation_interval)

    def family_size_variance(self) -> NeVarianceResult:
        """Ne_V's record, built once per memo, guard-free: Ne_V's row and Hill's collapse both read it."""
        return self._prerequisite(
            "family_size_variance", lambda: _variance_from(self.observed_cohorts(), self.generation_family_table())
        )

    def cohort_window(self) -> CohortWindow:
        return self._prerequisite("cohort_window", lambda: eligible_cohort_range(self.pg))

    # Estimators, memoized by name once they complete.

    def result(self, name: str) -> EffectiveSizeResult:
        if name not in self._results:
            self._results[name] = self._compute(name)
        return self._results[name]

    def _compute(self, name: str) -> EffectiveSizeResult:
        """Run one estimator's guards in its own context, then its build."""
        estimator = _REGISTRY[name]
        for guard in estimator.guards(self.pg):
            guard(self.pg, name)
        return estimator.build(self)


def _run_one[R: EffectiveSizeResult](pg: PedigreeGraph, name: str, kind: type[R], **options: Any) -> R:
    """One estimator through its registry row, over a memo of its own: what each public ``ne_X`` is."""
    result = _Prerequisites(pg, **options).result(name)
    assert isinstance(result, kind)
    return result


def ne_inbreeding(pg: PedigreeGraph) -> NeInbreedingResult:
    """Inbreeding-rate Ne (Ne_I).

    Computes per-cohort mean F over the observed generation labels.
    Each adjacent observed-cohort transition reports the gap-corrected
    ``Ne = 1 / (2·ΔF)`` of :func:`~pedigree_graph._ne_common._transition_ne`;
    the aggregate Ne comes from the regression slope of ``ln(1 − F̄)`` on the
    label offset, first observed cohort excluded.
    """
    return _run_one(pg, "ne_inbreeding", NeInbreedingResult)


def ne_coancestry(pg: PedigreeGraph) -> NeCoancestryResult:
    """Coancestry-rate Ne (Ne_C).

    Same regression form as Ne_I but on the per-cohort mean kinship θ over
    within-cohort unordered pairs of distinct genomes (excluding the
    diagonal, and counting MZ co-twins once) that
    :meth:`PedigreeGraph.mean_kinship_by_generation` reports, which holds
    no kinship and runs in memory linear in the rows.
    """
    return _run_one(pg, "ne_coancestry", NeCoancestryResult)


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
    single-sex pedigree), with ``no_estimate_code="too_few_parents"``.
    """
    return _run_one(pg, "ne_variance_family_size", NeVarianceResult)


def ne_sex_ratio(pg: PedigreeGraph) -> NeSexRatioResult:
    """Wright sex-ratio Ne (Ne_sr).

    ``Ne_g = 4·Nm_g·Nf_g / (Nm_g + Nf_g)`` per observed cohort; aggregate is
    the harmonic mean across cohorts with both sexes present.

    Emits a ``RuntimeWarning`` when ``pg.sex`` is uniformly 0 or 1.  The
    estimator still returns ``ne=None`` in that case (consistent with a
    legitimate single-sex pedigree), with
    ``no_estimate_code="no_cohort_with_both_sexes"``.
    """
    return _run_one(pg, "ne_sex_ratio", NeSexRatioResult)


def ne_individual_delta_f(pg: PedigreeGraph, *, reference: object | None = None) -> NeIndividualDeltaFResult:
    """Gutiérrez et al. 2008 individual increase in inbreeding (Ne_iΔF).

    Gutiérrez, Cervantes, Molina, Valera and Goyache, *Individual increase in
    inbreeding allows estimating effective sizes from pedigrees*, Genet. Sel.
    Evol. 40(4):359-378, eq. 2::

        ΔF_i = 1 − (1 − F_i)^(1/t_i)

    where ``t_i`` is the individual's equivalent complete generations, the sum
    over its known ancestors of ``(1/2)^n`` for meiotic distance ``n``
    (:func:`_equivalent_generations`).  A row is eligible
    when ``t_i > 0``: a founder has ``t = 0`` and no rate.  Rows with
    ``F_i = 1`` are dropped as well, which is this package's guard and not the
    paper's — eq. 2 is finite there and would report ``ΔF_i = 1``.

    Their §2.1 averages ΔF_i over a **reference subpopulation** to give ΔF̄ and
    reports ``Ne = 1/(2·ΔF̄)``, with standard error
    ``σ_Ne = (2/√N)·Ne²·σ_ΔF`` over that subpopulation's ``N`` individuals.
    σ_ΔF is the sample standard deviation (``ddof=1``): the paper does not
    state a ddof, and the difference is O(1/N).

    Equivalent complete generations already count each individual's own
    pedigree depth, so labels only group the per-cohort series; each cohort is
    itself a valid reference subpopulation, which is what ``ne_per_gen``
    reports.

    The result also carries ``ne_unrelated_founders``, a package-defined
    diagnostic that no line of the paper contains, dividing by ``t_i − 1``
    over the reference rows with ``t_i > 1``.  Its justification is
    measurement, not theory.  Eq. 1 is ``F_t = 1 − (1 − ΔF)^t``, so eq. 2
    recovers the census size only where the pedigree has accumulated ``t``
    generations of drift, and founders that really are unrelated and
    non-inbred leave generation 1 at ``F = 0`` exactly, one generation behind.
    The paper's own remedy is choosing the reference subpopulation by pedigree
    depth (its Table II) and reading ΔF_i against equivalent generations (its
    Figs. 3-4), which is what ``reference=`` does here.  A real pedigree's
    founders are merely where record-keeping stopped and are generally
    related, so there the lag does not exist and the field biases downward.
    ``ne`` remains the estimator; see
    :class:`~pedigree_graph.effective_size.NeIndividualDeltaFResult`.

    Args:
        pg: Pedigree graph.
        reference: Graph rows of the reference subpopulation, unique and in
            range.  ``None`` selects the rows of the last observed cohort.

    Returns:
        The :class:`~pedigree_graph.effective_size.NeIndividualDeltaFResult`.

    Raises:
        PedigreeValidationError: ``invalid_shape``, ``invalid_integer_value``,
            ``reference_row_out_of_range``, or ``duplicate_reference_row`` for
            a *reference* the graph cannot accept.
        MissingMetadataError: ``missing_generation_labels`` when the supplied
            labels are partly ``-1``.
    """
    rows = None if reference is None else _reference_rows(pg, reference)
    return _run_one(pg, "ne_individual_delta_f", NeIndividualDeltaFResult, delta_f_reference=rows)


def ne_long_term_contributions(pg: PedigreeGraph) -> NeLTCResult:
    """Founder-genome long-term contribution Ne (Ne_LTC).

    Mean founder-genome contributions ``c_b[f] = mean_{i ∈ cohort b} c[i, f]``
    are taken over the observed generation labels, and both effective sizes
    are read from the **last observed cohort**, named by
    ``final_generation``.

    ``n_effective_founders = 1 / Σ_f c_f²`` is Caballero & Toro 2000
    (Genet. Res. 75(3):331-343) eq. 19, ``N_ef = 1/[(1/N²)Σc²_{i(0,t)}]``,
    in normalised contributions.  It is the assumption-free quantity: what
    the founder-contribution vector measures directly.

    ``ne = 2 · n_effective_founders`` is Wray & Thompson 1990
    (Genet. Res. 55(1):41-54) eq. 31, ``Ne ≈ 2N/(μ_r² + σ_r²)``, which with
    ``μ_r = 1`` gives ``Σr² = N·Σc²`` and so ``Ne = 2/Σc²``; C&T state the
    same link as ``N_ef = Ne/2`` in the text after their eq. 20.  It is the
    derived quantity, and it holds under two assumptions this package cannot
    check against a pedigree: **regular random mating (α = 0)**, and
    **long-term contribution variance at its asymptote**.

    ``max_delta_final`` reports how far the contributions still moved into
    the last cohort, which is what a caller weighs against the second
    assumption; ``asymptote_reached`` is that same movement against a fixed
    tolerance and gates neither estimate.

    Requires complete generation labels (or none), then closed represented
    parentage: a row with exactly one represented parent raises
    ``incomplete_parentage``.
    """
    return _run_one(pg, "ne_long_term_contributions", NeLTCResult)


def ne_hill_overlapping(pg: PedigreeGraph, *, vk_scale: bool = False) -> NeHillResult:
    """Hill 1979 separate-sex overlapping-generation Ne (Ne_H).

    Dispatches on ``pg.birth_year``:

    * **Sentinel branch** (``pg.birth_year is None``): the
      discrete-generation form of Hill 1979 reduces to Ne_V (Caballero
      1994 eq. 6 = Hill 1979 eq. 8 under symmetric sexes), so ``ne`` is
      the :func:`ne_variance_family_size` passthrough and
      ``collapses_to_ne_v=True``.  This branch inherits Ne_V's guards:
      complete generation labels (or none), then complete sex.
    * **Birth-year branch**: ignores generation labels; requires complete
      sex, then a qualifying known-age edge for each parent role
      (:attr:`PedigreeGraph.generation_interval` raises
      ``insufficient_parent_age_data`` otherwise).  Per eligible
      birth-year cohort, computes
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
    return _run_one(pg, "ne_hill_overlapping", NeHillResult, hill_vk_scale=vk_scale)


def ne_group_coancestry(pg: PedigreeGraph) -> NeGroupCoancestryResult:
    """Group-coancestry rate Ne (Ne_GC).

    Caballero & Toro, *Interrelations between effective population size and
    other pedigree tools for the management of conserved populations*,
    Genet. Res. 75(3):331-343, 2000, eq. 3::

        f̄ = Σ_i Σ_j a_ij / 2N²

    over all ``N²`` ordered pairs of a group, self-coancestries and
    reciprocals included, with ``a_ij = 2φ_ij`` off the diagonal and
    ``a_ii = 1 + F_i`` on it.  Each observed cohort is one such group,
    taken over the genome-node pedigree of ADR 0008: an MZ pair is one
    genome, not two.  The paper has no MZ twins, so it is silent there
    rather than contradicted.

    The reduction to a scalar is **not** their eq. 11 ``Ne ≈ t/(2Δf₀,ₜ)``,
    which the authors themselves flag as "only accurate if F̄ₖ₋₁ is small".
    It is the regression of ``ln(1 − f̄_g)`` on the label offset that
    :func:`~pedigree_graph.effective_size.ne_inbreeding` and
    :func:`~pedigree_graph.effective_size.ne_coancestry` share, whose exact
    cumulative form ``1 − x_t = (1 − Δ)^t`` is Gutiérrez et al. 2008
    (Genet. Sel. Evol. 40:359) eq. 1.

    The first observed cohort is the regression baseline, and its f̄ is
    computed like every other cohort's rather than assumed.  Where that
    cohort is unrelated and non-inbred the computed value is ``1/(2n)``
    exactly — the ``f̄₀ = 1/(2N)`` the paper states for unrelated founders —
    and the first entry of ``ne_per_gen`` is then their eq. 11
    ``Δf₀,₁ = (f̄₁ − f̄₀)/(1 − f̄₀)``.

    **The scalar assumes a constant census**, and ``census_ratio`` on the
    record is the evidence for or against that.
    ``f̄_g = θ̄_g·(n_g − 1)/n_g + s̄_g/n_g`` is an identity whose second term
    is a self-coancestry near ``0.5`` over the cohort size; that term does
    not accumulate at the drift rate, so ``ln(1 − f̄)`` reads a change in
    census as drift.  Eq. 11 carries the same assumption, fixing ``N`` at
    ``f̄₀ = 1/(2N)`` and holding it after, so this is faithful to the source
    rather than a defect in the implementation.  Measured over 6 cohorts and
    8 seeds against :func:`~pedigree_graph.effective_size.ne_coancestry`, which
    reads the same drift without the diagonal term: a constant census of 40
    gives ``Ne_GC/Ne_C = 1.00``, a decline from 40 to 4 gives ``0.57``,
    growth from 10 to 40 gives ``1.28``, and a lone trailing individual
    gives ``0.08`` — ``ne`` of 3.40 where ``ne_coancestry`` is unmoved at
    40.73.  The number is always reported; weigh it against
    ``census_ratio``.

    Requires complete generation labels, or none, in which case structural
    depth groups the cohorts.  It reads no founder contributions, so unlike
    :func:`~pedigree_graph.effective_size.ne_long_term_contributions` it does
    not require closed represented parentage.

    Args:
        pg: Pedigree graph.

    Returns:
        The :class:`~pedigree_graph._ne_results.NeGroupCoancestryResult`.

    Raises:
        MissingMetadataError: ``missing_generation_labels`` when the supplied
            labels are partly ``-1``.
    """
    return _run_one(pg, "ne_group_coancestry", NeGroupCoancestryResult)


def _selected_names(estimators: object) -> tuple[str, ...]:
    """Materialize and validate a selector; canonical order, duplicates collapsed."""
    if estimators is None or isinstance(estimators, str | bytes):
        raise TypeError("estimators must be an iterable of estimator names, not a string or None")
    names = tuple(estimators)  # ty: ignore[invalid-argument-type]
    for name in names:
        if not isinstance(name, str):
            raise TypeError(f"estimator names must be str, got {type(name).__name__}")
        if name not in ALL_EFFECTIVE_SIZE_ESTIMATORS:
            raise ValueError(f"unknown estimator {name!r}; choose from {ALL_EFFECTIVE_SIZE_ESTIMATORS}")
    selected = frozenset(names)
    return tuple(name for name in ALL_EFFECTIVE_SIZE_ESTIMATORS if name in selected)


def estimate_effective_sizes(
    pg: PedigreeGraph,
    estimators: Iterable[str] = ALL_EFFECTIVE_SIZE_ESTIMATORS,
    *,
    hill_vk_scale: bool = False,
) -> EffectiveSizeResults:
    """Run the selected Ne estimators on ``pg`` over one prerequisite memo.

    Args:
        pg: Pedigree graph.
        estimators: Names from :data:`ALL_EFFECTIVE_SIZE_ESTIMATORS`.  Any
            finite iterable is materialized and validated before any work;
            order and duplicates do not matter; empty selects nothing.
        hill_vk_scale: Forwarded to the Hill estimator as ``vk_scale`` (Waples
            2002 eq. 5 rescaling of ``Vk``); must be a ``bool``.

    Returns:
        An :class:`EffectiveSizeResults` with all eight keys in canonical
        order.  An unselected estimator maps to
        ``UnavailableEffectiveSize(reason="not_requested")``; a selected
        estimator that refused the pedigree with
        :class:`~pedigree_graph.MissingMetadataError` maps to
        ``reason="missing_metadata"`` with that error's code and fields.  Any
        other exception propagates.

    Raises:
        TypeError: for a ``None`` or string selector, a non-string name, or
            a non-``bool`` ``hill_vk_scale``.
        ValueError: for an unknown estimator name.
    """
    selected = _selected_names(estimators)
    if type(hill_vk_scale) is not bool:
        raise TypeError(f"hill_vk_scale must be a bool, got {type(hill_vk_scale).__name__}")
    thread_budget()
    prerequisites = _Prerequisites(pg, hill_vk_scale=hill_vk_scale)
    values: dict[str, EffectiveSizeResult | UnavailableEffectiveSize] = {}
    for name in selected:
        try:
            values[name] = prerequisites.result(name)
        except MissingMetadataError as error:
            values[name] = UnavailableEffectiveSize.from_error(error)
    return EffectiveSizeResults(
        (name, values.get(name, UnavailableEffectiveSize.not_requested())) for name in ALL_EFFECTIVE_SIZE_ESTIMATORS
    )
