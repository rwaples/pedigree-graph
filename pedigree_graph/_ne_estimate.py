"""``estimate_effective_sizes``: the eight estimators over one prerequisite memo.

The orchestrator runs the selected estimators serially in canonical order.
:data:`_REGISTRY` gives each one a row naming its metadata guards and its
build, so what an estimator requires and what it assembles are read in one
place rather than reconstructed from a dispatch chain.

Each estimator first runs its guards in its own context, so a failure names
that estimator, then pulls what it needs from a per-call memo of named
prerequisites (:class:`_Prerequisites`): the observed cohorts, F, the
equivalent complete generations, the generation kinship summary, the
represented founders, the two family-size tables, the founder means, the
per-cohort group coancestry, the generation interval, and the cohort
window.  Each is built at most once per call and only when a selected
estimator asks for it, so an unselected estimator costs nothing and a
shared prerequisite is never built twice.  Completed estimator results are
memoized beside them in their own namespace, which is how Hill's
absent-birth-year collapse reuses a selected Ne_V result or computes one
privately while the public Ne_V key stays ``not_requested``.

An estimator is reachable directly as well, and the standalone function
wires its own guards and prerequisites.  The two wirings agree on every
degenerate graph measured, which
``tests/test_estimate_effective_sizes.py::TestPathEquivalence`` holds them
to.

There is no worker pool on this path (ADR 0007): the old pool dispatched
formulas only after eagerly building the expensive prerequisites, and
running the kinship and founder prerequisites concurrently would multiply
peak memory.  The package thread budget is committed once, after selection
is validated, and applies through the kernels the prerequisites call.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from pedigree_graph._cohort_utils import eligible_cohort_range, generation_interval
from pedigree_graph._cohorts import ObservedCohorts
from pedigree_graph._errors import MissingMetadataError
from pedigree_graph._kinship_depth import _compute_eqg
from pedigree_graph._ne_family_size import (
    _generation_family_table,
    _sex_column,
    _sex_ratio_from,
    _variance_from,
    _warn_if_uniform_sex,
    ne_sex_ratio,
    ne_variance_family_size,
)
from pedigree_graph._ne_founders import _founder_idx, _ltc_from, _per_gen_founder_means, ne_long_term_contributions
from pedigree_graph._ne_group_coancestry import _group_coancestry_by_cohort, _group_coancestry_from, ne_group_coancestry
from pedigree_graph._ne_hill import _birth_year_family_table, _hill_from, _hill_from_variance, ne_hill_overlapping
from pedigree_graph._ne_metadata import (
    _require_closed_parentage,
    _require_complete_generation_labels,
    _require_complete_sex,
)
from pedigree_graph._ne_rates import (
    _coancestry_from,
    _generation_kinship_summary,
    _inbreeding_from,
    _individual_delta_f_from,
    ne_coancestry,
    ne_inbreeding,
    ne_individual_delta_f,
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

_DIRECT: Mapping[str, Callable[..., EffectiveSizeResult]] = MappingProxyType(
    {
        "ne_inbreeding": ne_inbreeding,
        "ne_coancestry": ne_coancestry,
        "ne_variance_family_size": ne_variance_family_size,
        "ne_sex_ratio": ne_sex_ratio,
        "ne_individual_delta_f": ne_individual_delta_f,
        "ne_long_term_contributions": ne_long_term_contributions,
        "ne_hill_overlapping": ne_hill_overlapping,
        "ne_group_coancestry": ne_group_coancestry,
    }
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
    branch is Ne_V, and inherits Ne_V's requirements.
    """
    if pg.birth_year is not None:
        return (_require_complete_sex,)
    return _LABELS_AND_SEX


def _build_hill(memo: _Prerequisites) -> NeHillResult:
    """Hill's two branches over the shared memo.

    The collapse branch reaches Ne_V through the memo rather than recomputing
    it, which is how a selection holding both estimators pays for one variance
    table while Hill alone leaves the public Ne_V key ``not_requested``.
    """
    if memo.pg.birth_year is not None:
        gi = memo.generation_interval()
        assert gi is not None  # birth years are present, so the prerequisite returns or raises
        return _hill_from(memo.pg, gi, memo.cohort_window(), memo.birth_year_family_table(), memo.hill_vk_scale)
    with warnings.catch_warnings():
        # Only the duplicate uniform-sex notice from the nested call; the
        # caller already had it under this estimator's own name.
        warnings.filterwarnings(
            "ignore",
            message=r"ne_variance_family_size: pg\.sex is uniform",
            category=RuntimeWarning,
        )
        variance = memo.result("ne_variance_family_size")
    assert isinstance(variance, NeVarianceResult)
    return _hill_from_variance(variance, memo.hill_vk_scale)


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
        "ne_variance_family_size": _Estimator(
            _always(_LABELS_AND_SEX),
            lambda memo: _variance_from(memo.observed_cohorts(), memo.generation_family_table()),
        ),
        "ne_sex_ratio": _Estimator(
            _always(_LABELS_AND_SEX),
            lambda memo: _sex_ratio_from(memo.observed_cohorts(), _sex_column(memo.pg)),
        ),
        "ne_individual_delta_f": _Estimator(
            _always(_LABELS),
            lambda memo: _individual_delta_f_from(memo.observed_cohorts(), memo.inbreeding(), memo.eqg()),
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

    __slots__ = ("_built", "_results", "hill_vk_scale", "pg")

    def __init__(self, pg: PedigreeGraph, hill_vk_scale: bool) -> None:
        self.pg = pg
        self.hill_vk_scale = hill_vk_scale
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
            lambda: _compute_eqg(
                np.asarray(self.pg.mother_rows),
                np.asarray(self.pg.father_rows),
                np.asarray(self.pg.depth),
                self.pg.n_individuals,
            ),
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
        return self._prerequisite("generation_interval", lambda: generation_interval(self.pg))

    def cohort_window(self) -> CohortWindow:
        return self._prerequisite("cohort_window", lambda: eligible_cohort_range(self.pg))

    # Estimators, memoized by name once they complete.

    def result(self, name: str) -> EffectiveSizeResult:
        if name not in self._results:
            self._results[name] = self._compute(name)
        return self._results[name]

    def _compute(self, name: str) -> EffectiveSizeResult:
        """Run one estimator's guards in its own context, then its build."""
        if self.pg.n_individuals == 0:
            # Every estimator's no-estimate case, and Hill alone carries the
            # requested vk_scale into the record it returns for one.
            if name == "ne_hill_overlapping":
                return ne_hill_overlapping(self.pg, vk_scale=self.hill_vk_scale)
            return _DIRECT[name](self.pg)
        estimator = _REGISTRY[name]
        for guard in estimator.guards(self.pg):
            guard(self.pg, name)
        return estimator.build(self)


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
    prerequisites = _Prerequisites(pg, hill_vk_scale)
    values: dict[str, EffectiveSizeResult | UnavailableEffectiveSize] = {}
    for name in selected:
        try:
            values[name] = prerequisites.result(name)
        except MissingMetadataError as error:
            values[name] = UnavailableEffectiveSize.from_error(error)
    return EffectiveSizeResults(
        (name, values.get(name, UnavailableEffectiveSize.not_requested())) for name in ALL_EFFECTIVE_SIZE_ESTIMATORS
    )
