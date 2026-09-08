"""``estimate_effective_sizes``: the eight estimators over one prerequisite memo.

The orchestrator runs the selected estimators serially in canonical order.
Each one first runs its metadata guards in its own context, so a failure
names that estimator, then pulls what it needs from a per-call memo of
named prerequisites (:class:`_Prerequisites`): the observed cohorts, F, the
generation kinship summary, the represented founders, the two family-size
tables, the founder means, the Caballero-Toro accumulators, the generation
interval, and the cohort window.  Each is built at most once per call and
only when a selected estimator asks for it, so an unselected estimator
costs nothing and a shared prerequisite is never built twice.  Completed
estimator results are memoized the same way, which is how Hill's
absent-birth-year collapse reuses a selected Ne_V result or computes one
privately while the public Ne_V key stays ``not_requested``.

There is no worker pool on this path (ADR 0007): the old pool dispatched
formulas only after eagerly building the expensive prerequisites, and
running the kinship, founder, and Caballero-Toro prerequisites concurrently
would multiply peak memory.  The package thread budget is committed once,
after selection is validated, and applies through the kernels the
prerequisites call.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from pedigree_graph._cohort_utils import eligible_cohort_range, generation_interval
from pedigree_graph._cohorts import ObservedCohorts
from pedigree_graph._errors import MissingMetadataError
from pedigree_graph._kinship_kernel import _compute_eqg
from pedigree_graph._ne_caballero_toro import _caballero_toro_accumulators, _caballero_toro_from, ne_caballero_toro
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
    NeCaballeroToroResult,
    NeCoancestryResult,
    NeHillResult,
    NeInbreedingResult,
    NeIndividualDeltaFResult,
    NeLTCResult,
    NeSexRatioResult,
    NeVarianceResult,
)
from pedigree_graph._threads import thread_budget

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

    from pedigree_graph._core import PedigreeGraph

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
    "ne_caballero_toro",
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
    | NeCaballeroToroResult
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
        "ne_caballero_toro": ne_caballero_toro,
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


class _Prerequisites:
    """Per-call memo of named prerequisite thunks and completed results."""

    __slots__ = ("_memo", "hill_vk_scale", "pg")

    def __init__(self, pg: PedigreeGraph, hill_vk_scale: bool) -> None:
        self.pg = pg
        self.hill_vk_scale = hill_vk_scale
        self._memo: dict[str, Any] = {}

    def _once(self, name: str, build: Callable[[], Any]) -> Any:
        if name not in self._memo:
            self._memo[name] = build()
        return self._memo[name]

    def computed(self) -> frozenset[str]:
        """Names built so far; what the closure tests spy on."""
        return frozenset(self._memo)

    # Prerequisites.  Guards run in the estimator context, never here.

    def observed_cohorts(self) -> ObservedCohorts:
        labels = self.pg.generation_labels
        return self._once(
            "observed_cohorts", lambda: ObservedCohorts.from_labels(self.pg.depth if labels is None else labels)
        )

    def inbreeding(self) -> np.ndarray:
        return self._once("inbreeding", self.pg._inbreeding_values)

    def theta_summary(self):
        return self._once("theta_summary", lambda: _generation_kinship_summary(self.pg))

    def represented_founders(self) -> np.ndarray:
        return self._once("represented_founders", lambda: _founder_idx(self.pg))

    def generation_family_table(self):
        return self._once("generation_family_table", lambda: _generation_family_table(self.pg, self.observed_cohorts()))

    def birth_year_family_table(self):
        return self._once("birth_year_family_table", lambda: _birth_year_family_table(self.pg))

    def founder_means(self):
        return self._once(
            "founder_means",
            lambda: _per_gen_founder_means(
                self.pg, founder_idx=self.represented_founders(), cohorts=self.observed_cohorts()
            ),
        )

    def ct_accumulators(self):
        return self._once(
            "ct_accumulators",
            lambda: _caballero_toro_accumulators(
                self.pg, self.represented_founders(), self.inbreeding(), cohorts=self.observed_cohorts()
            ),
        )

    def generation_interval(self):
        return self._once("generation_interval", lambda: generation_interval(self.pg))

    def cohort_window(self):
        return self._once("cohort_window", lambda: eligible_cohort_range(self.pg))

    # Estimators, memoized by name once they complete.

    def result(self, name: str) -> EffectiveSizeResult:
        return self._once(name, lambda: self._compute(name))

    def _compute(self, name: str) -> EffectiveSizeResult:
        pg = self.pg
        if pg.n_individuals == 0:
            if name == "ne_hill_overlapping":
                return ne_hill_overlapping(pg, vk_scale=self.hill_vk_scale)
            return _DIRECT[name](pg)
        if name == "ne_hill_overlapping" and pg.birth_year is not None:
            _require_complete_sex(pg, name)
            gi = self.generation_interval()
            return _hill_from(pg, gi, self.cohort_window(), self.birth_year_family_table(), self.hill_vk_scale)
        _require_complete_generation_labels(pg, name)
        if name in ("ne_variance_family_size", "ne_sex_ratio", "ne_hill_overlapping"):
            _require_complete_sex(pg, name)
            _warn_if_uniform_sex(pg, name)
        if name in ("ne_long_term_contributions", "ne_caballero_toro"):
            _require_closed_parentage(pg, name)
        cohorts = self.observed_cohorts()
        if name == "ne_inbreeding":
            return _inbreeding_from(cohorts, self.inbreeding())
        if name == "ne_coancestry":
            return _coancestry_from(cohorts, self.theta_summary())
        if name == "ne_variance_family_size":
            return _variance_from(cohorts, self.generation_family_table())
        if name == "ne_sex_ratio":
            return _sex_ratio_from(cohorts, _sex_column(pg))
        if name == "ne_individual_delta_f":
            eqg = _compute_eqg(np.asarray(pg.mother), np.asarray(pg.father), pg.n_individuals)
            return _individual_delta_f_from(cohorts, self.inbreeding(), eqg)
        if name == "ne_long_term_contributions":
            return _ltc_from(cohorts, self.founder_means(), 1e-6)
        if name == "ne_hill_overlapping":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                variance = self.result("ne_variance_family_size")
            assert isinstance(variance, NeVarianceResult)
            return _hill_from_variance(variance, self.hill_vk_scale)
        if name == "ne_caballero_toro":
            return _caballero_toro_from(cohorts, self.ct_accumulators())
        raise KeyError(name)


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
