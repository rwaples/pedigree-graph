"""0.8.0-DELETE: the 0.7.1 effective-size result records and their dense fill.

The 0.7.1 records index every per-generation array by raw label value,
``0 .. max(label)``, with a gap slot for every label nobody carries.  The
final records (:mod:`pedigree_graph._ne_results`) index by observed cohort
instead.  The converters here scatter a final record back onto the dense
layout so the package-root estimators and :func:`compute_all_ne` keep the
0.7.1 field schemas until slice 7 deletes them.

Fill values follow the 0.7.1 arrays: mean-F gaps ``0.0`` (a cohort nobody
occupies had mean F ``0``), every other statistic ``NaN``, counts ``0``.
Rate transitions land at their ``transition_to`` label.  The variance
arrays keep the 0.7.1 length ``max(label)``, so the entry the final record
carries for the maximum label is necessarily dropped.  Corrected
sparse-transition values are intentional: the layout, not the old sparse
arithmetic, is the compatibility promise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from pedigree_graph._ne_results import GenerationInterval as GenerationInterval
from pedigree_graph._ne_results import NeHillResult as NeHillResult
from pedigree_graph._ne_results import _SerializableResult

if TYPE_CHECKING:
    from pedigree_graph._ne_results import NeCaballeroToroResult as _FinalCaballeroToro
    from pedigree_graph._ne_results import NeCoancestryResult as _FinalCoancestry
    from pedigree_graph._ne_results import NeInbreedingResult as _FinalInbreeding
    from pedigree_graph._ne_results import NeIndividualDeltaFResult as _FinalIndividualDeltaF
    from pedigree_graph._ne_results import NeLTCResult as _FinalLTC
    from pedigree_graph._ne_results import NeSexRatioResult as _FinalSexRatio
    from pedigree_graph._ne_results import NeVarianceResult as _FinalVariance


@dataclass(frozen=True, slots=True)
class NeInbreedingResult(_SerializableResult):
    """0.7.1 inbreeding-rate (Ne_I) record, dense over ``0 .. max(label)``."""

    ne: float | None
    ne_per_gen: np.ndarray
    mean_f_per_gen: np.ndarray
    slope: float
    n_generations_used: int


@dataclass(frozen=True, slots=True)
class NeCoancestryResult(_SerializableResult):
    """0.7.1 coancestry-rate (Ne_C) record, dense over ``0 .. max(label)``."""

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
    """0.7.1 variance-of-family-size (Ne_V) record, indexed by parent label ``< max(label)``."""

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
    """0.7.1 sex-ratio (Ne_sr) record, dense over ``0 .. max(label)``."""

    ne: float | None
    ne_per_gen: np.ndarray
    n_male_per_gen: np.ndarray
    n_female_per_gen: np.ndarray


@dataclass(frozen=True, slots=True)
class NeIndividualDeltaFResult(_SerializableResult):
    """0.7.1 individual-ΔF (Ne_iΔF) record, dense over ``0 .. max(label)``."""

    ne: float | None
    ne_per_gen: np.ndarray
    mean_eqg_per_gen: np.ndarray
    n_used_per_gen: np.ndarray


@dataclass(frozen=True, slots=True)
class NeLTCResult(_SerializableResult):
    """0.7.1 long-term contribution (Ne_LTC) record.

    ``n_iterations`` is the 0.7.1 meaning: the label of the cohort whose
    contribution vector produced ``sum_c_squared`` (``0`` when there was
    none), which under dense labels is also the number of comparisons.
    """

    ne: float | None
    asymptote_reached: bool
    n_iterations: int
    max_delta_final: float
    sum_c_squared: float


@dataclass(frozen=True, slots=True)
class NeCaballeroToroResult(_SerializableResult):
    """0.7.1 Caballero-Toro (Ne_CT) record, dense over ``0 .. max(label)``."""

    ne: float | None
    ne_per_gen: np.ndarray
    mean_self_coancestry_per_gen: np.ndarray
    n_founders_with_descendants_per_gen: np.ndarray
    slope: float


def _dense_length(labels: np.ndarray) -> int:
    return int(labels.max()) + 1 if labels.shape[0] else 0


def _scatter(values: np.ndarray, labels: np.ndarray, length: int, fill: float | int, dtype: type) -> np.ndarray:
    """Place ``values[i]`` at slot ``labels[i]`` of a ``fill``-initialised array."""
    out = np.full(length, fill, dtype=dtype)
    keep = labels < length
    out[labels[keep]] = values[keep]
    return out


def legacy_inbreeding(res: _FinalInbreeding) -> NeInbreedingResult:
    length = _dense_length(res.generations)
    return NeInbreedingResult(
        ne=res.ne,
        ne_per_gen=_scatter(res.ne_per_gen, res.transition_to, length, np.nan, np.float64),
        mean_f_per_gen=_scatter(res.mean_f_per_gen, res.generations, length, 0.0, np.float64),
        slope=res.slope,
        n_generations_used=res.n_generations_used,
    )


def legacy_coancestry(res: _FinalCoancestry) -> NeCoancestryResult:
    length = _dense_length(res.generations)
    return NeCoancestryResult(
        ne=res.ne,
        ne_per_gen=_scatter(res.ne_per_gen, res.transition_to, length, np.nan, np.float64),
        mean_theta_per_gen=_scatter(res.mean_theta_per_gen, res.generations, length, np.nan, np.float64),
        slope=res.slope,
        n_generations_used=res.n_generations_used,
    )


def legacy_variance(res: _FinalVariance) -> NeVarianceResult:
    length = max(_dense_length(res.parent_generations) - 1, 0)

    def dense(values: np.ndarray) -> np.ndarray:
        return _scatter(values, res.parent_generations, length, np.nan, np.float64)

    return NeVarianceResult(
        ne=res.ne,
        ne_per_transition=dense(res.ne_per_transition),
        v_mm=dense(res.v_mm),
        v_mf=dense(res.v_mf),
        v_fm=dense(res.v_fm),
        v_ff=dense(res.v_ff),
        cov_m=dense(res.cov_m),
        cov_f=dense(res.cov_f),
    )


def legacy_sex_ratio(res: _FinalSexRatio) -> NeSexRatioResult:
    length = _dense_length(res.generations)
    return NeSexRatioResult(
        ne=res.ne,
        ne_per_gen=_scatter(res.ne_per_gen, res.generations, length, np.nan, np.float64),
        n_male_per_gen=_scatter(res.n_male_per_gen, res.generations, length, 0, np.int64),
        n_female_per_gen=_scatter(res.n_female_per_gen, res.generations, length, 0, np.int64),
    )


def legacy_individual_delta_f(res: _FinalIndividualDeltaF) -> NeIndividualDeltaFResult:
    length = _dense_length(res.generations)
    return NeIndividualDeltaFResult(
        ne=res.ne,
        ne_per_gen=_scatter(res.ne_per_gen, res.generations, length, np.nan, np.float64),
        mean_eqg_per_gen=_scatter(res.mean_eqg_per_gen, res.generations, length, np.nan, np.float64),
        n_used_per_gen=_scatter(res.n_used_per_gen, res.generations, length, 0, np.int64),
    )


def legacy_ltc(res: _FinalLTC) -> NeLTCResult:
    return NeLTCResult(
        ne=res.ne,
        asymptote_reached=res.asymptote_reached,
        n_iterations=0 if res.final_generation is None else int(res.final_generation),
        max_delta_final=res.max_delta_final,
        sum_c_squared=res.sum_c_squared,
    )


def legacy_caballero_toro(res: _FinalCaballeroToro) -> NeCaballeroToroResult:
    length = _dense_length(res.generations)
    return NeCaballeroToroResult(
        ne=res.ne,
        ne_per_gen=_scatter(res.ne_per_gen, res.transition_to, length, np.nan, np.float64),
        mean_self_coancestry_per_gen=_scatter(
            res.mean_self_coancestry_per_gen, res.generations, length, np.nan, np.float64
        ),
        n_founders_with_descendants_per_gen=_scatter(
            res.n_founders_with_descendants_per_gen, res.generations, length, 0, np.int64
        ),
        slope=res.slope,
    )
