"""Shared numeric helpers for the Ne estimators (PGQ-006).

Pure functions used by more than one estimator module: the harmonic-mean
aggregator (variance / sex-ratio / individual-ΔF / Hill), the adjacent
observed-cohort rate and the ``ln(1 − x)`` OLS used by the rate-based
estimators (inbreeding, coancestry, Caballero-Toro), and the checked
``(k, n_founder_genomes)`` allocation the founder-based estimators share.
"""

from __future__ import annotations

import math

import numpy as np

from pedigree_graph._errors import ResourceError
from pedigree_graph._ne_metadata import _require_complete_generation_labels as _require_complete_generation_labels


def _harmonic_mean(values: np.ndarray) -> float:
    """Harmonic mean over finite, strictly positive entries; ``nan`` if none."""
    finite = np.isfinite(values) & (values > 0)
    if not finite.any():
        return float("nan")
    return float(finite.sum() / np.sum(1.0 / values[finite]))


def _transition_ne(x: np.ndarray, generations: np.ndarray) -> np.ndarray:
    """Ne for each adjacent observed-cohort transition of a cumulative series.

    ``x`` is a per-cohort mean of a quantity that accumulates like inbreeding
    (F̄, θ̄, or self-coancestry f̄_s) and ``generations`` the observed labels.
    For cohorts ``a < b`` separated by ``h = generations[b] − generations[a]``
    the per-generation rate follows the cumulative recurrence
    ``1 − x_t = (1 − Δ)^t`` of Gutiérrez et al. 2008 (Genet. Sel. Evol.
    40:359, eqs. 1–2)::

        Δ = 1 − ((1 − x_b) / (1 − x_a)) ** (1 / h)

    At ``h = 1`` this is the one-step ``(x_b − x_a) / (1 − x_a)`` and is
    evaluated exactly that way; for ``h > 1`` it is evaluated in log space
    so tiny rates over long gaps keep their precision.  A transition is NaN
    unless both means are finite and below 1, the gap is positive, and the
    rate is positive; otherwise ``Ne = 1 / (2Δ)``.

    Returns:
        float64 of length ``max(len(x) − 1, 0)``, entry ``i`` describing the
        transition ``generations[i] → generations[i + 1]``.
    """
    k = int(x.shape[0])
    out = np.full(max(k - 1, 0), np.nan, dtype=np.float64)
    for i in range(k - 1):
        a = float(x[i])
        b = float(x[i + 1])
        h = int(generations[i + 1]) - int(generations[i])
        if not (math.isfinite(a) and math.isfinite(b) and a < 1.0 and b < 1.0 and h > 0):
            continue
        if h == 1:
            delta = (b - a) / (1.0 - a)
        else:
            delta = -math.expm1((math.log1p(-b) - math.log1p(-a)) / h)
        if delta > 0:
            out[i] = 1.0 / (2.0 * delta)
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


def _scalar_ne_from_log_regression(series: np.ndarray, generations: np.ndarray) -> tuple[float | None, float, int]:
    """Aggregate Ne from the OLS slope of ``ln(1 − series)`` on the label offset.

    The three rate-based estimators reduce a per-cohort mean series — F̄
    (inbreeding), θ̄ (coancestry), or self-coancestry f̄_s (Caballero-Toro) —
    to a scalar Ne the same way: drop the first observed cohort (the
    baseline), regress ``ln(1 − series)`` on ``generations − generations[0]``,
    and report ``Ne = −1 / (2·slope)`` when the slope is finite and negative
    (a rising series ⇒ negative slope ⇒ positive Ne).  Adding a constant to
    every label changes nothing.

    Args:
        series: per-cohort mean series aligned with ``generations``.
        generations: observed labels, ascending.

    Returns:
        ``(ne, slope, n_generations_used)`` — ``ne`` is ``None`` when the
        slope is non-finite or non-negative; ``n_generations_used`` counts
        the post-baseline cohorts contributing a finite ``ln(1 − series)``
        term to the fit.
    """
    post_baseline = series[1:]
    t = (generations[1:] - generations[:1]).astype(np.float64) if series.shape[0] else np.empty(0, dtype=np.float64)
    slope, _ = _regress_log_one_minus(post_baseline, t)
    ne = -1.0 / (2.0 * slope) if np.isfinite(slope) and slope < 0 else None
    n_used = int(np.isfinite(np.log1p(-post_baseline)).sum())
    return ne, slope, n_used


def _checked_founder_matrix(k: int, n_founders: int, operation: str, dtype: type, fill: float | int) -> np.ndarray:
    """Allocate a ``(k, n_founders)`` matrix behind the structured resource errors.

    ``k * n_founders`` is formed in Python integers first, so a shape that no
    platform index can hold raises ``arithmetic_overflow`` rather than
    wrapping, and an allocation the host refuses raises ``allocation_failed``
    carrying the element count and dtype the caller asked for.
    """
    requested = int(k) * int(n_founders)
    if requested > np.iinfo(np.intp).max:
        raise ResourceError(
            "arithmetic_overflow",
            f"{operation}: {k} cohorts x {n_founders} founder genomes exceeds the platform index range",
            operation=operation,
            dtype=np.dtype(dtype).name,
        )
    try:
        return np.full((k, n_founders), fill, dtype=dtype)
    except MemoryError as exc:
        raise ResourceError(
            "allocation_failed",
            f"{operation}: could not allocate {requested} {np.dtype(dtype).name} elements",
            operation=operation,
            requested_elements=requested,
            dtype=np.dtype(dtype).name,
        ) from exc
