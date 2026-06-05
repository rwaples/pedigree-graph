"""Shared numeric helpers for the Ne estimators (PGQ-006).

Pure functions used by more than one estimator module: the harmonic-mean
aggregator (variance / sex-ratio / individual-ΔF / Hill) and the
``ln(1 − x)`` OLS used by the rate-based estimators (inbreeding,
coancestry, Caballero-Toro).
"""

from __future__ import annotations

import numpy as np


def _harmonic_mean(values: np.ndarray) -> float:
    """Harmonic mean over finite, strictly positive entries; ``nan`` if none."""
    finite = np.isfinite(values) & (values > 0)
    if not finite.any():
        return float("nan")
    return float(finite.sum() / np.sum(1.0 / values[finite]))


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


def _scalar_ne_from_log_regression(series: np.ndarray) -> tuple[float | None, float, int]:
    """Aggregate Ne from the OLS slope of ``ln(1 − series_t)`` on t ≥ 1.

    The three rate-based estimators reduce a per-generation mean series — F̄
    (inbreeding), θ̄ (coancestry), or self-coancestry f̄_s (Caballero-Toro) —
    to a scalar Ne the same way: drop the founder cohort (index 0), regress
    ``ln(1 − series)`` on the generation index, and report
    ``Ne = −1 / (2·slope)`` when the slope is finite and negative (a rising
    series ⇒ negative slope ⇒ positive Ne).

    Args:
        series: per-generation mean series indexed by generation, founders
            at index 0.

    Returns:
        ``(ne, slope, n_generations_used)`` — ``ne`` is ``None`` when the
        slope is non-finite or non-negative; ``n_generations_used`` counts
        the post-founder cohorts contributing a finite ``ln(1 − series)``
        term to the fit.
    """
    post_founder = series[1:]
    t = np.arange(1, len(series), dtype=np.float64)
    slope, _ = _regress_log_one_minus(post_founder, t)
    ne = -1.0 / (2.0 * slope) if np.isfinite(slope) and slope < 0 else None
    n_used = int(np.isfinite(np.log1p(-post_founder)).sum())
    return ne, slope, n_used
