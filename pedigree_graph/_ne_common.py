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
