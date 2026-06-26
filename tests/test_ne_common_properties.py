"""Property-based tests for the shared Ne numeric helpers in _ne_common.

These pure functions back five estimators (Ne_I / Ne_C / Ne_CT via the log
regression, and Ne_V / Ne_sr / Ne_iΔF / Ne_H via the harmonic mean), so a
defect here biases several estimators at once.  The log-regression reducer is
checked by *planted-signal recovery*: build a series whose true slope (hence
Ne) is known exactly and assert it is recovered — an oracle that inverts the
routine rather than re-implementing it.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pedigree_graph._ne_common import (
    _harmonic_mean,
    _regress_log_one_minus,
    _scalar_ne_from_log_regression,
)

_SETTINGS = settings(deadline=None, max_examples=100)

_positive = st.floats(min_value=1e-3, max_value=1e6, allow_nan=False, allow_infinity=False)
_junk = st.sampled_from([0.0, -1.0, -1e9, np.nan, np.inf, -np.inf])


@_SETTINGS
@given(
    pos=st.lists(_positive, min_size=1, max_size=12),
    junk=st.lists(_junk, max_size=4),
    data=st.data(),
)
def test_harmonic_mean_bounded_by_am_and_extremes(pos, junk, data):
    # HM is computed over the finite, strictly-positive entries only; the junk
    # (zeros, negatives, non-finite) must be ignored, not poison the result.
    vals = np.array(data.draw(st.permutations(pos + junk)), dtype=np.float64)
    hm = _harmonic_mean(vals)
    p = np.array(pos, dtype=np.float64)
    assert hm <= p.mean() + 1e-9  # AM >= HM
    assert p.min() - 1e-9 <= hm <= p.max() + 1e-9  # min <= HM <= max


@_SETTINGS
@given(pos=st.lists(_positive, min_size=1, max_size=12), alpha=st.floats(0.25, 8.0))
def test_harmonic_mean_scale_equivariant(pos, alpha):
    v = np.array(pos, dtype=np.float64)
    scaled = _harmonic_mean(alpha * v)
    assert scaled == pytest.approx(alpha * _harmonic_mean(v), rel=1e-9)


@_SETTINGS
@given(junk=st.lists(_junk, min_size=1, max_size=6))
def test_harmonic_mean_nan_when_no_positive(junk):
    assert np.isnan(_harmonic_mean(np.array(junk, dtype=np.float64)))


@_SETTINGS
@given(c=_positive, k=st.integers(min_value=1, max_value=10))
def test_harmonic_mean_of_constant_is_constant(c, k):
    assert _harmonic_mean(np.full(k, c)) == pytest.approx(c, rel=1e-9)


@_SETTINGS
@given(ne_true=st.floats(1.0, 500.0), g_max=st.integers(min_value=2, max_value=12))
def test_scalar_ne_recovers_planted_value(ne_true, g_max):
    # Plant a per-generation series with an exactly linear ln(1 - series_t):
    # series_t = 1 - exp(-t / (2 Ne)) gives slope = -1/(2 Ne), so the reducer's
    # Ne = -1/(2 slope) must recover Ne exactly (collinear -> perfect OLS).
    t = np.arange(g_max + 1, dtype=np.float64)
    series = 1.0 - np.exp(-t / (2.0 * ne_true))  # series[0] == 0 (founder, dropped)
    ne_rec, slope, n_used = _scalar_ne_from_log_regression(series)
    assert ne_rec is not None
    assert ne_rec == pytest.approx(ne_true, rel=1e-6)
    assert slope == pytest.approx(-1.0 / (2.0 * ne_true), rel=1e-6)
    assert n_used == g_max  # every post-founder cohort contributed


@_SETTINGS
@given(
    slope=st.floats(-0.5, -0.01),
    intercept=st.floats(-0.3, 0.0),
    g_max=st.integers(min_value=3, max_value=12),
)
def test_regress_log_one_minus_recovers_line(slope, intercept, g_max):
    t = np.arange(1, g_max + 1, dtype=np.float64)
    values = 1.0 - np.exp(slope * t + intercept)  # ln(1 - values) == slope*t + intercept
    s, b = _regress_log_one_minus(values, t)
    assert s == pytest.approx(slope, abs=1e-9)
    assert b == pytest.approx(intercept, abs=1e-9)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")  # log1p(<=-1) in the n_used count
def test_scalar_ne_none_when_degenerate():
    # Fewer than two post-founder points, or all post-founder values saturated
    # at >= 1 (log diverges), leave the slope undefined -> ne is None.
    assert _scalar_ne_from_log_regression(np.array([0.0]))[0] is None
    assert _scalar_ne_from_log_regression(np.array([0.0, 0.1]))[0] is None
    assert _scalar_ne_from_log_regression(np.array([0.0, 1.0, 1.5]))[0] is None


def test_regress_log_one_minus_nan_when_underdetermined():
    # < 2 finite points, and values >= 1 dropped, both yield (nan, nan).
    s, b = _regress_log_one_minus(np.array([0.5]), np.array([1.0]))
    assert np.isnan(s)
    assert np.isnan(b)
    s, b = _regress_log_one_minus(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
    assert np.isnan(s)
    assert np.isnan(b)
