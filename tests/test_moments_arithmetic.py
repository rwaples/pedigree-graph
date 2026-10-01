"""Core's moments boundary arithmetic against the 0.11 Python implementation, bit for bit (ADR 0015).

``oracle.moments_arithmetic`` is the packing, quantization, derivation, merge
and fold Python ran before core took them over.  Every comparison here is
exact: labels and levels equal, quantized integers and exponents equal,
exact table integers equal, and every derived float equal in its bits (NaN
where NaN), or both sides refusing.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from _support import _run_child
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from oracle import moments_arithmetic as oracle
from oracle.moments_arithmetic import OracleTable

from pedigree_graph import _native
from pedigree_graph._errors import ResourceError
from pedigree_graph.moments import _decode_exact, _encode_exact

STATISTICS = (
    "sum_first",
    "sum_second",
    "sumsq_first",
    "sumsq_second",
    "cross",
    "m2_first",
    "m2_second",
    "comoment",
    "mean_first",
    "mean_second",
    "pearson",
)
PER_PRODUCT = {"cross", "comoment", "pearson"}
OPERANDS = ((0, 0, 1, 0), (0, 1, 1, 1), (0, 0, 0, 1), (1, 1, 0, 0))

FAST = settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])


def _bits(values: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(values, dtype=np.float64).view(np.uint64)


# ---------------------------------------------------------------------------
# Scalars: division, exponent, quantization, packing
# ---------------------------------------------------------------------------


def _python_ratio(numerator: int, denominator: int, shift: int) -> float | None:
    try:
        return oracle.divide_exact(numerator, denominator, shift)
    except OverflowError:
        return None


def _same_float(got: float | None, want: float | None) -> bool:
    if got is None or want is None:
        return got is want
    return np.float64(got).view(np.uint64) == np.float64(want).view(np.uint64)


@FAST
@given(
    numerator=st.integers(-(2**300), 2**300),
    denominator=st.integers(1, 2**200),
    shift=st.integers(-2300, 2300),
)
def test_ratio_is_python_true_division(numerator, denominator, shift):
    got = _native.moments_ratio(str(numerator), str(denominator), shift)
    assert _same_float(got, _python_ratio(numerator, denominator, shift))


@FAST
@given(
    significand=st.integers(2**52, 2**53 - 1),
    low=st.integers(1, 80),
    shift=st.integers(-1200, 1150),
    negative=st.booleans(),
)
def test_ratio_rounds_half_ulp_ties_to_even(significand, low, shift, negative):
    # (2q + 1) · 2^low over 2^(low + 1) is q + 1/2 exactly: a tie at every scale.
    numerator = (2 * significand + 1) << low
    numerator = -numerator if negative else numerator
    got = _native.moments_ratio(str(numerator), "1", low + 1 + shift)
    assert _same_float(got, _python_ratio(numerator, 1, low + 1 + shift))


@pytest.mark.parametrize(
    ("numerator", "shift"),
    [
        (1, 1074),  # the smallest subnormal
        (1, 1075),  # half of it: a tie to zero
        (3, 1076),  # three quarters: up to the smallest subnormal
        (-1, 1080),  # an underflow keeps the sign
        (2**1024 - 2**970, 0),  # the tie between the largest float and 2^1024
        (2**1024 - 2**970 - 1, 0),  # just below it
        (2**53 - 1, -971),  # the largest float
        (2**53, -971),  # 2^1024
        (0, -5000),
    ],
)
def test_ratio_edges(numerator, shift):
    assert _same_float(_native.moments_ratio(str(numerator), "1", shift), _python_ratio(numerator, 1, shift))


FINITE = st.floats(allow_nan=False, allow_infinity=False)
REGIMES = st.sampled_from(["ordinary", "huge", "tiny", "subnormal", "near_max", "zero", "constant", "ties"])


@st.composite
def columns(draw, n=None):
    """A float64 column from one regime: magnitudes, subnormals, the float64 edge, constants, quantization ties."""
    n = draw(st.integers(0, 40)) if n is None else n
    regime = draw(REGIMES)
    if regime == "ordinary":
        values = draw(st.lists(st.floats(-1e6, 1e6), min_size=n, max_size=n))
    elif regime == "huge":
        values = draw(st.lists(st.floats(-1e300, 1e300), min_size=n, max_size=n))
        values = [v * 1e8 if abs(v) < 1e290 else v for v in values]
    elif regime == "tiny":
        values = draw(st.lists(st.floats(-1e-300, 1e-300), min_size=n, max_size=n))
    elif regime == "subnormal":
        values = [draw(st.integers(-(2**52), 2**52)) * 5e-324 for _ in range(n)]
    elif regime == "near_max":
        values = [
            draw(st.sampled_from([1.7976931348623157e308, -1.7976931348623157e308, 1.5e308, 1e308])) for _ in range(n)
        ]
    elif regime == "zero":
        values = [draw(st.sampled_from([0.0, -0.0])) for _ in range(n)]
    elif regime == "constant":
        values = [draw(FINITE)] * n
    else:
        # max|x| = 2^43 makes e = 0, so a half-integer quantizes on a tie.
        values = [draw(st.integers(-(2**20), 2**20)) + 0.5 for _ in range(n)]
        if n:
            values[0] = 2.0**43
    return np.array(values, dtype=np.float64)


@FAST
@given(column=columns())
def test_quantize_matches_frexp_ldexp_rint(column):
    want_q, want_e = oracle.quantize(column)
    out = np.full((len(column), 3), -7, dtype=np.int64)
    got_e = _native.moments_quantize(column, out, 1, "values['x']")
    assert got_e == want_e
    assert np.array_equal(out[:, 1], want_q)
    assert np.all(out[:, [0, 2]] == -7)


def test_quantize_refuses_a_non_finite_value():
    column = np.array([1.0, 2.0, np.inf])
    with pytest.raises(
        ValueError, match=r"values\['x'\] is not finite at position 2; mask with a factor level instead"
    ):
        _native.moments_quantize(column, np.zeros((3, 1), dtype=np.int64), 0, "values['x']")


@FAST
@given(
    data=st.data(),
    n=st.integers(0, 30),
    n_factors=st.integers(0, 3),
)
def test_pack_matches_np_unique(data, n, n_factors):
    factors = [
        np.array(
            data.draw(st.lists(st.integers(-(2**63), 2**63 - 1) | st.integers(-3, 3), min_size=n, max_size=n)),
            dtype=np.int64,
        )
        for _ in range(n_factors)
    ]
    labels, n_labels, levels = _native.moments_pack(factors, n, "first")
    want_labels, want_n, want_levels = oracle.pack(factors, n)
    assert n_labels == want_n
    assert np.array_equal(labels, want_labels)
    assert labels.dtype == np.int32
    assert len(levels) == len(want_levels)
    for got, want in zip(levels, want_levels, strict=True):
        assert np.array_equal(got, want)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


@st.composite
def tables(draw, shape=None, regime_columns=None):
    """A table accumulated from drawn pairs exactly as the engine would: quantized values summed per cell."""
    shape = draw(st.lists(st.integers(0, 3), min_size=1, max_size=3).map(tuple)) if shape is None else shape
    k = 2
    n_people = len(regime_columns[0]) if regime_columns else draw(st.integers(1, 30))
    cols = regime_columns or [draw(columns(n_people)) for _ in range(k)]
    quantized, exponents = zip(*(oracle.quantize(c) for c in cols), strict=True)
    q = np.stack(quantized, axis=1).tolist()
    operands = OPERANDS[: draw(st.integers(0, len(OPERANDS)))]
    n_cells = math.prod(shape)
    cells = {}
    for _ in range(draw(st.integers(0, 40)) if n_cells else 0):
        a, b = draw(st.integers(0, n_people - 1)), draw(st.integers(0, n_people - 1))
        cell = draw(st.integers(0, n_cells - 1))
        acc = cells.setdefault(cell, [0] * (1 + 4 * k + len(operands)))
        acc[0] += 1
        for c in range(k):
            x, y = q[a][c], q[b][c]
            acc[1 + c] += x
            acc[1 + k + c] += y
            acc[1 + 2 * k + c] += x * x
            acc[1 + 3 * k + c] += y * y
        for i, (sa, ca, sb, cb) in enumerate(operands):
            acc[1 + 4 * k + i] += (q[a] if sa == 0 else q[b])[ca] * (q[a] if sb == 0 else q[b])[cb]
    flat = np.empty((n_cells, 1 + 4 * k + len(operands)), dtype=object)
    for cell in range(n_cells):
        flat[cell] = cells.get(cell, [0] * flat.shape[1])
    return _split(flat.reshape(*shape, flat.shape[1]), np.array(exponents, dtype=np.int64), operands)


def _split(stacked: np.ndarray, exponents: np.ndarray, operands) -> OracleTable:
    k = len(exponents)
    at = 1
    slabs = []
    for width in (k, k, k, k, len(operands)):
        slabs.append(stacked[..., at : at + width])
        at += width
    return OracleTable(stacked[..., 0].astype(np.int64), *slabs, exponents=exponents, operands=tuple(operands))


def _stacked(table: OracleTable) -> np.ndarray:
    counts = np.asarray(table.counts, dtype=object)[..., np.newaxis]
    exact = (table.q_sum_first, table.q_sum_second, table.q_sumsq_first, table.q_sumsq_second, table.q_cross)
    return np.concatenate([counts, *(np.asarray(x, dtype=object) for x in exact)], axis=-1)


def _arg(table: OracleTable) -> tuple:
    width, data = _encode_exact(_stacked(table))
    k = len(table.exponents)
    return (list(table.shape), k, list(table.operands), [int(e) for e in table.exponents], width, data)


def _back(out: tuple, like: OracleTable) -> OracleTable:
    shape, exponents, width, data = out
    stride = 1 + 4 * len(like.exponents) + len(like.operands)
    values = _decode_exact(width, data).reshape(*shape, stride)
    return _split(values, np.array(exponents, dtype=np.int64), like.operands)


def _assert_same_table(got: OracleTable, want: OracleTable) -> None:
    assert got.shape == want.shape
    assert np.array_equal(got.exponents, want.exponents)
    assert np.array_equal(_stacked(got), _stacked(want))


def _assert_same_derivations(table: OracleTable) -> None:
    arg = _arg(table)
    for statistic in STATISTICS:
        indices = len(table.operands) if statistic in PER_PRODUCT else len(table.exponents)
        for index in range(indices):
            try:
                want = oracle.derive(table, statistic, index)
            except OverflowError:
                with pytest.raises(ValueError, match="is not representable in float64"):
                    _native.moments_table_derive(arg, statistic, index, "x")
                continue
            got = _native.moments_table_derive(arg, statistic, index, "x").reshape(table.shape)
            assert np.array_equal(_bits(got), _bits(want)), (statistic, index, got, want)


@FAST
@given(table=tables())
def test_derived_floats_match_python(table):
    _assert_same_derivations(table)


@FAST
@given(data=st.data(), n=st.integers(2, 12), slope=st.sampled_from([1.0, -1.0, 2.0, -0.5]))
def test_pearson_of_an_exact_line_is_one(data, n, slope):
    # Values on a line with exactly representable quantized integers.
    x = np.array(data.draw(st.lists(st.integers(-1000, 1000), min_size=n, max_size=n, unique=True)), dtype=np.float64)
    table = data.draw(tables(shape=(1,), regime_columns=[x, x * slope]))
    _assert_same_derivations(table)


@FAST
@given(table=tables(), data=st.data())
def test_sum_matches_the_object_array_fold(table, data):
    current = table
    axes = list(range(len(table.shape)))
    while axes:
        axis = data.draw(st.sampled_from(range(len(axes))))
        axes.pop(axis)
        got = _back(_native.moments_table_sum(_arg(current), axis), current)
        want = oracle.table_sum(current, axis)
        _assert_same_table(got, want)
        current = want
    _assert_same_derivations(current)


@FAST
@given(data=st.data(), shape=st.lists(st.integers(1, 3), min_size=1, max_size=2).map(tuple))
def test_merge_matches_aligned_merge_across_exponent_gaps(data, shape):
    left = data.draw(tables(shape=shape))
    # Same people count is not needed; the layouts match by shape and operands.
    right = data.draw(tables(shape=shape))
    operands = left.operands[: min(len(left.operands), len(right.operands))]
    left, right = _with_operands(left, operands), _with_operands(right, operands)
    got = _back(_native.moments_table_merge(_arg(left), _arg(right)), left)
    want = oracle.table_merge(left, right)
    _assert_same_table(got, want)
    _assert_same_derivations(want)


def test_merge_across_a_two_thousand_bit_exponent_gap():
    huge = np.array([1.5e300, -1e300, 7e299])
    tiny = np.array([3e-300, 1e-300, -2e-300])
    left = _fixed_table([huge, tiny])
    right = _fixed_table([tiny, huge])
    assert abs(int(left.exponents[0]) - int(right.exponents[0])) > 1900
    got = _back(_native.moments_table_merge(_arg(left), _arg(right)), left)
    _assert_same_table(got, oracle.table_merge(left, right))
    _assert_same_derivations(got)


def _fixed_table(cols: list[np.ndarray]) -> OracleTable:
    """Every ordered pair of three people in one cell."""
    q, exponents = zip(*(oracle.quantize(c) for c in cols), strict=True)
    k, n = len(cols), len(cols[0])
    acc = [0] * (1 + 4 * k + len(OPERANDS))
    for a in range(n):
        for b in range(n):
            acc[0] += 1
            for c in range(k):
                x, y = int(q[c][a]), int(q[c][b])
                acc[1 + c] += x
                acc[1 + k + c] += y
                acc[1 + 2 * k + c] += x * x
                acc[1 + 3 * k + c] += y * y
            for i, (sa, ca, sb, cb) in enumerate(OPERANDS):
                acc[1 + 4 * k + i] += int(q[ca][a if sa == 0 else b]) * int(q[cb][a if sb == 0 else b])
    stacked = np.array(acc, dtype=object).reshape(1, -1)
    return _split(stacked, np.array(exponents, dtype=np.int64), OPERANDS)


def _with_operands(table: OracleTable, operands) -> OracleTable:
    return OracleTable(
        table.counts,
        table.q_sum_first,
        table.q_sum_second,
        table.q_sumsq_first,
        table.q_sumsq_second,
        table.q_cross[..., : len(operands)],
        exponents=table.exponents,
        operands=tuple(operands),
    )


# ---------------------------------------------------------------------------
# Counts (ADR 0015, D9): per cell, exact, checked
# ---------------------------------------------------------------------------


def _counts_only(counts: list[int]) -> OracleTable:
    stacked = np.zeros((len(counts), 1 + 4 * 2), dtype=object)
    stacked[:, 0] = counts
    return _split(stacked, np.zeros(2, dtype=np.int64), ())


def test_a_cell_at_the_int64_maximum_is_accepted_and_one_past_is_refused():
    full = _counts_only([2**63 - 1])
    same = _back(_native.moments_table_merge(_arg(full), _arg(_counts_only([0]))), full)
    assert int(same.counts[0]) == 2**63 - 1
    with pytest.raises(ResourceError) as info:
        _native.moments_table_merge(_arg(full), _arg(_counts_only([1])))
    assert info.value.code == "arithmetic_overflow"
    with pytest.raises(ResourceError):
        _native.moments_table_sum(_arg(_counts_only([2**62, 2**62])), 0)
    folded = _back(_native.moments_table_sum(_arg(_counts_only([2**62, 2**62 - 1])), 0), full)
    assert int(folded.counts) == 2**63 - 1


def test_a_table_core_returns_is_at_its_minimal_width():
    table = _counts_only([1, 2, 3])
    _, _, width, data = _native.moments_table_select(_arg(table), 0, [2, 0])
    assert width == 1
    assert list(_decode_exact(width, data)) == [3] + [0] * 8 + [1] + [0] * 8


def test_a_refused_table_or_label_allocation_is_a_resource_error():
    """The two allocation families core's boundary arithmetic adds, each planted in a fresh process."""
    out = _run_child("""
        import numpy as np
        from pedigree_graph import ResourceError, _native
        table = ([3], 0, [], [], 1, np.zeros(3, dtype=np.uint8))
        for family, call in (
            ("moment_input", lambda: _native.moments_pack([np.arange(5)], 5, "first")),
            ("moment_table", lambda: _native.moments_table_sum(table, 0)),
        ):
            _native.fail_next_allocation(family, 1)
            try:
                call()
            except ResourceError as e:
                print(e.code, e.fields["operation"], e.fields["dtype"])
            _native.fail_next_allocation(None)
            call()
    """)
    assert out.splitlines() == ["allocation_failed moment_input int64", "allocation_failed moment_table uint8"]
