"""The relationship-moments boundary arithmetic as Python computed it through 0.11 (ADR 0013).

ADR 0015 moved packing, quantization, the exact table operations and every
derived float into the Rust core, shared by the Python and R hosts.  This
module keeps the Python implementation they replaced, unchanged in
behaviour, as the oracle ``tests/test_moments_arithmetic.py`` holds core to
bit for bit: ``np.unique`` packing, ``frexp``/``ldexp``/``rint``
quantization, Python ``int`` true division for every float, the
``_aligned`` merge and the object-array sum.

A table here is an :class:`OracleTable`: the per-cell count and the five
exact slabs as object arrays of Python ints, laid out as the 0.11
``RelationshipMoments`` held them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

QUANTIZED_BITS = 43
_INT32_MAX = 2**31 - 1


def pack(factors: list[np.ndarray], n: int) -> tuple[np.ndarray, int, list[np.ndarray]]:
    """Mixed-radix labels over ``np.unique`` levels; ``(labels, n_labels, levels)``; None past int32."""
    labels = np.zeros(n, dtype=np.int64)
    n_labels = 1
    levels = []
    for column in factors:
        distinct, index = np.unique(np.asarray(column, dtype=np.int64), return_inverse=True)
        n_labels *= max(len(distinct), 1)
        if n_labels > _INT32_MAX + 1:
            raise OverflowError(n_labels)
        labels = labels * len(distinct) + index.astype(np.int64)
        levels.append(distinct)
    return labels.astype(np.int32), n_labels, levels


def exponent(column: np.ndarray) -> int:
    """The largest integer ``e`` with ``max|x| · 2^e <= 2^43``, from ``frexp``; 0 for all zeros."""
    magnitude = max(float(column.max()), -float(column.min())) if len(column) else 0.0
    if magnitude == 0.0:
        return 0
    mantissa, e = np.frexp(magnitude)
    return int(QUANTIZED_BITS - e + (1 if mantissa == 0.5 else 0))


def quantize(column: np.ndarray) -> tuple[np.ndarray, int]:
    """``rint(ldexp(x, e))`` as int64, and ``e``."""
    buffer = np.asarray(column, dtype=np.float64).copy()
    e = exponent(buffer)
    np.ldexp(buffer, e, out=buffer)
    np.rint(buffer, out=buffer)
    return buffer.astype(np.int64), e


def divide_exact(numerator: int, count: int, e: int) -> float:
    """``numerator / (count · 2^e)`` by Python int true division; ``OverflowError`` past float64."""
    if e >= 0:
        return numerator / (count << e)
    return (numerator << -e) / count


@dataclass(frozen=True)
class OracleTable:
    """Counts and exact slabs over ``shape``; ``operands`` per product as ``(side_a, col_a, side_b, col_b)``."""

    counts: np.ndarray
    q_sum_first: np.ndarray
    q_sum_second: np.ndarray
    q_sumsq_first: np.ndarray
    q_sumsq_second: np.ndarray
    q_cross: np.ndarray
    exponents: np.ndarray
    operands: tuple[tuple[int, int, int, int], ...]

    @property
    def shape(self) -> tuple[int, ...]:
        return self.counts.shape

    def slabs(self) -> dict[str, np.ndarray]:
        return {
            "counts": self.counts,
            "q_sum_first": self.q_sum_first,
            "q_sum_second": self.q_sum_second,
            "q_sumsq_first": self.q_sumsq_first,
            "q_sumsq_second": self.q_sumsq_second,
            "q_cross": self.q_cross,
        }

    def product_exponent(self, index: int) -> int:
        _, ca, _, cb = self.operands[index]
        return int(self.exponents[ca] + self.exponents[cb])


def to_float(numerators: np.ndarray, e: int, divisor: np.ndarray | None) -> np.ndarray:
    """One column of ``numerator / (divisor · 2^e)``, 0 where the divisor is 0; ``OverflowError`` past float64."""
    numerators = np.asarray(numerators, dtype=object)
    flat = numerators.reshape(-1)
    counts = np.ones(flat.shape, dtype=np.int64) if divisor is None else np.asarray(divisor).reshape(-1)
    out = [divide_exact(int(v), int(n), e) if n > 0 else 0.0 for v, n in zip(flat, counts, strict=True)]
    return np.array(out, dtype=np.float64).reshape(numerators.shape)


def _n(table: OracleTable) -> np.ndarray:
    return np.asarray(table.counts, dtype=object)


def _numerator(n: np.ndarray, sumsq: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.asarray(n * sumsq - a * b, dtype=object)


def _sums(table: OracleTable, side: int) -> tuple[np.ndarray, np.ndarray]:
    if side == 0:
        return table.q_sum_first, table.q_sumsq_first
    return table.q_sum_second, table.q_sumsq_second


def _comoment_numerator(table: OracleTable, index: int) -> np.ndarray:
    sa, ca, sb, cb = table.operands[index]
    return _numerator(_n(table), table.q_cross[..., index], _sums(table, sa)[0][..., ca], _sums(table, sb)[0][..., cb])


def _own_numerator(table: OracleTable, side: int, column: int) -> np.ndarray:
    sums, sumsq = _sums(table, side)
    return _numerator(_n(table), sumsq[..., column], sums[..., column], sums[..., column])


def derive(table: OracleTable, statistic: str, index: int) -> np.ndarray:
    """One statistic for column or product *index*, per cell, as 0.11 derived it."""
    e = int(table.exponents[index]) if statistic not in ("cross", "comoment", "pearson") else 0
    if statistic in ("sum_first", "sum_second"):
        sums, _ = _sums(table, 0 if statistic == "sum_first" else 1)
        return to_float(sums[..., index], e, None)
    if statistic in ("sumsq_first", "sumsq_second"):
        _, sumsq = _sums(table, 0 if statistic == "sumsq_first" else 1)
        return to_float(sumsq[..., index], 2 * e, None)
    if statistic == "cross":
        return to_float(table.q_cross[..., index], table.product_exponent(index), None)
    if statistic in ("m2_first", "m2_second"):
        side = 0 if statistic == "m2_first" else 1
        return to_float(_own_numerator(table, side, index), 2 * e, table.counts)
    if statistic == "comoment":
        return to_float(_comoment_numerator(table, index), table.product_exponent(index), table.counts)
    if statistic in ("mean_first", "mean_second"):
        sums, _ = _sums(table, 0 if statistic == "mean_first" else 1)
        scaled = to_float(sums[..., index], e, table.counts)
        return np.where(table.counts > 0, scaled, np.nan)
    if statistic == "pearson":
        sa, ca, sb, cb = table.operands[index]
        n_ab = _comoment_numerator(table, index)
        n_aa = _own_numerator(table, sa, ca)
        n_bb = _own_numerator(table, sb, cb)
        out = np.full(table.shape, np.nan, dtype=np.float64)
        for cell in np.ndindex(*table.shape):
            aa, bb = n_aa[cell], n_bb[cell]
            if aa > 0 and bb > 0:
                ab = n_ab[cell]
                r = math.sqrt(ab * ab / (aa * bb))
                out[cell] = -r if ab < 0 else r
        return out
    raise ValueError(statistic)


def _sum_axis(array: np.ndarray, axis: int) -> np.ndarray:
    if array.shape[axis] == 0:
        shape = tuple(size for i, size in enumerate(array.shape) if i != axis)
        return np.zeros(shape, dtype=array.dtype)
    # A fold to no axes is a 0-d array, as the 0.11 result froze it.
    return np.asarray(np.sum(array, axis=axis), dtype=array.dtype)


def table_sum(table: OracleTable, axis: int) -> OracleTable:
    """The object-array fold of 0.11, counts summed as Python ints so a wrap cannot hide."""
    slabs = {key: _sum_axis(np.asarray(value, dtype=object), axis) for key, value in table.slabs().items()}
    return OracleTable(**slabs, exponents=table.exponents, operands=table.operands)


def _aligned(table: OracleTable, exponents: np.ndarray) -> dict[str, np.ndarray]:
    shift = [int(e) for e in exponents - table.exponents]
    one = np.array([2**s for s in shift], dtype=object)
    two = np.array([2 ** (2 * s) for s in shift], dtype=object)
    cross = np.array([2 ** (shift[ca] + shift[cb]) for _, ca, _, cb in table.operands], dtype=object)
    return {
        "counts": np.asarray(table.counts, dtype=object),
        "q_sum_first": table.q_sum_first * one,
        "q_sum_second": table.q_sum_second * one,
        "q_sumsq_first": table.q_sumsq_first * two,
        "q_sumsq_second": table.q_sumsq_second * two,
        "q_cross": table.q_cross * cross,
    }


def table_merge(a: OracleTable, b: OracleTable) -> OracleTable:
    """The 0.11 merge: both operands aligned to the larger exponent per column, then added."""
    exponents = np.maximum(a.exponents, b.exponents)
    left, right = _aligned(a, exponents), _aligned(b, exponents)
    return OracleTable(**{key: left[key] + right[key] for key in left}, exponents=exponents, operands=a.operands)
