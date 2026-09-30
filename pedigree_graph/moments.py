"""The relationship-moments result and its exact algebra (ADR 0013).

:class:`RelationshipMoments` is what ``PedigreeGraph.relationship_moments``
and ``PedigreeView.relationship_moments`` return: a labelled table over the
axes ``category``, one ``first_<factor>`` per first-member factor, one
``second_<factor>`` per second-member factor, and one ``same_<key>`` per
equality key.  Every pair of a category is counted once, in the category's
semantic orientation (``symmetric="canonical"``) or, for symmetric
categories under ``symmetric="both"``, once per orientation.

The table holds the engine's accumulators as they are: per cell the pair
count and, in quantized integer units, the sums of every value column over
each member, their sums of squares, and the cross sums of the requested
products, as NumPy object arrays of Python ``int`` with the per-column
exponents beside them.  :meth:`~RelationshipMoments.select` narrows an
axis, :meth:`~RelationshipMoments.sum` folds one away and
:meth:`~RelationshipMoments.merge` combines two results; all three are
integer additions, so they are exact and order-independent, and the
centered moments of a fold equal those of a direct engine call bit for bit.
Every float the table offers (``sum_first``, ``sumsq_first``, ``cross``,
``m2_first``, ``comoment``, ``mean``, ``pearson``, ...) is derived from the
integers by one path: an exact integer numerator, one correctly rounded
conversion to float64, a division by the count where the moment is
centered, and ``ldexp`` by the column exponents.
"""

from __future__ import annotations

__all__ = ["HOST_BYTES_PER_ACCUMULATOR", "MomentAxis", "RelationshipMoments", "side_column"]

import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, cast

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


#: Bytes the host keeps per accumulator, the term the engine's budget plan
#: adds once (``HOST_BYTES_PER_ACCUMULATOR`` in ``moments.rs``, which the
#: budget tests hold to this value): the two int64 halves the binding hands
#: over (16) plus the object-array slot (8) and a Python ``int`` of up to
#: ``2**126`` (48 on CPython 3.13).
HOST_BYTES_PER_ACCUMULATOR = 16 + 8 + 48


def _frozen(values: np.ndarray) -> np.ndarray:
    out = np.ascontiguousarray(values)
    out.setflags(write=False)
    return out


@dataclass(frozen=True, slots=True)
class MomentAxis:
    """One axis of a :class:`RelationshipMoments` table.

    Attributes:
        name: ``"category"``, ``"first_<factor>"``, ``"second_<factor>"`` or
            ``"same_<key>"``.
        levels: The axis labels in axis order: registry codes for
            ``category``, the factor's distinct input values (int64,
            ascending) for a factor, ``[0, 1]`` (members differ, members
            equal) for an equality key.
    """

    name: str
    levels: np.ndarray

    def position(self, value: object) -> int:
        """The index of *value* along this axis."""
        hits = np.flatnonzero(self.levels == value)
        if len(hits) != 1:
            raise ValueError(f"{value!r} is not a level of axis {self.name!r}")
        return int(hits[0])


def side_column(name: str, columns: Sequence[str]) -> tuple[int, int]:
    """Resolve ``"first.<column>"`` / ``"second.<column>"`` to ``(side, column index)``."""
    side, _, column = name.partition(".")
    if side not in ("first", "second") or column not in columns:
        raise ValueError(f"{name!r} is not 'first.<column>' or 'second.<column>' over {tuple(columns)}")
    return (0 if side == "first" else 1), columns.index(column)


def to_float(
    numerators: np.ndarray, exponents: np.ndarray, divisor: np.ndarray | None, what: str, names: Sequence[str]
) -> np.ndarray:
    """``float(numerator) / divisor · 2^-exponent`` per trailing column, the one integer-to-float path.

    *numerators* is an object array of Python ints with one trailing axis
    over *names*; *exponents* is one int per trailing entry; *divisor*
    (counts, broadcast over the trailing axis) is applied where it is
    positive and yields ``0.0`` elsewhere.  Each conversion rounds once, to
    nearest even; a value that does not fit float64 raises ``ValueError``
    naming the column.
    """
    out = np.empty(numerators.shape, dtype=np.float64)
    flat_out = out.reshape(-1, numerators.shape[-1]) if numerators.ndim else out.reshape(1, 1)
    flat_in = numerators.reshape(-1, numerators.shape[-1]) if numerators.ndim else numerators.reshape(1, 1)
    for j in range(flat_in.shape[1]):
        try:
            flat_out[:, j] = [float(v) for v in flat_in[:, j]]
        except OverflowError:
            raise ValueError(f"{what} of {names[j]!r} is not representable in float64 (an output overflowed)") from None
    if divisor is not None:
        safe = np.where(divisor > 0, divisor, 1).astype(np.float64)
        out = np.where(divisor[..., np.newaxis] > 0, out / safe[..., np.newaxis], 0.0)
    with np.errstate(over="ignore"):
        scaled = np.ldexp(out, -np.asarray(exponents, dtype=np.int64))
    if not np.all(np.isfinite(scaled)):
        bad = int(np.flatnonzero(~np.isfinite(scaled).reshape(-1, scaled.shape[-1]).all(axis=0))[0])
        raise ValueError(f"{what} of {names[bad]!r} is not representable in float64 (an output overflowed)")
    return _frozen(scaled)


def _ints(values: np.ndarray) -> np.ndarray:
    """*values* as an object array of Python ints."""
    return np.asarray(values, dtype=object)


@dataclass(frozen=True, slots=True, eq=False)
class RelationshipMoments:
    """Pair counts and value moments per category and pair-label cell.

    The exact accumulators are the ``q_*`` object arrays (Python ints, in
    quantized units of ``2^-exponents[c]`` per column ``c``); every other
    array is a float64 view derived from them on access.  ``shape`` is the
    tuple of axis sizes; the per-column arrays carry one trailing axis over
    :attr:`columns`, the per-product arrays one over :attr:`products`.
    Every array is read-only.

    Attributes:
        axes: The table's axes.
        columns: The value column names, in call order.
        products: The requested products as ``("<side>.<column>",
            "<side>.<column>")`` pairs, in call order.
        exponents: int64 scale exponent per column: the integers are the
            values times ``2**exponents[c]``.
        counts: int64 pairs per cell.
        q_sum_first: Σq of each column over the first member, per cell.
        q_sum_second: The same over the second member.
        q_sumsq_first: Σq² of each column over the first member.
        q_sumsq_second: The same over the second member.
        q_cross: Σ q_a q_b of each product over the pairs.
        symmetric: The orientation rule the result was computed under.
        lanes: The accumulator lanes the engine pass ran on (0 without a pass).
        lane_pairs: The pairs each lane reduced, a diagnostic.
        estimated_peak_bytes: The planned accumulator peak of that pass.
    """

    axes: tuple[MomentAxis, ...]
    columns: tuple[str, ...]
    products: tuple[tuple[str, str], ...]
    exponents: np.ndarray
    counts: np.ndarray
    q_sum_first: np.ndarray
    q_sum_second: np.ndarray
    q_sumsq_first: np.ndarray
    q_sumsq_second: np.ndarray
    q_cross: np.ndarray
    symmetric: str
    lanes: int = 0
    lane_pairs: tuple[int, ...] = ()
    estimated_peak_bytes: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "exponents", _frozen(np.asarray(self.exponents, dtype=np.int64)))
        object.__setattr__(self, "counts", _frozen(np.asarray(self.counts, dtype=np.int64)))
        for name in _EXACT:
            object.__setattr__(self, name, _frozen(_ints(getattr(self, name))))

    @property
    def shape(self) -> tuple[int, ...]:
        """The axis sizes, in :attr:`axes` order."""
        return tuple(len(axis.levels) for axis in self.axes)

    @property
    def categories(self) -> tuple[str, ...]:
        """The registry codes along the ``category`` axis, or ``()`` once it is folded away."""
        names = [axis.name for axis in self.axes]
        if "category" not in names:
            return ()
        return tuple(str(code) for code in self.axes[names.index("category")].levels)

    def __repr__(self) -> str:
        axes = ", ".join(f"{axis.name}={len(axis.levels)}" for axis in self.axes)
        return f"RelationshipMoments({axes}; pairs={int(self.counts.sum())})"

    def axis(self, name: str) -> MomentAxis:
        """The axis called *name*."""
        for axis in self.axes:
            if axis.name == name:
                return axis
        raise ValueError(f"no axis {name!r}; the axes are {tuple(a.name for a in self.axes)}")

    def _axis_index(self, name: str) -> int:
        return self.axes.index(self.axis(name))

    def _arrays(self) -> dict[str, np.ndarray]:
        return {name: getattr(self, name) for name in ("counts", *_EXACT)}

    def select(self, **levels: object) -> RelationshipMoments:
        """Narrow axes to the given levels, keeping every axis.

        Each keyword names an axis (``category="FS"``,
        ``first_generation=[3, 4]``, ``same_household=1``); a scalar (or a
        ``str`` or 0-d array) keeps one level, any other iterable keeps
        those levels in the given order.

        Raises:
            ValueError: An unknown axis, a value that is not a level of it,
                or a level named twice.
        """
        result = self
        for name, value in levels.items():
            axis = result._axis_index(name)
            wanted = _levels_of(value)
            positions = [result.axes[axis].position(v) for v in wanted]
            if len(set(positions)) != len(positions):
                raise ValueError(f"select({name}=...) names a level more than once")
            index = np.array(positions, dtype=np.intp)
            arrays = {key: np.take(array, index, axis=axis) for key, array in result._arrays().items()}
            axes = list(result.axes)
            axes[axis] = MomentAxis(axes[axis].name, _frozen(result.axes[axis].levels[index]))
            result = replace(result, axes=tuple(axes), **arrays)
        return result

    def sum(self, *names: str) -> RelationshipMoments:
        """Fold the named axes away by adding their cells exactly.

        ``sum("category")`` merges categories (``PO`` from ``MO`` and
        ``FO``), ``sum("first_sex")`` pools a stratum.  An axis with no
        levels folds to zero pairs.
        """
        result = self
        for name in names:
            axis = result._axis_index(name)
            arrays = {key: _sum_axis(array, axis) for key, array in result._arrays().items()}
            axes = tuple(a for i, a in enumerate(result.axes) if i != axis)
            result = replace(result, axes=axes, **arrays)
        return result

    def merge(self, other: RelationshipMoments) -> RelationshipMoments:
        """Add two results over identical axes, columns, products and orientation, cell by cell.

        Columns whose exponents differ are aligned exactly first: the
        coarser operand's integers are multiplied by the power of two that
        brings it to the finer exponent, so the merged result carries the
        larger exponent of each column and no rounding.

        Raises:
            ValueError: The layouts or the ``symmetric`` rules differ.
        """
        same_axes = len(self.axes) == len(other.axes) and all(
            a.name == b.name and a.levels.shape == b.levels.shape and bool(np.all(a.levels == b.levels))
            for a, b in zip(self.axes, other.axes, strict=True)
        )
        if not same_axes or self.columns != other.columns or self.products != other.products:
            raise ValueError("merge needs two results with the same axes, columns and products")
        if self.symmetric != other.symmetric:
            raise ValueError(f"merge needs one symmetric rule, got {self.symmetric!r} and {other.symmetric!r}")
        exponents = np.maximum(self.exponents, other.exponents)
        a, b = self._aligned(exponents), other._aligned(exponents)
        arrays = {key: a[key] + b[key] for key in a}
        return replace(self, exponents=exponents, lanes=0, lane_pairs=(), estimated_peak_bytes=0, **arrays)

    def _aligned(self, exponents: np.ndarray) -> dict[str, np.ndarray]:
        """The exact arrays rescaled to *exponents* (at or above this result's own)."""
        shift = [int(e) for e in exponents - self.exponents]
        if any(s < 0 for s in shift):
            raise ValueError("cannot align to a coarser exponent")
        one = np.array([2**s for s in shift], dtype=object)
        two = np.array([2 ** (2 * s) for s in shift], dtype=object)
        cross = np.array([2 ** (shift[ca] + shift[cb]) for _, ca, _, cb in self._product_operands()], dtype=object)
        return {
            "counts": self.counts,
            "q_sum_first": self.q_sum_first * one,
            "q_sum_second": self.q_sum_second * one,
            "q_sumsq_first": self.q_sumsq_first * two,
            "q_sumsq_second": self.q_sumsq_second * two,
            "q_cross": self.q_cross * cross,
        }

    def _product_operands(self) -> tuple[tuple[int, int, int, int], ...]:
        """Each product as ``(side_a, column_a, side_b, column_b)`` indices."""
        operands = []
        for a, b in self.products:
            side_a, column_a = side_column(a, self.columns)
            side_b, column_b = side_column(b, self.columns)
            operands.append((side_a, column_a, side_b, column_b))
        return tuple(operands)

    def _product_exponents(self) -> np.ndarray:
        return np.array(
            [self.exponents[ca] + self.exponents[cb] for _, ca, _, cb in self._product_operands()], dtype=np.int64
        )

    def _product_names(self) -> tuple[str, ...]:
        return tuple(f"{a} x {b}" for a, b in self.products)

    @property
    def sum_first(self) -> np.ndarray:
        """Σx of each column over the first member, per cell (float64)."""
        return to_float(self.q_sum_first, self.exponents, None, "sum_first", self.columns)

    @property
    def sum_second(self) -> np.ndarray:
        """Σx of each column over the second member, per cell (float64)."""
        return to_float(self.q_sum_second, self.exponents, None, "sum_second", self.columns)

    @property
    def sumsq_first(self) -> np.ndarray:
        """Σx² of each column over the first member, per cell (float64)."""
        return to_float(self.q_sumsq_first, 2 * self.exponents, None, "sumsq_first", self.columns)

    @property
    def sumsq_second(self) -> np.ndarray:
        """Σx² of each column over the second member, per cell (float64)."""
        return to_float(self.q_sumsq_second, 2 * self.exponents, None, "sumsq_second", self.columns)

    @property
    def cross(self) -> np.ndarray:
        """Σ x_a y_b of each product over the pairs, per cell (float64)."""
        return to_float(self.q_cross, self._product_exponents(), None, "cross", self._product_names())

    def _numerator(self, sumsq: np.ndarray, sum_a: np.ndarray, sum_b: np.ndarray) -> np.ndarray:
        """``n·Σq_a q_b − Σq_a·Σq_b`` exactly, per cell."""
        n = _ints(self.counts)[..., np.newaxis]
        return n * sumsq - sum_a * sum_b

    @property
    def m2_first(self) -> np.ndarray:
        """Exact centered second moment ``Σ(x − x̄)²`` of each column over the first member."""
        numerator = self._numerator(self.q_sumsq_first, self.q_sum_first, self.q_sum_first)
        return to_float(numerator, 2 * self.exponents, self.counts, "m2_first", self.columns)

    @property
    def m2_second(self) -> np.ndarray:
        """Exact centered second moment of each column over the second member."""
        numerator = self._numerator(self.q_sumsq_second, self.q_sum_second, self.q_sum_second)
        return to_float(numerator, 2 * self.exponents, self.counts, "m2_second", self.columns)

    def _comoment_numerators(self) -> np.ndarray:
        sums = (self.q_sum_first, self.q_sum_second)
        operands = self._product_operands()
        if not operands:
            return np.zeros((*self.shape, 0), dtype=object)
        columns = [
            self._numerator(self.q_cross[..., i : i + 1], sums[sa][..., ca : ca + 1], sums[sb][..., cb : cb + 1])
            for i, (sa, ca, sb, cb) in enumerate(operands)
        ]
        return np.concatenate(columns, axis=-1)

    @property
    def comoment(self) -> np.ndarray:
        """Exact centered co-moment ``Σ(x − x̄)(y − ȳ)`` of each product."""
        return to_float(
            self._comoment_numerators(), self._product_exponents(), self.counts, "comoment", self._product_names()
        )

    def count(self) -> np.ndarray:
        """The int64 pair count per cell, the same array as :attr:`counts`."""
        return self.counts

    def mean(self, name: str) -> np.ndarray:
        """The mean of ``"first.<column>"`` or ``"second.<column>"`` per cell, NaN where empty."""
        side, column = side_column(name, self.columns)
        total = (self.q_sum_first, self.q_sum_second)[side][..., column : column + 1]
        scaled = to_float(total, self.exponents[column : column + 1], self.counts, "mean", (name,))[..., 0]
        return _frozen(np.where(self.counts > 0, scaled, np.nan))

    def pearson(self, a: str, b: str) -> np.ndarray:
        """Pearson correlation of the product ``(a, b)`` per cell.

        *a* and *b* are ``"<side>.<column>"`` names of a requested product,
        in either order.  Computed from the exact numerators as
        ``N_ab / (sqrt(N_aa) · sqrt(N_bb))``, each converted to float64
        once, so column scales cancel and neither overflow nor underflow
        enters.  NaN where the cell is empty or either operand is constant.
        """
        if (a, b) in self.products:
            index = self.products.index((a, b))
        elif (b, a) in self.products:
            index = self.products.index((b, a))
        else:
            raise ValueError(f"({a!r}, {b!r}) is not a requested product; the products are {self.products}")
        n_ab = self._comoment_numerators()[..., index]
        n_aa = self._own_numerator(a)
        n_bb = self._own_numerator(b)
        out = np.full(self.shape, np.nan, dtype=np.float64)
        for cell in np.ndindex(*self.shape):
            aa, bb = n_aa[cell], n_bb[cell]
            if aa > 0 and bb > 0:
                out[cell] = float(n_ab[cell]) / (math.sqrt(float(aa)) * math.sqrt(float(bb)))
        return _frozen(out)

    def _own_numerator(self, name: str) -> np.ndarray:
        side, column = side_column(name, self.columns)
        sums = (self.q_sum_first, self.q_sum_second)[side][..., column : column + 1]
        sumsq = (self.q_sumsq_first, self.q_sumsq_second)[side][..., column : column + 1]
        return self._numerator(sumsq, sums, sums)[..., 0]

    def table(self, flag_a: str, flag_b: str) -> np.ndarray:
        """Pair counts by two factor axes, every other factor axis folded away.

        Returns an int64 array of shape ``(n_categories, len(a), len(b))``
        while the ``category`` axis is present, else ``(len(a), len(b))``:
        for two binary flags the 2×2 table.
        """
        if flag_a == flag_b:
            raise ValueError("table needs two different axes")
        keep = {"category", flag_a, flag_b}
        folded = self.sum(*(axis.name for axis in self.axes if axis.name not in keep))
        leading = ["category"] if any(axis.name == "category" for axis in folded.axes) else []
        order = [folded._axis_index(name) for name in (*leading, flag_a, flag_b)]
        return _frozen(np.transpose(folded.counts, order))

    def cell_levels(self) -> dict[str, np.ndarray]:
        """The level of every axis at every cell, each broadcast to :attr:`shape`."""
        grids = np.meshgrid(*(np.arange(size) for size in self.shape), indexing="ij")
        return {axis.name: _frozen(axis.levels[grid]) for axis, grid in zip(self.axes, grids, strict=True)}


_EXACT = ("q_sum_first", "q_sum_second", "q_sumsq_first", "q_sumsq_second", "q_cross")


def _levels_of(value: object) -> list[object]:
    if isinstance(value, np.ndarray):
        return [value.item()] if value.ndim == 0 else list(value)
    if isinstance(value, str | bytes) or not hasattr(value, "__iter__"):
        return [value]
    return list(cast("Iterable[object]", value))


def _sum_axis(array: np.ndarray, axis: int) -> np.ndarray:
    """Exact sum along *axis*; an empty axis gives zeros of the array's dtype."""
    if array.shape[axis] == 0:
        shape = tuple(size for i, size in enumerate(array.shape) if i != axis)
        return np.zeros(shape, dtype=np.int64) if array.dtype != object else _ints(np.zeros(shape, dtype=np.int64))
    return np.sum(array, axis=axis)
