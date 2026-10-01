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
products, with the per-column exponents beside them.
:meth:`~RelationshipMoments.select` narrows an axis,
:meth:`~RelationshipMoments.sum` folds one away and
:meth:`~RelationshipMoments.merge` combines two results; the fold and the
merge are integer additions, so they are exact and order-independent, and
the centered moments of a fold equal those of a direct engine call bit for
bit.  Every float the table offers (``sum_first``, ``sumsq_first``,
``cross``, ``m2_first``, ``comoment``, ``mean``, ``pearson``, ...) is
derived from the integers by one path: an exact integer numerator, divided
by the count where the moment is centered and by ``2**exponent``, as an
exact rational rounded once to float64.

The Rust core holds that arithmetic for the Python and R hosts alike (ADR
0015).  A result keeps the accumulators in core's encoding, little-endian
two's complement integers of one width per table, and hands them to core
for every fold, merge, selection and float; the ``q_*`` attributes decode
them to Python ints on access.
"""

from __future__ import annotations

__all__ = [
    "HOST_BYTES_PER_ACCUMULATOR",
    "MomentAxis",
    "RelationshipMoments",
    "host_bytes",
    "side_column",
]

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

import numpy as np

from pedigree_graph import _native

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


#: Bytes per accumulator the engine's budget plan adds once beyond its lanes
#: (``HOST_BYTES_PER_ACCUMULATOR`` in ``moments.rs``, which the budget tests
#: hold to this value): the encoded table, at most 16, and one host copy of
#: it, which R makes and Python does not.  One figure for both hosts.
HOST_BYTES_PER_ACCUMULATOR = 16 + 16


def host_bytes(accumulators: int) -> int:
    """The host term of the budget estimate for *accumulators* accumulators (``host_bytes`` in ``moments.rs``)."""
    return accumulators * HOST_BYTES_PER_ACCUMULATOR


def _frozen(values: np.ndarray) -> np.ndarray:
    # np.ascontiguousarray would promote a 0-d array (a fully folded cell) to shape (1,).
    out = np.asarray(values, order="C")
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

    def __post_init__(self) -> None:
        object.__setattr__(self, "levels", _frozen(np.array(self.levels, copy=True)))

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


def _encode_exact(values: np.ndarray) -> tuple[int, np.ndarray]:
    """Integers in core's encoding at a width that holds them all: ``(width, uint8 bytes)``."""
    ints = [int(v) for v in np.asarray(values).reshape(-1).tolist()]
    width = max(((v if v >= 0 else ~v).bit_length() // 8 + 1 for v in ints), default=1)
    data = b"".join(v.to_bytes(width, "little", signed=True) for v in ints)
    return width, np.frombuffer(data, dtype=np.uint8)


def _decode_exact(width: int, data: np.ndarray) -> np.ndarray:
    """Integers in core's encoding, ``(width, uint8 bytes)``, as an object array of Python ints."""
    if width > 8:
        raw = data.tobytes()
        return np.array(
            [int.from_bytes(raw[i : i + width], "little", signed=True) for i in range(0, len(raw), width)],
            dtype=object,
        )
    rows = data.reshape(-1, width)
    full = np.zeros((len(rows), 8), dtype=np.uint8)
    full[:, :width] = rows
    full[rows[:, width - 1] >= 0x80, width:] = 0xFF
    return full.view("<i8").reshape(-1).astype(object)


@dataclass(frozen=True, slots=True, eq=False)
class RelationshipMoments:
    """Pair counts and value moments per category and pair-label cell.

    The exact accumulators are in :attr:`encoded`, core's encoding; the
    ``q_*`` attributes decode them to object arrays of Python ints, in
    quantized units of ``2^-exponents[c]`` per column ``c``, and every float
    array is derived from them on access.  ``shape`` is the tuple of axis
    sizes; the per-column arrays carry one trailing axis over
    :attr:`columns`, the per-product arrays one over :attr:`products`.
    Every array is read-only.

    Attributes:
        axes: The table's axes.
        columns: The value column names, in call order.
        products: The requested products as ``("<side>.<column>",
            "<side>.<column>")`` pairs, in call order.
        exponents: int64 scale exponent per column: the integers are the
            values times ``2**exponents[c]``.
        width: Bytes per accumulator in :attr:`encoded`.
        encoded: The accumulators as uint8: per cell in row-major order over
            :attr:`shape`, the pair count, ``Σq`` of each column over the
            first member, over the second, ``Σq²`` of each over the first,
            over the second, and ``Σ q_a q_b`` of each product, each
            ``width`` bytes of little-endian two's complement.
        symmetric: The orientation rule the result was computed under.
        lanes: The accumulator lanes the engine pass ran on (0 without a pass).
        lane_pairs: The pairs each lane reduced, a diagnostic.
        estimated_peak_bytes: The planned accumulator peak of that pass.
        counts: int64 pairs per cell, decoded from :attr:`encoded`.
    """

    axes: tuple[MomentAxis, ...]
    columns: tuple[str, ...]
    products: tuple[tuple[str, str], ...]
    exponents: np.ndarray
    width: int
    encoded: np.ndarray
    symmetric: str
    lanes: int = 0
    lane_pairs: tuple[int, ...] = ()
    estimated_peak_bytes: int = 0
    counts: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "exponents", _frozen(np.asarray(self.exponents, dtype=np.int64)))
        object.__setattr__(self, "encoded", _frozen(np.asarray(self.encoded, dtype=np.uint8).reshape(-1)))
        # Core checks the layout and that every count is in [0, 2^63 - 1].
        counts = _native.moments_table_counts(self._core_table())
        object.__setattr__(self, "counts", _frozen(counts.reshape(self.shape)))

    @classmethod
    def from_exact(
        cls,
        *,
        axes: Sequence[MomentAxis],
        columns: Sequence[str],
        products: Sequence[tuple[str, str]],
        exponents: np.ndarray,
        counts: np.ndarray,
        q_sum_first: np.ndarray,
        q_sum_second: np.ndarray,
        q_sumsq_first: np.ndarray,
        q_sumsq_second: np.ndarray,
        q_cross: np.ndarray,
        symmetric: str,
    ) -> RelationshipMoments:
        """A table from its exact accumulators, for tables built without a pedigree.

        *counts* has the axes' shape; each ``q_*`` adds a trailing axis over
        *columns* (over *products* for ``q_cross``) of integers in quantized
        units, as the attributes of the same names hold them.

        Raises:
            ValueError: A count outside ``[0, 2**63 - 1]`` or arrays that do not
                fit the axes.
        """
        shape = tuple(len(axis.levels) for axis in axes)
        slabs = [np.asarray(counts, dtype=object).reshape(*shape, 1)]
        for name, values, width in (
            ("q_sum_first", q_sum_first, len(columns)),
            ("q_sum_second", q_sum_second, len(columns)),
            ("q_sumsq_first", q_sumsq_first, len(columns)),
            ("q_sumsq_second", q_sumsq_second, len(columns)),
            ("q_cross", q_cross, len(products)),
        ):
            array = np.asarray(values, dtype=object)
            if array.shape != (*shape, width):
                raise ValueError(f"{name} has shape {array.shape}, expected {(*shape, width)}")
            slabs.append(array)
        width, encoded = _encode_exact(np.concatenate(slabs, axis=-1))
        return cls(
            axes=tuple(axes),
            columns=tuple(columns),
            products=tuple(products),
            exponents=exponents,
            width=width,
            encoded=encoded,
            symmetric=symmetric,
        )

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
        # Python ints: several valid int64 cells can together pass 2^63.
        pairs = sum(self.counts.reshape(-1).tolist())
        return f"RelationshipMoments({axes}; pairs={pairs})"

    def axis(self, name: str) -> MomentAxis:
        """The axis called *name*."""
        return self.axes[self._axis_index(name)]

    def _axis_index(self, name: str) -> int:
        return _axis_index(self.axes, name)

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
            positions = [result.axes[axis].position(v) for v in _levels_of(value)]
            if len(set(positions)) != len(positions):
                raise ValueError(f"select({name}=...) names a level more than once")
            _, exponents, width, data = _native.moments_table_select(result._core_table(), axis, positions)
            axes = list(result.axes)
            axes[axis] = MomentAxis(axes[axis].name, axes[axis].levels[np.array(positions, dtype=np.intp)])
            result = result._from_core(tuple(axes), exponents, width, data, keep_pass=True)
        return result

    def sum(self, *names: str) -> RelationshipMoments:
        """Fold the named axes away by adding their cells exactly.

        ``sum("category")`` merges categories (``PO`` from ``MO`` and
        ``FO``), ``sum("first_sex")`` pools a stratum.  An axis with no
        levels folds to zero pairs.

        Raises:
            ResourceError: A cell's pair count would pass ``2**63 - 1``
                (``arithmetic_overflow``); nothing wraps.
        """
        if not names:
            return self
        axes = list(self.axes)
        shape, n_columns, operands, exponents, width, data = self._core_table()
        for name in names:
            axis = _axis_index(axes, name)
            shape, exponents, width, data = _native.moments_table_sum(
                (shape, n_columns, operands, exponents, width, data), axis
            )
            del axes[axis]
        return self._from_core(tuple(axes), exponents, width, data, keep_pass=True)

    def merge(self, other: RelationshipMoments) -> RelationshipMoments:
        """Add two results over identical axes, columns, products and orientation, cell by cell.

        Columns whose exponents differ are aligned exactly first: the
        coarser operand's integers are multiplied by the power of two that
        brings it to the finer exponent, so the merged result carries the
        larger exponent of each column and no rounding.

        Raises:
            ValueError: The layouts or the ``symmetric`` rules differ.
            ResourceError: A cell's pair count would pass ``2**63 - 1``
                (``arithmetic_overflow``); nothing wraps.
        """
        same_axes = len(self.axes) == len(other.axes) and all(
            a.name == b.name and a.levels.shape == b.levels.shape and bool(np.all(a.levels == b.levels))
            for a, b in zip(self.axes, other.axes, strict=True)
        )
        if not same_axes or self.columns != other.columns or self.products != other.products:
            raise ValueError("merge needs two results with the same axes, columns and products")
        if self.symmetric != other.symmetric:
            raise ValueError(f"merge needs one symmetric rule, got {self.symmetric!r} and {other.symmetric!r}")
        _, exponents, width, data = _native.moments_table_merge(self._core_table(), other._core_table())
        return self._from_core(self.axes, exponents, width, data)

    def _core_table(self) -> tuple:
        """This table as core reads it: ``(shape, n_columns, operands, exponents, width, bytes)``."""
        operands = list(self._product_operands())
        exponents = [int(e) for e in self.exponents]
        return (list(self.shape), len(self.columns), operands, exponents, self.width, self.encoded)

    def _from_core(
        self,
        axes: tuple[MomentAxis, ...],
        exponents: list[int],
        width: int,
        data: np.ndarray,
        *,
        keep_pass: bool = False,
    ) -> RelationshipMoments:
        """A result over *axes* from core's table; a merge has no single engine pass, so it keeps none."""
        return RelationshipMoments(
            axes=tuple(axes),
            columns=self.columns,
            products=self.products,
            exponents=np.asarray(exponents, dtype=np.int64),
            width=width,
            encoded=data,
            symmetric=self.symmetric,
            lanes=self.lanes if keep_pass else 0,
            lane_pairs=self.lane_pairs if keep_pass else (),
            estimated_peak_bytes=self.estimated_peak_bytes if keep_pass else 0,
        )

    def _exact(self, at: int, size: int) -> np.ndarray:
        """Accumulators ``at .. at + size`` of every cell, as Python ints over ``(*shape, size)``."""
        stride = 1 + 4 * len(self.columns) + len(self.products)
        slots = self.encoded.reshape(-1, stride, self.width)[:, at : at + size, :]
        return _frozen(_decode_exact(self.width, np.ascontiguousarray(slots).reshape(-1)).reshape(*self.shape, size))

    @property
    def q_sum_first(self) -> np.ndarray:
        """Σq of each column over the first member, per cell (Python ints)."""
        return self._exact(1, len(self.columns))

    @property
    def q_sum_second(self) -> np.ndarray:
        """Σq of each column over the second member, per cell (Python ints)."""
        k = len(self.columns)
        return self._exact(1 + k, k)

    @property
    def q_sumsq_first(self) -> np.ndarray:
        """Σq² of each column over the first member, per cell (Python ints)."""
        k = len(self.columns)
        return self._exact(1 + 2 * k, k)

    @property
    def q_sumsq_second(self) -> np.ndarray:
        """Σq² of each column over the second member, per cell (Python ints)."""
        k = len(self.columns)
        return self._exact(1 + 3 * k, k)

    @property
    def q_cross(self) -> np.ndarray:
        """Σ q_a q_b of each product over the pairs, per cell (Python ints)."""
        return self._exact(1 + 4 * len(self.columns), len(self.products))

    def _derive(self, statistic: str, names: Sequence[str]) -> np.ndarray:
        """*statistic* of every column or product in *names*, stacked on a trailing axis (core)."""
        table = self._core_table()
        out = np.empty((*self.shape, len(names)), dtype=np.float64)
        for j, name in enumerate(names):
            out[..., j] = _native.moments_table_derive(table, statistic, j, name).reshape(self.shape)
        return _frozen(out)

    def _product_operands(self) -> tuple[tuple[int, int, int, int], ...]:
        """Each product as ``(side_a, column_a, side_b, column_b)`` indices."""
        operands = []
        for a, b in self.products:
            side_a, column_a = side_column(a, self.columns)
            side_b, column_b = side_column(b, self.columns)
            operands.append((side_a, column_a, side_b, column_b))
        return tuple(operands)

    def _product_names(self) -> tuple[str, ...]:
        return tuple(f"{a} x {b}" for a, b in self.products)

    @property
    def sum_first(self) -> np.ndarray:
        """Σx of each column over the first member, per cell (float64)."""
        return self._derive("sum_first", self.columns)

    @property
    def sum_second(self) -> np.ndarray:
        """Σx of each column over the second member, per cell (float64)."""
        return self._derive("sum_second", self.columns)

    @property
    def sumsq_first(self) -> np.ndarray:
        """Σx² of each column over the first member, per cell (float64)."""
        return self._derive("sumsq_first", self.columns)

    @property
    def sumsq_second(self) -> np.ndarray:
        """Σx² of each column over the second member, per cell (float64)."""
        return self._derive("sumsq_second", self.columns)

    @property
    def cross(self) -> np.ndarray:
        """Σ x_a y_b of each product over the pairs, per cell (float64)."""
        return self._derive("cross", self._product_names())

    @property
    def m2_first(self) -> np.ndarray:
        """Exact centered second moment ``Σ(x − x̄)²`` of each column over the first member."""
        return self._derive("m2_first", self.columns)

    @property
    def m2_second(self) -> np.ndarray:
        """Exact centered second moment of each column over the second member."""
        return self._derive("m2_second", self.columns)

    @property
    def comoment(self) -> np.ndarray:
        """Exact centered co-moment ``Σ(x − x̄)(y − ȳ)`` of each product."""
        return self._derive("comoment", self._product_names())

    def mean(self, name: str) -> np.ndarray:
        """The mean of ``"first.<column>"`` or ``"second.<column>"`` per cell, NaN where empty."""
        side, column = side_column(name, self.columns)
        statistic = ("mean_first", "mean_second")[side]
        return _frozen(_native.moments_table_derive(self._core_table(), statistic, column, name).reshape(self.shape))

    def pearson(self, a: str, b: str) -> np.ndarray:
        """Pearson correlation of the product ``(a, b)`` per cell.

        *a* and *b* are ``"<side>.<column>"`` names of a requested product,
        in either order.  Computed from the exact numerators as
        ``sign(N_ab) · sqrt(N_ab² / (N_aa · N_bb))``: the ratio is exact
        integer arithmetic rounded once and at most 1 by Cauchy-Schwarz, so
        column scales cancel, nothing overflows or underflows, and
        ``|r| <= 1`` always.  NaN where the cell is empty or either operand
        is constant.
        """
        if (a, b) in self.products:
            index = self.products.index((a, b))
        elif (b, a) in self.products:
            index = self.products.index((b, a))
        else:
            raise ValueError(f"({a!r}, {b!r}) is not a requested product; the products are {self.products}")
        name = self._product_names()[index]
        return _frozen(_native.moments_table_derive(self._core_table(), "pearson", index, name).reshape(self.shape))

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


def _axis_index(axes: Sequence[MomentAxis], name: str) -> int:
    for i, axis in enumerate(axes):
        if axis.name == name:
            return i
    raise ValueError(f"no axis {name!r}; the axes are {tuple(a.name for a in axes)}")


def _levels_of(value: object) -> list[object]:
    if isinstance(value, np.ndarray):
        return [value.item()] if value.ndim == 0 else list(value)
    if isinstance(value, str | bytes) or not hasattr(value, "__iter__"):
        return [value]
    return list(cast("Iterable[object]", value))
