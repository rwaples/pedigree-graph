"""Relationship moments through the Rust row-streaming engine (ADR 0013).

``PedigreeGraph.relationship_moments`` and ``PedigreeView.relationship_moments``
end here.  This module is the boundary: it validates the named factors, value
columns, products and equality keys, packs each role's factors by mixed radix
into one int32 pair label per individual, quantizes every value column to a
fixed-point integer under the integer-exponent scale rule, hands the arrays
to ``_native.relationship_moments`` on the package pool, and rebuilds the
exact accumulators as a labelled
:class:`~pedigree_graph.moments.RelationshipMoments`.

Numeric contract (ADR 0013).  For a column with ``M = max|x|`` over the
receiver's rows, the exponent ``e`` is the largest integer with
``M · 2^e <= 2^43``, read from ``frexp`` rather than computed in floating
point, and ``0`` for an all-zero column.  ``q = rint(ldexp(x, e))`` rounds
half to even; ``ldexp`` scales in one exact step, so a subnormal ``M`` or one
near the float64 maximum quantizes without an intermediate overflow.  The
scale is taken over every receiver row, masked ones included, so a row that
a factor level masks out should still carry a finite value of ordinary
magnitude.  The engine accumulates the integers exactly and hands them back
as they are; the result derives every float from them.
"""

from __future__ import annotations

import logging
import math
import time
from typing import TYPE_CHECKING

import numpy as np

from pedigree_graph import _native
from pedigree_graph._errors import PedigreeValidationError
from pedigree_graph._input import _INT32_MAX, _INT64_MAX
from pedigree_graph._relationship_pairs import _should_compact_view
from pedigree_graph._threads import thread_budget
from pedigree_graph.moments import MomentAxis, RelationshipMoments, side_column

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from pedigree_graph._core import PedigreeGraph
    from pedigree_graph._selection import RelationshipSelection
    from pedigree_graph._view import PedigreeView

logger = logging.getLogger(__name__)

SYMMETRIC = ("canonical", "both")
DEFAULT_MEMORY_BUDGET_BYTES = 1 << 30
MAX_MEMORY_BUDGET_BYTES = (1 << 64) - 1
MAX_COLUMNS = 32
#: Each key doubles the cells; sixteen is 65,536 times the label product,
#: past which no cell table fits a sensible budget.
MAX_SAME_KEYS = 16
QUANTIZED_BITS = 43


def _coerce_column(kind: str, name: str, values: object, n: int, integer: bool) -> np.ndarray:
    """One per-individual array as int64 (factors, keys) or float64 (values), checked for shape, length and range."""
    field = f"{kind}[{name!r}]"
    array = np.asarray(values)
    if array.ndim != 1:
        raise PedigreeValidationError(
            "invalid_shape", f"{field} must be one-dimensional", field=field, expected_ndim=1, actual_shape=array.shape
        )
    if len(array) != n:
        raise PedigreeValidationError(
            "length_mismatch",
            f"{field} has length {len(array)}, expected {n} from the receiver",
            field=field,
            expected_length=n,
            actual_length=len(array),
        )
    if integer:
        if array.dtype.kind not in "iub":
            raise TypeError(f"{field} must be an integer or boolean array, got dtype {array.dtype}")
        if array.dtype.kind == "u" and len(array) and int(array.max()) > _INT64_MAX:
            position = int(np.flatnonzero(array > _INT64_MAX)[0])
            raise PedigreeValidationError(
                "value_out_of_range",
                f"{field} value at position {position} is outside the int64 range",
                field=field,
                position=position,
                value=int(array[position]),
                minimum=-_INT64_MAX - 1,
                maximum=_INT64_MAX,
            )
        return array.astype(np.int64, copy=False)
    if array.dtype.kind not in "iufb":
        raise TypeError(f"{field} must be a numeric array, got dtype {array.dtype}")
    out = array.astype(np.float64, copy=False)
    if not np.all(np.isfinite(out)):
        position = int(np.flatnonzero(~np.isfinite(out))[0])
        raise ValueError(f"{field} is not finite at position {position}; mask with a factor level instead")
    return out


def _check_names(kind: str, names: Iterable[str]) -> None:
    for name in names:
        if not isinstance(name, str) or not name:
            raise TypeError(f"{kind} names must be non-empty str, got {name!r}")


def _pack(kind: str, factors: Mapping[str, object] | None, n: int) -> tuple[np.ndarray, int, list[MomentAxis]]:
    """Pack named factors by mixed radix into one label per row; return (labels, n_labels, axes)."""
    if factors is None:
        factors = {}
    _check_names(f"{kind} factor", factors)
    labels = np.zeros(n, dtype=np.int64)
    n_labels = 1
    axes = []
    for name, values in factors.items():
        column = _coerce_column(kind, name, values, n, integer=True)
        levels, index = np.unique(column, return_inverse=True)
        n_labels *= max(len(levels), 1)
        if n_labels > _INT32_MAX + 1:
            raise PedigreeValidationError(
                "value_out_of_range",
                f"{kind} factors pack into {n_labels} labels, beyond the int32 label range",
                field=f"{kind} label",
                position=0,
                value=n_labels - 1,
                minimum=0,
                maximum=_INT32_MAX,
            )
        labels = labels * len(levels) + index.astype(np.int64)
        axes.append(MomentAxis(f"{kind}_{name}", levels))
    return labels.astype(np.int32), n_labels, axes


def _exponent(column: np.ndarray) -> int:
    """The largest integer ``e`` with ``max|x| · 2^e <= 2^43``, from the binary exponent; 0 for all zeros."""
    magnitude = float(np.max(np.abs(column))) if len(column) else 0.0
    if magnitude == 0.0:
        return 0
    mantissa, exponent = np.frexp(magnitude)
    # frexp gives |x| = m · 2^E with m in [0.5, 1); m · 2^(43 - E) < 2^43
    # always, and exactly 2^43 is allowed when m is 0.5, a power of two.
    return int(QUANTIZED_BITS - exponent + (1 if mantissa == 0.5 else 0))


def _quantize(columns: dict[str, np.ndarray], n: int) -> tuple[np.ndarray, np.ndarray]:
    """Row-major int64 ``[n, k]`` of quantized values and the int64 exponent per column."""
    exponents = np.array([_exponent(column) for column in columns.values()], dtype=np.int64)
    quantized = np.empty((n, len(columns)), dtype=np.int64)
    for j, column in enumerate(columns.values()):
        quantized[:, j] = np.rint(np.ldexp(column, int(exponents[j])))
    return quantized, exponents


def _parse_products(
    products: Iterable[tuple[str, str]] | None, columns: tuple[str, ...]
) -> tuple[tuple[tuple[str, str], ...], list[tuple[int, int, int, int]]]:
    """Resolve product names to (side, column) index pairs; default is the first×second diagonal."""
    if products is None:
        named = tuple((f"first.{c}", f"second.{c}") for c in columns)
    else:
        named = tuple(_product_entry(entry) for entry in products)
    resolved = []
    for a, b in named:
        side_a, column_a = _operand(a, columns)
        side_b, column_b = _operand(b, columns)
        resolved.append((side_a, column_a, side_b, column_b))
    return named, resolved


def _product_entry(entry: object) -> tuple[str, str]:
    if not isinstance(entry, tuple) or len(entry) != 2 or not all(isinstance(name, str) for name in entry):
        raise TypeError(f"each product must be a (str, str) tuple of operand names, got {entry!r}")
    return entry


def _operand(name: str, columns: tuple[str, ...]) -> tuple[int, int]:
    try:
        return side_column(name, columns)
    except ValueError:
        raise ValueError(
            f"product operand {name!r} must be 'first.<column>' or 'second.<column>' over {columns}"
        ) from None


def _split_halves(hi: np.ndarray, lo: np.ndarray) -> np.ndarray:
    """The exact ``i128`` values from their signed high and unsigned low halves, as Python ints."""
    return hi.astype(object) * (1 << 64) + lo.view(np.uint64).astype(object)


def relationship_moments(
    graph: PedigreeGraph,
    view: PedigreeView | None,
    selection: RelationshipSelection,
    *,
    first: Mapping[str, object] | None,
    second: Mapping[str, object] | None,
    values: Mapping[str, object] | None,
    products: Iterable[tuple[str, str]] | None,
    same: Mapping[str, object] | None,
    symmetric: str,
    memory_budget_bytes: int,
) -> RelationshipMoments:
    """Validate, pack, quantize, run the engine and label the result for *graph* or *view*.

    A view of fewer than two rows or an empty selection has no pairs and
    skips the engine, after committing the thread budget and planning the
    same sizes and budget the pass would, so it is refused the same way.
    """
    if symmetric not in SYMMETRIC:
        raise ValueError(f"symmetric must be one of {SYMMETRIC}, got {symmetric!r}")
    if (
        isinstance(memory_budget_bytes, bool)
        or not isinstance(memory_budget_bytes, int)
        or not 0 <= memory_budget_bytes <= MAX_MEMORY_BUDGET_BYTES
    ):
        raise ValueError(f"memory_budget_bytes must be an int in [0, 2**64 - 1], got {memory_budget_bytes!r}")
    n = graph.n_individuals if view is None else len(view)
    labels_first, n_first, axes_first = _pack("first", first, n)
    if second is None:
        labels_second, n_second, axes_second = (
            labels_first,
            n_first,
            [MomentAxis(f"second{axis.name[len('first') :]}", axis.levels) for axis in axes_first],
        )
    else:
        labels_second, n_second, axes_second = _pack("second", second, n)

    values = {} if values is None else values
    _check_names("value column", values)
    if len(values) > MAX_COLUMNS:
        raise ValueError(f"at most {MAX_COLUMNS} value columns are supported, got {len(values)}")
    columns = {name: _coerce_column("values", name, column, n, integer=False) for name, column in values.items()}
    names = tuple(columns)
    quantized, exponents = _quantize(columns, n)
    product_names, product_indices = _parse_products(products, names)

    same = {} if same is None else same
    _check_names("same key", same)
    if len(same) > MAX_SAME_KEYS:
        raise ValueError(f"at most {MAX_SAME_KEYS} same keys are supported, got {len(same)}")
    keys = np.empty((n, len(same)), dtype=np.int64)
    for j, (name, column) in enumerate(same.items()):
        keys[:, j] = _coerce_column("same", name, column, n, integer=True)
    axes_same = [MomentAxis(f"same_{name}", np.array([0, 1], dtype=np.int64)) for name in same]

    codes = selection.ordered
    axes = (MomentAxis("category", np.array(codes, dtype=object)), *axes_first, *axes_second, *axes_same)
    shape = tuple(len(axis.levels) for axis in axes)
    k, p = len(names), len(product_names)
    cells = math.prod(shape[1:])
    stride = 1 + 4 * k + p

    threads = thread_budget()
    if selection.top_degree is None or n < 2:
        lanes, peak = _native.moments_plan(
            n_categories=len(codes),
            n_labels_first=n_first,
            n_labels_second=n_second,
            n_columns=k,
            n_products=p,
            n_same=len(same),
            threads=threads,
            memory_budget_bytes=memory_budget_bytes,
        )
        exact = np.zeros((len(codes), cells, stride), dtype=object)
        lanes, lane_pairs = 0, ()
    else:
        view_rows = None if view is None else view._graph_to_view()
        compact = view is not None and _should_compact_view(graph.n_individuals, n)
        logger.info(
            "relationship_moments: max_degree=%d, %d categories, %d x %d labels, %d columns, %d products, "
            "%d same keys, symmetric=%s, threads=%d%s",
            selection.top_degree,
            len(codes),
            n_first,
            n_second,
            k,
            p,
            len(same),
            symmetric,
            threads,
            ", view" if view_rows is not None else "",
        )
        start = time.perf_counter()
        hi, lo, native_cells, native_stride, lanes, lane_pairs, peak = _native.relationship_moments(
            graph._built,
            max_degree=selection.top_degree,
            requested=list(codes),
            threads=threads,
            labels_first=labels_first,
            n_labels_first=n_first,
            labels_second=labels_second,
            n_labels_second=n_second,
            values=np.ascontiguousarray(quantized),
            products=product_indices,
            same=np.ascontiguousarray(keys),
            symmetric=symmetric,
            memory_budget_bytes=memory_budget_bytes,
            view_rows=view_rows,
            compact=compact,
        )
        assert (native_cells, native_stride) == (cells, stride), "native layout disagrees with the packing"
        exact = _split_halves(hi, lo).reshape(len(codes), cells, stride)
        del hi, lo
        logger.info(
            "relationship_moments total: %d pairs in %.3fs on %d lane(s), estimated accumulator peak %d bytes",
            sum(lane_pairs),
            time.perf_counter() - start,
            lanes,
            peak,
        )

    at = 1
    slabs = []
    for width in (k, k, k, k, p):
        slabs.append(exact[:, :, at : at + width].reshape(*shape, width))
        at += width
    return RelationshipMoments(
        axes=axes,
        columns=names,
        products=product_names,
        exponents=exponents,
        counts=exact[:, :, 0].astype(np.int64).reshape(shape),
        q_sum_first=slabs[0],
        q_sum_second=slabs[1],
        q_sumsq_first=slabs[2],
        q_sumsq_second=slabs[3],
        q_cross=slabs[4],
        symmetric=symmetric,
        lanes=int(lanes),
        lane_pairs=tuple(int(x) for x in lane_pairs),
        estimated_peak_bytes=int(peak),
    )
