"""Relationship moments through the Rust row-streaming engine (ADR 0013).

``PedigreeGraph.relationship_moments`` and ``PedigreeView.relationship_moments``
end here.  This module is the boundary: it validates the named factors, value
columns, products and equality keys, has core pack each role's factors by
mixed radix into one int32 pair label per individual and quantize every value
column to a fixed-point integer under the integer-exponent scale rule, hands
the arrays to ``_native.relationship_moments`` on the package pool, and
rebuilds the exact accumulators as a labelled
:class:`~pedigree_graph.moments.RelationshipMoments`.

Numeric contract (ADR 0013, ADR 0015).  For a column with ``M = max|x|`` over
the receiver's rows, the exponent ``e`` is the largest integer with
``M · 2^e <= 2^43``, read from the binary exponent of ``M``, and ``0`` for an
all-zero column; ``q`` is ``x · 2^e`` rounded half to even in exact integer
steps, so a subnormal ``M`` or one near the float64 maximum quantizes without
an intermediate overflow.  Core does both, for Python and R alike.  The scale
is taken over every receiver row, masked ones included, so a row that a
factor level masks out should still carry a finite value of ordinary
magnitude.  The engine accumulates the integers exactly and hands them back
as they are; the result derives every float from them through core.
"""

from __future__ import annotations

import logging
import math
import time
from typing import TYPE_CHECKING

import numpy as np

from pedigree_graph import _native
from pedigree_graph._errors import PedigreeValidationError
from pedigree_graph._input import _INT64_MAX
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
#: Core's limits, shared with the R binding.
MAX_COLUMNS = _native.MOMENTS_MAX_VALUE_COLUMNS
MAX_SAME_KEYS = _native.MOMENTS_MAX_SAME_KEYS


def _coerce_column(kind: str, name: str, values: object, n: int, integer: bool) -> np.ndarray:
    """One per-individual array, checked for shape, length and range.

    Factors and keys come back as int64.  Value columns keep their numeric
    dtype; :func:`_quantize` converts them to float64 one column at a time,
    so a float32 table is never held twice, and core refuses a value that is
    not finite.
    """
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
    return array


def _check_names(kind: str, names: Iterable[str]) -> None:
    for name in names:
        if not isinstance(name, str) or not name:
            raise TypeError(f"{kind} names must be non-empty str, got {name!r}")


def _pack(kind: str, factors: Mapping[str, object] | None, n: int) -> tuple[np.ndarray, int, list[MomentAxis]]:
    """Pack named factors by mixed radix into one label per row (core); return (labels, n_labels, axes)."""
    if factors is None:
        factors = {}
    _check_names(f"{kind} factor", factors)
    columns = [np.ascontiguousarray(_coerce_column(kind, name, v, n, integer=True)) for name, v in factors.items()]
    labels, n_labels, levels = _native.moments_pack(columns, n, kind)
    axes = [MomentAxis(f"{kind}_{name}", level) for name, level in zip(factors, levels, strict=True)]
    return labels, n_labels, axes


def _quantize(columns: dict[str, np.ndarray], n: int) -> tuple[np.ndarray, np.ndarray]:
    """Row-major int64 ``[n, k]`` of quantized values and the int64 exponent per column (core).

    Each column is widened to float64 in turn into one reused buffer, so the
    peak beyond the inputs and the result is a single float64 column.
    """
    exponents = np.empty(len(columns), dtype=np.int64)
    quantized = np.empty((n, len(columns)), dtype=np.int64)
    buffer = np.empty(n, dtype=np.float64)
    for j, (name, column) in enumerate(columns.items()):
        np.copyto(buffer, column, casting="unsafe")
        exponents[j] = _native.moments_quantize(buffer, quantized, j, f"values[{name!r}]")
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
        _, peak = _native.moments_plan(
            n_categories=len(codes),
            n_labels_first=n_first,
            n_labels_second=n_second,
            n_columns=k,
            n_products=p,
            n_same=len(same),
            threads=threads,
            memory_budget_bytes=memory_budget_bytes,
        )
        width, table = 1, np.zeros(len(codes) * cells * stride, dtype=np.uint8)
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
        width, table, native_cells, native_stride, lanes, lane_pairs, peak = _native.relationship_moments(
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
        logger.info(
            "relationship_moments total: %d pairs in %.3fs on %d lane(s), estimated accumulator peak %d bytes",
            sum(lane_pairs),
            time.perf_counter() - start,
            lanes,
            peak,
        )

    return RelationshipMoments(
        axes=axes,
        columns=names,
        products=product_names,
        exponents=exponents,
        width=width,
        encoded=table,
        symmetric=symmetric,
        lanes=int(lanes),
        lane_pairs=tuple(int(x) for x in lane_pairs),
        estimated_peak_bytes=int(peak),
    )
