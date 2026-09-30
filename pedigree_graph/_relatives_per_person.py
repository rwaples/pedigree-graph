"""Per-person relative counts through the Rust row-streaming engine (issue #33).

``PedigreeGraph.relatives_per_person`` and ``PedigreeView.relatives_per_person``
end here.  This module is the boundary: it validates the named threshold
columns, converts each side to contiguous float64 only when it is not
already (so the engine borrows the caller's arrays), passes a scalar
threshold as a float rather than an array, runs ``_native.relatives_per_person``
on the package pool, and wraps the counts it hands over, without a copy, as a
:class:`RelativesPerPerson`.

The crediting rule is the ADR 0013 pass with both orientations of every
symmetric pair: a symmetric category credits both members, a directional one
only its first (junior) member.  Column ``k`` credits a relative when
``relative_k[relative] <= threshold_k[person]``, compared as float64, so NaN
on either side never counts.
"""

from __future__ import annotations

import logging
import numbers
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from pedigree_graph import _native
from pedigree_graph._errors import PedigreeValidationError
from pedigree_graph._input import _own_native
from pedigree_graph._relationship_moments import MAX_COLUMNS, _check_names
from pedigree_graph._relationship_pairs import _should_compact_view
from pedigree_graph._threads import thread_budget

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from numpy.typing import ArrayLike

    from pedigree_graph._core import PedigreeGraph
    from pedigree_graph._selection import RelationshipSelection
    from pedigree_graph._view import PedigreeView

logger = logging.getLogger(__name__)

#: The pair-count column, always first; no threshold column may take its name.
RELATIVES = "relatives"
#: Integers beyond this magnitude would round when converted to float64.
EXACT_INTEGER = 1 << 53


@dataclass(frozen=True, slots=True, eq=False)
class RelativesPerPerson:
    """Per receiver row and requested category, the row's relatives and how many pass each threshold column.

    ``counts[row, c, 0]`` is the number of relatives of ``row`` in category
    ``categories[c]``; ``counts[row, c, 1 + k]`` how many of them pass
    threshold column ``k``.  A symmetric category credits both members of
    each pair and a directional one only its first (junior) member, so
    ``MO`` counts a row's mother and ``GP`` its grandparents, never its
    children or grandchildren.  Rows are graph rows for a graph and view
    rows for a view.

    Attributes:
        categories: The requested registry codes, in registry order.
        columns: ``("relatives", *threshold column names)``.
        counts: Read-only uint32 ``[rows, len(categories), len(columns)]``.
    """

    categories: tuple[str, ...]
    columns: tuple[str, ...]
    counts: np.ndarray

    def __repr__(self) -> str:
        totals = self.counts[:, :, 0].sum(axis=0, dtype=np.uint64)
        credited = ", ".join(f"{code}={int(total)}" for code, total in zip(self.categories, totals, strict=True))
        return f"RelativesPerPerson(rows={len(self.counts)}; {credited}; columns={self.columns})"

    def get(self, code: str, column: str = RELATIVES) -> np.ndarray:
        """One category's counts in one column, per row: a read-only uint32 view of :attr:`counts`.

        Raises:
            ValueError: *code* was not requested, or *column* is unknown.
        """
        return self.counts[:, self._slot(code), self._column(column)]

    def sum(self, codes: Iterable[str], column: str = RELATIVES) -> np.ndarray:
        """The per-row sum of several categories' counts in one column, as a new int64 array.

        Each code's column is added in place into one int64 result, so the
        call allocates the result alone.  Codes are summed as given, so a
        group of overlapping categories is folded by the caller.

        Raises:
            TypeError: *codes* is a bare ``str``.
            ValueError: A code appears twice or was not requested, or
                *column* is unknown.
        """
        if isinstance(codes, str):
            raise TypeError("codes must be an iterable of codes, not a single str")
        codes = tuple(codes)
        repeated = sorted({code for code in codes if codes.count(code) > 1})
        if repeated:
            raise ValueError(f"codes repeats {repeated}; each category is summed once")
        column_index = self._column(column)
        slots = [self._slot(code) for code in codes]
        total = np.zeros(len(self.counts), dtype=np.int64)
        for slot in slots:
            total += self.counts[:, slot, column_index]
        return total

    def _slot(self, code: str) -> int:
        try:
            return self.categories.index(code)
        except ValueError:
            raise ValueError(f"{code!r} was not requested; the categories are {self.categories}") from None

    def _column(self, name: str) -> int:
        try:
            return self.columns.index(name)
        except ValueError:
            raise ValueError(f"no column {name!r}; the columns are {self.columns}") from None


def _out_of_range(field: str, position: int, value: object) -> PedigreeValidationError:
    return PedigreeValidationError(
        "value_out_of_range",
        f"{field} value at position {position} is beyond ±2**53 and would round to float64",
        field=field,
        position=position,
        value=value,
        minimum=-EXACT_INTEGER,
        maximum=EXACT_INTEGER,
    )


def _rows(field: str, values: object, n: int, converted: dict[int, np.ndarray]) -> np.ndarray:
    """One real value per receiver row as contiguous float64, borrowed when it already is.

    *converted* maps the ``id`` of every input already converted in this
    call to its float64 array, so two columns passing one object share it.
    """
    if id(values) in converted:
        return converted[id(values)]
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
    if array.dtype.kind not in "iuf":
        raise TypeError(f"{field} must be a real numeric array, got dtype {array.dtype}")
    if array.dtype.kind == "f" and array.dtype.itemsize > 8:
        raise TypeError(f"{field} has dtype {array.dtype}, wider than float64, and would round")
    if array.dtype.kind in "iu":
        beyond = np.flatnonzero((array > EXACT_INTEGER) | (array < -EXACT_INTEGER))
        if len(beyond):
            raise _out_of_range(field, int(beyond[0]), int(array[beyond[0]]))
    converted[id(values)] = np.ascontiguousarray(array, dtype=np.float64)
    return converted[id(values)]


def _scalar(field: str, value: numbers.Real | np.bool_) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{field} must be a real number, not bool")
    if isinstance(value, np.floating) and value.dtype.itemsize > 8:
        raise TypeError(f"{field} has dtype {value.dtype}, wider than float64, and would round")
    if isinstance(value, numbers.Integral) and abs(int(value)) > EXACT_INTEGER:
        raise _out_of_range(field, 0, int(value))
    return float(value)


def _column(
    name: str, entry: object, n: int, converted: dict[int, np.ndarray]
) -> tuple[np.ndarray, np.ndarray | float]:
    """The engine's ``(relative, threshold)`` pair for one named column."""
    if not isinstance(entry, tuple) or len(entry) != 2:
        raise TypeError(f"thresholds[{name!r}] must be a (relative, threshold) tuple, got {entry!r}")
    relative, threshold = entry
    field = f"thresholds[{name!r}]"
    if isinstance(threshold, (numbers.Real, np.bool_)):
        side = _scalar(f"{field}.threshold", threshold)
    else:
        side = _rows(f"{field}.threshold", threshold, n, converted)
    return _rows(f"{field}.relative", relative, n, converted), side


def relatives_per_person(
    graph: PedigreeGraph,
    view: PedigreeView | None,
    selection: RelationshipSelection,
    thresholds: Mapping[str, tuple[ArrayLike, ArrayLike | float]] | None,
) -> RelativesPerPerson:
    """Validate the columns, run the engine and wrap the counts for *graph* or *view*.

    A receiver of fewer than two rows or an empty selection has no pairs
    and gets zero counts without running the engine.
    """
    thresholds = {} if thresholds is None else thresholds
    _check_names("threshold column", thresholds)
    if RELATIVES in thresholds:
        raise ValueError(f"{RELATIVES!r} is the pair-count column and cannot name a threshold column")
    if len(thresholds) > MAX_COLUMNS:
        raise ValueError(f"at most {MAX_COLUMNS} threshold columns are supported, got {len(thresholds)}")
    n = graph.n_individuals if view is None else len(view)
    converted: dict[int, np.ndarray] = {}
    columns = [_column(name, entry, n, converted) for name, entry in thresholds.items()]
    codes = selection.ordered
    names = (RELATIVES, *thresholds)

    threads = thread_budget()
    if selection.top_degree is None or n < 2:
        counts = np.zeros((n, len(codes), len(names)), dtype=np.uint32)
        counts.setflags(write=False)
        return RelativesPerPerson(codes, names, counts)
    view_rows = None if view is None else view._graph_to_view()
    compact = view is not None and _should_compact_view(graph.n_individuals, n)
    logger.info(
        "relatives_per_person: max_degree=%d, %d categories, %d threshold columns, threads=%d%s",
        selection.top_degree,
        len(codes),
        len(columns),
        threads,
        ", view" if view_rows is not None else "",
    )
    start = time.perf_counter()
    flat, rows, n_categories, stride, lanes, lane_pairs = _native.relatives_per_person(
        graph._built,
        max_degree=selection.top_degree,
        requested=list(codes),
        threads=threads,
        columns=columns,
        view_rows=view_rows,
        compact=compact,
    )
    assert (rows, n_categories, stride) == (n, len(codes), len(names)), "native layout disagrees with the request"
    logger.info(
        "relatives_per_person total: %d pairs in %.3fs on %d lane(s), pairs per lane %s",
        sum(lane_pairs),
        time.perf_counter() - start,
        lanes,
        lane_pairs,
    )
    return RelativesPerPerson(codes, names, _own_native(flat, np.uint32).reshape(rows, n_categories, stride))
