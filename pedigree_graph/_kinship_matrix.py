"""Complete, relationship-limited, and approximate-support kinship matrices.

All three public matrix families return the same CSC representation and the
same ADR 0009 float32 recurrence values.  They differ only in support:

* ``kinship_matrix`` keeps every nonzero pedigree kinship;
* ``relationship_kinship_matrix`` keeps closest-category pairs selected by the
  relationship engine;
* ``approximate_kinship_matrix`` keeps the old propagation-pruned support, then
  discards the propagated values and recomputes every retained coefficient.

Approximate-support values are captured during one complete retiring DP pass:
complete exact rows exist only while descendants need them, and only retained
candidate positions reach the output CSC. Sparse relationship-selected support
continues to use deterministic fixed-size pair chunks. Both paths implement the
same pinned recurrence bits.
"""

from __future__ import annotations

__all__ = [
    "approximate_kinship_matrix",
    "complete_kinship_matrix",
    "relationship_kinship_matrix",
]

import logging
import time
from typing import TYPE_CHECKING

import numba
import numpy as np
import scipy.sparse as sp

from pedigree_graph._errors import ResourceError
from pedigree_graph._kinship_dp import _build_kinship_csc, _fill_candidate_kinship_values
from pedigree_graph._kinship_pairwise import memoised_kinship
from pedigree_graph._pair_extractor import _requested_codes
from pedigree_graph._registry import RELATIONSHIPS
from pedigree_graph._threads import thread_budget

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from pedigree_graph._core import PedigreeGraph
    from pedigree_graph.relationships import RelationshipPairs

logger = logging.getLogger(__name__)

_INT32_MAX = int(np.iinfo(np.int32).max)

# Sparse relationship support uses bounded pairwise recurrence chunks. Dense
# approximate support instead uses the retiring DP below; see
# benchmarks/matrix_exactification.md for the measured crossover rationale.
_EXACT_VALUE_CHUNK_SIZE = 1 << 20


def _freeze_csc(matrix: sp.csc_matrix) -> sp.csc_matrix:
    """Sort and mark the three cached CSC arrays read-only."""
    matrix.sort_indices()
    matrix.data.setflags(write=False)
    matrix.indices.setflags(write=False)
    matrix.indptr.setflags(write=False)
    return matrix


def _as_csc(
    n: int,
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    *,
    operation: str,
) -> sp.csc_matrix:
    """Construct a CSC matrix and translate public-path capacity failures."""
    try:
        matrix = sp.csc_matrix((data, indices, indptr), shape=(n, n), copy=False)
    except MemoryError as exc:
        raise ResourceError(
            "allocation_failed",
            f"{operation}: allocation failed while constructing the CSC matrix",
            operation=operation,
            requested_elements=int(data.size),
            dtype="float32/int32",
        ) from exc
    return matrix


def _check_nnz(nnz: int) -> None:
    """Reject a CSC structure whose int32 pointer cannot represent its nnz."""
    if nnz > _INT32_MAX:
        raise ResourceError(
            "csc_index_overflow",
            "kinship matrix nnz exceeds the int32 CSC index range",
            nnz=nnz,
            maximum=_INT32_MAX,
        )


def _validate_propagated_threshold(value: float) -> float:
    """Return a finite propagation threshold in ``[0, 1]``."""
    try:
        threshold = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"min_propagated_kinship must be a real number in [0, 1], got {value!r}") from exc
    if not np.isfinite(threshold) or threshold < 0.0 or threshold > 1.0:
        raise ValueError(f"min_propagated_kinship must be finite and in [0, 1], got {value!r}")
    # Normalise negative zero so it shares the complete-matrix cache.
    return 0.0 if threshold == 0.0 else threshold


class PedigreeMatrixMethods:
    """Matrix receiver methods mixed into :class:`PedigreeGraph`.

    Keeping these docstring-heavy public boundaries beside their implementation
    prevents the already-large graph class from becoming the matrix engine.
    """

    def kinship_matrix(self: PedigreeGraph) -> sp.csc_matrix:
        """Return the cached complete pedigree-expected kinship matrix.

        The matrix contains every nonzero pedigree kinship plus the diagonal.
        It is a SciPy CSC matrix with float32 data, int32 indices and indptr,
        sorted rows per column, and read-only cached arrays.  Every coefficient
        is bit-identical to :meth:`pair_kinship` for the same graph rows.

        Returns:
            The complete ``n_individuals × n_individuals`` kinship matrix.
        """
        return complete_kinship_matrix(self)

    def relationship_kinship_matrix(
        self: PedigreeGraph,
        *,
        max_degree: int | None = None,
        categories: Iterable[str] | None = None,
    ) -> sp.csc_matrix:
        """Return kinship on selected closest-category pairs plus the diagonal.

        Exactly one selector is required, with the same output-filter and
        dependency-closure semantics as :meth:`relationship_pairs`.  All
        selected pairs are classified through the complete graph; every
        retained float32 value is bit-identical to :meth:`pair_kinship`.

        Args:
            max_degree: Select every category at or below this degree (0-5).
                Exclusive with *categories*.
            categories: Registry codes to select, any order.  Exclusive with
                *max_degree*.

        Returns:
            A cached full-symmetric CSC matrix with read-only float32 data and
            int32 indices/indptr.

        Raises:
            TypeError: Both selectors, neither, or malformed categories.
            PedigreeValidationError: As :meth:`relationship_pairs`.
            ResourceError: If CSC or allocation capacity is exceeded.
        """
        return relationship_kinship_matrix(self, max_degree=max_degree, categories=categories)

    def approximate_kinship_matrix(
        self: PedigreeGraph,
        *,
        min_propagated_kinship: float = 0.001,
    ) -> sp.csc_matrix:
        """Return exact values on propagation-pruned candidate support.

        The threshold is applied to intermediate values while propagating the
        candidate structure.  It is **not** a final pedigree-expected-value
        cutoff: compared with thresholding :meth:`pair_kinship`, the support
        can contain false positives or false negatives.  Once the support is
        chosen, every retained coefficient (including the always-present
        diagonal) is recomputed with the pinned float32 recurrence and is
        bit-identical to :meth:`pair_kinship`.

        ``min_propagated_kinship=0`` delegates to :meth:`kinship_matrix`.
        This operation is intentionally full-graph-only; views expose no
        matrix method.

        The threshold selects support only.  It does not bound run time or
        peak memory: retained values are captured during one complete
        retiring DP pass, so a raised threshold costs about what the complete
        matrix costs and can exhaust memory at the same pedigree size.  See
        ``benchmarks/matrix_exactification.md``.  Raising the threshold to
        fit a large pedigree in RAM gains nothing.

        Args:
            min_propagated_kinship: Finite propagation threshold in ``[0, 1]``.

        Returns:
            A cached full-symmetric CSC matrix with read-only float32 data and
            int32 indices/indptr.

        Raises:
            ValueError: The threshold is non-finite or outside ``[0, 1]``.
            ResourceError: If recurrence memo, CSC, or allocation capacity is
                exceeded.
        """
        return approximate_kinship_matrix(self, min_propagated_kinship)


@numba.njit(cache=True)
def _write_symmetric_values(
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    positions: np.ndarray,
    columns: np.ndarray,
    values: np.ndarray,
) -> bool:
    """Write one upper-triangle value chunk and its transposed positions.

    Returns false only if the supplied structure is not symmetric.  Public
    callers build the structure themselves, so that result is an internal
    invariant failure rather than a user error.
    """
    for k in range(positions.shape[0]):
        pos = positions[k]
        row = indices[pos]
        col = columns[k]
        value = values[k]
        data[pos] = value
        if row == col:
            continue
        lo = indptr[row]
        hi = indptr[row + 1]
        while lo < hi:
            mid = (lo + hi) // 2
            if indices[mid] < col:
                lo = mid + 1
            else:
                hi = mid
        if lo >= indptr[row + 1] or indices[lo] != col:
            return False
        data[lo] = value
    return True


def _upper_support_chunks(
    matrix: sp.csc_matrix,
    chunk_size: int,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Yield upper-triangle ``(rows, columns, data_positions)`` in CSC order."""
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    rows = np.empty(chunk_size, dtype=np.int32)
    columns = np.empty(chunk_size, dtype=np.int32)
    positions = np.empty(chunk_size, dtype=np.int32)
    used = 0

    for column in range(matrix.shape[1]):
        start = int(matrix.indptr[column])
        end = int(matrix.indptr[column + 1])
        column_rows = matrix.indices[start:end]
        upper_count = int(np.searchsorted(column_rows, column, side="right"))
        offset = 0
        while offset < upper_count:
            take = min(chunk_size - used, upper_count - offset)
            source_start = start + offset
            source_end = source_start + take
            target_end = used + take
            rows[used:target_end] = matrix.indices[source_start:source_end]
            columns[used:target_end] = column
            positions[used:target_end] = np.arange(source_start, source_end, dtype=np.int32)
            used = target_end
            offset += take
            if used == chunk_size:
                yield rows, columns, positions
                used = 0

    if used:
        yield rows[:used], columns[:used], positions[:used]


def _exactify_support(
    graph: PedigreeGraph,
    matrix: sp.csc_matrix,
    *,
    chunk_size: int = _EXACT_VALUE_CHUNK_SIZE,
) -> sp.csc_matrix:
    """Replace every value on a symmetric CSC support with pair-recurrence bits.

    Each chunk runs from the graph's retained pair memo and leaves it for the
    next, so the diagonal closure a preceding ``pair_kinship`` already walked
    is not walked again, and a following ``pair_kinship`` starts from the
    support's closure.
    """
    topology = graph._topology
    for first, second, positions in _upper_support_chunks(matrix, chunk_size):
        values = memoised_kinship(graph, topology.translate(first), topology.translate(second))
        if not _write_symmetric_values(matrix.indptr, matrix.indices, matrix.data, positions, second, values):
            raise AssertionError("kinship support must be symmetric")
    return matrix


def _topological_candidate_index(graph: PedigreeGraph, matrix: sp.csc_matrix) -> sp.csc_matrix:
    """Map upper graph-space support to stable-topology columns and output positions."""
    count = (matrix.nnz + graph.n_individuals) // 2
    lower = np.empty(count, dtype=np.int32)
    upper = np.empty(count, dtype=np.int32)
    output_positions = np.empty(count, dtype=np.int32)
    topology = graph._topology
    offset = 0
    for first, second, positions in _upper_support_chunks(matrix, _EXACT_VALUE_CHUNK_SIZE):
        end = offset + len(first)
        if end > count:
            raise AssertionError("upper candidate count must match symmetric CSC nnz")
        topo_first = np.asarray(topology.translate(first), dtype=np.int32)
        topo_second = np.asarray(topology.translate(second), dtype=np.int32)
        lower[offset:end] = np.minimum(topo_first, topo_second)
        upper[offset:end] = np.maximum(topo_first, topo_second)
        output_positions[offset:end] = positions
        offset = end
    if offset != count:
        raise AssertionError("upper candidate count must match symmetric CSC nnz")
    target = sp.coo_matrix(
        (output_positions, (lower, upper)),
        shape=matrix.shape,
        dtype=np.int32,
    ).tocsc()
    if target.nnz != count:
        raise AssertionError("topological candidate support must not contain duplicate pairs")
    return target


def _exactify_approximate_support(graph: PedigreeGraph, matrix: sp.csc_matrix) -> sp.csc_matrix:
    """Fill dense candidate support through one complete retiring DP pass."""
    try:
        target = _topological_candidate_index(graph, matrix)
        matrix.data.fill(np.nan)
        _fill_candidate_kinship_values(
            graph.n_individuals,
            graph.mother_rows,
            graph.father_rows,
            graph.twin_rows,
            graph.depth,
            target.indptr,
            target.indices,
            target.data,
            matrix.data,
        )
        for _first, second, positions in _upper_support_chunks(matrix, _EXACT_VALUE_CHUNK_SIZE):
            values = matrix.data[positions]
            if np.isnan(values).any():
                raise AssertionError("complete DP did not emit every approximate-support candidate")
            if not _write_symmetric_values(
                matrix.indptr,
                matrix.indices,
                matrix.data,
                positions,
                second,
                values,
            ):
                raise AssertionError("kinship support must be symmetric")
    except MemoryError as exc:
        raise ResourceError(
            "allocation_failed",
            "approximate_kinship_matrix: allocation failed while exactifying candidate values",
            operation="approximate_kinship_matrix",
            requested_elements=matrix.nnz,
            dtype="float32/int32",
        ) from exc
    return matrix


def _support_from_relationships(graph: PedigreeGraph, pairs: RelationshipPairs) -> sp.csc_matrix:
    """Build a symmetric CSC support from requested closest-category blocks."""
    n = graph.n_individuals
    pair_count = sum(len(block) for block in pairs.values())
    nnz = n + 2 * pair_count
    _check_nnz(nnz)
    try:
        first = np.empty(pair_count, dtype=np.int32)
        second = np.empty(pair_count, dtype=np.int32)
        offset = 0
        for block in pairs.values():
            count = len(block)
            first[offset : offset + count] = block.first_rows
            second[offset : offset + count] = block.second_rows
            offset += count
        diagonal = np.arange(n, dtype=np.int32)
        rows = np.concatenate((first, second, diagonal))
        columns = np.concatenate((second, first, diagonal))
        data = np.empty(nnz, dtype=np.float32)
        data.fill(np.nan)
        matrix = sp.coo_matrix((data, (rows, columns)), shape=(n, n)).tocsc()
    except MemoryError as exc:
        raise ResourceError(
            "allocation_failed",
            "relationship_kinship_matrix: allocation failed while assembling relationship support",
            operation="relationship_kinship_matrix",
            requested_elements=nnz,
            dtype="float32/int32",
        ) from exc
    if matrix.nnz != nnz:
        raise AssertionError("closest-category relationship support must not contain duplicate pairs")
    return matrix


def complete_kinship_matrix(graph: PedigreeGraph) -> sp.csc_matrix:
    """Return the cached complete pedigree-expected kinship matrix."""
    thread_budget()
    cached = graph._complete_kinship_cache
    if cached is not None:
        return cached
    started = time.perf_counter()
    try:
        indptr, indices, data = _build_kinship_csc(
            graph.n_individuals,
            graph.mother_rows,
            graph.father_rows,
            graph.twin_rows,
            graph.depth,
            0.0,
        )
    except MemoryError as exc:
        raise ResourceError(
            "allocation_failed",
            "kinship_matrix: allocation failed while computing the complete matrix",
            operation="kinship_matrix",
            requested_elements=graph.n_individuals,
            dtype="float32/int32",
        ) from exc
    matrix = _freeze_csc(_as_csc(graph.n_individuals, indptr, indices, data, operation="kinship_matrix"))
    graph._complete_kinship_cache = matrix
    logger.info(
        "kinship_matrix: n=%d, nnz=%d, %.2fs",
        graph.n_individuals,
        matrix.nnz,
        time.perf_counter() - started,
    )
    return matrix


def relationship_kinship_matrix(
    graph: PedigreeGraph,
    *,
    max_degree: int | None,
    categories: Iterable[str] | None,
) -> sp.csc_matrix:
    """Return the cached matrix on selected closest-category support."""
    # Materialise a one-shot iterable once: validation and pair extraction both
    # need to see the same selector.
    category_arg: Iterable[str] | None = categories
    if categories is not None and not isinstance(categories, str):
        category_arg = tuple(categories)
    requested = _requested_codes(max_degree, category_arg)
    if max_degree is not None:
        key: tuple[str, object] = ("max_degree", int(max_degree))
    else:
        key = ("categories", tuple(code for code in RELATIONSHIPS if code in requested))
    cached = graph._relationship_kinship_cache.get(key)
    if cached is not None:
        return cached

    started = time.perf_counter()
    pairs = graph.relationship_pairs(max_degree=max_degree, categories=category_arg)
    matrix = _support_from_relationships(graph, pairs)
    _exactify_support(graph, matrix)
    matrix = _freeze_csc(matrix)
    graph._relationship_kinship_cache[key] = matrix
    logger.info(
        "relationship_kinship_matrix: n=%d, nnz=%d, requested=%s, %.2fs",
        graph.n_individuals,
        matrix.nnz,
        ",".join(code for code in RELATIONSHIPS if code in requested),
        time.perf_counter() - started,
    )
    return matrix


def approximate_kinship_matrix(graph: PedigreeGraph, min_propagated_kinship: float) -> sp.csc_matrix:
    """Return exact values on the cached propagation-pruned candidate support."""
    threshold = _validate_propagated_threshold(min_propagated_kinship)
    if threshold == 0.0:
        return complete_kinship_matrix(graph)
    thread_budget()
    cached = graph._approximate_kinship_cache.get(threshold)
    if cached is not None:
        return cached

    started = time.perf_counter()
    try:
        indptr, indices, propagated = _build_kinship_csc(
            graph.n_individuals,
            graph.mother_rows,
            graph.father_rows,
            graph.twin_rows,
            graph.depth,
            threshold,
        )
    except MemoryError as exc:
        raise ResourceError(
            "allocation_failed",
            "approximate_kinship_matrix: allocation failed while computing candidate support",
            operation="approximate_kinship_matrix",
            requested_elements=graph.n_individuals,
            dtype="float32/int32",
        ) from exc
    matrix = _as_csc(
        graph.n_individuals,
        indptr,
        indices,
        propagated,
        operation="approximate_kinship_matrix",
    )
    _exactify_approximate_support(graph, matrix)
    matrix = _freeze_csc(matrix)
    graph._approximate_kinship_cache[threshold] = matrix
    logger.info(
        "approximate_kinship_matrix: n=%d, nnz=%d, min_propagated_kinship=%.4g, %.2fs",
        graph.n_individuals,
        matrix.nnz,
        threshold,
        time.perf_counter() - started,
    )
    return matrix
