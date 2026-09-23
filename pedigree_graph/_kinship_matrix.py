"""Complete, relationship-limited, and approximate-support kinship matrices.

All three public matrix families return the same CSC representation and the
same ADR 0009 float32 recurrence values.  They differ only in support:

* ``kinship_matrix`` keeps every nonzero pedigree kinship;
* ``relationship_kinship_matrix`` keeps closest-category pairs selected by the
  relationship engine;
* ``approximate_kinship_matrix`` keeps the old propagation-pruned support, then
  discards the propagated values and recomputes every retained coefficient.

The complete and approximate matrices are built by the core's depth-major DP
(``_native.kinship_csc`` and ``_native.approximate_kinship_csc``), which
returns the three CSC arrays in graph rows; the approximate support is
captured during one complete retiring pass of the same DP.  Sparse
relationship-selected support is filled by one native walk of the recurrence
over the CSC support itself (``_native.kinship_support_values``).  All three
implement the same pinned recurrence bits.
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

import numpy as np
import scipy.sparse as sp

from pedigree_graph import _native
from pedigree_graph._errors import ResourceError
from pedigree_graph._input import _own_native
from pedigree_graph._relationship_pairs import check_execution, relationship_pairs
from pedigree_graph._selection import RelationshipSelection
from pedigree_graph._threads import thread_budget

if TYPE_CHECKING:
    from collections.abc import Iterable

    from pedigree_graph._core import PedigreeGraph
    from pedigree_graph.relationships import RelationshipPairs

logger = logging.getLogger(__name__)

_INT32_MAX = int(np.iinfo(np.int32).max)


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
        Values follow represented parent edges alone, so they derive from
        structural depth and supplied generation labels never enter them.

        Returns:
            The complete ``n_individuals × n_individuals`` kinship matrix.
        """
        return complete_kinship_matrix(self)

    def relationship_kinship_matrix(
        self: PedigreeGraph,
        *,
        max_degree: int | None = None,
        categories: Iterable[str] | None = None,
        execution: str = "speed",
    ) -> sp.csc_matrix:
        """Return kinship on selected closest-category pairs plus the diagonal.

        Exactly one selector is required, with the same output-filter and
        dependency-closure semantics as :meth:`relationship_pairs`.  All
        selected pairs are classified through the complete graph; every
        retained float32 value is bit-identical to :meth:`pair_kinship`.
        Classification and values alike derive from structural depth, and
        supplied generation labels never enter either.

        Args:
            max_degree: Select every category at or below this degree (0-5).
                Exclusive with *categories*.
            categories: Registry codes to select, any order.  Exclusive with
                *max_degree*.
            execution: How the pair blocks this is built from are assembled,
                as :meth:`relationship_pairs` defines it.  This method holds
                the blocks and the support at once, so ``"memory"`` is worth
                having on a large selection; the matrix is identical either
                way, and so is the cache entry.

        Returns:
            A cached full-symmetric CSC matrix with read-only float32 data and
            int32 indices/indptr.

        Raises:
            TypeError: Both selectors, neither, or malformed categories.
            ValueError: *execution* is not ``"speed"`` or ``"memory"``.
            PedigreeValidationError: As :meth:`relationship_pairs`.
            ResourceError: If CSC or allocation capacity is exceeded.
        """
        return relationship_kinship_matrix(
            self, RelationshipSelection.parse(max_degree, categories), check_execution(execution)
        )

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
        bit-identical to :meth:`pair_kinship`.  The support and the values both
        derive from structural depth; supplied generation labels never enter
        either.

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


def _exactify_support(graph: PedigreeGraph, matrix: sp.csc_matrix) -> sp.csc_matrix:
    """Replace every value on a symmetric CSC support with pair-recurrence bits.

    One native walk over the support: each upper entry and the diagonal is
    evaluated once through a call-local memo and written with its mirror, so
    the only scratch beyond the memo is the ``data`` array the matrix owns.
    """
    values = _native.kinship_support_values(
        graph._built,
        graph.depth,
        np.ascontiguousarray(matrix.indptr, dtype=np.int64),
        np.ascontiguousarray(matrix.indices, dtype=np.int32),
    )
    matrix.data = _own_native(values, np.float32)
    return matrix


def _native_csc(
    graph: PedigreeGraph, arrays: tuple[np.ndarray, np.ndarray, np.ndarray], *, operation: str
) -> sp.csc_matrix:
    """Wrap the three arrays the core handed over, frozen and without a copy.

    The constructor is handed the arrays to validate the shape, then the
    matrix is pointed at the native allocations themselves rather than the
    views SciPy wraps them in, as the relationship path does.
    """
    indptr, indices, data = (
        _own_native(arrays[0], np.int32),
        _own_native(arrays[1], np.int32),
        _own_native(arrays[2], np.float32),
    )
    matrix = _as_csc(graph.n_individuals, indptr, indices, data, operation=operation)
    matrix.indptr, matrix.indices, matrix.data = indptr, indices, data
    return _freeze_csc(matrix)


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
    matrix = _native_csc(graph, _native.kinship_csc(graph._built, graph.depth), operation="kinship_matrix")
    graph._complete_kinship_cache = matrix
    logger.info(
        "kinship_matrix: n=%d, nnz=%d, %.2fs",
        graph.n_individuals,
        matrix.nnz,
        time.perf_counter() - started,
    )
    return matrix


def relationship_kinship_matrix(
    graph: PedigreeGraph, selection: RelationshipSelection, execution: str = "speed"
) -> sp.csc_matrix:
    """Return the cached matrix on selected closest-category support.

    The cache is keyed by the selection's canonical code order, so a cutoff
    and the explicit code list it names are one entry, not two.  *execution*
    is not part of the key: it changes what the pair blocks cost to build,
    never what they contain.
    """
    key = selection.ordered
    cached = graph._relationship_kinship_cache.get(key)
    if cached is not None:
        return cached

    started = time.perf_counter()
    pairs = relationship_pairs(graph, selection, execution)
    matrix = _support_from_relationships(graph, pairs)
    _exactify_support(graph, matrix)
    matrix = _freeze_csc(matrix)
    graph._relationship_kinship_cache[key] = matrix
    logger.info(
        "relationship_kinship_matrix: n=%d, nnz=%d, requested=%s, %.2fs",
        graph.n_individuals,
        matrix.nnz,
        ",".join(key),
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
    matrix = _native_csc(
        graph,
        _native.approximate_kinship_csc(graph._built, graph.depth, threshold),
        operation="approximate_kinship_matrix",
    )
    graph._approximate_kinship_cache[threshold] = matrix
    logger.info(
        "approximate_kinship_matrix: n=%d, nnz=%d, min_propagated_kinship=%.4g, %.2fs",
        graph.n_individuals,
        matrix.nnz,
        threshold,
        time.perf_counter() - started,
    )
    return matrix
