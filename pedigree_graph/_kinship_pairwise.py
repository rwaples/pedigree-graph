"""Pedigree-expected kinship for requested pairs: the pinned float32 recurrence (ADR 0009).

``pair_kinship`` answers a *specific set of pairs* without materialising the
``n x n`` matrix.  Its value is defined as one float32 Karigl recurrence with a
pinned peel rule, and the matrix DP implements the same definition, so a pair
value and the matrix entry for that pair are bit-identical within one graph.

The recurrence, over graph rows ``a`` and ``b`` with structural ``depth``:

* ``phi(a, a) = (1 + phi(mother_a, father_a)) / 2``; a missing parent gives
  ``phi(a, a) = 1/2``.
* MZ co-twins take the self-kinship of the peeled twin (they share parents, so
  either twin's self-kinship is the same value).
* Otherwise peel the endpoint ``c`` with the greater structural depth, ties
  broken by the greater row: ``phi(a, b) = (phi(mother_c, o) + phi(father_c, o)) / 2``
  where ``o`` is the other endpoint and a missing parent contributes ``0``.
* Two founders that are neither the same row nor co-twins have ``phi = 0``.

Every half-sum is the correctly rounded float32 of two float32 operands, and
the memo stores float32.  Consequences the public docstrings promise:

* reversed endpoint order gives identical bits (the peel rule does not read
  the argument order);
* a returned ``0`` means the exact kinship is ``0``;
* the result never depends on call history, because no cached matrix is read;
* two graphs built from the same pedigree in different row orders may differ
  on deep inbred pairs within ``2 * (depth_a + depth_b + 1) * 2**-25``, since
  the depth tie-break is by row (ADR 0009);
* a caller thresholding against a non-dyadic cutoff widens to float64 first.

The production kernel runs in the graph's private stable depth-major order
(:mod:`pedigree_graph._topology`), where "greater depth, then greater row" is
simply "greater row": the receiver boundary translates endpoints into that
order once, and the kernel never reads depth.  Reading depth per node instead
measured 7 to 9 percent slower on the 30k fixture.  The pure-Python oracle
works in graph space with explicit depth, so the parity test between the two
also checks the translation.

Cost is ``O(P + distinct ancestor-pairs reached)`` for ``P`` requested pairs,
shared across the request through one open-addressing memo keyed on the
canonical ``lo * n + hi`` row pair.  The memo, not the output, dominates memory
(12 bytes per entry).

Two implementations of the same recurrence live here: :func:`_pairwise_kinship_py`
is the readable recursive oracle used by the property tests, and
:func:`pairwise_kinship` is the ``@njit`` production kernel.
"""

from __future__ import annotations

__all__ = ["graph_pair_kinship", "pairwise_kinship", "view_pair_kinship"]

from dataclasses import dataclass
from functools import cache
from types import MappingProxyType
from typing import TYPE_CHECKING

import numba
import numpy as np

from pedigree_graph._errors import PedigreeValidationError, ResourceError
from pedigree_graph._input import _INT32_MAX, _check_shape, _coerce_to_int64, _FieldSpec, _invalid_integer
from pedigree_graph._kinship_depth import _check_topological
from pedigree_graph._threads import thread_budget
from pedigree_graph._topology import owned_readonly, readonly
from pedigree_graph.relationships import RelationshipPairBlock, RelationshipPairs

if TYPE_CHECKING:
    from collections.abc import Mapping

    from pedigree_graph._core import PedigreeGraph
    from pedigree_graph._view import CoordinateToken, PedigreeView

# Open-addressing memo load factor: grow when entries exceed this fraction of
# capacity.  0.7 keeps linear-probe chains short without wasting much memory.
_MEMO_LOAD_NUM = 7
_MEMO_LOAD_DEN = 10

# Hard backstop on memo capacity (slots): a runaway guard, not a working limit.
# Legitimate memos scale ~linearly with the requested pairs (145M entries for
# the degree-3 pairs of a 536k-row pedigree); this ceiling sits above that and
# only trips on pathological inbreeding where the memo heads toward n^2.
_MEMO_CAP_LIMIT = 1 << 31

# Stats array layout returned by the core kernel.
_STAT_ENTRIES = 0
_STAT_CAPACITY = 1
_STAT_GROWS = 2
_STAT_MAX_STACK = 3
_STAT_OVERFLOW = 4


def _pairwise_kinship_py(
    mother: np.ndarray,
    father: np.ndarray,
    twin: np.ndarray,
    depth: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
) -> np.ndarray:
    """Pedigree-expected kinship per requested pair, as a readable recursive oracle.

    Same recurrence, peel rule, and float32 arithmetic as :func:`pairwise_kinship`,
    written with ``functools.cache`` recursion over graph-space arrays and an
    explicit depth, so a reader can check it against the module docstring line
    by line.  Small pedigrees only: Python recursion grows with pedigree depth.

    Args:
        mother: Mother row per graph row, ``-1`` when absent.
        father: Father row per graph row, ``-1`` when absent.
        twin: MZ co-twin row per graph row, ``-1`` when absent.
        depth: Structural depth per graph row.
        first: First endpoint of each requested pair, as graph rows.
        second: Second endpoint of each requested pair, as graph rows.

    Returns:
        float32 array of length ``len(first)``, positionally aligned to the
        inputs.
    """
    mother = np.asarray(mother, dtype=np.int64)
    father = np.asarray(father, dtype=np.int64)
    twin = np.asarray(twin, dtype=np.int64)
    depth = np.asarray(depth, dtype=np.int64)
    half = np.float32(0.5)
    one = np.float32(1.0)
    zero = np.float32(0.0)

    @cache
    def _phi(a: int, b: int) -> np.float32:
        if a > b:
            a, b = b, a
        if depth[a] > depth[b]:
            other, peeled = b, a
        else:
            other, peeled = a, b
        m = int(mother[peeled])
        f = int(father[peeled])
        if other == peeled or twin[other] == peeled or twin[peeled] == other:
            if m < 0 or f < 0:
                return half
            return half * (one + _phi(m, f))
        left = _phi(m, other) if m >= 0 else zero
        right = _phi(f, other) if f >= 0 else zero
        return half * (left + right)

    out = np.empty(len(first), dtype=np.float32)
    for k in range(len(first)):
        out[k] = _phi(int(first[k]), int(second[k]))
    return out


# ---------------------------------------------------------------------------
# Numba production kernel: iterative work-stack DFS + open-addressing memo
# ---------------------------------------------------------------------------


@numba.njit(cache=True, inline="always")
def _next_pow2(x: int) -> int:
    """Smallest power of two >= max(x, 1)."""
    p = 1
    while p < x:
        p *= 2
    return p


@numba.njit(cache=True, inline="always")
def _canon_key(x: int, y: int, n: int) -> int:
    """Canonical ``lo * n + hi`` memo key for the unordered pair ``{x, y}``."""
    if x <= y:
        return x * n + y
    return y * n + x


@numba.njit(cache=True, inline="always")
def _memo_slot(memo_keys: np.ndarray, key: int) -> int:
    """Linear-probe to the slot holding ``key`` or the first empty (-1) slot.

    Capacity is a power of two, so ``& mask`` replaces the modulo.
    """
    mask = memo_keys.shape[0] - 1
    idx = key & mask
    while memo_keys[idx] != -1 and memo_keys[idx] != key:
        idx = (idx + 1) & mask
    return idx


@numba.njit(cache=True, inline="always")
def _grow_stack(stack: np.ndarray) -> np.ndarray:
    """Double the work-stack capacity, preserving contents."""
    new = np.empty(stack.shape[0] * 2, dtype=np.int64)
    new[: stack.shape[0]] = stack
    return new


@numba.njit(cache=True)
def _memo_grow(memo_keys: np.ndarray, memo_vals: np.ndarray):
    """Double the memo capacity and rehash live entries."""
    new_cap = memo_keys.shape[0] * 2
    new_keys = np.full(new_cap, -1, dtype=np.int64)
    new_vals = np.empty(new_cap, dtype=np.float32)
    mask = new_cap - 1
    for i in range(memo_keys.shape[0]):
        k = memo_keys[i]
        if k != -1:
            idx = k & mask
            while new_keys[idx] != -1:
                idx = (idx + 1) & mask
            new_keys[idx] = k
            new_vals[idx] = memo_vals[i]
    return new_keys, new_vals


@numba.njit(cache=True)
def _pairwise_kinship_core(
    mother: np.ndarray,
    father: np.ndarray,
    twin: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    n: int,
    cap_limit: int,
):
    """Pedigree-expected kinship per requested pair; the production recurrence kernel.

    Rows are in stable depth-major order, so peeling the greater row is the
    pinned depth-then-row rule.  Iterative (explicit work-stack) post-order
    evaluation of the module recurrence with a hand-rolled open-addressing
    ``int64 -> float32`` memo keyed on the canonical ``lo * n + hi`` row pair.  Returns ``(out, stats)``
    where ``stats`` is ``int64[5]`` = ``[entries, capacity, grows,
    max_stack_depth, overflow]``; ``overflow`` is ``1`` when the memo would
    exceed ``cap_limit`` slots, in which case ``out`` is unfinished.

    The combine expressions mirror :func:`_pairwise_kinship_py` term-for-term
    in float32, so the two agree to the bit whatever the traversal order.
    """
    half = np.float32(0.5)
    one = np.float32(1.0)
    zero = np.float32(0.0)
    p = first.shape[0]
    out = np.empty(p, dtype=np.float32)
    stats = np.zeros(5, dtype=np.int64)
    if p == 0:
        return out, stats

    # Memo sized from the request count, within the cap; grows geometrically on demand.
    cap = min(_next_pow2(4 * p if p > 4 else 16), cap_limit)
    memo_keys = np.full(cap, -1, dtype=np.int64)
    memo_vals = np.empty(cap, dtype=np.float32)
    entries = 0
    grows = 0

    # Explicit work stack of canonical keys.
    stack = np.empty(_next_pow2(4 * p if p > 16 else 64), dtype=np.int64)
    top = 0
    max_stack = 0

    # Two-phase iterative post-order: a key is pushed positive for the *expand*
    # phase (discover dependencies) and re-pushed as ``-(key + 1)`` for the
    # *compute* phase (combine resolved dependencies).  LIFO ordering guarantees
    # a node's whole subtree is computed before its compute marker is reached, so
    # each distinct node is computed exactly once even when shared across pairs.
    for k in range(p):
        root = _canon_key(first[k], second[k], n)
        if memo_keys[_memo_slot(memo_keys, root)] == root:
            continue  # already resolved by an earlier requested pair
        stack[0] = root
        top = 1
        while top > 0:
            if top > max_stack:
                max_stack = top
            s = stack[top - 1]
            top -= 1

            key = s if s >= 0 else -s - 1
            other = key // n
            peeled = key - other * n
            m = mother[peeled]
            f = father[peeled]
            self_like = other == peeled or twin[other] == peeled or twin[peeled] == other

            if s >= 0:
                # --- expand phase ---
                if memo_keys[_memo_slot(memo_keys, key)] == key:
                    continue  # already computed (shared dependency)
                if self_like:
                    if m < 0 or f < 0:
                        slot = _memo_slot(memo_keys, key)  # founder self-kinship
                        memo_keys[slot] = key
                        memo_vals[slot] = half
                        entries += 1
                    else:
                        if top + 2 > stack.shape[0]:
                            stack = _grow_stack(stack)
                        stack[top] = -(key + 1)  # schedule compute
                        top += 1
                        d0 = _canon_key(m, f, n)
                        if memo_keys[_memo_slot(memo_keys, d0)] != d0:
                            stack[top] = d0
                            top += 1
                elif m < 0 and f < 0:
                    slot = _memo_slot(memo_keys, key)  # unrelated founder
                    memo_keys[slot] = key
                    memo_vals[slot] = zero
                    entries += 1
                else:
                    if top + 3 > stack.shape[0]:
                        stack = _grow_stack(stack)
                    stack[top] = -(key + 1)  # schedule compute
                    top += 1
                    if m >= 0:
                        d0 = _canon_key(m, other, n)
                        if memo_keys[_memo_slot(memo_keys, d0)] != d0:
                            stack[top] = d0
                            top += 1
                    if f >= 0:
                        d1 = _canon_key(f, other, n)
                        if memo_keys[_memo_slot(memo_keys, d1)] != d1:
                            stack[top] = d1
                            top += 1
            else:
                # --- compute phase --- dependencies are now memoized
                if self_like:
                    v0 = memo_vals[_memo_slot(memo_keys, _canon_key(m, f, n))]
                    value = half * (one + v0)
                else:
                    v0 = memo_vals[_memo_slot(memo_keys, _canon_key(m, other, n))] if m >= 0 else zero
                    v1 = memo_vals[_memo_slot(memo_keys, _canon_key(f, other, n))] if f >= 0 else zero
                    value = half * (v0 + v1)
                slot = _memo_slot(memo_keys, key)
                memo_keys[slot] = key
                memo_vals[slot] = value
                entries += 1

            # Grow the memo if the load factor is exceeded.  Covers every insert
            # (leaf in the expand phase, value in the compute phase); the rehash
            # reassigns the tables before the next iteration's probes.
            if entries * _MEMO_LOAD_DEN >= cap * _MEMO_LOAD_NUM:
                if cap * 2 > cap_limit:
                    stats[_STAT_CAPACITY] = cap
                    stats[_STAT_OVERFLOW] = 1
                    return out, stats
                memo_keys, memo_vals = _memo_grow(memo_keys, memo_vals)
                cap = cap * 2
                grows += 1

    # Final scatter: every requested key is now memoized.
    for k in range(p):
        key = _canon_key(first[k], second[k], n)
        out[k] = memo_vals[_memo_slot(memo_keys, key)]

    stats[_STAT_ENTRIES] = entries
    stats[_STAT_CAPACITY] = cap
    stats[_STAT_GROWS] = grows
    stats[_STAT_MAX_STACK] = max_stack
    return out, stats


def _prepare_inputs(
    mother: np.ndarray,
    father: np.ndarray,
    twin: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Coerce to contiguous read-only int64 and reject inputs the kernel cannot walk.

    int64 throughout: the kernel mixes row values into int64 memo keys on every
    step, and int32 inputs measured 6 percent slower on the 30k fixture.  The
    public boundary has already validated pair rows; the checks here only keep
    a direct kernel caller from reading outside the arrays or feeding an order
    the peel cannot terminate on.
    """
    mother = owned_readonly(mother, np.int64)
    father = owned_readonly(father, np.int64)
    twin = owned_readonly(twin, np.int64)
    first = owned_readonly(first, np.int64)
    second = owned_readonly(second, np.int64)
    n = mother.shape[0]
    if not (father.shape[0] == twin.shape[0] == n):
        raise ValueError("pairwise_kinship: mother, father, and twin must have one entry per row")
    if not _check_topological(mother, father, n):
        raise ValueError("pairwise_kinship: rows must be in depth-major order (PedigreeGraph._topological_parents)")
    if first.shape[0] != second.shape[0]:
        raise ValueError("pairwise_kinship: first and second must have one entry per pair")
    if first.size and (
        int(first.min()) < 0 or int(second.min()) < 0 or int(first.max()) >= n or int(second.max()) >= n
    ):
        raise ValueError(f"pairwise_kinship: pair rows must lie in [0, {n})")
    return mother, father, twin, first, second, n


def _run_kernel(
    mother: np.ndarray,
    father: np.ndarray,
    twin: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    cap_limit: int = _MEMO_CAP_LIMIT,
) -> tuple[np.ndarray, np.ndarray]:
    mother, father, twin, first, second, n = _prepare_inputs(mother, father, twin, first, second)
    out, stats = _pairwise_kinship_core(mother, father, twin, first, second, n, cap_limit)
    if stats[_STAT_OVERFLOW]:
        raise ResourceError(
            "memo_capacity_exceeded",
            "pair_kinship: the recurrence memo would exceed its capacity limit; the pedigree is too "
            "inbred or too deep for the direct path",
            operation="pair_kinship",
            capacity=int(stats[_STAT_CAPACITY]),
            maximum=cap_limit,
        )
    out.setflags(write=False)
    return out, stats


def pairwise_kinship(
    mother: np.ndarray,
    father: np.ndarray,
    twin: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
) -> np.ndarray:
    """Pedigree-expected kinship per requested pair (numba production path).

    Same recurrence and bits as :func:`_pairwise_kinship_py`, iterative and
    memoized in a nopython kernel so it scales to large pedigrees.  Every array
    is in the graph's stable depth-major order
    (``PedigreeGraph._topological_parents`` and ``Topology.translate``), where
    peeling the greater row is the pinned depth-then-row rule.

    Args:
        mother: Mother row per row, ``-1`` when absent.
        father: Father row per row, ``-1`` when absent.
        twin: MZ co-twin row per row, ``-1`` when absent.
        first: First endpoint of each requested pair.
        second: Second endpoint of each requested pair.

    Returns:
        Read-only float32 array of length ``len(first)``, positionally aligned
        to the inputs.

    Raises:
        ResourceError: ``memo_capacity_exceeded`` when the memo would pass
            ``_MEMO_CAP_LIMIT`` slots.
    """
    out, _ = _run_kernel(mother, father, twin, first, second)
    return out


def _pairwise_kinship_with_stats(
    mother: np.ndarray,
    father: np.ndarray,
    twin: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
) -> tuple[np.ndarray, dict[str, int]]:
    """Benchmark/debug wrapper: returns ``(out, stats_dict)``.

    ``stats_dict`` holds ``memo_entries``, ``memo_capacity``, ``memo_grows``,
    and ``max_stack_depth``, used by the profiling harness to confirm the memo
    stays bounded.  Not part of the production path.
    """
    out, stats = _run_kernel(mother, father, twin, first, second)
    return out, {
        "memo_entries": int(stats[_STAT_ENTRIES]),
        "memo_capacity": int(stats[_STAT_CAPACITY]),
        "memo_grows": int(stats[_STAT_GROWS]),
        "max_stack_depth": int(stats[_STAT_MAX_STACK]),
    }


# ---------------------------------------------------------------------------
# Receiver boundary: validate a pair query, run one kernel call, shape the result
# ---------------------------------------------------------------------------

_FIRST = _FieldSpec("first_rows", True, 0, _INT32_MAX, np.int32)
_SECOND = _FieldSpec("second_rows", True, 0, _INT32_MAX, np.int32)


@dataclass(frozen=True, slots=True)
class _PairQuery:
    """A validated pair request in receiver rows.

    Attributes:
        first: int32 first endpoints.
        second: int32 second endpoints, same length.
        block_lengths: ``(code, length)`` per registry code when the request was
            a :class:`RelationshipPairs`, so the flat result can be split back;
            ``None`` for a flat request.
    """

    first: np.ndarray
    second: np.ndarray
    block_lengths: tuple[tuple[str, int], ...] | None


def _row_out_of_range(argument: str, row: object, position: int, n_individuals: int) -> PedigreeValidationError:
    return PedigreeValidationError(
        "pair_row_out_of_range",
        f"{argument} row {row} at position {position} is outside the {n_individuals}-row receiver",
        argument=argument,
        row=row,
        position=position,
        n_individuals=n_individuals,
    )


def _coerce_rows(spec: _FieldSpec, values: object, n_individuals: int) -> np.ndarray:
    """Return one endpoint argument as int32 receiver rows.

    Checks run as shape, integer form, then range.  A value with no int64 form
    reads as out of range, so an endpoint argument fails with ``invalid_shape``,
    ``invalid_integer_value``, or ``pair_row_out_of_range`` and nothing else.
    """
    arr = np.asarray(values)
    _check_shape(spec, arr)
    try:
        rows, nulls = _coerce_to_int64(spec, arr)
    except PedigreeValidationError as err:
        if err.code != "value_out_of_range":
            raise
        position = err.fields["position"]
        assert isinstance(position, int)
        raise _row_out_of_range(spec.name, err.fields["value"], position, n_individuals) from None
    if nulls.any():
        raise _invalid_integer(spec.name, int(np.argmax(nulls)), "null")
    outside = (rows < 0) | (rows >= n_individuals)
    if outside.any():
        position = int(np.argmax(outside))
        raise _row_out_of_range(spec.name, int(rows[position]), position, n_individuals)
    return readonly(rows.astype(np.int32))


def _check_token(block: RelationshipPairBlock, token: CoordinateToken, receiver_type: str) -> None:
    if block._coordinate_token is not token:
        raise PedigreeValidationError(
            "coordinate_space_mismatch",
            f"pair_kinship: this {block.code} block was not produced by this {receiver_type}",
            operation="pair_kinship",
            receiver_type=receiver_type,
            result_type=type(block).__name__,
        )


def _resolve_query(
    first: object,
    second: object | None,
    *,
    token: CoordinateToken,
    n_individuals: int,
    receiver_type: str,
) -> _PairQuery:
    """Turn the three call forms into one validated :class:`_PairQuery`.

    Raises:
        TypeError: A block or collection given with a second argument, or a
            first row array without a second.
        PedigreeValidationError: ``coordinate_space_mismatch`` for a block from
            another receiver; ``invalid_shape``, ``invalid_integer_value``, or
            ``pair_row_out_of_range`` per row argument; then
            ``pair_length_mismatch``.
    """
    if isinstance(first, RelationshipPairs):
        if second is not None:
            raise TypeError("pair_kinship(pairs) takes no second argument")
        for block in first.values():
            _check_token(block, token, receiver_type)
        lengths = tuple((code, len(block)) for code, block in first.items())
        rows = [block.first_rows for block in first.values() if len(block)]
        cols = [block.second_rows for block in first.values() if len(block)]
        empty = np.zeros(0, dtype=np.int32)
        return _PairQuery(
            readonly(np.concatenate(rows)) if rows else empty,
            readonly(np.concatenate(cols)) if cols else empty,
            lengths,
        )
    if isinstance(first, RelationshipPairBlock):
        if second is not None:
            raise TypeError("pair_kinship(block) takes no second argument")
        _check_token(first, token, receiver_type)
        return _PairQuery(first.first_rows, first.second_rows, None)
    if second is None:
        raise TypeError("pair_kinship(first_rows, second_rows) needs both row arrays")
    rows = _coerce_rows(_FIRST, first, n_individuals)
    cols = _coerce_rows(_SECOND, second, n_individuals)
    if rows.shape[0] != cols.shape[0]:
        raise PedigreeValidationError(
            "pair_length_mismatch",
            f"first_rows has {rows.shape[0]} entries but second_rows has {cols.shape[0]}",
            first_length=rows.shape[0],
            second_length=cols.shape[0],
        )
    return _PairQuery(rows, cols, None)


def _shape_result(query: _PairQuery, values: np.ndarray) -> np.ndarray | Mapping[str, np.ndarray]:
    """Return *values* flat, or split per registry code for a collection query."""
    if query.block_lengths is None:
        return values
    offsets = np.cumsum([length for _, length in query.block_lengths])[:-1]
    pieces = np.split(values, offsets)
    return MappingProxyType(dict(zip((code for code, _ in query.block_lengths), pieces, strict=True)))


def _evaluate(graph: PedigreeGraph, first: np.ndarray, second: np.ndarray, *, commit_threads: bool) -> np.ndarray:
    if commit_threads:
        thread_budget()
    topology = graph._topology
    mother, father, twin = graph._topological_parents
    return pairwise_kinship(mother, father, twin, topology.translate(first), topology.translate(second))


def graph_pair_kinship(
    graph: PedigreeGraph,
    first: object,
    second: object | None,
    *,
    commit_threads: bool = True,  # 0.8.0-DELETE: the 0.7.1 adapter leaves the budget open.
) -> np.ndarray | Mapping[str, np.ndarray]:
    """Answer ``PedigreeGraph.pair_kinship`` in graph rows."""
    query = _resolve_query(
        first, second, token=graph._coordinate_token, n_individuals=graph.n_individuals, receiver_type="PedigreeGraph"
    )
    return _shape_result(query, _evaluate(graph, query.first, query.second, commit_threads=commit_threads))


def view_pair_kinship(
    view: PedigreeView,
    first: object,
    second: object | None,
    *,
    commit_threads: bool = True,  # 0.8.0-DELETE: the 0.7.1 adapter leaves the budget open.
) -> np.ndarray | Mapping[str, np.ndarray]:
    """Answer ``PedigreeView.pair_kinship``: validate in view rows, evaluate in graph rows."""
    query = _resolve_query(
        first, second, token=view._coordinate_token, n_individuals=view.n_individuals, receiver_type="PedigreeView"
    )
    rows = view.graph_rows
    values = _evaluate(
        view._graph, readonly(rows[query.first]), readonly(rows[query.second]), commit_threads=commit_threads
    )
    return _shape_result(query, values)
