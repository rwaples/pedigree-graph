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
* the result never depends on call history: no cached matrix is read, and the
  memo a graph retains between calls only ever returns the cold walk's bits;
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
(12 bytes per slot).

The memo outlives the call.  A graph keeps the table its last kernel call left
behind (:class:`_PairMemo` on ``PedigreeGraph._pair_memo``) and hands it to the
next call as the starting table, so a second query on the same graph pays for
the ancestor pairs it newly reaches, not for the closure it already walked.  On
the 30k parity fixture a degree-3 query reaches 23 million ancestor pairs, and
every query touching the deepest generation re-derives most of them.  Reuse
cannot change a bit: each key is computed exactly once, from its dependencies'
memoised values with the same float32 operations, whatever order the requests
arrive in, so a memo hit returns the value the cold walk would have stored.
The retained table is the memory trade.  It is kept only while it fits under
:data:`_MEMO_RETAIN_LIMIT` bytes (a per-graph override lives on
``PedigreeGraph._pair_memo_limit``) and is dropped otherwise, so a call that
completes never fails on retention; ``PedigreeGraph._release_pair_memo`` frees
it explicitly.

Two implementations of the same recurrence live here: :func:`_pairwise_kinship_py`
is the readable recursive oracle used by the property tests, and
:func:`pairwise_kinship` is the ``@njit`` production kernel.
"""

from __future__ import annotations

__all__ = ["graph_pair_kinship", "memoised_kinship", "pairwise_kinship", "view_pair_kinship"]

from dataclasses import dataclass, field
from functools import cache
from types import MappingProxyType
from typing import TYPE_CHECKING

import numba
import numpy as np

from pedigree_graph._errors import PedigreeValidationError, ResourceError
from pedigree_graph._input import _INT32_MAX, _coerce_row_selection, _FieldSpec
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

# Largest memo a graph retains between calls, in bytes of table (12 per slot).
# The degree-3 closure of ``random_30k`` ends at 2**25 slots, 384 MiB, and is
# kept (benchmarks/bench_pair_kinship.md); the 145M-entry degree-3 memo of a
# 536k-row pedigree needs 2**28 slots, 3 GiB, and is dropped, so a graph that
# size keeps its pre-slice memory profile unless the caller raises
# ``PedigreeGraph._pair_memo_limit``.
_MEMO_RETAIN_LIMIT = 1 << 30

# Stats array layout returned by the core kernel.
_STAT_ENTRIES = 0
_STAT_CAPACITY = 1
_STAT_GROWS = 2
_STAT_MAX_STACK = 3
_STAT_OVERFLOW = 4


@dataclass(frozen=True, slots=True)
class _PairMemo:
    """The open-addressing table one kernel call left behind, ready for the next.

    ``keys`` and ``vals`` are the live table (capacity a power of two, ``-1``
    marks an empty slot) and ``entries`` how many slots are filled.  The empty
    memo has zero-length tables, which :func:`_run_kernel` reads as "start cold".

    Attributes:
        keys: int64 canonical ``lo * n + hi`` keys, ``-1`` where empty.
        vals: float32 values aligned to ``keys``.
        entries: Number of filled slots.
    """

    keys: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    vals: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    entries: int = 0

    @property
    def capacity(self) -> int:
        return int(self.keys.shape[0])

    @property
    def nbytes(self) -> int:
        return int(self.keys.nbytes + self.vals.nbytes)


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
    memo_keys: np.ndarray,
    memo_vals: np.ndarray,
    entries: int,
):
    """Pedigree-expected kinship per requested pair; the production recurrence kernel.

    Rows are in stable depth-major order, so peeling the greater row is the
    pinned depth-then-row rule.  Iterative (explicit work-stack) post-order
    evaluation of the module recurrence over a caller-owned open-addressing
    ``int64 -> float32`` memo keyed on the canonical ``lo * n + hi`` row pair.

    ``memo_keys``, ``memo_vals`` and ``entries`` are the table and its fill:
    the kernel reads and writes them in place and never allocates, grows, or
    returns them.  When the load factor would be exceeded it stops and reports
    ``overflow = 1``; the caller (:func:`_run_kernel`) doubles the table and
    re-enters, which is cheap because every node resolved so far is a memo
    hit.  Keeping the tables out of the kernel's return value is what keeps
    the walk at the pre-memo speed: a kernel that grew and returned them
    measured 6 percent slower on the cold 30k walk, and this shape 5 percent
    faster than the pre-memo kernel (``benchmarks/bench_pair_kinship.md``).

    Returns ``(out, stats)`` where ``stats`` is ``int64[5]`` = ``[entries,
    capacity, grows, max_stack_depth, overflow]`` with ``entries`` the table's
    fill after this call and ``grows`` always ``0`` here; ``out`` is
    unfinished when ``overflow`` is ``1``.

    The combine expressions mirror :func:`_pairwise_kinship_py` term-for-term
    in float32, so the two agree to the bit whatever the traversal order, and
    a reused entry is the bit a cold walk would store.
    """
    half = np.float32(0.5)
    one = np.float32(1.0)
    zero = np.float32(0.0)
    p = first.shape[0]
    out = np.empty(p, dtype=np.float32)
    stats = np.zeros(5, dtype=np.int64)
    stats[_STAT_ENTRIES] = entries
    stats[_STAT_CAPACITY] = memo_keys.shape[0]
    if p == 0:
        return out, stats

    cap = memo_keys.shape[0]
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
                stats[_STAT_ENTRIES] = entries
                stats[_STAT_CAPACITY] = cap
                stats[_STAT_OVERFLOW] = 1
                return out, stats

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
    a direct kernel caller from reading outside the arrays, overflowing the pair
    key, or feeding an order the peel cannot terminate on.

    Parents-before-children is all that is *checked*, because that is all
    termination needs.  Depth-major is the caller's responsibility and is what
    the ADR 0009 bit-parity contract rests on: a merely topological order still
    returns a valid recurrence, but one whose rounding no longer matches the
    matrix.  ``PedigreeGraph._topological_parents`` supplies the right order.
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
        raise ValueError(
            "pairwise_kinship: every parent row must precede its child; pass the depth-major arrays from "
            "PedigreeGraph._topological_parents"
        )
    if n > 0 and n > (np.iinfo(np.int64).max // n):
        raise ValueError(f"pairwise_kinship: pedigree size n={n} overflows the int64 pair-key encoding (lo * n + hi)")
    if first.shape[0] != second.shape[0]:
        raise ValueError("pairwise_kinship: first and second must have one entry per pair")
    if first.size and (
        int(first.min()) < 0 or int(second.min()) < 0 or int(first.max()) >= n or int(second.max()) >= n
    ):
        raise ValueError(f"pairwise_kinship: pair rows must lie in [0, {n})")
    return mother, father, twin, first, second, n


def _memo_ceiling(cap_limit: int) -> int:
    """Largest power of two at or below *cap_limit*.

    The memo indexes with ``key & (capacity - 1)``, so any capacity that is not
    a power of two masks most of the table away and the probe loop spins
    forever once the reachable slots fill.  Flooring here keeps that invariant
    a property of the one Python entry point rather than a rule every caller
    has to know.
    """
    if cap_limit < 1:
        raise ValueError(f"pair_kinship: the memo capacity limit must be at least one slot, got {cap_limit}")
    return 1 << (int(cap_limit).bit_length() - 1)


def _cold_capacity(pair_count: int, cap_limit: int) -> int:
    """Initial table size for a cold walk: four slots per requested pair, within the cap."""
    wanted = 4 * pair_count if pair_count > 4 else 16
    return min(1 << (wanted - 1).bit_length(), cap_limit)


def _run_kernel(
    mother: np.ndarray,
    father: np.ndarray,
    twin: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    cap_limit: int = _MEMO_CAP_LIMIT,
    memo: _PairMemo | None = None,
) -> tuple[np.ndarray, np.ndarray, _PairMemo]:
    """Validate, walk to completion, and return ``(out, stats, memo)``.

    *memo* is the starting table, ``None`` or empty for cold; the returned
    memo is the table the walk left behind, for the caller to retain or drop.
    Growth lives here: each time the kernel reports its table full, the table
    is doubled and the kernel re-entered, until it completes or the next
    doubling would pass *cap_limit*.  A starting table already past the cap is
    discarded rather than handed to a kernel that could never fill it.
    """
    cap_limit = _memo_ceiling(cap_limit)
    mother, father, twin, first, second, n = _prepare_inputs(mother, father, twin, first, second)
    if memo is None or memo.capacity == 0 or memo.capacity > cap_limit:
        capacity = _cold_capacity(first.shape[0], cap_limit)
        keys = np.full(capacity, -1, dtype=np.int64)
        vals = np.empty(capacity, dtype=np.float32)
        entries = 0
    else:
        keys, vals, entries = memo.keys, memo.vals, memo.entries
    grows = 0
    max_stack = 0
    while True:
        out, stats = _pairwise_kinship_core(mother, father, twin, first, second, n, keys, vals, entries)
        entries = int(stats[_STAT_ENTRIES])
        max_stack = max(max_stack, int(stats[_STAT_MAX_STACK]))
        if not stats[_STAT_OVERFLOW]:
            break
        if keys.shape[0] * 2 > cap_limit:
            raise ResourceError(
                "memo_capacity_exceeded",
                "pair_kinship: the recurrence memo would exceed its capacity limit; the pedigree is too "
                "inbred or too deep for the direct path",
                operation="pair_kinship",
                capacity=int(keys.shape[0]),
                maximum=cap_limit,
            )
        keys, vals = _memo_grow(keys, vals)
        grows += 1
    stats[_STAT_GROWS] = grows
    stats[_STAT_MAX_STACK] = max_stack
    out.setflags(write=False)
    return out, stats, _PairMemo(keys, vals, entries)


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
    out, _, _ = _run_kernel(mother, father, twin, first, second)
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
    out, stats, _ = _run_kernel(mother, father, twin, first, second)
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
        block_lengths: How many pairs each registry code contributed, in
            registry order, when the request was a :class:`RelationshipPairs`,
            so the flat result can be split back; ``None`` for a flat request.
    """

    first: np.ndarray
    second: np.ndarray
    block_lengths: dict[str, int] | None


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

    The view's row selection and a pair endpoint share one rule
    (:func:`pedigree_graph._input._coerce_row_selection`), so an endpoint fails
    with ``invalid_shape``, ``invalid_integer_value``, or
    ``pair_row_out_of_range`` and nothing else, in that order.
    """
    rows = _coerce_row_selection(
        spec,
        values,
        n_individuals,
        lambda value, position: _row_out_of_range(spec.name, value, position, n_individuals),
    )
    return readonly(rows.astype(np.int32))


def _join_rows(blocks: list[np.ndarray]) -> np.ndarray:
    """Join the rows of every non-empty block, or return the empty array for none."""
    if not blocks:
        return np.zeros(0, dtype=np.int32)
    return readonly(np.concatenate(blocks))


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
        block_lengths: dict[str, int] = {}
        firsts: list[np.ndarray] = []
        seconds: list[np.ndarray] = []
        for code, block in first.items():
            _check_token(block, token, receiver_type)
            block_lengths[code] = len(block)
            if len(block):
                firsts.append(block.first_rows)
                seconds.append(block.second_rows)
        return _PairQuery(_join_rows(firsts), _join_rows(seconds), block_lengths)
    if isinstance(first, RelationshipPairBlock):
        if second is not None:
            raise TypeError("pair_kinship(block) takes no second argument")
        _check_token(first, token, receiver_type)
        return _PairQuery(first.first_rows, first.second_rows, None)
    if second is None:
        raise TypeError("pair_kinship(first_rows, second_rows) needs both row arrays")
    first_rows = _coerce_rows(_FIRST, first, n_individuals)
    second_rows = _coerce_rows(_SECOND, second, n_individuals)
    if first_rows.shape[0] != second_rows.shape[0]:
        raise PedigreeValidationError(
            "pair_length_mismatch",
            f"first_rows has {first_rows.shape[0]} entries but second_rows has {second_rows.shape[0]}",
            first_length=first_rows.shape[0],
            second_length=second_rows.shape[0],
        )
    return _PairQuery(first_rows, second_rows, None)


def _shape_result(query: _PairQuery, values: np.ndarray) -> np.ndarray | Mapping[str, np.ndarray]:
    """Return *values* flat, or split back per registry code for a collection query."""
    if query.block_lengths is None:
        return values
    offsets = np.cumsum(list(query.block_lengths.values()))[:-1]
    pieces = np.split(values, offsets)
    return MappingProxyType(dict(zip(query.block_lengths, pieces, strict=True)))


def memoised_kinship(graph: PedigreeGraph, first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Run the kernel on *graph* from its retained memo and retain what it leaves.

    *first* and *second* are already in the graph's depth-major order.  This is
    the one place a graph's memo is read and written, so ``pair_kinship`` and
    the relationship matrix share it.  The table is kept when it fits under the
    graph's retention limit and dropped otherwise; either way the values are
    those of a cold call.
    """
    mother, father, twin = graph._topological_parents
    out, _, memo = _run_kernel(mother, father, twin, first, second, memo=graph._pair_memo)
    graph._pair_memo = memo if memo.nbytes <= graph._pair_memo_limit else None
    return out


def _evaluate(graph: PedigreeGraph, first: np.ndarray, second: np.ndarray, *, commit_threads: bool) -> np.ndarray:
    if commit_threads:
        thread_budget()
    topology = graph._topology
    return memoised_kinship(graph, topology.translate(first), topology.translate(second))


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
