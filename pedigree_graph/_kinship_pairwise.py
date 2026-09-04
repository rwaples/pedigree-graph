"""Exact on-demand pairwise kinship via a direct memoized recurrence.

Computes the exact kinship coefficient ``phi(a, b)`` for a *specific set of
requested pairs* without ever materializing the full ``n x n`` kinship matrix.
This is the scalable replacement for the full-matrix path in
:meth:`PedigreeGraph.compute_pair_kinship`: the matrix path is the package's
dominant super-linear cost (a 16k-row pedigree builds a ~53M-nonzero matrix and
OOMs on larger simACE-like inputs), whereas the requested extracted pairs are a
sparse subset of all related pairs.

Algorithm (Karigl tabular kinship, on demand + memoized)
--------------------------------------------------------
For graph-space individuals ``a, b`` whose parent indices are topologically
ordered (parent index < child index, guaranteed by ``PedigreeGraph``
construction):

* ``phi(a, a) = (1 + F_a) / 2``
* ``F_a = phi(mother_a, father_a)``; a ``-1`` parent gives ``F_a = 0``
* MZ pair (``twin[a] == b``): ``phi(a, b) = self-kinship of the MZ row`` — MZ
  twins share a genome, so their cross-kinship equals self-kinship and the
  doubling propagates to descendants exactly as the matrix path's twin
  off-diagonals do.
* otherwise, with ``c = max(a, b)`` (more recent / topologically later) and
  ``o = min(a, b)``::

      phi(a, b) = 0.5 * (phi(mother_c, o) + phi(father_c, o))

  a ``-1`` parent term contributes ``0.0``.
* base: if ``c`` is a founder and ``a != b``, ``phi(a, b) = 0``.

This is exact-by-construction against ``kinship_matrix(0.0)``: that path's DP
computes the diagonal ``F`` as ``phi(mother, father)`` *inside the kernel*
(``_kinship_dp._dp_kinship``), not from ``compute_inbreeding`` (which agrees, ADR 0008);
its merge walk is literally ``0.5 * (K[m, k] + K[f, k])``; and its MZ pass writes
the twin off-diagonal to the inbred self-kinship. So this recurrence reproduces
every one of those rules. ``compute_inbreeding`` is therefore *not* consulted
here.

Why not threshold-prune the matrix instead
-------------------------------------------
DP threshold-pruning is lossy for *cross-generation* propagation and cannot be
made exact: a sub-threshold kinship between two mates feeds their descendants'
above-threshold kinship, and pruning deletes it at the parents' generation.
Half-first-cousin parents are the disproof — ``phi`` between the mates is
``1/32``, yet pruning at any threshold that drops ``1/32`` makes their child's
exact parent-offspring kinship ``0.265625`` collapse to ``0.25``. No global
magnitude threshold can be exact, because ``phi(i, j)`` needs the full kinship
sub-matrix over ``ancestors(i) U ancestors(j)``.

Cost
----
``O(P + distinct ancestor-pairs reached)`` for the requested-pair count ``P``.
The honest worst case is ``O(P * A^2)`` where ``A`` is the max distinct
ancestors per individual: resolving ``phi(a, b)`` recurses over
``|anc(a)| x |anc(b)|`` ancestor-pairs (``<= 4^g`` at generation ``g``, capped
by the founder count and terminated by ``phi(founder_x, founder_y) = 0``),
shared across requests via the memo. For shallow random-mating pedigrees
(``G_ped ~ 8``) ``A`` is small; deeply inbred / high-overlap pedigrees can
inflate the shared memo and are out of scope for the scaling guarantee.

This module ships two implementations of the same recurrence:

* :func:`_pairwise_kinship_py` — a pure-Python recursive ``functools.cache``
  reference. Readable, used as the test bit-oracle; not a production hot path.
* ``pairwise_kinship`` — the ``@njit`` production kernel (added in stage 2).
"""

from __future__ import annotations

from functools import cache

import numba
import numpy as np

# Open-addressing memo load factor: grow when entries exceed this fraction of
# capacity.  0.7 keeps linear-probe chains short without wasting much memory.
_MEMO_LOAD_NUM = 7
_MEMO_LOAD_DEN = 10

# Hard backstop on memo capacity (slots): a runaway guard, not a working limit.
# Legitimate memos scale ~linearly with the requested pairs (~12.7M entries at
# 16k individuals, so the ~600k-individual target lands in the hundreds of
# millions); this ceiling sits above that. It only trips on pathological
# inbreeding where the memo heads toward n^2 — at which point the table would be
# ~tens of GB and we raise instead of silently thrashing toward OOM.
_MEMO_CAP_LIMIT = 1 << 31

# Stats array layout returned by the core kernel.
_STAT_ENTRIES = 0
_STAT_CAPACITY = 1
_STAT_GROWS = 2
_STAT_MAX_STACK = 3


def _pairwise_kinship_py(
    mother: np.ndarray,
    father: np.ndarray,
    twin: np.ndarray,
    pair_a: np.ndarray,
    pair_b: np.ndarray,
) -> np.ndarray:
    """Exact kinship for each requested pair (pure-Python reference).

    Recursive ``functools.cache`` implementation of the module recurrence.
    Readable and used as the bit-exact oracle for the numba kernel; intended for
    small pedigrees / tests only (Python recursion depth scales with depth).

    Args:
        mother: int row-index array; ``-1`` for founder/missing.
        father: int row-index array; ``-1`` for founder/missing.
        twin: int MZ-partner row-index array; ``-1`` for non-twin.
        pair_a: first endpoint of each requested pair (graph-space row indices).
        pair_b: second endpoint of each requested pair (graph-space row indices).

    Returns:
        ``float64`` array of length ``len(pair_a)`` with exact ``phi`` per pair,
        positionally aligned to the inputs. Input pair orientation is preserved
        (no canonical reordering of the output).
    """
    mother = np.asarray(mother, dtype=np.int64)
    father = np.asarray(father, dtype=np.int64)
    twin = np.asarray(twin, dtype=np.int64)
    pair_a = np.asarray(pair_a, dtype=np.int64)
    pair_b = np.asarray(pair_b, dtype=np.int64)

    @cache
    def _f(i: int) -> float:
        m = mother[i]
        f = father[i]
        if m < 0 or f < 0:
            return 0.0
        return _phi(int(m), int(f))

    @cache
    def _phi(a: int, b: int) -> float:
        # Canonicalize so the memo shares phi(a, b) == phi(b, a).
        if a > b:
            a, b = b, a
        if a == b:
            return 0.5 * (1.0 + _f(a))
        # MZ pair: cross-kinship == self-kinship of the MZ row.  F_a == F_b for
        # true twins (shared parents), so either index gives the same value; use
        # the larger (b) to mirror the matrix MZ pass's max-index write.
        if twin[a] == b or twin[b] == a:
            return 0.5 * (1.0 + _f(b))
        # Peel the more-recent (larger-index) endpoint c = b down to its parents.
        m = mother[b]
        f = father[b]
        left = _phi(int(m), a) if m >= 0 else 0.0
        right = _phi(int(f), a) if f >= 0 else 0.0
        return 0.5 * (left + right)

    out = np.empty(pair_a.shape[0], dtype=np.float64)
    for k in range(pair_a.shape[0]):
        out[k] = _phi(int(pair_a[k]), int(pair_b[k]))
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
    new_vals = np.empty(new_cap, dtype=np.float64)
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
    pair_a: np.ndarray,
    pair_b: np.ndarray,
    n: int,
):
    """Exact kinship per requested pair; the production recurrence kernel.

    Iterative (explicit work-stack) post-order evaluation of the module
    recurrence with a hand-rolled open-addressing ``int64 -> float64`` memo
    keyed on canonical ``lo * n + hi``.  Returns ``(out, stats)`` where
    ``stats`` is ``int64[4]`` = ``[entries, capacity, grows, max_stack_depth]``.

    The combine expressions mirror :func:`_pairwise_kinship_py` term-for-term
    (``left`` = mother-side, ``right`` = father-side, ``0.5 * (left + right)``),
    so the float64 output is bit-identical regardless of traversal order.
    """
    p = pair_a.shape[0]
    out = np.empty(p, dtype=np.float64)
    stats = np.zeros(4, dtype=np.int64)
    if p == 0:
        return out, stats

    # Memo sized from the request count; grows geometrically on demand.
    cap = _next_pow2(4 * p if p > 4 else 16)
    memo_keys = np.full(cap, -1, dtype=np.int64)
    memo_vals = np.empty(cap, dtype=np.float64)
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
        root = _canon_key(pair_a[k], pair_b[k], n)
        if memo_keys[_memo_slot(memo_keys, root)] == root:
            continue  # already resolved by an earlier requested pair
        stack[0] = root
        top = 1
        while top > 0:
            if top > max_stack:
                max_stack = top
            s = stack[top - 1]
            top -= 1

            if s >= 0:
                # --- expand phase ---
                key = s
                if memo_keys[_memo_slot(memo_keys, key)] == key:
                    continue  # already computed (shared dependency)
                lo = key // n
                hi = key % n
                m = mother[hi]
                f = father[hi]
                if lo == hi or twin[lo] == hi or twin[hi] == lo:
                    # self / MZ -> self-kinship of hi
                    if m < 0 or f < 0:
                        slot = _memo_slot(memo_keys, key)  # founder leaf -> 0.5
                        memo_keys[slot] = key
                        memo_vals[slot] = 0.5
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
                    slot = _memo_slot(memo_keys, key)  # founder, hi != lo -> 0.0
                    memo_keys[slot] = key
                    memo_vals[slot] = 0.0
                    entries += 1
                else:
                    # peel hi -> 0.5 * (phi(m, lo) + phi(f, lo))
                    if top + 3 > stack.shape[0]:
                        stack = _grow_stack(stack)
                    stack[top] = -(key + 1)  # schedule compute
                    top += 1
                    if m >= 0:
                        d0 = _canon_key(m, lo, n)
                        if memo_keys[_memo_slot(memo_keys, d0)] != d0:
                            stack[top] = d0
                            top += 1
                    if f >= 0:
                        d1 = _canon_key(f, lo, n)
                        if memo_keys[_memo_slot(memo_keys, d1)] != d1:
                            stack[top] = d1
                            top += 1
            else:
                # --- compute phase --- dependencies are now memoized
                key = -s - 1
                lo = key // n
                hi = key % n
                m = mother[hi]
                f = father[hi]
                if lo == hi or twin[lo] == hi or twin[hi] == lo:
                    v0 = memo_vals[_memo_slot(memo_keys, _canon_key(m, f, n))]
                    value = 0.5 * (1.0 + v0)
                else:
                    v0 = memo_vals[_memo_slot(memo_keys, _canon_key(m, lo, n))] if m >= 0 else 0.0
                    v1 = memo_vals[_memo_slot(memo_keys, _canon_key(f, lo, n))] if f >= 0 else 0.0
                    value = 0.5 * (v0 + v1)
                slot = _memo_slot(memo_keys, key)
                memo_keys[slot] = key
                memo_vals[slot] = value
                entries += 1

            # Grow the memo if the load factor is exceeded.  Covers every insert
            # (leaf in the expand phase, value in the compute phase); the rehash
            # reassigns the tables before the next iteration's probes.
            if entries * _MEMO_LOAD_DEN >= cap * _MEMO_LOAD_NUM:
                new_cap = cap * 2
                if new_cap > _MEMO_CAP_LIMIT:
                    raise ValueError(
                        "pairwise_kinship: memo exceeded the capacity limit; the "
                        "pedigree is too inbred/deep for the direct path"
                    )
                memo_keys, memo_vals = _memo_grow(memo_keys, memo_vals)
                cap = new_cap
                grows += 1

    # Final scatter: every requested key is now memoized.
    for k in range(p):
        key = _canon_key(pair_a[k], pair_b[k], n)
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
    pair_a: np.ndarray,
    pair_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Coerce to contiguous int64 and validate the key-space against overflow."""
    mother = np.ascontiguousarray(mother, dtype=np.int64)
    father = np.ascontiguousarray(father, dtype=np.int64)
    twin = np.ascontiguousarray(twin, dtype=np.int64)
    pair_a = np.ascontiguousarray(pair_a, dtype=np.int64)
    pair_b = np.ascontiguousarray(pair_b, dtype=np.int64)
    n = mother.shape[0]
    # Canonical key lo * n + hi must fit int64.  PedigreeGraph already rejects
    # n > int32 max, so n*n <= 2**62 here, but guard explicitly anyway.
    if n > 0 and n > (np.iinfo(np.int64).max // n):
        raise ValueError(f"pairwise_kinship: pedigree size n={n} overflows the int64 pair-key encoding (lo * n + hi)")
    return mother, father, twin, pair_a, pair_b, n


def pairwise_kinship(
    mother: np.ndarray,
    father: np.ndarray,
    twin: np.ndarray,
    pair_a: np.ndarray,
    pair_b: np.ndarray,
) -> np.ndarray:
    """Exact kinship per requested pair (numba production path).

    Same recurrence and result as :func:`_pairwise_kinship_py`, but iterative
    and memoized in a nopython kernel so it scales to large pedigrees. Output is
    ``float64``, positionally aligned to the inputs, with input orientation
    preserved.

    Args:
        mother: int row-index array; ``-1`` for founder/missing.
        father: int row-index array; ``-1`` for founder/missing.
        twin: int MZ-partner row-index array; ``-1`` for non-twin.
        pair_a: first endpoint of each requested pair (graph-space row indices).
        pair_b: second endpoint of each requested pair (graph-space row indices).

    Returns:
        ``float64`` array of length ``len(pair_a)`` with exact ``phi`` per pair.
    """
    mother, father, twin, pair_a, pair_b, n = _prepare_inputs(mother, father, twin, pair_a, pair_b)
    out, _ = _pairwise_kinship_core(mother, father, twin, pair_a, pair_b, n)
    return out


def _pairwise_kinship_with_stats(
    mother: np.ndarray,
    father: np.ndarray,
    twin: np.ndarray,
    pair_a: np.ndarray,
    pair_b: np.ndarray,
) -> tuple[np.ndarray, dict[str, int]]:
    """Benchmark/debug wrapper: returns ``(out, stats_dict)``.

    ``stats_dict`` holds ``memo_entries``, ``memo_capacity``, ``memo_grows``,
    and ``max_stack_depth`` — used by the profiling harness to confirm the memo
    stays bounded. Not part of the production path.
    """
    mother, father, twin, pair_a, pair_b, n = _prepare_inputs(mother, father, twin, pair_a, pair_b)
    out, stats = _pairwise_kinship_core(mother, father, twin, pair_a, pair_b, n)
    return out, {
        "memo_entries": int(stats[_STAT_ENTRIES]),
        "memo_capacity": int(stats[_STAT_CAPACITY]),
        "memo_grows": int(stats[_STAT_GROWS]),
        "max_stack_depth": int(stats[_STAT_MAX_STACK]),
    }
