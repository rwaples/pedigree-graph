"""Caballero & Toro 2002 self-coancestry rate Ne (Ne_CT) (PGQ-006).

Owns the Numba ancestor-set arena plumbing, the streaming accumulator
(:func:`_caballero_toro_accumulators` → :class:`CTAccumulators`), and the
estimator :func:`ne_caballero_toro`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numba
import numpy as np

from pedigree_graph._ne_common import _regress_log_one_minus
from pedigree_graph._ne_founders import _founder_idx
from pedigree_graph._ne_results import NeCaballeroToroResult

if TYPE_CHECKING:
    from pedigree_graph._core import PedigreeGraph


@dataclass(frozen=True, slots=True)
class CTAccumulators:
    """Caballero & Toro per-(generation, founder) self-coancestry sums.

    Produced by :func:`_caballero_toro_accumulators` and consumed by
    :func:`ne_caballero_toro`.  ``sums`` and ``counts`` share the same
    ``(g_max + 1, n_founders)`` layout; founder columns align with
    ``founder_idx``.  The ``peak_*`` / ``total_*`` fields are descriptive
    arena telemetry (scaling diagnostics), not used in the Ne formula.

    Attributes:
        sums: ``(g_max + 1, n_founders)`` float64 — Σ self-coancestry of
            founder f's descendants in generation g.
        counts: ``(g_max + 1, n_founders)`` int64 — descendant count of
            founder f in generation g.
        peak_ancestor_set_size: largest single founder-ancestor set seen.
        peak_live_ancestor_sets: max simultaneously live sets in the arena.
        total_ancestor_pair_visits: Σ over individuals of ancestor-set size.
        founder_idx: ``(n_founders,)`` intp — founder row indices.
    """

    sums: np.ndarray
    counts: np.ndarray
    peak_ancestor_set_size: int
    peak_live_ancestor_sets: int
    total_ancestor_pair_visits: int
    founder_idx: np.ndarray


@numba.njit(cache=True)
def _ct_ensure_pool_capacity(pool, cursor, needed):
    """Grow the CT ancestor-set arena when ``cursor + needed`` would overflow."""
    required = cursor + needed
    if required <= pool.shape[0]:
        return pool

    new_capacity = pool.shape[0] * 2
    while new_capacity < required:
        new_capacity *= 2
    new_pool = np.empty(new_capacity, dtype=np.int32)
    new_pool[:cursor] = pool[:cursor]
    return new_pool


@numba.njit(cache=True)
def _ct_merge_to_pool(pool, cursor, a_start, a_len, b_start, b_len):
    """Merge two sorted unique CT ancestor sets into ``pool[cursor:]``."""
    pool = _ct_ensure_pool_capacity(pool, cursor, a_len + b_len)
    i = 0
    j = 0
    out = cursor
    # Founder local indices are always >= 0, so -1 is a safe "no previous" sentinel.
    last = np.int32(-1)

    while i < a_len and j < b_len:
        av = pool[a_start + i]
        bv = pool[b_start + j]
        if av < bv:
            v = av
            i += 1
        elif bv < av:
            v = bv
            j += 1
        else:
            v = av
            i += 1
            j += 1
        if v != last:
            pool[out] = v
            out += 1
            last = v

    while i < a_len:
        v = pool[a_start + i]
        i += 1
        if v != last:
            pool[out] = v
            out += 1
            last = v

    while j < b_len:
        v = pool[b_start + j]
        j += 1
        if v != last:
            pool[out] = v
            out += 1
            last = v

    return pool, out - cursor


@numba.njit(cache=True)
def _ct_accumulators_kernel(gen, mother, father, founder_local_of, self_coancestry, g_max, n_founders):
    """Numba core for Caballero-Toro descendant self-coancestry accumulators."""
    n = len(gen)
    sums = np.zeros((g_max + 1, n_founders), dtype=np.float64)
    counts = np.zeros((g_max + 1, n_founders), dtype=np.int64)

    n_children = np.zeros(n, dtype=np.int64)
    for i in range(n):
        m = mother[i]
        if m >= 0:
            n_children[m] += 1
        f = father[i]
        if f >= 0:
            n_children[f] += 1
    n_remaining = n_children.copy()

    starts = np.full(n, -1, dtype=np.int64)
    lens = np.zeros(n, dtype=np.int32)
    pool = np.empty(max(n, 16), dtype=np.int32)
    cursor = 0

    peak_set_size = 0
    peak_live = 0
    active_count = 0
    total_pair_visits = 0

    # PedigreeGraph guarantees topological row order (parents precede children),
    # so a single forward row sweep is sufficient even when generation labels are
    # sparse or skip-generation edges are present.
    for i in range(n):
        start = -1
        length = 0
        f_local = founder_local_of[i]
        if f_local >= 0:
            pool = _ct_ensure_pool_capacity(pool, cursor, 1)
            start = cursor
            pool[cursor] = np.int32(f_local)
            cursor += 1
            length = 1
        else:
            m = mother[i]
            f = father[i]
            m_start = -1
            m_len = 0
            f_start = -1
            f_len = 0
            if m >= 0:
                m_start = starts[m]
                m_len = lens[m]
            if f >= 0:
                f_start = starts[f]
                f_len = lens[f]

            if m_len > 0 and f_len > 0:
                if m_start == f_start and m_len == f_len:
                    start = m_start
                    length = m_len
                else:
                    start = cursor
                    pool, length = _ct_merge_to_pool(pool, cursor, m_start, m_len, f_start, f_len)
                    cursor += length
            elif m_len > 0:
                start = m_start
                length = m_len
            elif f_len > 0:
                start = f_start
                length = f_len

        if length > 0:
            g = gen[i]
            sc = self_coancestry[i]
            for k in range(length):
                a = pool[start + k]
                sums[g, a] += sc
                counts[g, a] += 1
            total_pair_visits += length
            if length > peak_set_size:
                peak_set_size = length

        m = mother[i]
        if m >= 0:
            n_remaining[m] -= 1
            if n_remaining[m] == 0 and lens[m] > 0:
                lens[m] = 0
                active_count -= 1
        f = father[i]
        if f >= 0:
            n_remaining[f] -= 1
            if n_remaining[f] == 0 and lens[f] > 0:
                lens[f] = 0
                active_count -= 1

        if n_children[i] > 0 and length > 0:
            starts[i] = start
            lens[i] = length
            active_count += 1
            if active_count > peak_live:
                peak_live = active_count

    return sums, counts, peak_set_size, peak_live, total_pair_visits


def _caballero_toro_accumulators(
    pg: PedigreeGraph,
    founder_idx: np.ndarray,
    F: np.ndarray,
) -> CTAccumulators:
    """Streaming forward sweep producing per-(g, f) self-coancestry sums.

    For each generation g and founder f, accumulates the count of
    descendants of f in gen g and the sum of their self-coancestry
    ``(1 + F_i) / 2``.  Avoids materializing the dense
    ``(n × n_founders)`` contribution matrix by maintaining sorted
    per-individual Founder-Ancestor sets in a Numba arena and retiring
    them from the live frontier once the last child has been processed.

    "Descendant of f" is graph reachability — equivalent to ``c[i, f] >
    0`` because the forward recursion only adds non-negatives, so a
    non-zero ⇔ at least one ancestor path exists.

    Args:
        pg: Pedigree graph.
        founder_idx: Founder indices (output of :func:`_founder_idx`).
        F: Per-individual inbreeding coefficients (length ``pg.n``).

    Returns:
        A :class:`CTAccumulators` record.
    """
    n = pg.n
    n_founders = len(founder_idx)
    gen = np.asarray(pg.generation, dtype=np.int64)
    mother = np.asarray(pg.mother, dtype=np.int64)
    father = np.asarray(pg.father, dtype=np.int64)
    g_max = int(gen.max()) if n > 0 else 0

    if n_founders == 0:
        return CTAccumulators(
            sums=np.zeros((g_max + 1, 0), dtype=np.float64),
            counts=np.zeros((g_max + 1, 0), dtype=np.int64),
            peak_ancestor_set_size=0,
            peak_live_ancestor_sets=0,
            total_ancestor_pair_visits=0,
            founder_idx=founder_idx,
        )

    founder_local_of = np.full(n, -1, dtype=np.int64)
    founder_local_of[np.asarray(founder_idx, dtype=np.int64)] = np.arange(n_founders, dtype=np.int64)
    self_coancestry = (1.0 + np.asarray(F, dtype=np.float64)) / 2.0

    sums, counts, peak_set_size, peak_live, total_pair_visits = _ct_accumulators_kernel(
        gen,
        mother,
        father,
        founder_local_of,
        self_coancestry,
        g_max,
        n_founders,
    )
    return CTAccumulators(
        sums=sums,
        counts=counts,
        peak_ancestor_set_size=int(peak_set_size),
        peak_live_ancestor_sets=int(peak_live),
        total_ancestor_pair_visits=int(total_pair_visits),
        founder_idx=founder_idx,
    )


def ne_caballero_toro(
    pg: PedigreeGraph,
    ct_accumulators: CTAccumulators | None = None,
) -> NeCaballeroToroResult:
    """Caballero & Toro 2002 self-coancestry rate Ne (Ne_CT).

    For each founder f and generation g > 0, descendants are detected
    via graph reachability — equivalently, ``c[i, f] > 0`` under the
    Mendelian recursion.  Self-coancestry per descendant is
    ``(1 + F_i) / 2``; averaged within each founder's descendant set,
    then averaged across founders that have descendants at gen g.  Ne
    from the regression slope of ``ln(1 − f̄_s,g)`` on g.
    """
    if ct_accumulators is None:
        founder_idx = _founder_idx(pg)
        F = pg.compute_inbreeding()
        ct_accumulators = _caballero_toro_accumulators(pg, founder_idx, F)

    sums = ct_accumulators.sums
    counts = ct_accumulators.counts
    g_max = sums.shape[0] - 1

    valid = counts > 0
    # per_founder_mean[g, f] = mean self-coancestry of f's descendants in gen g
    per_founder_mean = np.where(valid, sums / np.maximum(counts, 1), 0.0)
    n_with_desc_per_gen = valid.sum(axis=1).astype(np.int64)
    mean_fs_per_gen = np.full(g_max + 1, np.nan, dtype=np.float64)
    nz = n_with_desc_per_gen > 0
    if nz.any():
        mean_fs_per_gen[nz] = per_founder_mean.sum(axis=1)[nz] / n_with_desc_per_gen[nz]
    # Gen 0 has no "descendants" in the CT regression sense; force NaN/0
    # to match the historical contract (regression starts at g=1).
    mean_fs_per_gen[0] = np.nan
    n_with_desc_per_gen[0] = 0

    ne_per_gen = np.full(g_max + 1, np.nan, dtype=np.float64)
    for g in range(1, g_max + 1):
        # For g=1 we anchor `prev = 0.5` (the natural self-coancestry floor for
        # non-inbred individuals, since fs = (1+F)/2 and founder F = 0).  An
        # earlier version anchored at 0.0, which produced a meaningless
        # ``Ne = 1/(2·0.5) = 1`` artifact at g=1; the actual drift signal is the
        # deviation of fs above 0.5, which only starts accumulating from g=2
        # onward.  At g=1 ``d == 0`` and the ``d > 0`` guard now correctly
        # leaves the entry as NaN.
        prev = mean_fs_per_gen[g - 1] if g >= 2 else 0.5
        if not np.isfinite(prev) or prev >= 1.0 or not np.isfinite(mean_fs_per_gen[g]):
            continue
        d = (mean_fs_per_gen[g] - prev) / (1.0 - prev)
        if d > 0:
            ne_per_gen[g] = 1.0 / (2.0 * d)

    t = np.arange(1, g_max + 1, dtype=np.float64)
    slope, _ = _regress_log_one_minus(mean_fs_per_gen[1:], t)
    if np.isfinite(slope) and slope < 0:
        ne_scalar: float | None = -1.0 / (2.0 * slope)
    else:
        ne_scalar = None

    return NeCaballeroToroResult(
        ne=ne_scalar,
        ne_per_gen=ne_per_gen,
        mean_self_coancestry_per_gen=mean_fs_per_gen,
        n_founders_with_descendants_per_gen=n_with_desc_per_gen,
        slope=slope,
    )
