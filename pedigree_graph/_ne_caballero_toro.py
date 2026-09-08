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

from pedigree_graph._cohorts import ObservedCohorts
from pedigree_graph._ne_common import _checked_founder_matrix, _scalar_ne_from_log_regression, _transition_ne
from pedigree_graph._ne_founders import _founder_columns, _founder_idx
from pedigree_graph._ne_metadata import _require_closed_parentage
from pedigree_graph._ne_results import NeCaballeroToroResult

if TYPE_CHECKING:
    from pedigree_graph._core import PedigreeGraph


@dataclass(frozen=True, slots=True)
class CTAccumulators:
    """Caballero & Toro per-(generation, founder) self-coancestry sums.

    Produced by :func:`_caballero_toro_accumulators` and consumed by
    :func:`ne_caballero_toro`.  ``sums`` and ``counts`` share the same
    ``(k, n_founders)`` layout, one row per observed cohort; columns align
    with ``founder_idx``, one per represented founder genome.  A founder row
    seeds its genome's lineage but is not its own descendant, so it never
    enters the sums.  The ``peak_*`` / ``total_*`` fields are descriptive
    arena telemetry (scaling diagnostics), not used in the Ne formula.

    Attributes:
        sums: ``(k, n_founders)`` float64 — Σ self-coancestry of founder
            genome f's descendants in observed cohort b.
        counts: ``(k, n_founders)`` int64 — descendant count of founder
            genome f in observed cohort b.
        peak_ancestor_set_size: largest single founder-ancestor set seen.
        peak_live_ancestor_sets: max simultaneously live sets in the arena.
        total_ancestor_pair_visits: Σ over descendants of ancestor-set size.
        founder_idx: ``(n_founders,)`` intp — canonical founder-genome rows.
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
def _ct_accumulators_kernel(bucket, mother, father, founder_local_of, self_coancestry, sums, counts):
    """Numba core for Caballero-Toro descendant self-coancestry accumulators.

    Fills the caller-allocated ``sums`` / ``counts`` (one row per cohort
    bucket) and returns the arena telemetry.
    """
    n = len(bucket)

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

    # Rows arrive in the caller's private topological order (parents precede
    # children), so a single forward sweep is sufficient even when generation
    # labels are sparse or skip-generation edges are present.
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

        if length > 0 and f_local < 0:
            g = bucket[i]
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

    return peak_set_size, peak_live, total_pair_visits


def _caballero_toro_accumulators(
    pg: PedigreeGraph,
    founder_idx: np.ndarray,
    F: np.ndarray,
    cohorts: ObservedCohorts | None = None,
) -> CTAccumulators:
    """Streaming forward sweep producing per-(cohort, genome) self-coancestry sums.

    For each observed cohort b and founder genome f, accumulates the count
    of descendants of f in cohort b and the sum of their self-coancestry
    ``(1 + F_i) / 2``.  Avoids materializing the dense
    ``(n × n_founders)`` contribution matrix by maintaining sorted
    per-individual Founder-Ancestor sets in a Numba arena and retiring
    them from the live frontier once the last child has been processed.

    "Descendant of f" is graph reachability — equivalent to ``c[i, f] >
    0`` because the forward recursion only adds non-negatives, so a
    non-zero ⇔ at least one ancestor path exists.  A founder row seeds
    its genome's set but is not its own descendant; parentless MZ co-twins
    seed the same genome column.

    The sweep runs in the graph's private topological order; ``sums`` and
    ``counts`` are indexed by cohort bucket and local founder index, so
    they carry their graph-space meaning unchanged.

    Args:
        pg: Pedigree graph.
        founder_idx: Canonical founder-genome rows (:func:`_founder_idx`).
        F: Per-individual inbreeding coefficients (length ``pg.n_individuals``), in
            graph rows.
        cohorts: Optional precomputed grouping; defaults to the graph's.

    Returns:
        A :class:`CTAccumulators` record.
    """
    if cohorts is None:
        cohorts = ObservedCohorts.for_graph(pg, "ne_caballero_toro")
    n = pg.n_individuals
    n_founders = int(founder_idx.shape[0])
    sums = _checked_founder_matrix(cohorts.k, n_founders, "ct_accumulators", np.float64, 0.0)
    counts = _checked_founder_matrix(cohorts.k, n_founders, "ct_accumulators", np.int64, 0)
    if n_founders == 0 or n == 0:
        return CTAccumulators(
            sums=sums,
            counts=counts,
            peak_ancestor_set_size=0,
            peak_live_ancestor_sets=0,
            total_ancestor_pair_visits=0,
            founder_idx=founder_idx,
        )

    topo = pg._topology
    bucket = np.asarray(topo.gather(cohorts.dense), dtype=np.int64)
    m_idx, f_idx, _ = pg._topological_parents
    mother = np.asarray(m_idx, dtype=np.int64)
    father = np.asarray(f_idx, dtype=np.int64)
    # Local founder numbering is graph-space, so gathering keeps genome k as
    # genome k while the sweep itself runs in topological rows.  Both arrays
    # are copied so the kernel sees one writeability whether or not the gather
    # was a no-op (see _topology.readonly).
    founder_local_of = np.array(topo.gather(_founder_columns(pg, founder_idx)), dtype=np.int64)
    self_coancestry = np.array(topo.gather((1.0 + np.asarray(F, dtype=np.float64)) / 2.0), dtype=np.float64)

    peak_set_size, peak_live, total_pair_visits = _ct_accumulators_kernel(
        bucket,
        mother,
        father,
        founder_local_of,
        self_coancestry,
        sums,
        counts,
    )
    return CTAccumulators(
        sums=sums,
        counts=counts,
        peak_ancestor_set_size=int(peak_set_size),
        peak_live_ancestor_sets=int(peak_live),
        total_ancestor_pair_visits=int(total_pair_visits),
        founder_idx=founder_idx,
    )


def _caballero_toro_from(cohorts: ObservedCohorts, acc: CTAccumulators) -> NeCaballeroToroResult:
    sums = acc.sums
    counts = acc.counts
    k = cohorts.k
    if sums.shape[0] != k:
        raise ValueError("Caballero-Toro accumulators do not describe the estimator's observed cohorts")

    valid = counts > 0
    # per_founder_mean[b, f] = mean self-coancestry of f's descendants in cohort b
    per_founder_mean = np.where(valid, sums / np.maximum(counts, 1), 0.0)
    n_with_desc = valid.sum(axis=1).astype(np.int64)
    mean_fs = np.full(k, np.nan, dtype=np.float64)
    nz = n_with_desc > 0
    if nz.any():
        mean_fs[nz] = per_founder_mean.sum(axis=1)[nz] / n_with_desc[nz]
    if k:
        # The first observed cohort is the baseline: no descendants in the CT
        # regression sense, whatever its rows' labels or parentage.
        mean_fs[0] = np.nan
        n_with_desc[0] = 0

    # The first transition starts from the self-coancestry of a non-inbred
    # individual, 0.5, not from 0: fs = (1 + F) / 2 and founder F = 0, so the
    # drift signal is the rise above 0.5 (anchoring at 0 yields a spurious
    # Ne = 1 at the first transition).
    rate_series = mean_fs.copy()
    if k:
        rate_series[0] = 0.5
    ne_scalar, slope, _ = _scalar_ne_from_log_regression(mean_fs, cohorts.generations)

    return NeCaballeroToroResult(
        ne=ne_scalar,
        generations=cohorts.generations,
        mean_self_coancestry_per_gen=mean_fs,
        n_founders_with_descendants_per_gen=n_with_desc,
        transition_from=cohorts.transition_from(),
        transition_to=cohorts.transition_to(),
        ne_per_gen=_transition_ne(rate_series, cohorts.generations),
        slope=slope,
    )


def ne_caballero_toro(pg: PedigreeGraph) -> NeCaballeroToroResult:
    """Caballero & Toro 2002 self-coancestry rate Ne (Ne_CT).

    For each represented founder genome f and observed cohort after the
    first, descendants are detected via graph reachability — equivalently,
    ``c[i, f] > 0`` under the Mendelian recursion.  Self-coancestry per
    descendant is ``(1 + F_i) / 2``; averaged within each genome's
    descendant set, then averaged across genomes that have descendants in
    the cohort.  Ne from the regression slope of ``ln(1 − f̄_s)`` on the
    label offset; each adjacent transition is gap-corrected.

    Requires complete generation labels (or none), then closed represented
    parentage: a row with exactly one represented parent raises
    ``incomplete_parentage``.
    """
    cohorts = ObservedCohorts.for_graph(pg, "ne_caballero_toro")
    _require_closed_parentage(pg, "ne_caballero_toro")
    founder_idx = _founder_idx(pg)
    acc = _caballero_toro_accumulators(pg, founder_idx, pg._inbreeding_values(), cohorts=cohorts)
    return _caballero_toro_from(cohorts, acc)
