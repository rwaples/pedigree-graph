"""Memory-bounded relationship-pair estimates via scalar arithmetic.

``StreamingPairCounter`` is a read-only collaborator over a
:class:`~pedigree_graph._core.PedigreeGraph`: it reads the graph's parent
matrices and adjacency powers and returns per-code counts computed with
per-anchor ``C(k, 2)`` sums and lineal-edge ``.nnz`` reads — no pair-key
arrays are materialized, so peak memory is O(N) regardless of pedigree
density.  It never writes the graph's caches (ADR 0002); it reports the codes
whose inclusion-exclusion residual it floored at zero as data, so the result
can carry them.

:func:`estimate_relationship_counts` is the 0.8 operation over the counter:
validation, the thread budget, the per-cutoff result cache on the graph, the
typed :class:`~pedigree_graph.relationships.RelationshipCountResult`, matrix
release, and the one ``RuntimeWarning`` per clamped computation (ADR 0006).
Full-graph only.  Which codes are exact is ``REL_PLAN`` in ``_registry``.
"""

from __future__ import annotations

import logging
import time
import warnings
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from pedigree_graph._registry import RELATIONSHIPS, _validate_max_degree, estimate_exact_codes
from pedigree_graph._threads import thread_budget
from pedigree_graph.relationships import RelationshipCountResult

if TYPE_CHECKING:
    from pedigree_graph._core import PedigreeGraph

logger = logging.getLogger(__name__)


class ScalarCounts(NamedTuple):
    """What :meth:`StreamingPairCounter.count` returns.

    Attributes:
        raw: The unfolded scalar counts, all 23 codes, ``0`` above the cutoff.
        overlaps: Per code, how many of ``raw`` a parent-offspring category
            claims under the precedence fold; non-zero only for MHS and PHS.
        clamped: Codes whose residual was floored at zero.
    """

    raw: dict[str, int]
    overlaps: dict[str, int]
    clamped: frozenset[str]


class CachedEstimate(NamedTuple):
    """One cutoff's entry in ``PedigreeGraph._estimate_cache``.

    Attributes:
        result: The public fold-aware result.
    """

    result: RelationshipCountResult


def estimate_relationship_counts(graph: PedigreeGraph, *, max_degree: int) -> RelationshipCountResult:
    """Estimate the number of pairs in every category up to *max_degree*.

    The body of :meth:`PedigreeGraph.estimate_relationship_counts`.

    Args:
        graph: The full graph to count on.
        max_degree: Degree cutoff, validated against ``[0, 5]``.

    Returns:
        The cached :class:`RelationshipCountResult` for this cutoff.

    Raises:
        PedigreeValidationError: ``max_degree_out_of_range``.
    """
    md = _validate_max_degree(max_degree)
    thread_budget()
    return _estimate(graph, md).result


def _estimate(graph: PedigreeGraph, max_degree: int) -> CachedEstimate:
    """Validate and return the cached estimate for *max_degree*, computing it once.

    The warning is raised before the cache write, so under an "error"
    warning filter the first call raises without caching and a retry
    recomputes; a cached retrieval is silent and each cutoff warns on its
    own.  ``stacklevel=4`` reaches the caller of the public method through
    the module function and the graph method.
    """
    md = _validate_max_degree(max_degree)
    cached = graph._estimate_cache.get(md)
    if cached is not None:
        return cached

    raw, overlaps, clamped = StreamingPairCounter(graph).count(md)
    requested = frozenset(code for code, category in RELATIONSHIPS.items() if category.degree <= md)
    exact = requested & estimate_exact_codes()
    values = {code: raw[code] - overlaps[code] if code in requested else None for code in RELATIONSHIPS}
    result = RelationshipCountResult(values, requested, exact, requested - exact, clamped)
    if clamped:
        names = ", ".join(code for code in RELATIONSHIPS if code in clamped)
        warnings.warn(
            f"estimate_relationship_counts(max_degree={md}): the scalar residual for {names} "
            "underflowed and was clamped to 0, so those counts are unreliable rather than a true "
            "absence (typically inbreeding or complex mating); use relationship_counts for exact values",
            RuntimeWarning,
            stacklevel=4,
        )
    entry = CachedEstimate(result)
    graph._estimate_cache[md] = entry
    # _A…_A5 would otherwise stay resident for the graph's lifetime and inflate
    # later inbreeding / Ne work (issue #4).
    graph._release_pair_matrices()
    return entry


class StreamingPairCounter:
    """Count relationship pairs from a PedigreeGraph without materializing pairs.

    ``count`` returns fresh objects; the caller persists them to the graph's
    caches.
    """

    def __init__(self, pg: PedigreeGraph) -> None:
        self.pg = pg
        self._clamped: set[str] = set()

    def count(self, max_degree: int) -> ScalarCounts:
        """Return the raw counts, the fold overlaps, and the clamped codes up to *max_degree*.

        *max_degree* is already validated.  Every dict covers all 23 codes,
        ``0`` above the cutoff.  Full-graph only.
        """
        pg = self.pg
        t_total = time.perf_counter()
        n = pg.n_individuals
        self._clamped = set()

        counts: dict[str, int] = dict.fromkeys(RELATIONSHIPS, 0)
        overlaps: dict[str, int] = dict.fromkeys(RELATIONSHIPS, 0)

        # ---- Degree 0: MZ ---------------------------------------------
        mz_i, _ = pg._mz_twin_pairs()
        counts["MZ"] = len(mz_i)
        if max_degree < 1:
            return self._finalise(counts, overlaps, t_total)

        # ---- Degree 1: MO, FO, FS -------------------------------------
        counts["MO"] = int(np.count_nonzero(pg.mother_rows >= 0))
        counts["FO"] = int(np.count_nonzero(pg.father_rows >= 0))

        sm = pg.mother_ids
        sf = pg.father_ids
        nt = ((sm >= 0) | (sf >= 0)) & (pg.twin_rows < 0)
        nt_idx = np.where(nt)[0]
        nt_m = sm[nt_idx]
        nt_f = sf[nt_idx]

        mating_pair_id = np.full(n, -1, dtype=np.int64)
        pair_k = np.array([], dtype=np.int64)
        fs_count = 0
        both = (nt_m >= 0) & (nt_f >= 0)
        if both.any():
            bk_idx = nt_idx[both]
            bk_m = nt_m[both]
            bk_f = nt_f[both]
            max_p = int(max(bk_m.max(), bk_f.max())) + 1
            family_key = bk_m.astype(np.int64) * max_p + bk_f.astype(np.int64)
            _, inverse, sizes = np.unique(
                family_key,
                return_inverse=True,
                return_counts=True,
            )
            mating_pair_id[bk_idx] = inverse.astype(np.int64)
            pair_k = sizes.astype(np.int64)
            fs_count = int(((pair_k * (pair_k - 1)) // 2).sum())
        counts["FS"] = fs_count

        if max_degree < 2:
            return self._finalise(counts, overlaps, t_total)

        # Sex-side and mating-pair member arrays are reused across every
        # higher-degree branch. Empty arrays make those blocks skip cleanly.
        m_known = nt_m >= 0
        f_known = nt_f >= 0
        has_m = bool(m_known.any())
        has_f = bool(f_known.any())
        # Parent ids are original ids, so the per-parent bincounts below index
        # by the dense group of each parent, never by the id itself (a
        # ten-digit id would otherwise allocate a table of that size).
        _, m_parents, m_sizes = np.unique(nt_m[m_known], return_inverse=True, return_counts=True)
        _, f_parents, f_sizes = np.unique(nt_f[f_known], return_inverse=True, return_counts=True)
        m_anchors = nt_idx[m_known]
        f_anchors = nt_idx[f_known]
        members = mating_pair_id >= 0
        members_pid = mating_pair_id[members]
        has_pairs = len(pair_k) > 0

        # ---- Degree 2: MHS, PHS, GP, Av -------------------------------
        if has_m:
            counts["MHS"] = int(((m_sizes * (m_sizes - 1)) // 2).sum()) - fs_count
        if has_f:
            counts["PHS"] = int(((f_sizes * (f_sizes - 1)) // 2).sum()) - fs_count
        nontwin = pg.twin_rows < 0
        overlaps["MHS"] = _half_sibs_that_are_parent_offspring(pg.father_rows, sm, nontwin)
        overlaps["PHS"] = _half_sibs_that_are_parent_offspring(pg.mother_rows, sf, nontwin)

        # Lazily rebuild _Am / _Af if a pair extraction released them; needed for
        # adjacency powers from degree 2 onward.
        pg._ensure_parent_csr()
        children_count = np.diff(pg._A.tocsc().indptr).astype(np.int64)
        counts["GP"] = int(pg._A2.nnz)

        pair_sum_d1 = np.array([], dtype=np.int64)
        pair_sum_d1_sq = np.array([], dtype=np.int64)
        if has_pairs:
            pair_sum_d1 = np.bincount(
                members_pid,
                weights=children_count[members],
                minlength=len(pair_k),
            ).astype(np.int64)
            pair_sum_d1_sq = np.bincount(
                members_pid,
                weights=children_count[members].astype(np.int64) ** 2,
                minlength=len(pair_k),
            ).astype(np.int64)
            counts["Av"] = int(((pair_k - 1) * pair_sum_d1).sum())

        if max_degree < 3:
            return self._finalise(counts, overlaps, t_total)

        # ---- Degree 3: GGP, HAv, GAv, 1C ------------------------------
        counts["GGP"] = int(pg._A3.nnz)
        d2_count = np.diff(pg._A2.tocsc().indptr).astype(np.int64)
        d3_count = np.diff(pg._A3.tocsc().indptr).astype(np.int64)

        m_av, m_kp, _ = self._per_sex_anchor_sums(
            has_m,
            m_parents,
            children_count,
            m_anchors,
        )
        f_av, f_kp, _ = self._per_sex_anchor_sums(
            has_f,
            f_parents,
            children_count,
            f_anchors,
        )
        counts["HAv"] = m_av + f_av - 2 * counts["Av"]

        pair_sum_d2 = np.array([], dtype=np.int64)
        pair_sum_d3 = np.array([], dtype=np.int64)
        if has_pairs:
            pair_sum_d2 = np.bincount(
                members_pid,
                weights=d2_count[members],
                minlength=len(pair_k),
            ).astype(np.int64)
            counts["GAv"] = int(((pair_k - 1) * pair_sum_d2).sum())
            # 1C: cross-pair grandchildren-via-pair (non-inbred exact).
            counts["1C"] = int(
                ((pair_sum_d1 * pair_sum_d1 - pair_sum_d1_sq) // 2).sum(),
            )

        if max_degree < 4:
            return self._finalise(counts, overlaps, t_total)

        # ---- Degree 4: GGGP, HGAv, GGAv, H1C, 1C1R --------------------
        # H1C: pairs sharing exactly one distinct grandparent.
        h1c_naive = int(((d2_count * (d2_count - 1)) // 2).sum())
        counts["H1C"] = self._clamp_residual(
            "H1C",
            h1c_naive - 4 * counts["FS"] - 2 * counts["MHS"] - 2 * counts["PHS"] - 2 * counts["1C"],
        )

        counts["GGGP"] = int(pg._A4.nnz)
        d4_count = np.diff(pg._A4.tocsc().indptr).astype(np.int64)

        if has_pairs:
            pair_sum_d3 = np.bincount(
                members_pid,
                weights=d3_count[members],
                minlength=len(pair_k),
            ).astype(np.int64)
            counts["GGAv"] = int(((pair_k - 1) * pair_sum_d3).sum())
            naive_1c1r = int((pair_sum_d1 * pair_sum_d2).sum())
            counts["1C1R"] = self._clamp_residual(
                "1C1R",
                naive_1c1r - counts["Av"] - counts["GAv"],
            )

        m_hgav, _, _ = self._per_sex_anchor_sums(has_m, m_parents, d2_count, m_anchors, m_kp)
        f_hgav, _, _ = self._per_sex_anchor_sums(has_f, f_parents, d2_count, f_anchors, f_kp)
        counts["HGAv"] = m_hgav + f_hgav - 2 * counts["GAv"]

        if max_degree < 5:
            return self._finalise(counts, overlaps, t_total)

        # ---- Degree 5: G3GP, HGGAv, G3Av, H1C1R, 1C2R, 2C -------------
        counts["G3GP"] = int(pg._A5.nnz)

        if has_pairs:
            pair_sum_d4 = np.bincount(
                members_pid,
                weights=d4_count[members],
                minlength=len(pair_k),
            ).astype(np.int64)
            counts["G3Av"] = int(((pair_k - 1) * pair_sum_d4).sum())
            naive_1c2r = int((pair_sum_d1 * pair_sum_d3).sum())
            counts["1C2R"] = self._clamp_residual(
                "1C2R",
                naive_1c2r - counts["GAv"] - counts["GGAv"],
            )
            counts["2C"] = int(((pair_sum_d2 * (pair_sum_d2 - 1)) // 2).sum())

        m_hggav, _, _ = self._per_sex_anchor_sums(has_m, m_parents, d3_count, m_anchors, m_kp)
        f_hggav, _, _ = self._per_sex_anchor_sums(has_f, f_parents, d3_count, f_anchors, f_kp)
        counts["HGGAv"] = m_hggav + f_hggav - 2 * counts["GGAv"]

        h1c1r_naive = int((d2_count * d3_count).sum())
        counts["H1C1R"] = self._clamp_residual(
            "H1C1R",
            h1c1r_naive - 2 * counts["1C1R"] - counts["HAv"] - counts["HGAv"],
        )

        return self._finalise(counts, overlaps, t_total)

    @staticmethod
    def _per_sex_anchor_sums(
        has_side: bool,
        parents: np.ndarray,
        weights: np.ndarray,
        anchors: np.ndarray,
        kp_cached: np.ndarray | None = None,
    ) -> tuple[int, np.ndarray, np.ndarray]:
        """Σ over single-parent anchor i: (kp_i − 1) · Σ_d weights[child].

        Per-degree HAv/HGAv/HGGAv all collapse to this form, parameterised
        by the depth-d ``weights`` array (``children_count``,
        ``d2_count``, …).  ``kp_cached`` lets the d>1 caller reuse the
        per-parent child count computed at d=1, avoiding a redundant
        ``np.bincount(parents)`` per degree.

        Returns ``(scalar_sum, kp, weighted_sum_per_parent)`` so the d=1
        caller can capture ``kp`` and ``weighted_sum`` for downstream
        diagnostics; later calls discard them.
        """
        if not has_side:
            empty = np.array([], dtype=np.int64)
            return 0, empty, empty
        weighted_per_parent = np.bincount(parents, weights=weights[anchors]).astype(np.int64)
        kp = kp_cached if kp_cached is not None else np.bincount(parents).astype(np.int64)
        total = int(((kp - 1).clip(min=0) * weighted_per_parent).sum())
        return total, kp, weighted_per_parent

    def _clamp_residual(self, code: str, raw: int) -> int:
        """Floor an inclusion-exclusion residual at zero, recording the code on underflow.

        The cousin/collateral residual codes (``H1C``, ``1C1R``, ``1C2R``,
        ``H1C1R``) subtract closer-relationship contributions with fixed
        coefficients that are exact only on non-inbred, single-mating
        pedigrees.  A negative raw residual means those corrections
        over-counted: the clamped ``0`` is unreliable rather than a true
        absence, which is why the code is reported in the result's ``clamped``.
        """
        if raw < 0:
            self._clamped.add(code)
        return max(0, raw)

    def _finalise(self, counts: dict[str, int], overlaps: dict[str, int], t_total: float) -> ScalarCounts:
        """Log timing and return copies of the counts, the overlaps, and the clamped set."""
        logger.info(
            "estimate_relationship_counts total: %.3fs",
            time.perf_counter() - t_total,
        )
        return ScalarCounts(dict(counts), dict(overlaps), frozenset(self._clamped))


def _half_sibs_that_are_parent_offspring(
    other_parent: np.ndarray,
    shared_parent_id: np.ndarray,
    nontwin: np.ndarray,
) -> int:
    """Count half-sib pairs the precedence fold files as parent-offspring.

    A child and its father who have the same mother are MHS by the sibling
    formula and FO by the lineal one; the fold keeps FO.  *other_parent* is the
    parent row array on the lineal side (``father`` for MHS), *shared_parent_id*
    the original-ID array of the shared side (``_orig_mother``), matching the
    sibling formula's own grouping.  Such a pair can never be a full sib (the
    child's other parent would have to be the parent itself, a cycle).
    """
    child = np.flatnonzero((other_parent >= 0) & nontwin)
    parent = other_parent[child]
    shared = shared_parent_id[child]
    return int(np.count_nonzero(nontwin[parent] & (shared >= 0) & (shared_parent_id[parent] == shared)))
