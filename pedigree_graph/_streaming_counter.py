"""Memory-bounded relationship-pair counts via scalar arithmetic.

``StreamingPairCounter`` is a read-only collaborator over a
:class:`~pedigree_graph._core.PedigreeGraph`: it reads the graph's parent
matrices and adjacency powers and returns per-code counts computed with
per-anchor ``C(k, 2)`` sums and lineal-edge ``.nnz`` reads — no pair-key
arrays are materialized, so peak memory is O(N) regardless of pedigree
density.  It never writes the graph's result cache; the thin
``PedigreeGraph.count_pairs_streaming`` wrapper owns cache lookup/persist
and the scope/subsample validation.  See ADR 0002.

The scalar path is full-graph only.  See the wrapper docstring for the
exact / approximate precision contract per relationship code.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import numpy as np

from pedigree_graph._registry import REL_REGISTRY

if TYPE_CHECKING:
    from pedigree_graph._core import PedigreeGraph

logger = logging.getLogger(__name__)


class StreamingPairCounter:
    """Count relationship pairs from a PedigreeGraph without materializing pairs.

    ``count`` returns a fresh dict; the caller persists it to the graph's
    count cache.
    """

    def __init__(self, pg: PedigreeGraph) -> None:
        self.pg = pg

    def count(self, max_degree: int) -> dict[str, int]:
        """Return per-code counts up to *max_degree* (already validated).

        Codes above *max_degree* are ``0``.  Full-graph scope only — the
        wrapper rejects ``scope='subsample'`` on ``from_subsample`` graphs.
        """
        pg = self.pg
        t_total = time.perf_counter()
        n = pg.n

        # Lazily rebuild _Am / _Af if extract_pairs deleted them.
        pg._ensure_parent_csr()

        counts: dict[str, int] = dict.fromkeys(REL_REGISTRY, 0)
        children_count = np.diff(pg._A.tocsc().indptr).astype(np.int64)

        # ---- Degree 0: MZ ---------------------------------------------
        mz_i, _ = pg._mz_twin_pairs()
        counts["MZ"] = len(mz_i)

        # ---- Degree 1: MO, FO, FS -------------------------------------
        counts["MO"] = int(pg._Am.nnz)
        counts["FO"] = int(pg._Af.nnz)

        sm = pg._orig_mother
        sf = pg._orig_father
        nt = ((sm >= 0) | (sf >= 0)) & (pg.twin < 0)
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
                family_key, return_inverse=True, return_counts=True,
            )
            mating_pair_id[bk_idx] = inverse.astype(np.int64)
            pair_k = sizes.astype(np.int64)
            fs_count = int(((pair_k * (pair_k - 1)) // 2).sum())
        counts["FS"] = fs_count

        # Sex-side and mating-pair member arrays are reused across every
        # degree branch — compute them once.  Empty when no sample has the
        # corresponding known parent (the per-degree blocks skip cleanly).
        m_known = nt_m >= 0
        f_known = nt_f >= 0
        has_m = bool(m_known.any())
        has_f = bool(f_known.any())
        m_parents = nt_m[m_known]
        f_parents = nt_f[f_known]
        m_anchors = nt_idx[m_known]
        f_anchors = nt_idx[f_known]
        members = mating_pair_id >= 0
        members_pid = mating_pair_id[members]
        has_pairs = len(pair_k) > 0

        if has_m:
            _, m_sizes = np.unique(m_parents, return_counts=True)
            counts["MHS"] = (
                int(((m_sizes * (m_sizes - 1)) // 2).sum()) - fs_count
            )
        if has_f:
            _, f_sizes = np.unique(f_parents, return_counts=True)
            counts["PHS"] = (
                int(((f_sizes * (f_sizes - 1)) // 2).sum()) - fs_count
            )

        if max_degree < 2:
            return self._finalise(counts, t_total)

        # ---- Degree 2: GP, Av -----------------------------------------
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
            return self._finalise(counts, t_total)

        # ---- Degree 3: GGP, HAv, GAv, 1C, H1C -------------------------
        counts["GGP"] = int(pg._A3.nnz)
        d2_count = np.diff(pg._A2.tocsc().indptr).astype(np.int64)
        d3_count = np.diff(pg._A3.tocsc().indptr).astype(np.int64)

        m_av, m_kp, _ = self._per_sex_anchor_sums(
            has_m, m_parents, children_count, m_anchors,
        )
        f_av, f_kp, _ = self._per_sex_anchor_sums(
            has_f, f_parents, children_count, f_anchors,
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

        # H1C: pairs sharing exactly one distinct grandparent.
        h1c_naive = int(((d2_count * (d2_count - 1)) // 2).sum())
        counts["H1C"] = max(
            0,
            h1c_naive
            - 4 * counts["FS"]
            - 2 * counts["MHS"]
            - 2 * counts["PHS"]
            - 2 * counts["1C"],
        )

        if max_degree < 4:
            return self._finalise(counts, t_total)

        # ---- Degree 4: GGGP, HGAv, GGAv, 1C1R -------------------------
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
            counts["1C1R"] = max(
                0, naive_1c1r - counts["Av"] - counts["GAv"],
            )

        m_hgav, _, _ = self._per_sex_anchor_sums(has_m, m_parents, d2_count, m_anchors, m_kp)
        f_hgav, _, _ = self._per_sex_anchor_sums(has_f, f_parents, d2_count, f_anchors, f_kp)
        counts["HGAv"] = m_hgav + f_hgav - 2 * counts["GAv"]

        if max_degree < 5:
            return self._finalise(counts, t_total)

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
            counts["1C2R"] = max(
                0, naive_1c2r - counts["GAv"] - counts["GGAv"],
            )
            counts["2C"] = int(((pair_sum_d2 * (pair_sum_d2 - 1)) // 2).sum())

        m_hggav, _, _ = self._per_sex_anchor_sums(has_m, m_parents, d3_count, m_anchors, m_kp)
        f_hggav, _, _ = self._per_sex_anchor_sums(has_f, f_parents, d3_count, f_anchors, f_kp)
        counts["HGGAv"] = m_hggav + f_hggav - 2 * counts["GGAv"]

        h1c1r_naive = int((d2_count * d3_count).sum())
        counts["H1C1R"] = max(
            0,
            h1c1r_naive
            - 2 * counts["1C1R"]
            - counts["HAv"]
            - counts["HGAv"],
        )

        return self._finalise(counts, t_total)

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
        kp = (
            kp_cached if kp_cached is not None
            else np.bincount(parents).astype(np.int64)
        )
        total = int(((kp - 1).clip(min=0) * weighted_per_parent).sum())
        return total, kp, weighted_per_parent

    @staticmethod
    def _finalise(counts: dict[str, int], t_total: float) -> dict[str, int]:
        """Log timing and return a copy of the counts.

        Cache persistence is the wrapper's responsibility (read-only
        engine contract, ADR 0002).
        """
        logger.info(
            "count_pairs_streaming total: %.3fs",
            time.perf_counter() - t_total,
        )
        return dict(counts)
