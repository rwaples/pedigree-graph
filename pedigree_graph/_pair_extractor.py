"""Exact relationship-pair extraction via sparse matrix products.

``MatrixPairExtractor`` is a read-only collaborator over a
:class:`~pedigree_graph._core.PedigreeGraph`: it holds a ``pg`` reference,
reads the graph's cached adjacency powers / sibling matrices, and returns
relationship-pair arrays.  It never writes the graph's result cache — the
thin ``PedigreeGraph.extract_pairs`` wrapper owns persisting counts and
releasing the transient matrices.  See ADR 0002.

One extractor instance spans a single ``extract()`` call, so degree-gated
run-state (the half-1C pairs found at degree 3 and consumed at degree 4)
lives as instance state and cannot leak between calls.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.sparse as sp

from pedigree_graph._pair_utils import (
    dedup_pairs,
    extract_from_sparse,
    pairs_from_groups,
    remap_pairs_to_caller,
)
from pedigree_graph._registry import PAIR_KINSHIP

if TYPE_CHECKING:
    from pedigree_graph._core import PedigreeGraph

logger = logging.getLogger(__name__)


class MatrixPairExtractor:
    """Extract exact relationship-pair arrays from a PedigreeGraph.

    Holds a reference to the owning graph and reads its adjacency powers
    (``pg._A`` … ``pg._A5``), sibling matrices, and parent arrays.  The
    extractor is side-effect-free with respect to the graph's result
    cache; ``extract`` returns ``(pairs, raw_counts, subsample_counts)``
    and the caller persists them.
    """

    def __init__(self, pg: PedigreeGraph) -> None:
        self.pg = pg
        # Half-1C pairs (share exactly one grandparent) discovered while
        # extracting full 1st cousins at degree 3; consumed by H1C
        # extraction at degree 4.  Instance state → fresh per extract() run.
        self._h1c_pairs_cache: tuple[np.ndarray, np.ndarray] = (
            np.array([], dtype=np.intp),
            np.array([], dtype=np.intp),
        )

    # ------------------------------------------------------------------
    # Per-relationship extraction primitives
    # ------------------------------------------------------------------

    def _lineal_pairs(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Direct ancestor-descendant pairs at exactly k hops."""
        Ak = self.pg._get_Ak(k)
        desc_i, anc_j = Ak.nonzero()
        if len(desc_i) == 0:
            return np.array([], dtype=np.intp), np.array([], dtype=np.intp)
        return desc_i.astype(np.intp), anc_j.astype(np.intp)

    def _collateral_pairs(
        self,
        sib_matrix: sp.spmatrix,
        up: int,
        down: int,
        subtract: list[tuple[np.ndarray, np.ndarray]] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Pairs connected through a sibling link at depths (up, down).

        Individual B is (down-1) hops below a sibling of someone (up-1)
        hops above individual A, where sibling type is determined by
        *sib_matrix* (full-sib or half-sib).
        """
        if sib_matrix.nnz == 0:
            return np.array([], dtype=np.intp), np.array([], dtype=np.intp)
        A_down_1 = self.pg._get_Ak(down - 1)
        A_up_1 = self.pg._get_Ak(up - 1)
        M = A_down_1 @ sib_matrix @ A_up_1.T
        return extract_from_sparse(M, subtract=subtract)

    def _cousin_pairs(self) -> tuple[np.ndarray, np.ndarray]:
        """Full 1st cousin pairs: share exactly 2 grandparents (a mated pair) but not a parent.

        Uses group-by-grandparent enumeration. Each pair sharing a grandparent
        is counted — pairs appearing ≥ 2 times share 2+ grandparents (full 1C).
        Pairs appearing exactly once share 1 grandparent (half-1C); these are
        cached in ``_h1c_pairs_cache`` for use by H1C extraction at degree 4.
        """
        pg = self.pg
        t0 = time.perf_counter()
        empty = np.array([], dtype=np.intp), np.array([], dtype=np.intp)
        gc_i, gp_j = pg._A2.nonzero()
        if len(gc_i) == 0:
            self._h1c_pairs_cache = empty
            return empty

        # Enumerate all (i < j) pairs sharing a grandparent
        p1, p2 = pairs_from_groups(gc_i.astype(np.intp), gp_j)
        if len(p1) == 0:
            self._h1c_pairs_cache = empty
            return empty

        logger.debug(
            "Cousin group-by: %d candidate pairs from %d edges (%.3fs)",
            len(p1),
            len(gc_i),
            time.perf_counter() - t0,
        )

        # Remove sibling/half-sib pairs (those sharing a parent)
        share_mother = (pg._orig_mother[p1] >= 0) & (pg._orig_mother[p1] == pg._orig_mother[p2])
        share_father = (pg._orig_father[p1] >= 0) & (pg._orig_father[p1] == pg._orig_father[p2])
        is_sib = share_mother | share_father
        p1, p2 = p1[~is_sib], p2[~is_sib]

        if len(p1) == 0:
            logger.debug("Cousins: 0 pairs after sibling removal (%.3fs)", time.perf_counter() - t0)
            self._h1c_pairs_cache = empty
            return empty

        # Count shared grandparents per pair using int64 keys
        lo = np.minimum(p1, p2).astype(np.intp)
        hi = np.maximum(p1, p2).astype(np.intp)
        max_id = int(hi.max()) + 1
        keys = lo.astype(np.int64) * max_id + hi.astype(np.int64)
        unique_keys, _inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)

        # Full 1C: pairs sharing >= 2 grandparents
        full_mask = counts >= 2
        full_idx = np.where(full_mask)[0]
        # Map unique keys back to (lo, hi)
        full_lo = (unique_keys[full_idx] // max_id).astype(np.intp)
        full_hi = (unique_keys[full_idx] % max_id).astype(np.intp)

        # Half 1C: pairs sharing exactly 1 grandparent — cache for H1C extraction
        half_mask = counts == 1
        half_idx = np.where(half_mask)[0]
        half_lo = (unique_keys[half_idx] // max_id).astype(np.intp)
        half_hi = (unique_keys[half_idx] % max_id).astype(np.intp)
        self._h1c_pairs_cache = (half_lo, half_hi)

        logger.debug(
            "Cousins: %d full 1C, %d half 1C (%.3fs)",
            len(full_lo),
            len(half_lo),
            time.perf_counter() - t0,
        )
        return full_lo, full_hi

    def _grandparent_grandchild_pairs(self) -> tuple[np.ndarray, np.ndarray]:
        """Grandparent-grandchild pairs: 2-hop ancestor links."""
        return self._lineal_pairs(2)

    def _avuncular_pairs(self, full_sib: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Avuncular (uncle/aunt-nephew/niece) pairs.

        An avuncular pair (child C, uncle U) exists when C's parent P is a
        full sibling of U. In matrix form: A @ S_full, then exclude
        parent-child pairs (which share the same edge structure).
        """
        pg = self.pg
        pg._ensure_sibling_matrices()
        if pg._full_sib_matrix.nnz == 0:
            return np.array([], dtype=np.intp), np.array([], dtype=np.intp)

        avunc = pg._A @ pg._full_sib_matrix
        avunc.setdiag(0)

        # Exclude parent-child pairs
        parent_child = (pg._A + pg._A.T) > 0
        avunc = avunc - avunc.multiply(parent_child)
        avunc.eliminate_zeros()

        if avunc.nnz == 0:
            return np.array([], dtype=np.intp), np.array([], dtype=np.intp)

        return dedup_pairs(*avunc.nonzero())

    def _second_cousin_matrix(self) -> sp.spmatrix:
        """Symmetric sparse matrix with nonzeros at full 2nd cousin pairs.

        Full 2nd cousins share ≥ 2 great-grandparents (a mated pair) but no
        grandparents.  Half-2nd-cousins (1 shared great-grandparent) are
        excluded — they fall beyond degree 5.
        """
        pg = self.pg
        t0 = time.perf_counter()
        D_raw = pg._A3 @ pg._A3.T
        logger.debug("A3 @ A3.T computed in %.3fs (nnz=%d)", time.perf_counter() - t0, D_raw.nnz)
        # Keep only pairs sharing ≥ 2 great-grandparents (full 2C), then booleanise
        D_raw.data[D_raw.data < 2] = 0
        D_raw.eliminate_zeros()
        D_raw.data[:] = 1.0
        C_raw = pg._A2_shared.copy()
        C_raw.data[:] = 1.0

        second_cousins = D_raw - D_raw.multiply(C_raw)
        second_cousins.setdiag(0)
        second_cousins.eliminate_zeros()
        logger.debug("2nd cousin matrix: nnz=%d (%.3fs total)", second_cousins.nnz, time.perf_counter() - t0)
        return second_cousins

    def _second_cousin_pairs(self) -> tuple[np.ndarray, np.ndarray]:
        """2nd cousin pairs: share a great-grandparent but not a grandparent."""
        second_cousins = self._second_cousin_matrix()

        sc_upper = sp.triu(second_cousins, k=1)
        sc_i, sc_j = sc_upper.nonzero()

        if len(sc_i) == 0:
            return np.array([], dtype=np.intp), np.array([], dtype=np.intp)
        return sc_i.astype(np.intp), sc_j.astype(np.intp)

    # ------------------------------------------------------------------
    # Top-level extraction
    # ------------------------------------------------------------------

    def extract(
        self,
        max_degree: int,
        min_kinship: float = 0.0,
    ) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, int], dict[str, int]]:
        """Extract all relationship categories up to *max_degree*.

        *max_degree* must already be validated by the caller.  Returns
        ``(pairs, raw_counts, subsample_counts)``:

        - ``pairs``: code → (idx1, idx2) in caller-input coordinates.
        - ``raw_counts``: per-code counts before sample-mask filtering
          (full-graph scope).
        - ``subsample_counts``: per-code counts after mask + remap
          (caller scope).

        The extractor does not persist these; the ``extract_pairs``
        wrapper writes them to the graph's count cache and releases the
        transient matrices.
        """
        pg = self.pg
        t_total = time.perf_counter()
        pairs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        empty = np.array([], dtype=np.intp)

        def _needed(code: str) -> bool:
            return PAIR_KINSHIP.get(code, 0) >= min_kinship

        need_hs = _needed("MHS") and max_degree >= 2

        # Pre-trigger cached properties needed by downstream extractions.
        # _Am/_Af are only needed to build _A; delete after to free memory.
        pg._ensure_parent_csr()
        if max_degree >= 2:
            _ = pg._A2  # chains: _Am, _Af → _A → _A2
        else:
            _ = pg._A
        del pg._Am, pg._Af

        pairs["MZ"] = pg._mz_twin_pairs()

        mo, fo = pg._parent_offspring_pairs()
        pairs["MO"] = mo
        pairs["FO"] = fo

        t0 = time.perf_counter()
        full_sib, mat_hs, pat_hs = pg.sibling_pairs()
        pairs["FS"] = full_sib
        pairs["MHS"] = mat_hs if need_hs else (empty, empty)
        pairs["PHS"] = pat_hs if need_hs else (empty, empty)
        logger.info(
            "Siblings: %d full, %d maternal HS, %d paternal HS (%.3fs)",
            len(pairs["FS"][0]),
            len(pairs["MHS"][0]),
            len(pairs["PHS"][0]),
            time.perf_counter() - t0,
        )

        # ---- Degree 2 (kinship 1/8): GP, Av, 1C ----
        if max_degree >= 2:
            t0 = time.perf_counter()
            need_cousins = _needed("1C")
            need_gp = _needed("GP")
            need_avunc = _needed("Av")

            futures = {}
            with ThreadPoolExecutor(max_workers=3) as pool:
                if need_cousins:
                    futures["1C"] = pool.submit(self._cousin_pairs)
                if need_gp:
                    futures["GP"] = pool.submit(self._grandparent_grandchild_pairs)
                if need_avunc:
                    futures["Av"] = pool.submit(self._avuncular_pairs, full_sib)
                for k, fut in futures.items():
                    pairs[k] = fut.result()

            for k in ("1C", "GP", "Av"):
                if k not in pairs:
                    pairs[k] = (empty, empty)

            logger.info(
                "Degree 2: cousins=%d, grandparent=%d, avuncular=%d (%.3fs)%s",
                len(pairs["1C"][0]),
                len(pairs["GP"][0]),
                len(pairs["Av"][0]),
                time.perf_counter() - t0,
                f" [min_kinship={min_kinship}]" if min_kinship > 0 else "",
            )
        else:
            for k in ("1C", "GP", "Av"):
                pairs[k] = (empty, empty)

        # ---- Degree 3+ setup (deferred to avoid work at default degree 2) ----
        if max_degree >= 3:
            po_pairs = (
                np.concatenate([pairs["MO"][0], pairs["FO"][0]]),
                np.concatenate([pairs["MO"][1], pairs["FO"][1]]),
            )
            gp_pairs = pairs["GP"]
            fsm = pg._full_sib_matrix
            pg._build_half_sib_matrix(mat_hs, pat_hs)
            hsm = pg._half_sib_matrix
        # sib_all only needed at degree 4+ (1C1R, H1C1R, 1C2R subtract lists)
        if max_degree >= 4:
            sib_all = (
                np.concatenate([pairs["FS"][0], pairs["MHS"][0], pairs["PHS"][0]]),
                np.concatenate([pairs["FS"][1], pairs["MHS"][1], pairs["PHS"][1]]),
            )

        # ---- Degree 3 (kinship 1/16): GGP, HAv, GAv ----
        if max_degree >= 3:
            t0 = time.perf_counter()
            _ = pg._A3  # pre-trigger

            futures: dict[str, Any] = {}
            with ThreadPoolExecutor(max_workers=3) as pool:
                if _needed("GGP"):
                    futures["GGP"] = pool.submit(self._lineal_pairs, 3)
                if _needed("HAv"):
                    futures["HAv"] = pool.submit(self._collateral_pairs, hsm, 1, 2, [po_pairs, gp_pairs])
                if _needed("GAv"):
                    futures["GAv"] = pool.submit(self._collateral_pairs, fsm, 1, 3, [po_pairs, gp_pairs, pairs["Av"]])
                for k, fut in futures.items():
                    pairs[k] = fut.result()
            for code in ("GGP", "HAv", "GAv"):
                if code not in pairs:
                    pairs[code] = (empty, empty)

            logger.info(
                "Degree 3: GGP=%d, HAv=%d, GAv=%d (%.3fs)",
                len(pairs["GGP"][0]),
                len(pairs["HAv"][0]),
                len(pairs["GAv"][0]),
                time.perf_counter() - t0,
            )
        else:
            for code in ("GGP", "HAv", "GAv"):
                pairs[code] = (empty, empty)

        # ---- Degree 4 (kinship 1/32): GGGP, HGAv, GGAv, H1C, 1C1R ----
        if max_degree >= 4:
            t0 = time.perf_counter()
            # Lazy: _A4 and A2_A3T triggered by types that need them
            A2_A3T = None
            if _needed("1C1R"):
                A2_A3T = pg._A2 @ pg._A3.T

            def _extract_h1c() -> tuple[np.ndarray, np.ndarray]:
                # Use cached half-cousin pairs from _cousin_pairs() — already
                # identified as pairs sharing exactly 1 grandparent, with
                # sibling pairs excluded.
                return self._h1c_pairs_cache

            def _extract_1c1r() -> tuple[np.ndarray, np.ndarray]:
                P_full = A2_A3T.copy()
                P_full.setdiag(0)
                P_full.data[P_full.data < 2] = 0
                P_full.eliminate_zeros()
                return extract_from_sparse(
                    P_full,
                    subtract=[po_pairs, gp_pairs, pairs["GGP"], pairs["Av"], pairs["GAv"], sib_all, pairs["1C"]],
                )

            futures = {}
            with ThreadPoolExecutor(max_workers=5) as pool:
                if _needed("GGGP"):
                    futures["GGGP"] = pool.submit(self._lineal_pairs, 4)
                if _needed("HGAv"):
                    futures["HGAv"] = pool.submit(
                        self._collateral_pairs,
                        hsm,
                        1,
                        3,
                        [po_pairs, gp_pairs, pairs["GGP"], pairs["HAv"]],
                    )
                if _needed("GGAv"):
                    futures["GGAv"] = pool.submit(
                        self._collateral_pairs,
                        fsm,
                        1,
                        4,
                        [po_pairs, gp_pairs, pairs["GGP"], pairs["Av"], pairs["GAv"]],
                    )
                if _needed("H1C"):
                    futures["H1C"] = pool.submit(_extract_h1c)
                if _needed("1C1R"):
                    futures["1C1R"] = pool.submit(_extract_1c1r)
                for k, fut in futures.items():
                    pairs[k] = fut.result()
            for code in ("GGGP", "HGAv", "GGAv", "H1C", "1C1R"):
                if code not in pairs:
                    pairs[code] = (empty, empty)

            logger.info(
                "Degree 4: GGGP=%d, HGAv=%d, GGAv=%d, H1C=%d, 1C1R=%d (%.3fs)",
                len(pairs["GGGP"][0]),
                len(pairs["HGAv"][0]),
                len(pairs["GGAv"][0]),
                len(pairs["H1C"][0]),
                len(pairs["1C1R"][0]),
                time.perf_counter() - t0,
            )
        else:
            A2_A3T = None
            for code in ("GGGP", "HGAv", "GGAv", "H1C", "1C1R"):
                pairs[code] = (empty, empty)

        # ---- Degree 5 (kinship 1/64): 2C, G3GP, HGGAv, G3Av, H1C1R, 1C2R ----
        if max_degree >= 5:
            t0 = time.perf_counter()
            # _A5 triggered lazily by G3GP (_lineal_pairs(5))
            # A2_A3T needed by H1C1R only
            if _needed("H1C1R") and A2_A3T is None:
                A2_A3T = pg._A2 @ pg._A3.T

            def _extract_h1c1r() -> tuple[np.ndarray, np.ndarray]:
                P_half = A2_A3T.copy()
                P_half.setdiag(0)
                P_half.data[P_half.data != 1] = 0
                P_half.eliminate_zeros()
                return extract_from_sparse(
                    P_half,
                    subtract=[
                        po_pairs,
                        gp_pairs,
                        pairs["GGP"],
                        pairs["GGGP"],
                        pairs["HAv"],
                        pairs["HGAv"],
                        sib_all,
                        pairs["1C"],
                        pairs["H1C"],
                        pairs["1C1R"],
                    ],
                )

            def _extract_1c2r() -> tuple[np.ndarray, np.ndarray]:
                P_full = pg._A2 @ pg._A4.T
                P_full.setdiag(0)
                P_full.data[P_full.data < 2] = 0
                P_full.eliminate_zeros()
                return extract_from_sparse(
                    P_full,
                    subtract=[
                        po_pairs,
                        gp_pairs,
                        pairs["GGP"],
                        pairs["GGGP"],
                        pairs["Av"],
                        pairs["GAv"],
                        pairs["GGAv"],
                        sib_all,
                        pairs["1C"],
                        pairs["H1C"],
                        pairs["1C1R"],
                    ],
                )

            futures = {}
            with ThreadPoolExecutor(max_workers=6) as pool:
                if _needed("2C"):
                    futures["2C"] = pool.submit(self._second_cousin_pairs)
                if _needed("G3GP"):
                    futures["G3GP"] = pool.submit(self._lineal_pairs, 5)
                if _needed("HGGAv"):
                    futures["HGGAv"] = pool.submit(
                        self._collateral_pairs,
                        hsm,
                        1,
                        4,
                        [po_pairs, gp_pairs, pairs["GGP"], pairs["GGGP"], pairs["HAv"], pairs["HGAv"]],
                    )
                if _needed("G3Av"):
                    futures["G3Av"] = pool.submit(
                        self._collateral_pairs,
                        fsm,
                        1,
                        5,
                        [po_pairs, gp_pairs, pairs["GGP"], pairs["GGGP"], pairs["Av"], pairs["GAv"], pairs["GGAv"]],
                    )
                if _needed("H1C1R"):
                    futures["H1C1R"] = pool.submit(_extract_h1c1r)
                if _needed("1C2R"):
                    futures["1C2R"] = pool.submit(_extract_1c2r)
                for k, fut in futures.items():
                    pairs[k] = fut.result()
            for code in ("2C", "G3GP", "HGGAv", "G3Av", "H1C1R", "1C2R"):
                if code not in pairs:
                    pairs[code] = (empty, empty)

            logger.info(
                "Degree 5: 2C=%d, G3GP=%d, HGGAv=%d, G3Av=%d, H1C1R=%d, 1C2R=%d (%.3fs)",
                len(pairs["2C"][0]),
                len(pairs["G3GP"][0]),
                len(pairs["HGGAv"][0]),
                len(pairs["G3Av"][0]),
                len(pairs["H1C1R"][0]),
                len(pairs["1C2R"][0]),
                time.perf_counter() - t0,
            )
        else:
            for code in ("2C", "G3GP", "HGGAv", "G3Av", "H1C1R", "1C2R"):
                pairs[code] = (empty, empty)

        # Save raw counts before sample_mask filtering (used by count_pairs(scope="full"))
        raw_pair_counts = {k: len(v[0]) for k, v in pairs.items()}

        # Restrict to active (sampled) individuals when a mask is set
        if pg._sample_mask is not None:
            for k, (idx1, idx2) in pairs.items():
                if len(idx1) > 0:
                    mask = pg._sample_mask[idx1] & pg._sample_mask[idx2]
                    pairs[k] = (idx1[mask].astype(np.intp), idx2[mask].astype(np.intp))
                else:
                    pairs[k] = (empty, empty)
            logger.info(
                "Filtered to sample_mask: %s",
                ", ".join(f"{k}: {len(v[0])}" for k, v in pairs.items()),
            )

        # Remap graph-space row indices to caller-space when a remap is set.
        # After this, pair indices are in caller-input coordinates regardless
        # of which constructor was used.  remap_pairs_to_caller re-canonicalizes
        # (lo, hi) since the remap can permute rows (see PGQ-001).
        if pg._subsample_remap is not None:
            remap_pairs_to_caller(pairs, pg._subsample_remap)

        # Subsample-filtered counts so count_pairs(scope="subsample") is O(1).
        subsample_pair_counts = {k: len(v[0]) for k, v in pairs.items()}

        logger.info("extract_pairs total: %.3fs", time.perf_counter() - t_total)
        return pairs, raw_pair_counts, subsample_pair_counts
