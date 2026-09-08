"""Exact relationship-pair extraction via sparse matrix products.

``MatrixPairExtractor`` is a read-only collaborator over a
:class:`~pedigree_graph._core.PedigreeGraph`: it holds a ``pg`` reference,
reads the graph's cached adjacency powers / sibling matrices, and returns
graph-space relationship pairs in the semantic orientation of each
:class:`~pedigree_graph._registry.RelationshipCategory`.  It never writes the
graph's result cache and never releases the transient matrices; its callers
(``relationship_pairs`` here) own that.
See ADR 0002 and ADR 0006.

One extractor instance spans a single ``extract()`` call, so degree-gated
run-state (the half-1C pairs found at degree 3 and consumed at degree 4)
lives as instance state and cannot leak between calls.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sp

from pedigree_graph._input import _own
from pedigree_graph._pair_utils import (
    canonical_keys,
    oriented_pairs_from_sparse,
    pairs_from_groups,
    project_pairs,
    sort_by_canonical_key,
)
from pedigree_graph._registry import RELATIONSHIPS, categories_up_to_degree, select_categories
from pedigree_graph._threads import thread_budget
from pedigree_graph.relationships import RelationshipPairBlock, RelationshipPairs

if TYPE_CHECKING:
    from collections.abc import Iterable

    from pedigree_graph._core import PedigreeGraph
    from pedigree_graph._view import CoordinateToken, PedigreeView

logger = logging.getLogger(__name__)

# A per-code extraction thunk: returns one relationship's (first, second) arrays.
_Thunk = Callable[[], tuple[np.ndarray, np.ndarray]]

_REGISTRY_INDEX = {code: index for index, code in enumerate(RELATIONSHIPS)}


def dependency_closure(requested: frozenset[str]) -> frozenset[str]:
    """Return the codes the engine must compute to produce *requested* exactly.

    Every engine dependency is of strictly lower degree (the subtract lists,
    the H1C cache filled by 1C, the sibling and parent-offspring blocks), so
    the closure is the registry prefix ending at the last requested code:
    every code of lower degree plus the same-degree codes up to it.

    Args:
        requested: Registry codes the caller wants.

    Returns:
        *requested* plus its dependencies; empty when *requested* is empty.
    """
    if not requested:
        return frozenset()
    top = max(requested, key=_REGISTRY_INDEX.__getitem__)
    return frozenset(list(RELATIONSHIPS)[: _REGISTRY_INDEX[top] + 1])


class MatrixPairExtractor:
    """Extract exact, oriented relationship-pair arrays from a PedigreeGraph.

    Holds a reference to the owning graph and reads its adjacency powers
    (``pg._A`` … ``pg._A5``), sibling matrices, and parent arrays.  The
    extractor is side-effect-free with respect to the graph's result cache.

    Args:
        pg: The graph to read.
        max_workers: Thread cap for the per-degree parallel stage.
    """

    def __init__(self, pg: PedigreeGraph, *, max_workers: int) -> None:
        self.pg = pg
        self._max_workers = max_workers
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
        desc_i, anc_j = Ak.nonzero()  # ty: ignore[unresolved-attribute]
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
        if sib_matrix.nnz == 0:  # ty: ignore[unresolved-attribute]
            return np.array([], dtype=np.intp), np.array([], dtype=np.intp)
        A_down_1 = self.pg._get_Ak(down - 1)
        A_up_1 = self.pg._get_Ak(up - 1)
        M = A_down_1 @ sib_matrix @ A_up_1.T  # ty: ignore[unsupported-operator, unresolved-attribute]
        # Rows sit (down-1) hops below the sibling link, so they carry the
        # niece_nephew role whenever up == 1, which is every registry use.
        return oriented_pairs_from_sparse(M, row_is_first=True, subtract=subtract)

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
        share_mother = (pg.mother_ids[p1] >= 0) & (pg.mother_ids[p1] == pg.mother_ids[p2])
        share_father = (pg.father_ids[p1] >= 0) & (pg.father_ids[p1] == pg.father_ids[p2])
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
        full sibling of U. In matrix form: A @ S_full (row C, column U), then
        exclude parent-child pairs (which share the same edge structure).
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

        return oriented_pairs_from_sparse(avunc, row_is_first=True)

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

    def _run_parallel(self, tasks: dict[str, _Thunk]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Run each per-code extraction thunk concurrently; return ``{code: pairs}``.

        numpy/scipy release the GIL for the heavy sparse products, so the
        per-degree codes overlap up to the worker cap.  Only the codes in
        *tasks* are computed; the caller pre-seeds every registry code to
        empty, so any code omitted here stays empty.
        """
        if not tasks:
            return {}
        workers = min(self._max_workers, len(tasks))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {code: pool.submit(fn) for code, fn in tasks.items()}
            return {code: fut.result() for code, fut in futures.items()}

    @staticmethod
    def _log_counts(
        label: str,
        pairs: dict[str, tuple[np.ndarray, np.ndarray]],
        codes: tuple[str, ...],
        t0: float,
    ) -> None:
        """Emit an INFO line summarising per-code pair counts and elapsed time."""
        summary = ", ".join(f"{code}={len(pairs[code][0])}" for code in codes)
        logger.info("%s: %s (%.3fs)", label, summary, time.perf_counter() - t0)

    def extract(self, codes: frozenset[str]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Compute the relationship pairs of every code in *codes*.

        Args:
            codes: Registry codes to compute.  Must equal its own
                :func:`dependency_closure`; the callers build it that way.

        Returns:
            ``{code: (first, second)}`` over every registry code, in graph
            rows and the category's semantic orientation (symmetric codes
            canonical ``first < second``), each block sorted by canonical
            unordered key.  Codes outside *codes* map to empty arrays.
        """
        assert codes == dependency_closure(codes), "extract() needs the dependency closure of its codes"
        pg = self.pg
        t_total = time.perf_counter()
        empty = np.array([], dtype=np.intp)
        # Every registry code appears in the output; codes that are not
        # computed stay empty.  Pre-seeding here makes that contract
        # structural and removes per-degree "fill in the missing codes"
        # bookkeeping along with its mirror-image ``else`` branches.
        pairs: dict[str, tuple[np.ndarray, np.ndarray]] = dict.fromkeys(RELATIONSHIPS, (empty, empty))

        def _needed(code: str) -> bool:
            return code in codes

        needs_degree1_plus = any(RELATIONSHIPS[code].degree >= 1 for code in codes)
        needs_degree2_plus = any(RELATIONSHIPS[code].degree >= 2 for code in codes)
        needs_degree3_plus = any(RELATIONSHIPS[code].degree >= 3 for code in codes)
        needs_degree4_plus = any(RELATIONSHIPS[code].degree >= 4 for code in codes)
        needs_degree5 = any(RELATIONSHIPS[code].degree >= 5 for code in codes)

        # Pre-trigger cached properties needed by degree-2+ extractions.
        # _Am/_Af are only needed to build _A; delete after to free memory.
        if needs_degree2_plus:
            pg._ensure_parent_csr()
            _ = pg._A2  # chains: _Am, _Af → _A → _A2
        pg.__dict__.pop("_Am", None)
        pg.__dict__.pop("_Af", None)

        if _needed("MZ"):
            pairs["MZ"] = pg._mz_twin_pairs()

        # Tuple defaults match sibling_pairs()'s (maternal, paternal) shape; only
        # ever read after the needs_degree1_plus branch repopulates them.
        full_sib, mat_hs, pat_hs = (empty, empty), (empty, empty), (empty, empty)
        if needs_degree1_plus:
            mo, fo = pg._parent_offspring_pairs()
            if _needed("MO"):
                pairs["MO"] = mo
            if _needed("FO"):
                pairs["FO"] = fo

            t0 = time.perf_counter()
            full_sib, mat_hs, pat_hs = pg._sibling_pairs()
            if _needed("FS"):
                pairs["FS"] = full_sib
            if _needed("MHS"):
                pairs["MHS"] = mat_hs
            if _needed("PHS"):
                pairs["PHS"] = pat_hs
            self._log_counts("Siblings", pairs, ("FS", "MHS", "PHS"), t0)

        # ---- Degree 2 (kinship 1/8): GP, Av ----
        if _needed("GP") or _needed("Av"):
            t0 = time.perf_counter()
            tasks: dict[str, _Thunk] = {}
            if _needed("GP"):
                tasks["GP"] = self._grandparent_grandchild_pairs
            if _needed("Av"):
                tasks["Av"] = partial(self._avuncular_pairs, full_sib)
            pairs.update(self._run_parallel(tasks))
            self._log_counts("Degree 2", pairs, ("GP", "Av"), t0)

        # ---- Degree 3+ setup (deferred to avoid work below the 1C/GGP cutoff) ----
        if needs_degree3_plus:
            po_pairs = (
                np.concatenate([pairs["MO"][0], pairs["FO"][0]]),
                np.concatenate([pairs["MO"][1], pairs["FO"][1]]),
            )
            gp_pairs = pairs["GP"]
            fsm = pg._full_sib_matrix
            pg._build_half_sib_matrix(mat_hs, pat_hs)
            hsm = pg._half_sib_matrix
        # sib_all only needed at degree 4+ (1C1R, H1C1R, 1C2R subtract lists)
        if needs_degree4_plus:
            sib_all = (
                np.concatenate([pairs["FS"][0], pairs["MHS"][0], pairs["PHS"][0]]),
                np.concatenate([pairs["FS"][1], pairs["MHS"][1], pairs["PHS"][1]]),
            )

        # ---- Degree 3 (kinship 1/16): GGP, HAv, GAv, 1C ----
        if needs_degree3_plus:
            t0 = time.perf_counter()
            _ = pg._A3  # pre-trigger
            tasks = {}
            if _needed("GGP"):
                tasks["GGP"] = partial(self._lineal_pairs, 3)
            if _needed("HAv"):
                tasks["HAv"] = partial(self._collateral_pairs, hsm, 1, 2, [po_pairs, gp_pairs])
            if _needed("GAv"):
                tasks["GAv"] = partial(self._collateral_pairs, fsm, 1, 3, [po_pairs, gp_pairs, pairs["Av"]])
            if _needed("1C"):
                tasks["1C"] = self._cousin_pairs
            pairs.update(self._run_parallel(tasks))
            self._log_counts("Degree 3", pairs, ("GGP", "HAv", "GAv", "1C"), t0)

        # ---- Degree 4 (kinship 1/32): GGGP, HGAv, GGAv, H1C, 1C1R ----
        # A2_A3T is built lazily by 1C1R (here) and/or H1C1R (degree 5); seed
        # it before the degree-4 gate so the degree-5 block can reuse it.
        A2_A3T = None
        if needs_degree4_plus:
            t0 = time.perf_counter()
            if _needed("1C1R"):
                A2_A3T = pg._A2 @ pg._A3.T

            def _extract_h1c() -> tuple[np.ndarray, np.ndarray]:
                # Half-1C pairs cached by _cousin_pairs(): share exactly one
                # grandparent, with sibling pairs already excluded.
                return self._h1c_pairs_cache

            def _extract_1c1r() -> tuple[np.ndarray, np.ndarray]:
                assert A2_A3T is not None  # set above under the same _needed("1C1R") guard
                P_full = A2_A3T.copy()
                P_full.setdiag(0)
                P_full.data[P_full.data < 2] = 0
                P_full.eliminate_zeros()
                # Rows have the shared ancestor two meioses up, columns three:
                # the column is the junior cousin.
                return oriented_pairs_from_sparse(
                    P_full,
                    row_is_first=False,
                    subtract=[po_pairs, gp_pairs, pairs["GGP"], pairs["Av"], pairs["GAv"], sib_all, pairs["1C"]],
                )

            tasks = {}
            if _needed("GGGP"):
                tasks["GGGP"] = partial(self._lineal_pairs, 4)
            if _needed("HGAv"):
                tasks["HGAv"] = partial(
                    self._collateral_pairs,
                    hsm,
                    1,
                    3,
                    [po_pairs, gp_pairs, pairs["GGP"], pairs["HAv"]],
                )
            if _needed("GGAv"):
                tasks["GGAv"] = partial(
                    self._collateral_pairs,
                    fsm,
                    1,
                    4,
                    [po_pairs, gp_pairs, pairs["GGP"], pairs["Av"], pairs["GAv"]],
                )
            if _needed("H1C"):
                tasks["H1C"] = _extract_h1c
            if _needed("1C1R"):
                tasks["1C1R"] = _extract_1c1r
            pairs.update(self._run_parallel(tasks))
            self._log_counts("Degree 4", pairs, ("GGGP", "HGAv", "GGAv", "H1C", "1C1R"), t0)

        # ---- Degree 5 (kinship 1/64): 2C, G3GP, HGGAv, G3Av, H1C1R, 1C2R ----
        if needs_degree5:
            t0 = time.perf_counter()
            # _A5 triggered lazily by G3GP (_lineal_pairs(5))
            # A2_A3T needed by H1C1R only
            if _needed("H1C1R") and A2_A3T is None:
                A2_A3T = pg._A2 @ pg._A3.T

            def _extract_h1c1r() -> tuple[np.ndarray, np.ndarray]:
                assert A2_A3T is not None  # set above when _needed("H1C1R")
                P_half = A2_A3T.copy()
                P_half.setdiag(0)
                P_half.data[P_half.data != 1] = 0
                P_half.eliminate_zeros()
                return oriented_pairs_from_sparse(
                    P_half,
                    row_is_first=False,
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
                return oriented_pairs_from_sparse(
                    P_full,
                    row_is_first=False,
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

            tasks = {}
            if _needed("2C"):
                tasks["2C"] = self._second_cousin_pairs
            if _needed("G3GP"):
                tasks["G3GP"] = partial(self._lineal_pairs, 5)
            if _needed("HGGAv"):
                tasks["HGGAv"] = partial(
                    self._collateral_pairs,
                    hsm,
                    1,
                    4,
                    [po_pairs, gp_pairs, pairs["GGP"], pairs["GGGP"], pairs["HAv"], pairs["HGAv"]],
                )
            if _needed("G3Av"):
                tasks["G3Av"] = partial(
                    self._collateral_pairs,
                    fsm,
                    1,
                    5,
                    [po_pairs, gp_pairs, pairs["GGP"], pairs["GGGP"], pairs["Av"], pairs["GAv"], pairs["GGAv"]],
                )
            if _needed("H1C1R"):
                tasks["H1C1R"] = _extract_h1c1r
            if _needed("1C2R"):
                tasks["1C2R"] = _extract_1c2r
            pairs.update(self._run_parallel(tasks))
            self._log_counts("Degree 5", pairs, ("2C", "G3GP", "HGGAv", "G3Av", "H1C1R", "1C2R"), t0)

        n = pg.n_individuals
        for code, (first, second) in pairs.items():
            pairs[code] = sort_by_canonical_key(first, second, n)
        logger.info("pair extraction total: %.3fs", time.perf_counter() - t_total)
        return pairs


# ----------------------------------------------------------------------
# Public assembly: PedigreeGraph.relationship_pairs and PedigreeView.relationship_pairs
# ----------------------------------------------------------------------

_DEBUG_EXCLUSIVITY_ENV = "PEDIGREE_GRAPH_DEBUG_EXCLUSIVITY"

_PairArrays = tuple[np.ndarray, np.ndarray]


def _requested_codes(max_degree: int | None, categories: Iterable[str] | None) -> frozenset[str]:
    """Validate the one-of-two selector and return the codes it names.

    Raises:
        TypeError: Both selectors, neither, a bare ``str`` for *categories*,
            or a non-``str`` code.
        PedigreeValidationError: ``max_degree_out_of_range`` or
            ``unknown_relationship_category``.
    """
    if (max_degree is None) == (categories is None):
        raise TypeError("relationship_pairs() takes exactly one of max_degree= or categories=")
    if max_degree is not None:
        selected = categories_up_to_degree(max_degree)
    else:
        if isinstance(categories, str):
            raise TypeError("categories must be an iterable of codes, not a single str")
        assert categories is not None
        selected = select_categories(categories)
    return frozenset(category.code for category in selected)


def _classify(graph: PedigreeGraph, requested: frozenset[str]) -> dict[str, _PairArrays]:
    """Return the closest-category graph-row pairs of every code *requested* depends on."""
    computed = dependency_closure(requested)
    pairs = MatrixPairExtractor(graph, max_workers=thread_budget()).extract(computed)
    graph._release_pair_matrices()
    _fold_precedence(pairs, [code for code in RELATIONSHIPS if code in computed], graph.n_individuals)
    return pairs


def _build_result(
    pairs: dict[str, _PairArrays], requested: frozenset[str], token: CoordinateToken
) -> RelationshipPairs:
    """Wrap the requested codes of *pairs* as owned int32 blocks carrying *token*."""
    empty = np.array([], dtype=np.int32)
    blocks = {}
    for code, category in RELATIONSHIPS.items():
        first, second = pairs[code] if code in requested else (empty, empty)
        blocks[code] = RelationshipPairBlock(
            category,
            _own(first, np.int32),
            _own(second, np.int32),
            code in requested,
            token,
        )
    result = RelationshipPairs(blocks)
    if os.environ.get(_DEBUG_EXCLUSIVITY_ENV) == "1":
        check_exclusive(result)
    return result


def relationship_pairs(
    graph: PedigreeGraph,
    *,
    max_degree: int | None = None,
    categories: Iterable[str] | None = None,
) -> RelationshipPairs:
    """Build the :class:`RelationshipPairs` of *graph* for one selector.

    Args:
        graph: The receiver; results are in its graph rows.
        max_degree: Select every category at or below this degree.
        categories: Select these registry codes.

    Returns:
        All 23 blocks; the unselected ones are empty and unrequested.

    Raises:
        TypeError: Both selectors, neither, a bare ``str`` for *categories*,
            or a non-``str`` code.
        PedigreeValidationError: ``max_degree_out_of_range`` or
            ``unknown_relationship_category``.
    """
    requested = _requested_codes(max_degree, categories)
    return _build_result(_classify(graph, requested), requested, graph._coordinate_token)


def view_relationship_pairs(
    view: PedigreeView,
    *,
    max_degree: int | None = None,
    categories: Iterable[str] | None = None,
) -> RelationshipPairs:
    """Build the :class:`RelationshipPairs` of *view* for one selector.

    Classification runs over the full graph; a pair is kept when both
    endpoints are selected and is relabelled into view rows.  Asymmetric
    blocks keep the graph-space role orientation, symmetric blocks are
    re-canonicalised to ``first < second`` in view rows, and every block is
    re-sorted by the canonical view-row key (ADR 0006 pair contract 6).

    Args:
        view: The receiver; results are in its view rows.
        max_degree: Select every category at or below this degree.
        categories: Select these registry codes.

    Returns:
        All 23 blocks carrying the view's token; the unselected ones are
        empty and unrequested.

    Raises:
        TypeError: As :func:`relationship_pairs`.
        PedigreeValidationError: As :func:`relationship_pairs`.
    """
    requested = _requested_codes(max_degree, categories)
    n = len(view)
    empty = np.array([], dtype=np.intp)
    pairs: dict[str, _PairArrays] = dict.fromkeys(requested, (empty, empty))
    if n >= 2:
        graph_pairs = _classify(view._graph, requested)
        graph_to_view = view._graph_to_view()
        for code in requested:
            first, second = project_pairs(*graph_pairs[code], graph_to_view)
            if RELATIONSHIPS[code].symmetric:
                first, second = np.minimum(first, second), np.maximum(first, second)
            pairs[code] = sort_by_canonical_key(first, second, n)
    return _build_result(pairs, requested, view._coordinate_token)


def _fold_precedence(pairs: dict[str, tuple[np.ndarray, np.ndarray]], order: list[str], n: int) -> None:
    """Drop every pair already claimed by an earlier code of *order*, in place.

    Pairs are keyed by canonical unordered key; ``lexsort((rank, key))`` puts
    each key's lowest-ranked occurrence first, and masking a block keeps its
    canonical-key order intact.
    """
    parts = [(code, *pairs[code]) for code in order if len(pairs[code][0]) > 0]
    if not parts:
        return
    keys = np.concatenate([canonical_keys(first, second, n) for _, first, second in parts])
    rank = np.repeat(np.arange(len(parts)), [len(first) for _, first, _ in parts])
    by_rank_within_key = np.lexsort((rank, keys))
    sorted_keys = keys[by_rank_within_key]
    leads = np.ones(sorted_keys.size, dtype=bool)
    leads[1:] = sorted_keys[1:] != sorted_keys[:-1]
    keep = np.zeros(keys.size, dtype=bool)
    keep[by_rank_within_key[leads]] = True
    start = 0
    for code, first, second in parts:
        stop = start + len(first)
        mask = keep[start:stop]
        if not mask.all():
            pairs[code] = (first[mask], second[mask])
        start = stop


def check_exclusive(pairs: RelationshipPairs) -> None:
    """Assert the ADR 0006 pair invariants on *pairs*.

    Every block is sorted by canonical unordered key with no repeated pair
    (so no asymmetric block holds both orientations), and no unordered pair
    appears in two blocks.

    Args:
        pairs: The result to check.

    Raises:
        AssertionError: Naming the offending pair and its block(s).
    """
    n = 1 + max(
        (int(max(block.first_rows.max(), block.second_rows.max())) for block in pairs.values() if len(block)), default=0
    )
    seen: dict[int, str] = {}
    for code, block in pairs.items():
        keys = canonical_keys(block.first_rows, block.second_rows, n)
        assert np.all(keys[1:] > keys[:-1]), f"{code}: block is not strictly sorted by canonical key"
        for key, first, second in zip(
            keys.tolist(), block.first_rows.tolist(), block.second_rows.tolist(), strict=True
        ):
            other = seen.setdefault(key, code)
            assert other == code, f"pair ({first}, {second}) appears in both {other} and {code}"
