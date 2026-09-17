"""The retired SciPy matrix pair extractor, kept as the differential oracle for the Rust engine.

Test code only (ADR 0007): ``pedigree_graph`` never imports this, and it is
not a fallback.  It is the 0.8 production engine moved here verbatim when
slice 12 put ``relationship_pairs`` on the row-streaming Rust engine, so it
reaches the frozen ADR 0006 pair contract through a different algorithm:
global sparse products, path-multiplicity thresholds, per-category
subtraction lists, then a whole-block precedence fold.  :class:`_Matrices`
stands in for the graph attributes the extractor used to read.

Use :func:`oracle_pairs` and :func:`oracle_view_pairs` for the folded,
sorted blocks of a selector, :func:`check_exclusive` for the block
invariants, and :func:`sibling_pairs` for the three sibling categories on
their own.  The array helpers (:func:`canonical_keys`, :func:`subtract_pairs`,
:func:`oriented_pairs_from_sparse`, :func:`pairs_from_groups`,
:func:`project_pairs`) are the former ``_pair_utils``.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property, partial
from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sp

from pedigree_graph._registry import RELATIONSHIPS
from pedigree_graph._selection import RelationshipSelection

if TYPE_CHECKING:
    from collections.abc import Iterable

    from pedigree_graph import PedigreeGraph, PedigreeView, RelationshipPairs

logger = logging.getLogger(__name__)

_PairArrays = tuple[np.ndarray, np.ndarray]

# A per-code extraction thunk: returns one relationship's (first, second) arrays.
_Thunk = Callable[[], tuple[np.ndarray, np.ndarray]]


def canonical_keys(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    """Return the canonical unordered int64 key ``min(a, b) * n + max(a, b)`` per pair.

    Args:
        a: First member of each pair, any orientation.
        b: Second member of each pair, any orientation.
        n: Key base, at least ``max(a, b) + 1``; ``n`` is bounded by the int32
            row capacity so the product fits int64.

    Returns:
        One int64 key per pair, equal for the two orientations of a pair.
    """
    return np.minimum(a, b).astype(np.int64) * n + np.maximum(a, b).astype(np.int64)


def sort_by_canonical_key(a: np.ndarray, b: np.ndarray, n: int) -> _PairArrays:
    """Return ``(a, b)`` reordered by :func:`canonical_keys`, orientation kept."""
    if len(a) == 0:
        return a, b
    order = np.argsort(canonical_keys(a, b, n), kind="stable")
    return a[order], b[order]


def subtract_pairs(keep: _PairArrays, remove: list[_PairArrays]) -> _PairArrays:
    """Drop from *keep* every unordered pair that occurs in any of *remove*.

    Membership is decided on the canonical unordered key ``min * m + max``, so
    the inputs may be in any orientation and *keep* comes back in the
    orientation it arrived in (ADR 0006 pair contract 3).

    Args:
        keep: ``(a, b)`` candidate pair arrays.
        remove: Pair arrays whose unordered pairs are dropped from *keep*.

    Returns:
        The surviving ``(a, b)`` rows of *keep*, order preserved.
    """
    a, b = keep
    parts = [pair for pair in remove if len(pair[0]) > 0]
    if len(a) == 0 or not parts:
        return keep
    rm_a = np.concatenate([pair[0] for pair in parts])
    rm_b = np.concatenate([pair[1] for pair in parts])
    m = int(max(a.max(), b.max(), rm_a.max(), rm_b.max())) + 1
    # Membership only: sorting the raw remove keys plus searchsorted beats
    # np.unique + np.isin at scale, and duplicate remove keys are harmless.
    rm_keys = np.sort(canonical_keys(rm_a, rm_b, m))
    keys = canonical_keys(a, b, m)
    pos = np.searchsorted(rm_keys, keys)
    hit = pos < rm_keys.size
    hit[hit] = rm_keys[pos[hit]] == keys[hit]
    return a[~hit], b[~hit]


def oriented_pairs_from_sparse(
    M: sp.spmatrix,
    *,
    row_is_first: bool,
    subtract: list[_PairArrays] | None = None,
) -> _PairArrays:
    """Read an asymmetric relationship product as oriented ``(first, second)`` pairs.

    Each nonzero ``M[r, c]`` is one pair; *row_is_first* says which side of
    the product holds the ``first`` role.  A pair valid in both orientations
    (both ``M[a, b]`` and ``M[b, a]`` nonzero, through different paths of an
    inbred pedigree) is kept once with the lower row as ``first`` (ADR 0006
    pair contract 5).  Mutates *M* in place (zeroes the diagonal).

    Args:
        M: Square sparse product whose nonzeros are the candidate pairs.
        row_is_first: ``True`` when the row index carries the ``first`` role.
        subtract: Closer-category pairs to drop, in any orientation.

    Returns:
        Oriented intp ``(first, second)`` arrays, one entry per unordered
        pair, sorted by canonical key.
    """
    M.setdiag(0)  # ty: ignore[unresolved-attribute]
    M.eliminate_zeros()  # ty: ignore[unresolved-attribute]
    if M.nnz == 0:  # ty: ignore[unresolved-attribute]
        return np.array([], dtype=np.intp), np.array([], dtype=np.intp)
    rows, cols = M.nonzero()  # ty: ignore[unresolved-attribute]
    first, second = (rows, cols) if row_is_first else (cols, rows)
    first = first.astype(np.intp)
    second = second.astype(np.intp)
    keys = canonical_keys(first, second, M.shape[0])
    order = np.lexsort((first, keys))
    sorted_keys = keys[order]
    unique = np.ones(order.size, dtype=bool)
    unique[1:] = sorted_keys[1:] != sorted_keys[:-1]
    kept = order[unique]
    first, second = first[kept], second[kept]
    if subtract:
        first, second = subtract_pairs((first, second), subtract)
    return first, second


def pairs_from_groups(indices: np.ndarray, group_key: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Generate all (i < j) pairs of indices within each group.

    Uses batch-by-size triu_indices for vectorized pair generation.
    """
    if len(indices) == 0:
        return np.array([], dtype=np.intp), np.array([], dtype=np.intp)

    sort_idx = np.argsort(group_key, kind="mergesort")
    sorted_keys = group_key[sort_idx]
    sorted_indices = indices[sort_idx]

    # sorted_keys is already sorted; diff-based run detection avoids
    # np.unique re-sorting/hashing the array it was just handed.
    starts = np.concatenate(([0], np.flatnonzero(sorted_keys[1:] != sorted_keys[:-1]) + 1))
    counts = np.diff(np.append(starts, len(sorted_keys)))

    multi = counts >= 2
    starts = starts[multi]
    counts = counts[multi]

    if len(starts) == 0:
        return np.array([], dtype=np.intp), np.array([], dtype=np.intp)

    pair_i_parts = []
    pair_j_parts = []
    for size in np.unique(counts):
        gs = starts[counts == size]
        ii, jj = np.triu_indices(size, k=1)
        all_i = (gs[:, np.newaxis] + ii[np.newaxis, :]).ravel()
        all_j = (gs[:, np.newaxis] + jj[np.newaxis, :]).ravel()
        pair_i_parts.append(sorted_indices[all_i])
        pair_j_parts.append(sorted_indices[all_j])

    p1 = np.concatenate(pair_i_parts)
    p2 = np.concatenate(pair_j_parts)

    lo = np.minimum(p1, p2)
    hi = np.maximum(p1, p2)
    return lo.astype(np.intp), hi.astype(np.intp)


def project_pairs(first: np.ndarray, second: np.ndarray, graph_to_view: np.ndarray) -> _PairArrays:
    """Keep the pairs with both members selected, relabelled into view rows.

    Args:
        first: Graph rows of the first member of each pair.
        second: Graph rows of the second member of each pair.
        graph_to_view: View row of each graph row, ``-1`` where unselected.

    Returns:
        Intp ``(first, second)`` view rows of the retained pairs, orientation
        and order preserved.
    """
    view_first = graph_to_view[first]
    view_second = graph_to_view[second]
    keep = (view_first >= 0) & (view_second >= 0)
    return view_first[keep].astype(np.intp), view_second[keep].astype(np.intp)


class _Matrices:
    """The adjacency powers and sibling matrices of one graph, as the extractor reads them.

    Built from the graph's public columns; nothing here touches the graph's
    caches.  The methods are the former ``PedigreeGraph`` privates.
    """

    def __init__(self, graph: PedigreeGraph) -> None:
        self.n_individuals = graph.n_individuals
        self.mother_rows = graph.mother_rows
        self.father_rows = graph.father_rows
        self.twin_rows = graph.twin_rows
        self.mother_ids = graph.mother_ids
        self.father_ids = graph.father_ids

    @cached_property
    def _A(self):
        """Child → both parents adjacency matrix, from one pass over both edge lists.

        Assembled as a single COO rather than as a CSR per parent that are then
        summed.  The two agree entry for entry: ``check_same_parent``
        (``crates/core/src/graph.rs:244``) rejects a row naming one id in both
        roles, so no ``(child, parent)`` pair can appear twice and every stored
        value is ``1``.  Building the halves eagerly cost every graph two
        matrices no other production reader consumed (issue #18).

        Every edge is used, so a partial pedigree still contributes whichever
        side it knows.
        """
        t0 = time.perf_counter()
        n = self.n_individuals
        has_mother = self.mother_rows >= 0
        has_father = self.father_rows >= 0
        # Not ``np.where``, whose index is ``intp``: scipy widens the whole COO
        # to its widest input, which cost more transient memory than the eager
        # two-matrix build this replaces.
        rows = np.arange(n, dtype=self.mother_rows.dtype)
        children = np.concatenate((rows[has_mother], rows[has_father]))
        parents = np.concatenate((self.mother_rows[has_mother], self.father_rows[has_father]))
        result = sp.csr_matrix(
            (np.ones(len(children), dtype=np.int32), (children, parents)),
            shape=(n, n),
        )
        logger.debug("_A built from %d parent edges in %.3fs", len(children), time.perf_counter() - t0)
        return result

    @cached_property
    def _A2(self):
        """2-hop parent reach (grandparents): A @ A."""
        t0 = time.perf_counter()
        result = self._A @ self._A
        logger.debug("_A2 = A @ A computed in %.3fs (nnz=%d)", time.perf_counter() - t0, result.nnz)
        return result

    @cached_property
    def _A2_shared(self):
        """Shared-grandparent matrix: A² @ (A²).T.

        Only needed when 2nd cousin extraction is enabled.
        """
        t0 = time.perf_counter()
        result = self._A2 @ self._A2.T
        logger.debug("_A2_shared = A2 @ A2.T computed in %.3fs (nnz=%d)", time.perf_counter() - t0, result.nnz)
        return result

    @cached_property
    def _A3(self):
        """3-hop parent reach (great-grandparents): A² @ A."""
        t0 = time.perf_counter()
        result = self._A2 @ self._A
        logger.debug("_A3 = A2 @ A computed in %.3fs (nnz=%d)", time.perf_counter() - t0, result.nnz)
        return result

    @cached_property
    def _A4(self):
        """4-hop parent reach (great²-grandparents): A³ @ A."""
        t0 = time.perf_counter()
        result = self._A3 @ self._A
        logger.debug("_A4 = A3 @ A computed in %.3fs (nnz=%d)", time.perf_counter() - t0, result.nnz)
        return result

    @cached_property
    def _A5(self):
        """5-hop parent reach (great³-grandparents): A⁴ @ A."""
        t0 = time.perf_counter()
        result = self._A4 @ self._A
        logger.debug("_A5 = A4 @ A computed in %.3fs (nnz=%d)", time.perf_counter() - t0, result.nnz)
        return result

    def _get_Ak(self, k: int) -> sp.spmatrix:
        """Return the k-hop parent-reach matrix (k=0 returns identity)."""
        if k == 0:
            return sp.eye(self.n_individuals, format="csr")
        if k == 1:
            return self._A
        return getattr(self, f"_A{k}")

    def _ensure_sibling_matrices(self) -> None:
        """Ensure _full_sib_matrix and _half_sib_matrix are computed."""
        if hasattr(self, "_full_sib_matrix"):
            return
        # Trigger sibling extraction which sets _full_sib_matrix
        self._sibling_pairs()

    def _build_half_sib_matrix(
        self,
        mat_hs: tuple[np.ndarray, np.ndarray],
        pat_hs: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Build and cache _half_sib_matrix from extracted half-sib pairs."""
        hs1 = np.concatenate([mat_hs[0], pat_hs[0]])
        hs2 = np.concatenate([mat_hs[1], pat_hs[1]])
        if len(hs1) > 0:
            ones = np.ones(len(hs1), dtype=np.int32)
            H = sp.csr_matrix((ones, (hs1, hs2)), shape=(self.n_individuals, self.n_individuals))
            self._half_sib_matrix = H + H.T
        else:
            self._half_sib_matrix = sp.csr_matrix((self.n_individuals, self.n_individuals))

    # ------------------------------------------------------------------
    # Relationship extraction
    # ------------------------------------------------------------------

    def _mz_twin_pairs(self) -> tuple[np.ndarray, np.ndarray]:
        """MZ twin pairs: twin != -1, deduplicated with id < twin_id."""
        has_twin = self.twin_rows >= 0
        ids = np.where(has_twin)[0]
        partners = self.twin_rows[has_twin]
        mask = ids < partners
        return ids[mask], partners[mask].astype(np.intp)

    def _parent_offspring_pairs(
        self,
    ) -> tuple[
        tuple[np.ndarray, np.ndarray],
        tuple[np.ndarray, np.ndarray],
    ]:
        """Mother-offspring and Father-offspring pairs.

        Each parent link is reported independently, so a child with only
        one parent in the sample still contributes a PO pair.  Graph-data
        accessor read by the matrix pair extractor.
        """
        m_mask = self.mother_rows >= 0
        m_children = np.where(m_mask)[0]

        f_mask = self.father_rows >= 0
        f_children = np.where(f_mask)[0]

        return (m_children, self.mother_rows[m_children].astype(np.intp)), (
            f_children,
            self.father_rows[f_children].astype(np.intp),
        )

    def _sibling_pairs(
        self,
    ) -> tuple[
        tuple[np.ndarray, np.ndarray],
        tuple[np.ndarray, np.ndarray],
        tuple[np.ndarray, np.ndarray],
    ]:
        """Full sib, maternal half sib, and paternal half sib pairs.

        Uses numpy sort+group for direct enumeration — faster than sparse
        matmul for 1-hop relationships since it avoids materializing N×N
        shared-parent matrices.

        Groups by ORIGINAL pedigree parent IDs (not remapped row indices)
        so that siblings are correctly detected even when parents are absent
        from a subsampled dataset.

        Individuals with only one known parent can participate in half-sib
        detection through that parent (but not full-sib detection, which
        requires both parents known).

        Twin individuals are excluded entirely (matching legacy semantics).
        Returns (full_sib, maternal_hs, paternal_hs) tuples of (idx1, idx2).
        """
        empty = np.array([], dtype=np.intp), np.array([], dtype=np.intp)

        # Non-twin individuals with at least one known parent
        has_parent = (self.mother_ids >= 0) | (self.father_ids >= 0)
        nt_mask = has_parent & (self.twin_rows < 0)
        nt_idx = np.where(nt_mask)[0]

        if len(nt_idx) < 2:
            self._full_sib_matrix = sp.csr_matrix((self.n_individuals, self.n_individuals))
            self._half_sib_matrix = sp.csr_matrix((self.n_individuals, self.n_individuals))
            return empty, empty, empty

        nt_mother = self.mother_ids[nt_idx]
        nt_father = self.father_ids[nt_idx]

        # --- Full sibs: same KNOWN mother AND same KNOWN father ---
        both_known = (nt_mother >= 0) & (nt_father >= 0)
        bk_idx = nt_idx[both_known]
        bk_mother = nt_mother[both_known]
        bk_father = nt_father[both_known]

        if len(bk_idx) >= 2:
            max_parent = max(int(bk_mother.max()), int(bk_father.max())) + 1
            # int64 cast required: max_id² overflows int32
            family_key = bk_mother.astype(np.int64) * max_parent + bk_father.astype(np.int64)
            full_sib = pairs_from_groups(bk_idx, family_key)
        else:
            full_sib = empty

        # --- Maternal half sibs: all pairs sharing known mother, minus full-sib pairs ---
        has_mother = nt_mother >= 0
        m_idx = nt_idx[has_mother]
        m_mother = nt_mother[has_mother]
        if len(m_idx) >= 2:
            mat_all = pairs_from_groups(m_idx, m_mother)
            mat_hs = subtract_pairs(mat_all, [full_sib])
        else:
            mat_hs = empty

        # --- Paternal half sibs: all pairs sharing known father, minus full-sib pairs ---
        has_father = nt_father >= 0
        f_idx = nt_idx[has_father]
        f_father = nt_father[has_father]
        if len(f_idx) >= 2:
            pat_all = pairs_from_groups(f_idx, f_father)
            pat_hs = subtract_pairs(pat_all, [full_sib])
        else:
            pat_hs = empty

        # Build full-sib sparse matrix for _avuncular_pairs and collateral methods
        sib1, sib2 = full_sib
        if len(sib1) > 0:
            ones = np.ones(len(sib1), dtype=np.int32)
            F = sp.csr_matrix((ones, (sib1, sib2)), shape=(self.n_individuals, self.n_individuals))
            self._full_sib_matrix = F + F.T
        else:
            self._full_sib_matrix = sp.csr_matrix((self.n_individuals, self.n_individuals))

        return full_sib, mat_hs, pat_hs


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
        needs_degree3_plus = any(RELATIONSHIPS[code].degree >= 3 for code in codes)
        needs_degree4_plus = any(RELATIONSHIPS[code].degree >= 4 for code in codes)
        needs_degree5 = any(RELATIONSHIPS[code].degree >= 5 for code in codes)

        # GP is the first code in the dependency closure that consumes _A2.
        # Keep its build timing unchanged without charging MHS/PHS-only queries.
        if _needed("GP"):
            _ = pg._A2

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


def sibling_pairs(graph: PedigreeGraph) -> tuple[_PairArrays, _PairArrays, _PairArrays]:
    """``(full_sib, maternal_hs, paternal_hs)`` graph-row pairs, ``lo < hi``, unfolded."""
    return _Matrices(graph)._sibling_pairs()


def _classify(graph: PedigreeGraph, requested: frozenset[str]) -> dict[str, _PairArrays]:
    """The closest-category graph-row pairs of every code *requested* depends on."""
    computed = dependency_closure(requested)
    pairs = MatrixPairExtractor(_Matrices(graph), max_workers=1).extract(computed)
    _fold_precedence(pairs, [code for code in RELATIONSHIPS if code in computed], graph.n_individuals)
    return pairs


def oracle_pairs(
    graph: PedigreeGraph, *, max_degree: int | None = None, categories: Iterable[str] | None = None
) -> dict[str, _PairArrays]:
    """``{code: (first, second)}`` intp graph rows for the selector, every code present, unrequested empty."""
    selection = RelationshipSelection.parse(max_degree, categories)
    pairs = _classify(graph, selection.codes)
    empty = np.array([], dtype=np.intp)
    return {code: pairs[code] if code in selection.codes else (empty, empty) for code in RELATIONSHIPS}


def oracle_view_pairs(
    view: PedigreeView, *, max_degree: int | None = None, categories: Iterable[str] | None = None
) -> dict[str, _PairArrays]:
    """As :func:`oracle_pairs`, in view rows: both endpoints selected, symmetric re-canonicalised, view-key sorted."""
    selection = RelationshipSelection.parse(max_degree, categories)
    n = len(view)
    empty = np.array([], dtype=np.intp)
    pairs: dict[str, _PairArrays] = dict.fromkeys(RELATIONSHIPS, (empty, empty))
    if n < 2:
        return pairs
    graph_pairs = _classify(view._graph, selection.codes)
    graph_to_view = view._graph_to_view()
    for code in selection.codes:
        first, second = project_pairs(*graph_pairs[code], graph_to_view)
        if RELATIONSHIPS[code].symmetric:
            first, second = np.minimum(first, second), np.maximum(first, second)
        pairs[code] = sort_by_canonical_key(first, second, n)
    return pairs
