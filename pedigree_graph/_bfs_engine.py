"""Experimental BFS relationship-pair counting engine (PGQ-009).

Implementation of :func:`count_pairs_bfs`, the BFS / boolean-matmul /
numba relationship pair counter.  This is the engine module; the public
*experimental* surface is :mod:`pedigree_graph.experimental`, which
re-exports :func:`count_pairs_bfs` and carries the breaking-API contract.

The engine is intentionally decoupled from :mod:`pedigree_graph._core`:
it takes a :class:`PedigreeGraph` as a parameter (typed under
``TYPE_CHECKING``), reads only the shared read-only collaborators the
matrix engine also uses (``pg._mz_twin_pairs`` / ``pg._parent_offspring_pairs``
/ ``pg.sibling_pairs`` / the static ``pg._subtract_pairs``; ADR 0002),
and sources the relationship code set from :mod:`pedigree_graph._registry`.

Counts-only API.  Differs from the matrix engine
(:meth:`PedigreeGraph.count_pairs`) on inbred pedigrees: BFS counts
*distinct* shared ancestors at depth ≥ 2 while the matrix engine counts
*paths* (multiplicity).  Identical on non-inbred pedigrees.  The codes
that may diverge are exactly :func:`pedigree_graph._registry.bfs_divergent_codes`.
"""

from __future__ import annotations

__all__ = ["count_pairs_bfs"]

import logging
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import numba
import numpy as np
import scipy.sparse as sp

from pedigree_graph._bfs_kernel import _enumerate_pairs_kernel
from pedigree_graph._pair_utils import dedup_pairs
from pedigree_graph._registry import REL_REGISTRY

if TYPE_CHECKING:
    from collections.abc import Callable

    from pedigree_graph._core import PedigreeGraph

logger = logging.getLogger(__name__)

_BFS_DEFAULT_THREADS = 8


def _lineal_pairs_lo_hi(Pk: sp.csr_matrix) -> tuple[np.ndarray, np.ndarray]:
    """Lineal pairs at the depth encoded in ``Pk`` as ``(lo, hi)``."""
    rows, cols = Pk.nonzero()
    if rows.size == 0:
        return np.array([], dtype=np.intp), np.array([], dtype=np.intp)
    lo = np.minimum(rows, cols).astype(np.intp)
    hi = np.maximum(rows, cols).astype(np.intp)
    return lo, hi


def _run_per_anchor(
    worker: Callable[[int, int], tuple[list[np.ndarray], list[np.ndarray]]],
    n: int,
    n_threads: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Partition ``range(n)`` across ``n_threads`` workers and concatenate parts.

    ``worker(start, stop)`` returns ``(i_parts, j_parts)``. Numpy ops release
    the GIL, so threading gives near-linear speedup for the per-anchor loops
    in :func:`count_pairs_bfs`.
    """
    if n == 0:
        return [], []
    n_threads = max(1, min(n_threads, n))
    if n_threads == 1:
        return worker(0, n)
    chunk = (n + n_threads - 1) // n_threads
    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        futures = [pool.submit(worker, k * chunk, min((k + 1) * chunk, n)) for k in range(n_threads)]
        all_i: list[np.ndarray] = []
        all_j: list[np.ndarray] = []
        for f in futures:
            i_p, j_p = f.result()
            all_i.extend(i_p)
            all_j.extend(j_p)
    return all_i, all_j


def _count_after_subtract(
    lo: np.ndarray,
    hi: np.ndarray,
    subtracts: list[tuple[np.ndarray, np.ndarray]],
) -> int:
    """Count canonical (lo, hi) pairs not present in any of *subtracts*.

    Computes a unified ``max_id`` across the candidate set and all subtract
    sets up front, so all int64 keys share a single base. Avoids the
    rebuild-on-overflow path that the original pedsum implementation had as
    latent dead code (a list-comprehension that silently dropped accumulated
    rm_parts on a base bump).
    """
    if lo.size == 0:
        return 0
    max_idx = int(hi.max())
    for r_lo, r_hi in subtracts:
        if r_lo.size == 0:
            continue
        max_idx = max(max_idx, int(r_hi.max()), int(r_lo.max()))
    max_id = max_idx + 1
    keys = lo.astype(np.int64) * max_id + hi.astype(np.int64)
    rm_parts: list[np.ndarray] = []
    for r_lo, r_hi in subtracts:
        if r_lo.size == 0:
            continue
        r_lo_s = np.minimum(r_lo, r_hi).astype(np.int64)
        r_hi_s = np.maximum(r_lo, r_hi).astype(np.int64)
        rm_parts.append(r_lo_s * max_id + r_hi_s)
    if rm_parts:
        rm_keys = np.concatenate(rm_parts)
        keep = ~np.isin(keys, rm_keys)
        keys = keys[keep]
    return int(keys.size)


def count_pairs_bfs(
    pg: PedigreeGraph,
    *,
    max_degree: int = 5,
    n_threads: int | None = None,
) -> dict[str, int]:
    """BFS / boolean-matmul / numba relationship pair counter (experimental).

    Returns counts for the 23 named relationship codes (``MZ`` + 22 path codes
    through degree 5). Does NOT return ``PO`` or ``by_degree`` aggregates —
    callers compose those themselves.

    On non-inbred pedigrees the counts match
    :meth:`PedigreeGraph.count_pairs` exactly. On inbred pedigrees BFS counts
    *distinct* shared ancestors at depth ≥ 2 while the matrix engine counts
    *paths* (multiplicity); the codes that may diverge are exactly those in
    :func:`pedigree_graph._registry.bfs_divergent_codes`
    (``1C1R``, ``H1C1R``, ``1C2R``, ``2C``).

    Args:
        pg: A :class:`PedigreeGraph` built directly (not via
            :meth:`PedigreeGraph.from_subsample`). Subsample support is not
            implemented for BFS.
        max_degree: Must be 5. Lower values raise :class:`NotImplementedError`
            — use :meth:`PedigreeGraph.count_pairs` for partial extractions.
        n_threads: Optional override for numba's prange thread count. Numba
            caches the thread count at first JIT compilation, so this only
            takes effect on the first call in a process. To control threading
            on all calls, set ``NUMBA_NUM_THREADS`` in the environment.

    Raises:
        NotImplementedError: If ``max_degree != 5`` or ``pg`` was built via
            ``from_subsample``.

    Note:
        Emits a :class:`FutureWarning` on every call (Python's default
        once-per-call-site behavior). Pin the warning filter if it's noisy
        in your test runner.
    """
    if max_degree != 5:
        raise NotImplementedError(
            "count_pairs_bfs only supports max_degree=5; use PedigreeGraph.count_pairs for partial extractions",
        )
    if pg._sample_mask is not None or pg._subsample_remap is not None:
        raise NotImplementedError(
            "count_pairs_bfs does not support subsampled graphs yet; use PedigreeGraph.count_pairs() instead",
        )
    warnings.warn(
        "count_pairs_bfs is experimental — API and semantics may change or be removed in any minor release",
        FutureWarning,
        stacklevel=2,
    )
    if n_threads is not None:
        numba.set_num_threads(int(n_threads))
    threads = n_threads if n_threads is not None else _BFS_DEFAULT_THREADS

    n = pg.n
    mother = pg.mother.astype(np.int64)
    father = pg.father.astype(np.int64)

    # ---- P_1: parent matrix; P_k via boolean (set-union) matmul.
    t0 = time.perf_counter()
    rows_idx = np.concatenate(
        [np.flatnonzero(mother != -1), np.flatnonzero(father != -1)],
    )
    cols_idx = np.concatenate(
        [mother[mother != -1], father[father != -1]],
    )
    if rows_idx.size:
        P1 = sp.csr_matrix(
            (np.ones(rows_idx.size, dtype=np.int8), (rows_idx, cols_idx)),
            shape=(n, n),
        )
    else:
        P1 = sp.csr_matrix((n, n), dtype=np.int8)
    logger.info("[bfs]   P_1 built in %.2fs (nnz=%d)", time.perf_counter() - t0, P1.nnz)

    def _bool_matmul(A: sp.csr_matrix, B: sp.csr_matrix) -> sp.csr_matrix:
        # Boolean (set-union) matmul: clamp values to 1 to avoid path-count
        # multiplicity inflating nnz. Output is csr int8.
        M = A @ B
        M.data[:] = 1
        M.eliminate_zeros()
        return M.astype(np.int8)

    P: list[sp.csr_matrix | None] = [None]  # 1-indexed: P[k] = P_k
    P.append(P1)
    for k in range(2, 6):
        t0 = time.perf_counter()
        Pk = _bool_matmul(P[k - 1], P1)
        P.append(Pk)
        logger.info("[bfs]   P_%d built in %.2fs (nnz=%d)", k, time.perf_counter() - t0, Pk.nnz)

    # Pre-build CSR views of P_k.T (rows = anchor X, cols = descendants at depth k).
    t0 = time.perf_counter()
    PT_csr: list[sp.csr_matrix | None] = [None] + [P[k].T.tocsr() for k in range(1, 6)]
    logger.info("[bfs]   PT_csr (5 transposes) in %.2fs", time.perf_counter() - t0)

    def _desc(k: int, X: int) -> np.ndarray:
        return PT_csr[k].indices[PT_csr[k].indptr[X] : PT_csr[k].indptr[X + 1]].astype(np.intp)

    # ---- Degree 0: MZ twins (delegate to PedigreeGraph for correctness).
    mz_lo, _ = pg._mz_twin_pairs()
    n_mz = len(mz_lo)

    # ---- Degree 1: PO, FS, MHS, PHS — delegate to PedigreeGraph.
    t0 = time.perf_counter()
    (mo_i, mo_j), (fo_i, fo_j) = pg._parent_offspring_pairs()
    n_mo = len(mo_i)
    n_fo = len(fo_i)
    po_i = np.concatenate([mo_i, fo_i]).astype(np.intp)
    po_j = np.concatenate([mo_j, fo_j]).astype(np.intp)
    if po_i.size:
        po_lo = np.minimum(po_i, po_j)
        po_hi = np.maximum(po_i, po_j)
    else:
        po_lo = po_hi = np.array([], dtype=np.intp)

    fs, mat_hs, pat_hs = pg.sibling_pairs()
    fs_lo, fs_hi = fs
    mat_hs_lo, mat_hs_hi = mat_hs
    pat_hs_lo, pat_hs_hi = pat_hs
    logger.info("[bfs]   FS/MHS/PHS/PO blocks in %.2fs", time.perf_counter() - t0)

    enum_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def _enumerate_shared(a: int, b: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Pairs (i, j, c) where i and j share c distinct ancestors at depth (a, b).

        Calls the parallel ``_enumerate_pairs_kernel`` (numba ``prange``) to
        emit a flat ``(i, j)`` array, then deduplicates via ``np.unique``.
        For ``a == b`` each unique pair appears once per shared anchor; for
        ``a != b`` counts include both directional contributions, matching
        the matrix engine's ``M[i, j] + M[j, i]`` total.
        """
        if (a, b) in enum_cache:
            return enum_cache[(a, b)]
        t_total = time.perf_counter()
        PT_a = PT_csr[a]
        PT_b = PT_csr[b]
        t_k = time.perf_counter()
        out_i, out_j = _enumerate_pairs_kernel(
            PT_a.indptr.astype(np.int64),
            PT_a.indices.astype(np.int64),
            PT_b.indptr.astype(np.int64),
            PT_b.indices.astype(np.int64),
            n,
            a == b,
        )
        t_kernel = time.perf_counter() - t_k
        if a != b:
            mask = out_i != out_j
            out_i = out_i[mask]
            out_j = out_j[mask]
        if out_i.size == 0:
            empty = (
                np.array([], dtype=np.intp),
                np.array([], dtype=np.intp),
                np.array([], dtype=np.int64),
            )
            enum_cache[(a, b)] = empty
            logger.info(
                "[bfs]   _enumerate_shared(%d,%d) empty in %.2fs",
                a,
                b,
                time.perf_counter() - t_total,
            )
            return empty
        t_d = time.perf_counter()
        lo = np.minimum(out_i, out_j).astype(np.intp)
        hi = np.maximum(out_i, out_j).astype(np.intp)
        max_id = int(hi.max()) + 1 if hi.size else 1
        keys = lo.astype(np.int64) * max_id + hi.astype(np.int64)
        u_keys, counts = np.unique(keys, return_counts=True)
        u_lo = (u_keys // max_id).astype(np.intp)
        u_hi = (u_keys % max_id).astype(np.intp)
        result = (u_lo, u_hi, counts.astype(np.int64))
        enum_cache[(a, b)] = result
        logger.info(
            "[bfs]   _enumerate_shared(%d,%d) %.2fs (kernel=%.2fs dedup=%.2fs raw=%d uniq=%d)",
            a,
            b,
            time.perf_counter() - t_total,
            t_kernel,
            time.perf_counter() - t_d,
            out_i.size,
            u_lo.size,
        )
        return result

    # ---- 1C / H1C from depth-2 shared ancestors.
    c1_lo_all, c1_hi_all, c1_counts = _enumerate_shared(2, 2)
    if c1_lo_all.size:
        # Drop sib pairs (share a parent, not a grandparent).
        share_mother = (mother[c1_lo_all] != -1) & (mother[c1_lo_all] == mother[c1_hi_all])
        share_father = (father[c1_lo_all] != -1) & (father[c1_lo_all] == father[c1_hi_all])
        is_sib = share_mother | share_father
        c1_lo_all = c1_lo_all[~is_sib]
        c1_hi_all = c1_hi_all[~is_sib]
        c1_counts = c1_counts[~is_sib]
    full_mask = c1_counts >= 2
    half_mask = c1_counts == 1
    one_c_lo = c1_lo_all[full_mask]
    one_c_hi = c1_hi_all[full_mask]
    h1c_lo = c1_lo_all[half_mask]
    h1c_hi = c1_hi_all[half_mask]

    # ---- Lineal counts at each depth.
    gp_count = int(P[2].nnz)
    ggp_count = int(P[3].nnz)
    gggp_count = int(P[4].nnz)
    g3gp_count = int(P[5].nnz)

    # ---- Avuncular: A @ full_sib_matrix - parent_child. Equivalent BFS form:
    # for each FS pair (s1, s2), all (child_of_s1, s2) and (child_of_s2, s1).
    t0 = time.perf_counter()
    if fs_lo.size:

        def _av_worker(start: int, stop: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
            local_i: list[np.ndarray] = []
            local_j: list[np.ndarray] = []
            for k in range(start, stop):
                s1, s2 = int(fs_lo[k]), int(fs_hi[k])
                for parent, niece in ((s1, s2), (s2, s1)):
                    kids = _desc(1, parent)
                    if kids.size == 0:
                        continue
                    others = np.full(kids.size, niece, dtype=np.intp)
                    lo = np.minimum(kids, others)
                    hi = np.maximum(kids, others)
                    neq = lo != hi
                    local_i.append(lo[neq])
                    local_j.append(hi[neq])
            return local_i, local_j

        av_lo_parts, av_hi_parts = _run_per_anchor(_av_worker, fs_lo.size, threads)
    else:
        av_lo_parts, av_hi_parts = [], []
    if av_lo_parts:
        av_lo = np.concatenate(av_lo_parts)
        av_hi = np.concatenate(av_hi_parts)
        av_lo, av_hi = dedup_pairs(av_lo, av_hi)
        av_lo, av_hi = pg._subtract_pairs((av_lo, av_hi), (po_lo, po_hi))
    else:
        av_lo = av_hi = np.array([], dtype=np.intp)
    logger.info("[bfs]   Av in %.2fs (n=%d)", time.perf_counter() - t0, av_lo.size)

    def _collateral_pairs(
        sib_lo: np.ndarray,
        sib_hi: np.ndarray,
        down: int,
        up: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Pairs joined through a sibling-pair link at depths (down-1, up-1).

        For each sib pair (s1, s2), emits all (i, j) where i is a depth-(down-1)
        descendant of s1 and j is a depth-(up-1) descendant of s2 (and the
        symmetric (s2, s1) cross), filtering out i == j.
        """
        if sib_lo.size == 0:
            return np.array([], dtype=np.intp), np.array([], dtype=np.intp)
        a = down - 1
        b = up - 1

        def _worker(start: int, stop: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
            local_i: list[np.ndarray] = []
            local_j: list[np.ndarray] = []
            for k in range(start, stop):
                s1, s2 = int(sib_lo[k]), int(sib_hi[k])
                for x, y in ((s1, s2), (s2, s1)):
                    dx = _desc(a, x) if a > 0 else np.array([x], dtype=np.intp)
                    dy = _desc(b, y) if b > 0 else np.array([y], dtype=np.intp)
                    if dx.size == 0 or dy.size == 0:
                        continue
                    ax = np.repeat(dx, dy.size)
                    ay = np.tile(dy, dx.size)
                    mask = ax != ay
                    local_i.append(ax[mask])
                    local_j.append(ay[mask])
            return local_i, local_j

        i_parts, j_parts = _run_per_anchor(_worker, sib_lo.size, threads)
        if not i_parts:
            return np.array([], dtype=np.intp), np.array([], dtype=np.intp)
        ax = np.concatenate(i_parts)
        ay = np.concatenate(j_parts)
        return dedup_pairs(ax, ay)

    gp_lo, gp_hi = _lineal_pairs_lo_hi(P[2])
    ggp_lo, ggp_hi = _lineal_pairs_lo_hi(P[3])
    gggp_lo, gggp_hi = _lineal_pairs_lo_hi(P[4])

    # ---- HAv: half-sib avuncular: down=1, up=2.
    t0 = time.perf_counter()
    hs_lo = np.concatenate([mat_hs_lo, pat_hs_lo])
    hs_hi = np.concatenate([mat_hs_hi, pat_hs_hi])
    hav_lo, hav_hi = _collateral_pairs(hs_lo, hs_hi, down=1, up=2)
    hav_count = _count_after_subtract(hav_lo, hav_hi, [(po_lo, po_hi), (gp_lo, gp_hi)])
    logger.info("[bfs]   HAv in %.2fs", time.perf_counter() - t0)

    # ---- GAv: full-sib great-avuncular: down=1, up=3.
    t0 = time.perf_counter()
    gav_lo, gav_hi = _collateral_pairs(fs_lo, fs_hi, down=1, up=3)
    gav_count = _count_after_subtract(
        gav_lo,
        gav_hi,
        [(po_lo, po_hi), (gp_lo, gp_hi), (av_lo, av_hi)],
    )
    logger.info("[bfs]   GAv in %.2fs", time.perf_counter() - t0)

    # ---- HGAv: half-sib great-avuncular: down=1, up=3.
    t0 = time.perf_counter()
    hgav_lo, hgav_hi = _collateral_pairs(hs_lo, hs_hi, down=1, up=3)
    hgav_count = _count_after_subtract(
        hgav_lo,
        hgav_hi,
        [(po_lo, po_hi), (gp_lo, gp_hi), (ggp_lo, ggp_hi), (hav_lo, hav_hi)],
    )
    logger.info("[bfs]   HGAv in %.2fs", time.perf_counter() - t0)

    # ---- GGAv: full-sib great-great-avuncular: down=1, up=4.
    t0 = time.perf_counter()
    ggav_lo, ggav_hi = _collateral_pairs(fs_lo, fs_hi, down=1, up=4)
    ggav_count = _count_after_subtract(
        ggav_lo,
        ggav_hi,
        [(po_lo, po_hi), (gp_lo, gp_hi), (ggp_lo, ggp_hi), (av_lo, av_hi), (gav_lo, gav_hi)],
    )
    logger.info("[bfs]   GGAv in %.2fs", time.perf_counter() - t0)

    # ---- 1C1R / H1C1R: pairs sharing depth (2, 3) ancestors.
    p23_lo, p23_hi, p23_counts = _enumerate_shared(2, 3)
    sib_all_lo = np.concatenate([fs_lo, mat_hs_lo, pat_hs_lo])
    sib_all_hi = np.concatenate([fs_hi, mat_hs_hi, pat_hs_hi])
    p23_full_mask = p23_counts >= 2
    p23_half_mask = p23_counts == 1
    c1r_lo = p23_lo[p23_full_mask]
    c1r_hi = p23_hi[p23_full_mask]
    c1r_count = _count_after_subtract(
        c1r_lo,
        c1r_hi,
        [
            (po_lo, po_hi),
            (gp_lo, gp_hi),
            (ggp_lo, ggp_hi),
            (av_lo, av_hi),
            (gav_lo, gav_hi),
            (sib_all_lo, sib_all_hi),
            (one_c_lo, one_c_hi),
        ],
    )
    h1c1r_lo = p23_lo[p23_half_mask]
    h1c1r_hi = p23_hi[p23_half_mask]

    # ---- HGGAv: half-sib gggrand-avuncular: down=1, up=4.
    t0 = time.perf_counter()
    hggav_lo, hggav_hi = _collateral_pairs(hs_lo, hs_hi, down=1, up=4)
    hggav_count = _count_after_subtract(
        hggav_lo,
        hggav_hi,
        [(po_lo, po_hi), (gp_lo, gp_hi), (ggp_lo, ggp_hi), (gggp_lo, gggp_hi), (hav_lo, hav_hi), (hgav_lo, hgav_hi)],
    )
    logger.info("[bfs]   HGGAv in %.2fs", time.perf_counter() - t0)

    # ---- G3Av: full-sib gggrand-avuncular: down=1, up=5.
    t0 = time.perf_counter()
    g3av_lo, g3av_hi = _collateral_pairs(fs_lo, fs_hi, down=1, up=5)
    g3av_count = _count_after_subtract(
        g3av_lo,
        g3av_hi,
        [
            (po_lo, po_hi),
            (gp_lo, gp_hi),
            (ggp_lo, ggp_hi),
            (gggp_lo, gggp_hi),
            (av_lo, av_hi),
            (gav_lo, gav_hi),
            (ggav_lo, ggav_hi),
        ],
    )
    logger.info("[bfs]   G3Av in %.2fs", time.perf_counter() - t0)

    # ---- H1C1R: half-2C-1R = depth-(2,3) shared with count == 1.
    h1c1r_count = _count_after_subtract(
        h1c1r_lo,
        h1c1r_hi,
        [
            (po_lo, po_hi),
            (gp_lo, gp_hi),
            (ggp_lo, ggp_hi),
            (gggp_lo, gggp_hi),
            (hav_lo, hav_hi),
            (hgav_lo, hgav_hi),
            (sib_all_lo, sib_all_hi),
            (one_c_lo, one_c_hi),
            (h1c_lo, h1c_hi),
            (c1r_lo, c1r_hi),
        ],
    )

    # ---- 1C2R: pairs sharing depth (2, 4) ancestors with count >= 2.
    p24_lo, p24_hi, p24_counts = _enumerate_shared(2, 4)
    p24_full_mask = p24_counts >= 2
    c2r_lo = p24_lo[p24_full_mask]
    c2r_hi = p24_hi[p24_full_mask]
    c2r_count = _count_after_subtract(
        c2r_lo,
        c2r_hi,
        [
            (po_lo, po_hi),
            (gp_lo, gp_hi),
            (ggp_lo, ggp_hi),
            (gggp_lo, gggp_hi),
            (av_lo, av_hi),
            (gav_lo, gav_hi),
            (ggav_lo, ggav_hi),
            (sib_all_lo, sib_all_hi),
            (one_c_lo, one_c_hi),
            (h1c_lo, h1c_hi),
            (c1r_lo, c1r_hi),
        ],
    )

    # ---- 2C: pairs sharing depth-3 ancestors with count >= 2, minus those
    # sharing any depth-2 ancestor (matches the matrix engine's
    # subtract-A2_shared semantic).
    p33_lo, p33_hi, p33_counts = _enumerate_shared(3, 3)
    full_2c_mask = p33_counts >= 2
    sc_lo = p33_lo[full_2c_mask]
    sc_hi = p33_hi[full_2c_mask]
    sc2_lo, sc2_hi, _ = _enumerate_shared(2, 2)
    sc_count = _count_after_subtract(sc_lo, sc_hi, [(sc2_lo, sc2_hi)])

    named: dict[str, int] = {
        "MZ": n_mz,
        "MO": n_mo,
        "FO": n_fo,
        "FS": int(fs_lo.size),
        "MHS": int(mat_hs_lo.size),
        "PHS": int(pat_hs_lo.size),
        "GP": gp_count,
        "Av": int(av_lo.size),
        "GGP": ggp_count,
        "HAv": hav_count,
        "GAv": gav_count,
        "1C": int(one_c_lo.size),
        "GGGP": gggp_count,
        "HGAv": hgav_count,
        "GGAv": ggav_count,
        "H1C": int(h1c_lo.size),
        "1C1R": c1r_count,
        "G3GP": g3gp_count,
        "HGGAv": hggav_count,
        "G3Av": g3av_count,
        "H1C1R": h1c1r_count,
        "1C2R": c2r_count,
        "2C": sc_count,
    }
    # Sanity: the produced code set must match the relationship registry.
    assert set(named) == set(REL_REGISTRY), "BFS produced an unexpected code set"
    return named
