"""Pedigree relationship extraction via sparse matrix products.

Builds parent→child CSR matrices and extracts relationship categories
using sparse matrix algebra (A @ A.T for siblings, A² @ (A²).T for cousins, etc.).

Each relationship type is parameterised by (up, down, n_ancestors):
  - up:   meioses from individual A up to common ancestor(s), canonicalised up ≤ down
  - down: meioses from common ancestor(s) down to individual B
  - n_ancestors: 1 (half / lineal) or 2 (full, i.e. mated pair)
  - kinship = n_ancestors × (1/2)^(up + down + 1)
"""

from __future__ import annotations

__all__ = [
    "PAIR_KINSHIP",
    "REL_REGISTRY",
    "PedigreeGraph",
    "RelType",
]

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

import numpy as np
import scipy.sparse as sp

from pedigree_graph._effective_size import _per_gen_mean_kinship
from pedigree_graph._kinship_kernel import (
    _build_kinship_csc,
    _check_topological,
    _compute_depth,
    _compute_F_meuwissen_luo,
    _compute_theta_per_gen,
)

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


_REQUIRED_COLUMNS: tuple[str, ...] = ("id", "mother", "father", "twin", "sex", "generation")
_OPTIONAL_COLUMNS: tuple[str, ...] = ("birth_year",)


def _coerce_to_array_dict(data: dict[str, np.ndarray] | pd.DataFrame) -> dict[str, np.ndarray]:
    """Normalize input to a dict of numpy arrays.

    Accepts either a ``dict[str, np.ndarray]`` (returned as-is) or a
    pandas ``DataFrame`` (each required column extracted via ``.values``;
    optional columns extracted when present).  No pandas import is
    performed in the dict path.
    """
    if isinstance(data, dict):
        return data
    result = {col: data[col].values for col in _REQUIRED_COLUMNS}
    for col in _OPTIONAL_COLUMNS:
        if col in data.columns:
            result[col] = data[col].values
    return result


def _known_parent_edges(
    parent_arr: np.ndarray,
    birth_year: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Edges where both endpoints have a known ``birth_year``.

    Shared by every overlapping-generation utility that needs the
    parent-age distribution (topological validation,
    ``pg.generation_interval``, ``eligible_cohort_range``, age-table
    diagnostic).

    Args:
        parent_arr: per-row parent row-index array (``self.mother`` or
            ``self.father``).
        birth_year: per-row birth_year array (sentinel ``-1`` = unknown).

    Returns:
        ``(child_rows, age_diffs)`` — child row indices (where the edge
        exists *and* both endpoints have ``birth_year >= 0``), and the
        corresponding ``child.birth_year - parent.birth_year`` values
        as ``int32``.  Empty arrays when no qualifying edges exist.
    """
    edge_rows = np.where(parent_arr >= 0)[0]
    if edge_rows.size == 0:
        return np.array([], dtype=np.intp), np.array([], dtype=np.int32)
    parents = parent_arr[edge_rows]
    by_child = birth_year[edge_rows]
    by_parent = birth_year[parents]
    both_known = (by_child >= 0) & (by_parent >= 0)
    if not np.any(both_known):
        return np.array([], dtype=np.intp), np.array([], dtype=np.int32)
    return edge_rows[both_known], (by_child[both_known] - by_parent[both_known]).astype(np.int32)


# ---------------------------------------------------------------------------
# Relationship type registry
# ---------------------------------------------------------------------------


class RelType(NamedTuple):
    """Relationship category defined by path through pedigree."""

    up: int  # meioses A → common ancestor(s)
    down: int  # meioses common ancestor(s) → B
    n_anc: int  # 1 = half/lineal, 2 = full (mated-pair ancestors)
    code: str  # short dict key
    label: str  # human-readable display label

    @property
    def kinship(self) -> float:
        """Kinship coefficient derived from path length and ancestor count."""
        if self.code == "MZ":
            return 0.5
        return self.n_anc * 0.5 ** (self.up + self.down + 1)

    @property
    def degree(self) -> int:
        """Kinship degree (0 for MZ, 1 for parent-offspring/full-sib, etc.)."""
        if self.code == "MZ":
            return 0
        return round(-1 - np.log2(self.kinship))


# Ordered registry: kinship-descending, degree-ascending.
# MZ twins are a special case (up=down=n_anc=0).
REL_REGISTRY: dict[str, RelType] = {}
for _rt in [
    # --- special ---
    RelType(0, 0, 0, "MZ", "MZ twin"),
    # --- degree 1 (kinship 1/4) ---
    RelType(1, 0, 1, "MO", "Mother-offspring"),
    RelType(1, 0, 1, "FO", "Father-offspring"),
    RelType(1, 1, 2, "FS", "Full sib"),
    # --- degree 2 (kinship 1/8) ---
    RelType(1, 1, 1, "MHS", "Maternal half sib"),
    RelType(1, 1, 1, "PHS", "Paternal half sib"),
    RelType(2, 0, 1, "GP", "Grandparent"),
    RelType(1, 2, 2, "Av", "Avuncular"),
    # --- degree 3 (kinship 1/16) ---
    RelType(3, 0, 1, "GGP", "Great-grandparent"),
    RelType(1, 2, 1, "HAv", "Half-avuncular"),
    RelType(1, 3, 2, "GAv", "Great-avuncular"),
    RelType(2, 2, 2, "1C", "1st cousin"),
    # --- degree 4 (kinship 1/32) ---
    RelType(4, 0, 1, "GGGP", "Great\u00b2-grandparent"),
    RelType(1, 3, 1, "HGAv", "Half-great-avuncular"),
    RelType(1, 4, 2, "GGAv", "Great\u00b2-avuncular"),
    RelType(2, 2, 1, "H1C", "Half-1st-cousin"),
    RelType(2, 3, 2, "1C1R", "1st cousin 1R"),
    # --- degree 5 (kinship 1/64) ---
    RelType(5, 0, 1, "G3GP", "Great\u00b3-grandparent"),
    RelType(1, 4, 1, "HGGAv", "Half-great\u00b2-avuncular"),
    RelType(1, 5, 2, "G3Av", "Great\u00b3-avuncular"),
    RelType(2, 3, 1, "H1C1R", "Half-1st-cousin 1R"),
    RelType(2, 4, 2, "1C2R", "1st cousin 2R"),
    RelType(3, 3, 2, "2C", "2nd cousin"),
]:
    REL_REGISTRY[_rt.code] = _rt

# Kinship lookup by code — single source of truth for all consumers
PAIR_KINSHIP: dict[str, float] = {rt.code: rt.kinship for rt in REL_REGISTRY.values()}


class PedigreeGraph:
    """Parent→child DAG for efficient relationship queries.

    Each individual is a vertex whose index equals its row index in the
    input.  Sparse CSR matrices encode parent-child edges for O(nnz)
    relationship extraction via matrix products.

    Args:
        data: Either a ``dict[str, np.ndarray]`` keyed by ``id``, ``mother``,
            ``father``, ``twin``, ``sex``, ``generation``, or a pandas
            ``DataFrame`` with those columns.  The dict path requires no
            pandas import.
    """

    def __init__(self, data: dict[str, np.ndarray] | pd.DataFrame) -> None:
        arrays = _coerce_to_array_dict(data)
        ids_arr = np.asarray(arrays["id"])
        n = len(ids_arr)
        self.n = n

        # Subsample state — set only by from_subsample. When _sample_mask is
        # set, extract_pairs filters to pairs where both endpoints are active.
        # When _subsample_remap is set, extract_pairs additionally remaps
        # graph row indices to caller-input row indices.
        self._sample_mask: np.ndarray | None = None
        self._subsample_remap: np.ndarray | None = None

        # Lazy kinship cache — populated by kinship_matrix(); keyed by
        # the resolved min_kinship threshold.
        self._kinship_cache: dict[float, sp.csc_matrix] = {}
        # Lazy per-generation mean kinship cache — populated by
        # per_gen_mean_kinship(); keyed by min_kinship so callers using a
        # non-default threshold do not get a stale θ̄.
        self._theta_per_gen_cache: dict[float, np.ndarray] = {}
        self._inbreeding: np.ndarray | None = None
        # Topological depth recomputed from edges; user-supplied
        # ``self.generation`` may be sparse/skipped/post-filtered and is
        # not safe as a substitute for the ML F kernel.
        self._depth: np.ndarray | None = None

        # Build ID → row index mapping (int32: row indices fit in 2.1B)
        self._ids = ids_arr
        if n > np.iinfo(np.int32).max:
            raise ValueError(f"Pedigree has {n:,} rows, exceeding int32 max for row indices")
        id_to_row = np.full(int(ids_arr.max()) + 1, -1, dtype=np.int32)
        id_to_row[ids_arr] = np.arange(n, dtype=np.int32)

        def _remap(col_vals: np.ndarray) -> np.ndarray:
            """Map IDs to row indices; -1 stays -1."""
            out = np.full(len(col_vals), -1, dtype=np.int32)
            valid = (col_vals >= 0) & (col_vals < len(id_to_row))
            out[valid] = id_to_row[col_vals[valid]]
            return out

        # Original pedigree parent IDs (for sibling classification —
        # valid even when the parent isn't in the sample).
        self._orig_mother = np.asarray(arrays["mother"])
        self._orig_father = np.asarray(arrays["father"])

        # Remap parent/twin IDs to row indices (for sparse matrices)
        self.mother = _remap(self._orig_mother)
        self.father = _remap(self._orig_father)
        self.twin = _remap(np.asarray(arrays["twin"]))
        self.sex = np.asarray(arrays["sex"]).astype(np.int8)
        self.generation = np.asarray(arrays["generation"]).astype(np.int32)

        # Optional birth_year (sentinel -1 for unknown). NaN floats coerced
        # to -1 so callers can pass a pandas Series with missing values.
        if "birth_year" in arrays:
            by_raw = np.asarray(arrays["birth_year"])
            if by_raw.dtype.kind == "f":
                by_raw = np.where(np.isnan(by_raw), -1, by_raw)
            self.birth_year: np.ndarray | None = by_raw.astype(np.int32)
        else:
            self.birth_year = None

        if not _check_topological(self.mother, self.father, n):
            raise ValueError("PedigreeGraph requires topological order: parents must precede children")

        self._validate_birth_year_topology()

        # Build parent→child matrices using ALL available edges.
        # Each matrix is built independently so partial-pedigree data
        # (e.g. after subsampling) still contributes edges.
        m_mask = self.mother >= 0
        m_idx = np.where(m_mask)[0]
        f_mask = self.father >= 0
        f_idx = np.where(f_mask)[0]

        self._Am = sp.csr_matrix(
            (np.ones(len(m_idx), dtype=np.float64), (m_idx, self.mother[m_idx])),
            shape=(n, n),
        )
        self._Af = sp.csr_matrix(
            (np.ones(len(f_idx), dtype=np.float64), (f_idx, self.father[f_idx])),
            shape=(n, n),
        )

    def _validate_birth_year_topology(self) -> None:
        """Reject parent-child edges with child.birth_year < parent.birth_year.

        Only checks edges where both endpoints have known birth_year
        (sentinel ``-1`` skipped). Unknown-parent and unknown-child rows
        contribute no constraints.
        """
        if self.birth_year is None:
            return
        for parent_arr, parent_label in ((self.mother, "mother"), (self.father, "father")):
            edge_rows, diffs = _known_parent_edges(parent_arr, self.birth_year)
            if diffs.size == 0:
                continue
            violations = diffs < 0
            if np.any(violations):
                n_bad = int(violations.sum())
                first = int(np.argmax(violations))
                raise ValueError(
                    f"birth_year topology violation: {n_bad} {parent_label}-child "
                    f"edge(s) have child.birth_year < {parent_label}.birth_year "
                    f"(first: row {int(edge_rows[first])}, diff={int(diffs[first])})"
                )

    @cached_property
    def generation_interval(self):
        """Sex-split generation interval (Hill 1979 ``L``).

        Returns a :class:`~pedigree_graph._effective_size.GenerationInterval`
        ``(T, T_m, T_f, n_edges)`` computed over all parent-child edges
        where both endpoints have known ``birth_year`` (sentinel ``-1``
        skipped).

        * ``T_m`` = mean ``child.birth_year − sire.birth_year`` over
          sire-offspring edges (parent = father field).
        * ``T_f`` = symmetric for dam-offspring edges (parent = mother).
        * ``T = (T_m + T_f) / 2``.
        * Skip-generation edges included unconditionally.

        Returns ``None`` if ``self.birth_year is None`` or if either sex
        has zero qualifying edges.  Cached on the instance on first read.
        """
        if self.birth_year is None:
            return None

        _, diffs_m = _known_parent_edges(self.father, self.birth_year)
        _, diffs_f = _known_parent_edges(self.mother, self.birth_year)
        if diffs_m.size == 0 or diffs_f.size == 0:
            return None

        from pedigree_graph._effective_size import GenerationInterval

        T_m = float(diffs_m.mean())
        T_f = float(diffs_f.mean())
        return GenerationInterval(
            T=(T_m + T_f) / 2.0,
            T_m=T_m,
            T_f=T_f,
            n_edges=int(diffs_m.size + diffs_f.size),
        )

    # ------------------------------------------------------------------
    # Lazy sparse products (computed on first access)
    # ------------------------------------------------------------------

    @cached_property
    def _A(self):
        """Child → both parents adjacency matrix."""
        t0 = time.perf_counter()
        result = self._Am + self._Af
        logger.debug("_A (Am + Af) computed in %.3fs", time.perf_counter() - t0)
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
            return sp.eye(self.n, format="csr")
        if k == 1:
            return self._A
        return getattr(self, f"_A{k}")

    def _ensure_sibling_matrices(self) -> None:
        """Ensure _full_sib_matrix and _half_sib_matrix are computed."""
        if hasattr(self, "_full_sib_matrix"):
            return
        # Trigger sibling extraction which sets _full_sib_matrix
        self.sibling_pairs()

    def _build_half_sib_matrix(
        self,
        mat_hs: tuple[np.ndarray, np.ndarray],
        pat_hs: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Build and cache _half_sib_matrix from extracted half-sib pairs."""
        hs1 = np.concatenate([mat_hs[0], pat_hs[0]])
        hs2 = np.concatenate([mat_hs[1], pat_hs[1]])
        if len(hs1) > 0:
            ones = np.ones(len(hs1), dtype=np.float64)
            H = sp.csr_matrix((ones, (hs1, hs2)), shape=(self.n, self.n))
            self._half_sib_matrix = H + H.T
        else:
            self._half_sib_matrix = sp.csr_matrix((self.n, self.n))

    # ------------------------------------------------------------------
    # Shared extraction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _dedup_pairs(a_i: np.ndarray, a_j: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Canonicalize (lo, hi) and deduplicate pair arrays via int64 keys."""
        if len(a_i) == 0:
            return np.array([], dtype=np.intp), np.array([], dtype=np.intp)
        lo = np.minimum(a_i, a_j).astype(np.intp)
        hi = np.maximum(a_i, a_j).astype(np.intp)
        max_id = int(hi.max()) + 1
        keys = lo.astype(np.int64) * max_id + hi.astype(np.int64)
        _, unique_idx = np.unique(keys, return_index=True)
        return lo[unique_idx], hi[unique_idx]

    def _extract_from_sparse(
        self,
        M: sp.spmatrix,
        subtract: list[tuple[np.ndarray, np.ndarray]] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract nonzero pairs from sparse matrix, dedup, and subtract closer pairs.

        Mutates *M* in place (zeroes diagonal). Callers should not reuse *M*.
        All subtract pairs are batched into a single ``np.isin`` call.
        """
        M.setdiag(0)
        M.eliminate_zeros()
        if M.nnz == 0:
            return np.array([], dtype=np.intp), np.array([], dtype=np.intp)

        a_i, a_j = M.nonzero()
        lo, hi = self._dedup_pairs(a_i, a_j)

        if subtract and len(lo) > 0:
            # Collect all subtract pairs into one key set, then filter once
            rm_lo_parts: list[np.ndarray] = []
            rm_hi_parts: list[np.ndarray] = []
            for rm_pair in subtract:
                if len(rm_pair[0]) > 0:
                    r1, r2 = rm_pair
                    rm_lo_parts.append(np.minimum(r1, r2))
                    rm_hi_parts.append(np.maximum(r1, r2))
            if rm_lo_parts:
                all_rm_lo = np.concatenate(rm_lo_parts)
                all_rm_hi = np.concatenate(rm_hi_parts)
                max_id = int(max(lo.max(), hi.max(), all_rm_lo.max(), all_rm_hi.max())) + 1
                rm_keys = all_rm_lo.astype(np.int64) * max_id + all_rm_hi.astype(np.int64)
                cand_keys = lo.astype(np.int64) * max_id + hi.astype(np.int64)
                keep = ~np.isin(cand_keys, rm_keys)
                lo, hi = lo[keep], hi[keep]
        return lo, hi

    def _lineal_pairs(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Direct ancestor-descendant pairs at exactly k hops."""
        Ak = self._get_Ak(k)
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
        A_down_1 = self._get_Ak(down - 1)
        A_up_1 = self._get_Ak(up - 1)
        M = A_down_1 @ sib_matrix @ A_up_1.T
        return self._extract_from_sparse(M, subtract=subtract)

    # ------------------------------------------------------------------
    # Relationship extraction
    # ------------------------------------------------------------------

    def _mz_twin_pairs(self) -> tuple[np.ndarray, np.ndarray]:
        """MZ twin pairs: twin != -1, deduplicated with id < twin_id."""
        has_twin = self.twin >= 0
        ids = np.where(has_twin)[0]
        partners = self.twin[has_twin]
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
        one parent in the sample still contributes a PO pair.
        """
        m_mask = self.mother >= 0
        m_children = np.where(m_mask)[0]

        f_mask = self.father >= 0
        f_children = np.where(f_mask)[0]

        return (m_children, self.mother[m_children].astype(np.intp)), (
            f_children,
            self.father[f_children].astype(np.intp),
        )

    def sibling_pairs(
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
        has_parent = (self._orig_mother >= 0) | (self._orig_father >= 0)
        nt_mask = has_parent & (self.twin < 0)
        nt_idx = np.where(nt_mask)[0]

        if len(nt_idx) < 2:
            self._full_sib_matrix = sp.csr_matrix((self.n, self.n))
            self._half_sib_matrix = sp.csr_matrix((self.n, self.n))
            return empty, empty, empty

        nt_mother = self._orig_mother[nt_idx]
        nt_father = self._orig_father[nt_idx]

        # --- Full sibs: same KNOWN mother AND same KNOWN father ---
        both_known = (nt_mother >= 0) & (nt_father >= 0)
        bk_idx = nt_idx[both_known]
        bk_mother = nt_mother[both_known]
        bk_father = nt_father[both_known]

        if len(bk_idx) >= 2:
            max_parent = max(int(bk_mother.max()), int(bk_father.max())) + 1
            # int64 cast required: max_id² overflows int32
            family_key = bk_mother.astype(np.int64) * max_parent + bk_father.astype(np.int64)
            full_sib = self._pairs_from_groups(bk_idx, family_key)
        else:
            full_sib = empty

        # --- Maternal half sibs: all pairs sharing known mother, minus full-sib pairs ---
        has_mother = nt_mother >= 0
        m_idx = nt_idx[has_mother]
        m_mother = nt_mother[has_mother]
        if len(m_idx) >= 2:
            mat_all = self._pairs_from_groups(m_idx, m_mother)
            mat_hs = self._subtract_pairs(mat_all, full_sib)
        else:
            mat_hs = empty

        # --- Paternal half sibs: all pairs sharing known father, minus full-sib pairs ---
        has_father = nt_father >= 0
        f_idx = nt_idx[has_father]
        f_father = nt_father[has_father]
        if len(f_idx) >= 2:
            pat_all = self._pairs_from_groups(f_idx, f_father)
            pat_hs = self._subtract_pairs(pat_all, full_sib)
        else:
            pat_hs = empty

        # Build full-sib sparse matrix for _avuncular_pairs and collateral methods
        sib1, sib2 = full_sib
        if len(sib1) > 0:
            ones = np.ones(len(sib1), dtype=np.float64)
            F = sp.csr_matrix((ones, (sib1, sib2)), shape=(self.n, self.n))
            self._full_sib_matrix = F + F.T
        else:
            self._full_sib_matrix = sp.csr_matrix((self.n, self.n))

        return full_sib, mat_hs, pat_hs

    @staticmethod
    def _subtract_pairs(
        all_pairs: tuple[np.ndarray, np.ndarray],
        remove_pairs: tuple[np.ndarray, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Remove pairs in *remove_pairs* from *all_pairs* using set subtraction.

        Both inputs must be canonically ordered (lo, hi).
        Encodes each pair as lo * max_id + hi for O(1) lookup.
        """
        a1, a2 = all_pairs
        r1, r2 = remove_pairs

        if len(a1) == 0:
            return all_pairs
        if len(r1) == 0:
            return all_pairs

        # int64 cast required: max_id² overflows int32
        max_id = int(max(a1.max(), a2.max(), r1.max(), r2.max())) + 1
        remove_keys = r1.astype(np.int64) * max_id + r2.astype(np.int64)
        all_keys = a1.astype(np.int64) * max_id + a2.astype(np.int64)

        keep = ~np.isin(all_keys, remove_keys)

        return a1[keep].astype(np.intp), a2[keep].astype(np.intp)

    @staticmethod
    def _pairs_from_groups(indices: np.ndarray, group_key: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Generate all (i < j) pairs of indices within each group.

        Uses batch-by-size triu_indices for vectorized pair generation.
        """
        sort_idx = np.argsort(group_key, kind="mergesort")
        sorted_keys = group_key[sort_idx]
        sorted_indices = indices[sort_idx]

        _, starts, counts = np.unique(sorted_keys, return_index=True, return_counts=True)

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

    @staticmethod
    def _pairs_from_groups_filtered(
        indices: np.ndarray,
        group_key: np.ndarray,
        filter_key: np.ndarray,
        keep_same: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate pairs within groups, filtered by a secondary key.

        Like _pairs_from_groups but also checks the filter_key for each pair
        and keeps only pairs where filter_key matches (keep_same=True) or
        differs (keep_same=False).
        """
        sort_idx = np.argsort(group_key, kind="mergesort")
        sorted_keys = group_key[sort_idx]
        sorted_indices = indices[sort_idx]
        sorted_filter = filter_key[sort_idx]

        _, starts, counts = np.unique(sorted_keys, return_index=True, return_counts=True)

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

            if keep_same:
                mask = sorted_filter[all_i] == sorted_filter[all_j]
            else:
                mask = sorted_filter[all_i] != sorted_filter[all_j]

            pair_i_parts.append(sorted_indices[all_i[mask]])
            pair_j_parts.append(sorted_indices[all_j[mask]])

        if not pair_i_parts:
            return np.array([], dtype=np.intp), np.array([], dtype=np.intp)

        p1 = np.concatenate(pair_i_parts)
        p2 = np.concatenate(pair_j_parts)

        lo = np.minimum(p1, p2)
        hi = np.maximum(p1, p2)
        return lo.astype(np.intp), hi.astype(np.intp)

    def _cousin_pairs(self) -> tuple[np.ndarray, np.ndarray]:
        """Full 1st cousin pairs: share exactly 2 grandparents (a mated pair) but not a parent.

        Uses group-by-grandparent enumeration. Each pair sharing a grandparent
        is counted — pairs appearing ≥ 2 times share 2+ grandparents (full 1C).
        Pairs appearing exactly once share 1 grandparent (half-1C); these are
        cached in ``_h1c_pairs_cache`` for use by H1C extraction at degree 4.
        """
        t0 = time.perf_counter()
        empty = np.array([], dtype=np.intp), np.array([], dtype=np.intp)
        gc_i, gp_j = self._A2.nonzero()
        if len(gc_i) == 0:
            self._h1c_pairs_cache = empty
            return empty

        # Enumerate all (i < j) pairs sharing a grandparent
        p1, p2 = self._pairs_from_groups(gc_i.astype(np.intp), gp_j)
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
        share_mother = (self._orig_mother[p1] >= 0) & (self._orig_mother[p1] == self._orig_mother[p2])
        share_father = (self._orig_father[p1] >= 0) & (self._orig_father[p1] == self._orig_father[p2])
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
        self._ensure_sibling_matrices()
        if self._full_sib_matrix.nnz == 0:
            return np.array([], dtype=np.intp), np.array([], dtype=np.intp)

        avunc = self._A @ self._full_sib_matrix
        avunc.setdiag(0)

        # Exclude parent-child pairs
        parent_child = (self._A + self._A.T) > 0
        avunc = avunc - avunc.multiply(parent_child)
        avunc.eliminate_zeros()

        if avunc.nnz == 0:
            return np.array([], dtype=np.intp), np.array([], dtype=np.intp)

        return self._dedup_pairs(*avunc.nonzero())

    def _second_cousin_matrix(self) -> sp.spmatrix:
        """Symmetric sparse matrix with nonzeros at full 2nd cousin pairs.

        Full 2nd cousins share ≥ 2 great-grandparents (a mated pair) but no
        grandparents.  Half-2nd-cousins (1 shared great-grandparent) are
        excluded — they fall beyond degree 5.
        """
        t0 = time.perf_counter()
        D_raw = self._A3 @ self._A3.T
        logger.debug("A3 @ A3.T computed in %.3fs (nnz=%d)", time.perf_counter() - t0, D_raw.nnz)
        # Keep only pairs sharing ≥ 2 great-grandparents (full 2C), then booleanise
        D_raw.data[D_raw.data < 2] = 0
        D_raw.eliminate_zeros()
        D_raw.data[:] = 1.0
        C_raw = self._A2_shared.copy()
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

    def extract_pairs(
        self,
        max_degree: int = 2,
        min_kinship: float = 0.0,
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Extract all relationship categories.

        Returned indices are always in caller-input coordinates:

        - ``__init__(df)``: indices into *df* rows.
        - ``from_subsample(full_pedigree, df)``: indices into *df* rows;
          pairs are filtered to those with both endpoints in *df*.
        - ``from_arrays(ids, ...)``: positions in the input *ids* array.

        Args:
            max_degree: Maximum kinship degree to extract (1-5). Degree 2
                covers through 1st cousins, degree 5 through 2nd cousins.
                Higher degrees require more expensive matrix products.
            min_kinship: Skip pair types with kinship coefficient below this
                threshold. E.g., 0.125 skips 1st cousins (0.0625) and 2nd
                cousins (0.016), avoiding their expensive sparse products.

        Returns:
            Dict mapping relationship code to (idx1, idx2) row-index arrays.
        """
        t_total = time.perf_counter()
        pairs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        empty = np.array([], dtype=np.intp)

        def _needed(code: str) -> bool:
            return PAIR_KINSHIP.get(code, 0) >= min_kinship

        need_hs = _needed("MHS") and max_degree >= 2

        # Pre-trigger cached properties needed by downstream extractions.
        # _Am/_Af are only needed to build _A; delete after to free memory.
        if max_degree >= 2:
            _ = self._A2  # chains: _Am, _Af → _A → _A2
        else:
            _ = self._A
        del self._Am, self._Af

        pairs["MZ"] = self._mz_twin_pairs()

        mo, fo = self._parent_offspring_pairs()
        pairs["MO"] = mo
        pairs["FO"] = fo

        t0 = time.perf_counter()
        full_sib, mat_hs, pat_hs = self.sibling_pairs()
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
            fsm = self._full_sib_matrix
            self._build_half_sib_matrix(mat_hs, pat_hs)
            hsm = self._half_sib_matrix
        # sib_all only needed at degree 4+ (1C1R, H1C1R, 1C2R subtract lists)
        if max_degree >= 4:
            sib_all = (
                np.concatenate([pairs["FS"][0], pairs["MHS"][0], pairs["PHS"][0]]),
                np.concatenate([pairs["FS"][1], pairs["MHS"][1], pairs["PHS"][1]]),
            )

        # ---- Degree 3 (kinship 1/16): GGP, HAv, GAv ----
        if max_degree >= 3:
            t0 = time.perf_counter()
            _ = self._A3  # pre-trigger

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
                A2_A3T = self._A2 @ self._A3.T

            def _extract_h1c() -> tuple[np.ndarray, np.ndarray]:
                # Use cached half-cousin pairs from _cousin_pairs() — already
                # identified as pairs sharing exactly 1 grandparent, with
                # sibling pairs excluded.
                return getattr(self, "_h1c_pairs_cache", (empty, empty))

            def _extract_1c1r() -> tuple[np.ndarray, np.ndarray]:
                P_full = A2_A3T.copy()
                P_full.setdiag(0)
                P_full.data[P_full.data < 2] = 0
                P_full.eliminate_zeros()
                return self._extract_from_sparse(
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
                A2_A3T = self._A2 @ self._A3.T

            def _extract_h1c1r() -> tuple[np.ndarray, np.ndarray]:
                P_half = A2_A3T.copy()
                P_half.setdiag(0)
                P_half.data[P_half.data != 1] = 0
                P_half.eliminate_zeros()
                return self._extract_from_sparse(
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
                P_full = self._A2 @ self._A4.T
                P_full.setdiag(0)
                P_full.data[P_full.data < 2] = 0
                P_full.eliminate_zeros()
                return self._extract_from_sparse(
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
        self._raw_pair_counts = {k: len(v[0]) for k, v in pairs.items()}

        # Restrict to active (sampled) individuals when a mask is set
        if self._sample_mask is not None:
            for k, (idx1, idx2) in pairs.items():
                if len(idx1) > 0:
                    mask = self._sample_mask[idx1] & self._sample_mask[idx2]
                    pairs[k] = (idx1[mask].astype(np.intp), idx2[mask].astype(np.intp))
                else:
                    pairs[k] = (empty, empty)
            logger.info(
                "Filtered to sample_mask: %s",
                ", ".join(f"{k}: {len(v[0])}" for k, v in pairs.items()),
            )

        # Remap graph row indices to caller-input row indices when a remap is set.
        # After this, pair indices are in caller-input coordinates regardless
        # of which constructor was used.
        if self._subsample_remap is not None:
            remap = self._subsample_remap
            for k, (idx1, idx2) in pairs.items():
                if len(idx1) > 0:
                    pairs[k] = (
                        remap[idx1].astype(np.intp),
                        remap[idx2].astype(np.intp),
                    )

        # Cache subsample-filtered counts so count_pairs(scope="subsample") is O(1).
        self._subsample_pair_counts = {k: len(v[0]) for k, v in pairs.items()}

        # Free all cached sparse matrices — only pair arrays and count caches
        # are needed after this point.
        for attr in ("_A", "_A2", "_A3", "_A4", "_A5", "_A2_shared", "_full_sib_matrix", "_half_sib_matrix"):
            self.__dict__.pop(attr, None)

        logger.info("extract_pairs total: %.3fs", time.perf_counter() - t_total)
        return pairs

    def count_pairs(self, max_degree: int = 2, scope: Literal["subsample", "full"] = "subsample") -> dict[str, int]:
        """Count all relationship categories.

        If ``extract_pairs()`` was already called on this instance, returns
        the matching cached counts (nearly free).  Otherwise runs
        ``extract_pairs()`` to compute all types up to *max_degree*.

        Args:
            max_degree: Maximum kinship degree to compute when extract_pairs
                has not yet been called on this instance.
            scope: ``"subsample"`` (default) returns counts that match
                ``extract_pairs`` output (mask-filtered, in caller-input
                coordinates).  ``"full"`` returns the pre-mask counts over
                the underlying graph — the cache-reuse fast path used when a
                full-pedigree summary is needed alongside subsample-restricted
                pairs.  For graphs not constructed via ``from_subsample`` the
                two scopes are equivalent.
        """
        if scope not in ("subsample", "full"):
            raise ValueError(f"scope must be 'subsample' or 'full', got {scope!r}")

        if not hasattr(self, "_raw_pair_counts"):
            self.extract_pairs(max_degree=max_degree)

        if scope == "full":
            return dict(self._raw_pair_counts)
        return dict(self._subsample_pair_counts)

    # ------------------------------------------------------------------
    # Alternative constructor
    # ------------------------------------------------------------------

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> PedigreeGraph:
        """Construct from a pandas ``DataFrame``.

        Explicit DataFrame entry point.  ``__init__`` also accepts
        DataFrames directly via type dispatch; this classmethod is
        provided for callers that want the intent in the call site.
        """
        return cls(df)

    @classmethod
    def from_arrays(
        cls,
        ids: np.ndarray,
        mothers: np.ndarray,
        fathers: np.ndarray,
        twins: np.ndarray | None = None,
        generation: np.ndarray | None = None,
        birth_year: np.ndarray | None = None,
    ) -> PedigreeGraph:
        """Construct a PedigreeGraph directly from numpy arrays.

        Used by hot-loop callers (PA-FGRS, external-tool exports) that
        don't have a ``pedigree.parquet`` DataFrame handy.  When
        *generation* is None, it is derived from the parent graph via a
        fixed-point sweep (founders = 0, offspring = max(parent_gen)+1).

        *birth_year* is optional; sentinel ``-1`` marks unknown.  When
        supplied, parent-child edges with both endpoints known are
        validated to satisfy ``child.birth_year >= parent.birth_year``.
        """
        ids_arr = np.asarray(ids)
        mothers_arr = np.asarray(mothers)
        fathers_arr = np.asarray(fathers)
        n = len(ids_arr)
        twins_arr = np.full(n, -1, dtype=np.int64) if twins is None else np.asarray(twins)

        # Generation column may be unknown here — fill a placeholder,
        # instantiate the graph (which remaps parents to row indices),
        # then derive depth from the already-remapped parent arrays.
        derive_generation = generation is None
        data: dict[str, np.ndarray] = {
            "id": ids_arr,
            "mother": mothers_arr,
            "father": fathers_arr,
            "twin": twins_arr,
            "sex": np.zeros(n, dtype=np.int8),
            "generation": (
                np.zeros(n, dtype=np.int32) if derive_generation else np.asarray(generation, dtype=np.int32)
            ),
        }
        if birth_year is not None:
            data["birth_year"] = np.asarray(birth_year)
        pg = cls(data)
        if derive_generation:
            pg.generation = _compute_depth(pg.mother, pg.father, n)
        return pg

    @classmethod
    def from_subsample(
        cls,
        full_pedigree: dict[str, np.ndarray] | pd.DataFrame,
        df: dict[str, np.ndarray] | pd.DataFrame,
    ) -> PedigreeGraph:
        """Construct a graph over *full_pedigree*, restricted to *df*.

        Builds the full-pedigree graph (so multi-hop relationships are
        detected through ancestors absent from *df*), then sets a private
        sample mask + remap so that ``extract_pairs`` returns indices into
        *df* (filtered to pairs where both endpoints are in *df*).

        Args:
            full_pedigree: Complete pedigree as ``dict[str, np.ndarray]`` or
                pandas ``DataFrame``.
            df: Subsample of *full_pedigree* in the same form.  Must have
                unique IDs and each ID must appear in *full_pedigree*'s
                ``id`` column.  Empty *df* is permitted and yields a graph
                whose ``extract_pairs`` returns empty arrays.

        Raises:
            ValueError: if *df* has duplicate IDs, or if any ID in *df* is
                missing from *full_pedigree*.
        """
        df_arrays = _coerce_to_array_dict(df)
        df_ids = np.asarray(df_arrays["id"])
        if len(df_ids) != len(np.unique(df_ids)):
            dup_count = len(df_ids) - len(np.unique(df_ids))
            raise ValueError(f"from_subsample: df has {dup_count} duplicate id(s)")

        full_arrays = _coerce_to_array_dict(full_pedigree)
        full_ids = np.asarray(full_arrays["id"])
        if len(df_ids) > 0:
            in_full = np.isin(df_ids, full_ids)
            if not in_full.all():
                missing = df_ids[~in_full]
                preview = missing[:10].tolist()
                raise ValueError(
                    f"from_subsample: {len(missing)} id(s) in df not present in "
                    f"full_pedigree (first {min(len(missing), 10)}: {preview})"
                )

        pg = cls(full_arrays)

        if len(df_ids) == 0:
            # Empty subsample → mask filters everything; remap unused.
            pg._sample_mask = np.zeros(len(full_ids), dtype=bool)
            pg._subsample_remap = np.full(len(full_ids), -1, dtype=np.intp)
            return pg

        pg._sample_mask = np.isin(full_ids, df_ids)

        # Build full-graph-row → df-row table.
        max_id = int(max(int(full_ids.max()), int(df_ids.max()))) + 1
        id_to_df_row = np.full(max_id, -1, dtype=np.intp)
        id_to_df_row[df_ids] = np.arange(len(df_ids), dtype=np.intp)
        pg._subsample_remap = id_to_df_row[full_ids]

        return pg

    # ------------------------------------------------------------------
    # Sparse kinship, inbreeding, and exact pair kinship
    # ------------------------------------------------------------------

    def kinship_matrix(
        self,
        min_kinship: float = 0.0,
        max_degree: int | None = None,
    ) -> sp.csc_matrix:
        """Build and cache the full-symmetric sparse kinship matrix (φ-scale).

        Diagonal is ``(1 + F_i) / 2``; MZ off-diagonals are set to the
        corresponding twin's self-kinship (= 0.5 without inbreeding).

        Args:
            min_kinship: kernel-side pruning threshold.  Off-diagonal
                entries with ``value <= min_kinship`` are dropped during
                DP propagation.  Diagonal always kept.
            max_degree: convenience shortcut for ``min_kinship``.  Sets
                the threshold to ``0.5 ** (max_degree + 1) - 1e-9`` so
                that the boundary kinship (e.g. 1/16 at degree 3) is
                retained.  The stricter of the two applies.

        Returns:
            ``scipy.sparse.csc_matrix`` cached under the resolved
            ``min_kinship`` in ``self._kinship_cache``.
        """
        if max_degree is not None:
            deg_threshold = 0.5 ** (max_degree + 1) - 1e-9
            min_kinship = max(min_kinship, deg_threshold)

        key = float(min_kinship)
        cached = self._kinship_cache.get(key)
        if cached is not None:
            return cached

        t0 = time.perf_counter()
        indptr, indices, data = _build_kinship_csc(
            self.n,
            self.mother,
            self.father,
            self.twin,
            self.generation,
            min_kinship,
        )
        K = sp.csc_matrix((data, indices, indptr), shape=(self.n, self.n))
        self._kinship_cache[key] = K

        logger.info(
            "kinship_matrix: n=%d, nnz=%d, min_kinship=%.4g, %.2fs",
            self.n,
            K.nnz,
            min_kinship,
            time.perf_counter() - t0,
        )
        return K

    def per_gen_mean_kinship(self, min_kinship: float = 0.0) -> np.ndarray:
        """Per-generation mean kinship θ̄_g, computed without building K.

        Streams the DP row storage directly through
        :func:`_per_gen_mean_kinship_from_dp`; no CSC matrix is ever
        materialized.  This is the K-free path for
        :func:`pedigree_graph.ne_coancestry` and scales to N where
        ``kinship_matrix()`` would OOM (nnz beyond int32, or simply too
        large for memory).

        Result is cached under ``min_kinship`` in
        ``self._theta_per_gen_cache``.  Mirrors ``kinship_matrix()``'s
        cache keying so a caller requesting a different threshold gets a
        fresh computation rather than a stale θ̄.

        Args:
            min_kinship: kernel-side pruning threshold.  Off-diagonal
                entries with ``value <= min_kinship`` are dropped during
                DP propagation.  Diagonal always kept.

        Returns:
            ``np.ndarray`` of dtype float64, length ``g_max + 1``, with
            mean within-cohort kinship per generation (NaN for cohorts
            with fewer than 2 non-twin members).
        """
        key = float(min_kinship)
        cached = self._theta_per_gen_cache.get(key)
        if cached is not None:
            return cached

        t0 = time.perf_counter()
        # If K is already cached at this threshold, derive θ̄ from the
        # existing sparse matrix (seconds) rather than re-running the DP
        # (tens of seconds + multi-GB scratch).  Streaming intent is "K-free
        # at scale"; reusing a cached K when one happens to exist is
        # complementary.
        K = self._kinship_cache.get(key)
        if K is not None:
            theta = _per_gen_mean_kinship(
                K, np.asarray(self.generation), np.asarray(self.twin),
            )
        else:
            theta = _compute_theta_per_gen(
                self.n,
                self.mother,
                self.father,
                self.twin,
                self.generation,
                min_kinship,
            )
        self._theta_per_gen_cache[key] = theta

        logger.info(
            "per_gen_mean_kinship: n=%d, g_max=%d, min_kinship=%.4g, %.2fs",
            self.n,
            theta.shape[0] - 1,
            min_kinship,
            time.perf_counter() - t0,
        )
        return theta

    def compute_inbreeding(self) -> np.ndarray:
        """Return the inbreeding coefficient *F* per individual.

        Computed via Meuwissen & Luo (1992) ancestor-walking on the LDL'
        decomposition of the numerator relationship matrix; result is
        cached on the graph (``self._inbreeding``).  MZ-naive — twins are
        treated as full sibs.  Differs from the matrix-derived F only
        for individuals whose only inbreeding path runs through both
        members of an MZ twin pair as ancestors.
        """
        if self._inbreeding is None:
            if self._depth is None:
                self._depth = _compute_depth(self.mother, self.father, self.n)
            self._inbreeding = _compute_F_meuwissen_luo(
                self.mother, self.father, self._depth, self.n
            )
        return self._inbreeding

    def compute_pair_kinship(
        self,
        pairs: dict[str, tuple[np.ndarray, np.ndarray]],
    ) -> dict[str, np.ndarray]:
        """Exact kinship per extracted pair.

        Fast path (non-inbred, MZ-free): returns ``PAIR_KINSHIP[code]``
        for each pair (MZ case handled via the registry's special case
        = 0.5).  The MZ-free guard is required because ``compute_inbreeding``
        is MZ-naive: an individual whose only inbreeding is via MZ
        co-coalescence has ML F = 0 but the matrix kinship still records
        ``K[twin_a, twin_b] = (1 + F)/2``.  Without the guard, the fast
        path would return ``PAIR_KINSHIP[code]`` defaults that miss this
        inbred-MZ-coalescence case.

        Inbred path: reads values from the cached full kinship matrix.
        MZ-correct because the kernel sets twin off-diagonals to the
        inbred self-kinship ``(1 + F) / 2``.

        Call *after* :meth:`extract_pairs`.
        """
        F = self.compute_inbreeding()

        if np.all(F == 0) and not np.any(self.twin >= 0):
            return {code: np.full(len(idx1), PAIR_KINSHIP.get(code, 0.0)) for code, (idx1, _) in pairs.items()}

        K = self.kinship_matrix(min_kinship=0.0).tocsr()
        result: dict[str, np.ndarray] = {}
        for code, (idx1, idx2) in pairs.items():
            if len(idx1) == 0:
                result[code] = np.array([], dtype=np.float64)
                continue
            result[code] = np.array(K[idx1, idx2]).ravel()
        return result
