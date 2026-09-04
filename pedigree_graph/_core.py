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
    "FrameLike",
    "PedigreeGraph",
    "RelType",
]

import logging
import time
from functools import cached_property
from typing import Literal

import numpy as np
import scipy.sparse as sp

from pedigree_graph._effective_size import _per_gen_mean_kinship
from pedigree_graph._errors import PedigreeValidationError, ResourceError
from pedigree_graph._frames import (
    FrameLike,
    _coerce_to_array_dict,
)
from pedigree_graph._input import (
    _map_ids_to_rows,
    parse_pedigree_input,
    validate_id_field,
)
from pedigree_graph._kinship_kernel import (
    _build_kinship_csc,
    _check_topological,
    _compute_depth,
    _compute_F_meuwissen_luo,
    _compute_theta_per_gen,
)
from pedigree_graph._kinship_pairwise import pairwise_kinship
from pedigree_graph._lineage_kernel import (
    _compute_n_ancestors,
    _compute_n_descendants,
)
from pedigree_graph._pair_extractor import MatrixPairExtractor
from pedigree_graph._pair_utils import pairs_from_groups
from pedigree_graph._registry import (
    PAIR_KINSHIP,
    REL_REGISTRY,
    RelType,
    _validate_max_degree,
)
from pedigree_graph._streaming_counter import StreamingPairCounter

logger = logging.getLogger(__name__)


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


class PedigreeGraph:
    """Parent→child DAG for efficient relationship queries.

    Each individual is a vertex whose index equals its row index in the
    input.  Sparse CSR matrices encode parent-child edges for O(nnz)
    relationship extraction via matrix products.

    Args:
        data: Either a ``dict[str, np.ndarray]`` or any :class:`FrameLike`
            table — pandas and polars DataFrames both satisfy the structural
            protocol, and neither library is imported by this package.
            ``id``, ``mother``, and ``father`` are required; ``twin``,
            ``sex``, ``generation``, and ``birth_year`` are optional and
            other keys are ignored.  :mod:`pedigree_graph._input` owns every
            guard on the values.
    """

    # 0.8.0-DELETE: the loose dict-or-frame constructor; 0.8 constructs through
    # from_frame / from_arrays only.
    def __init__(self, data: dict[str, np.ndarray] | FrameLike) -> None:
        parsed = parse_pedigree_input(data)
        self._input = parsed
        n = parsed.n_individuals
        self.n = n

        # Subsample state — set only by from_subsample. When _sample_mask is
        # set, extract_pairs filters to pairs where both endpoints are active.
        # When _subsample_remap is set, extract_pairs additionally remaps
        # graph row indices to caller-input row indices.  _subsample_inverse is
        # the inverse table (caller/df row -> graph row); compute_pair_kinship
        # uses it to map caller-coordinate pairs back onto the full-graph
        # kinship matrix, which is always built in graph coordinates.
        self._sample_mask: np.ndarray | None = None
        self._subsample_remap: np.ndarray | None = None
        self._subsample_inverse: np.ndarray | None = None

        # Lazy kinship cache — populated by kinship_matrix(); keyed by
        # the resolved min_kinship threshold.
        self._kinship_cache: dict[float, sp.csc_matrix] = {}
        # Lazy per-generation mean kinship cache — populated by
        # per_gen_mean_kinship(); keyed by min_kinship so callers using a
        # non-default threshold do not get a stale θ̄.
        self._theta_per_gen_cache: dict[float, np.ndarray] = {}
        # Lazy pair-count cache — populated by extract_pairs() and
        # count_pairs_streaming(); keyed on (engine, max_degree,
        # min_kinship).  Value is a (raw_counts, subsample_counts) pair
        # so the scope='full' and scope='subsample' fast paths read the
        # same entry.  The streaming engine fixes min_kinship=0.0 (its
        # scalar formulas don't prune on kinship); the matrix engine
        # uses the requested min_kinship.
        self._pair_count_cache: dict[tuple[str, int, float], tuple[dict[str, int], dict[str, int]]] = {}
        self._inbreeding: np.ndarray | None = None
        # Topological depth recomputed from edges; user-supplied
        # ``self.generation`` may be sparse/skipped/post-filtered and is
        # not safe as a substitute for the ML F kernel.
        self._depth: np.ndarray | None = None
        # Lazy lineage caches populated by compute_n_ancestors() and
        # compute_n_descendants().
        self._n_ancestors: np.ndarray | None = None
        self._n_descendants: np.ndarray | None = None
        # Lazy cache of known-parent edge filters, keyed "mother"/"father"
        # (see _known_parent_edges_for); shared by the overlapping-generation
        # diagnostics so the full-pedigree edge scan runs once per side.
        self._known_parent_edges_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        self._ids = parsed.ids
        # Original pedigree parent IDs (for sibling classification — valid
        # even when the parent isn't represented).
        self._orig_mother = parsed.mother_ids
        self._orig_father = parsed.father_ids
        self.mother = parsed.mother_rows
        self.father = parsed.father_rows
        self.twin = parsed.twin_rows
        self.birth_year: np.ndarray | None = parsed.birth_year
        # 0.8.0-DELETE: the 0.7.1 all-female sex default and depth-derived
        # generation fallback; the 0.8 properties report absent metadata instead.
        self.sex = parsed.sex if parsed.sex is not None else np.zeros(n, dtype=np.int8)
        self.generation = (
            parsed.generation if parsed.generation is not None else _compute_depth(self.mother, self.father, n)
        )

        # 0.8.0-DELETE: slice 1b routes arbitrary acyclic row order.
        if not _check_topological(self.mother, self.father, n):
            raise ValueError("PedigreeGraph requires topological order: parents must precede children")

        self._validate_birth_year_topology()

        # Build parent→child matrices using ALL available edges.
        # Each matrix is built independently so partial-pedigree data
        # (e.g. after subsampling) still contributes edges.
        self._build_parent_csr()

    def _validate_birth_year_topology(self) -> None:
        """Reject parent-child edges with child.birth_year < parent.birth_year.

        Only checks edges where both endpoints have known birth_year
        (sentinel ``-1`` skipped). Unknown-parent and unknown-child rows
        contribute no constraints.  The first violating edge in row order,
        for the first violating parent role, is reported.
        """
        if self.birth_year is None:
            return
        for parent_role in ("mother", "father"):
            edge_rows, diffs = self._known_parent_edges_for(parent_role)
            if diffs.size == 0:
                continue
            violations = diffs < 0
            if not violations.any():
                continue
            first = int(np.argmax(violations))
            child_row = int(edge_rows[first])
            parent_arr = self.mother if parent_role == "mother" else self.father
            parent_row = int(parent_arr[child_row])
            raise PedigreeValidationError(
                "birth_year_topology",
                f"birth_year topology violation: {parent_role}-child edge at row {child_row} "
                f"has child.birth_year below {parent_role}.birth_year",
                parent_role=parent_role,
                child_row=child_row,
                parent_row=parent_row,
                child_id=int(self._ids[child_row]),
                parent_id=int(self._ids[parent_row]),
                child_birth_year=int(self.birth_year[child_row]),
                parent_birth_year=int(self.birth_year[parent_row]),
                violation_count=int(violations.sum()),
            )

    def _ensure_parent_csr(self) -> None:
        """Idempotent (re)build of ``self._Am`` and ``self._Af``.

        No-op when both matrices are already present.  Called by
        ``__init__`` (initial build), by :meth:`extract_pairs` (rebuild
        after a prior call dropped them), and by
        :meth:`count_pairs_streaming` (same reason).  Single guarded
        helper avoids duplicating the ``hasattr`` check at each
        rebuild site.
        """
        if hasattr(self, "_Am") and hasattr(self, "_Af"):
            return
        self._build_parent_csr()

    def _build_parent_csr(self) -> None:
        """Unconditionally (re)build ``self._Am`` and ``self._Af``.

        Re-called by :meth:`count_pairs_streaming` after
        :meth:`extract_pairs` drops the matrices to free memory.  Keeping
        one canonical builder prevents the dtype/shape contract from
        drifting between constructor and re-builder.
        """
        n = self.n
        m_idx = np.where(self.mother >= 0)[0]
        f_idx = np.where(self.father >= 0)[0]
        self._Am = sp.csr_matrix(
            (np.ones(len(m_idx), dtype=np.int32), (m_idx, self.mother[m_idx])),
            shape=(n, n),
        )
        self._Af = sp.csr_matrix(
            (np.ones(len(f_idx), dtype=np.int32), (f_idx, self.father[f_idx])),
            shape=(n, n),
        )

    def _known_parent_edges_for(
        self,
        parent_label: Literal["mother", "father"],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Cached :func:`_known_parent_edges` lookup by parent label.

        ``_validate_birth_year_topology`` (in __init__),
        ``generation_interval``, ``_cohort_utils.eligible_cohort_range``,
        and ``_hill_age_table`` all call the same edge-filter and
        age-diff computation against the same arrays — at scale that's
        eight passes over the full pedigree.  Cached per-graph keyed on
        ``"mother"``/``"father"`` (cache initialised in ``__init__``).
        Bypassed when ``birth_year is None`` (the underlying helper still
        runs but the result is small).
        """
        hit = self._known_parent_edges_cache.get(parent_label)
        if hit is not None:
            return hit
        parent_arr = self.mother if parent_label == "mother" else self.father
        if self.birth_year is None:
            result = (np.array([], dtype=np.intp), np.array([], dtype=np.int32))
        else:
            result = _known_parent_edges(parent_arr, self.birth_year)
        self._known_parent_edges_cache[parent_label] = result
        return result

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

        _, diffs_m = self._known_parent_edges_for("father")
        _, diffs_f = self._known_parent_edges_for("mother")
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
            ones = np.ones(len(hs1), dtype=np.int32)
            H = sp.csr_matrix((ones, (hs1, hs2)), shape=(self.n, self.n))
            self._half_sib_matrix = H + H.T
        else:
            self._half_sib_matrix = sp.csr_matrix((self.n, self.n))

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
        one parent in the sample still contributes a PO pair.  Shared
        graph-data accessor: read by both the matrix extractor and the
        experimental BFS counter.
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
            full_sib = pairs_from_groups(bk_idx, family_key)
        else:
            full_sib = empty

        # --- Maternal half sibs: all pairs sharing known mother, minus full-sib pairs ---
        has_mother = nt_mother >= 0
        m_idx = nt_idx[has_mother]
        m_mother = nt_mother[has_mother]
        if len(m_idx) >= 2:
            mat_all = pairs_from_groups(m_idx, m_mother)
            mat_hs = self._subtract_pairs(mat_all, full_sib)
        else:
            mat_hs = empty

        # --- Paternal half sibs: all pairs sharing known father, minus full-sib pairs ---
        has_father = nt_father >= 0
        f_idx = nt_idx[has_father]
        f_father = nt_father[has_father]
        if len(f_idx) >= 2:
            pat_all = pairs_from_groups(f_idx, f_father)
            pat_hs = self._subtract_pairs(pat_all, full_sib)
        else:
            pat_hs = empty

        # Build full-sib sparse matrix for _avuncular_pairs and collateral methods
        sib1, sib2 = full_sib
        if len(sib1) > 0:
            ones = np.ones(len(sib1), dtype=np.int32)
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
        Encodes each pair as lo * max_id + hi, then tests membership by
        sorting the remove keys and binary-searching the candidates —
        much faster than ``np.unique`` + ``np.isin`` at scale, and
        duplicate remove keys are harmless (same rationale as
        ``extract_from_sparse``).
        """
        a1, a2 = all_pairs
        r1, r2 = remove_pairs

        if len(a1) == 0:
            return all_pairs
        if len(r1) == 0:
            return all_pairs

        # int64 cast required: max_id² overflows int32
        max_id = int(max(a1.max(), a2.max(), r1.max(), r2.max())) + 1
        remove_keys = np.sort(r1.astype(np.int64) * max_id + r2.astype(np.int64))
        all_keys = a1.astype(np.int64) * max_id + a2.astype(np.int64)

        pos = np.searchsorted(remove_keys, all_keys)
        hit = pos < remove_keys.size
        hit[hit] = remove_keys[pos[hit]] == all_keys[hit]
        keep = ~hit

        return a1[keep].astype(np.intp), a2[keep].astype(np.intp)

    def extract_pairs(
        self,
        max_degree: int = 3,
        min_kinship: float = 0.0,
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Extract all relationship categories.

        Returned indices are always in caller-input coordinates:

        - ``__init__(df)``: indices into *df* rows.
        - ``from_subsample(full_pedigree, df)``: indices into *df* rows;
          pairs are filtered to those with both endpoints in *df*.
        - ``from_arrays(ids, ...)``: positions in the input *ids* array.

        Args:
            max_degree: Maximum kinship degree to extract (0-5). Includes
                relationship categories whose registry degree is <= this
                cutoff: 0=MZ only, 1=parent-offspring/full-sib, 2 adds
                half-sibs/grandparent/avuncular, 3 adds 1st cousins and
                other degree-3 categories, and 5 reaches 2nd cousins.
                Higher degrees require more
                expensive matrix products.
            min_kinship: Skip pair types with kinship coefficient below this
                threshold. E.g., 0.125 skips 1st cousins (0.0625) and 2nd
                cousins (0.016), avoiding their expensive sparse products.

        Returns:
            Dict mapping relationship code to (idx1, idx2) row-index arrays.

        Delegates extraction to :class:`MatrixPairExtractor` (a read-only
        collaborator); this wrapper persists the resulting counts and
        releases the transient adjacency/sibling matrices.  See ADR 0002.
        """
        max_degree = _validate_max_degree(max_degree)
        pairs, raw_counts, sub_counts = MatrixPairExtractor(self).extract(max_degree, min_kinship)
        self._pair_count_cache[("matrix", int(max_degree), float(min_kinship))] = (
            raw_counts,
            sub_counts,
        )
        self._release_pair_matrices()
        return pairs

    def _release_pair_matrices(self) -> None:
        """Drop the transient adjacency / sibling matrices built for pair work.

        Only the pair arrays and count caches are needed after an
        extraction; the cached sparse matrices can be large, so they are
        released here.  Idempotent — missing attributes are ignored.
        """
        for attr in ("_A", "_A2", "_A3", "_A4", "_A5", "_A2_shared", "_full_sib_matrix", "_half_sib_matrix"):
            self.__dict__.pop(attr, None)

    def count_pairs(self, max_degree: int = 3, scope: Literal["subsample", "full"] = "subsample") -> dict[str, int]:
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
        max_degree = _validate_max_degree(max_degree)

        key = ("matrix", int(max_degree), 0.0)
        entry = self._pair_count_cache.get(key)
        if entry is None:
            self.extract_pairs(max_degree=max_degree)
            entry = self._pair_count_cache[key]
        raw, sub = entry
        return dict(raw) if scope == "full" else dict(sub)

    def count_pairs_streaming(
        self,
        max_degree: int = 3,
        scope: Literal["subsample", "full"] = "full",
    ) -> dict[str, int]:
        """Memory-bounded relationship pair counts via pure scalar arithmetic.

        Computes counts via per-anchor ``C(k, 2)`` sums and lineal-edge
        ``.nnz`` reads.  No pair-key arrays are materialized; peak
        memory is O(N) regardless of pedigree density.  Verified to
        run in seconds on stallion-heavy 783K-row livestock pedigrees
        that OOM the matrix and BFS engines.

        The scalar path is **full-graph only** — the cache slot it
        populates holds identical raw/subsample dicts.  On graphs built
        via :meth:`from_subsample`, ``scope='subsample'`` raises
        ``NotImplementedError``; use :meth:`count_pairs` for
        subsample-restricted counts.

        **Precision contract** (single source of truth:
        ``REL_PLAN`` / :func:`streaming_exact_codes` in ``_registry``):

        - Exact (bit-identical to :meth:`count_pairs` on every input)
          for the lineal and sibling codes — exactly the set returned by
          :func:`~pedigree_graph._registry.streaming_exact_codes`, the
          single source of truth (ADR 0003).  The codes are not
          re-enumerated here so the docstring cannot drift from the registry.
        - Approximate for the remaining cousin / collateral codes
          (:func:`~pedigree_graph._registry.streaming_approximate_codes`).
          The scalar formulas assume each individual has the full
          complement of known grandparents at the relevant depth and
          diverge from the matrix engine on:
          (a) shallow pedigrees where many founders have unknown
              grandparents (constants like ``4*FS`` over-subtract);
              ``H1C`` in particular may clamp to ``0`` on pedigrees
              with depth ≤ 3.
          (b) inbred input (sib-mating creates pairs with multi-anchor
              contributions that constants don't capture).
          (c) twin-having pedigrees (twin individuals' children
              contribute to cousin sums in ways the formulas miss).
        - On deep livestock pedigrees (depth 5+, low inbreeding) the
          scalar formulas are accurate to better than 1%.  The
          horse-pedigree benchmark (N=783K, mean F=0.007) completes
          in ~5 seconds with peak RSS ~730 MB.

        Returns a dict containing all 23 codes; codes above
        ``max_degree`` are ``0``.  Populates
        ``self._pair_count_cache[("streaming", max_degree, 0.0)]`` and, like
        :meth:`extract_pairs`, releases the transient adjacency powers it
        builds before returning (:meth:`_release_pair_matrices`); a later
        pair-work call rebuilds them lazily via :meth:`_ensure_parent_csr`.

        Args:
            max_degree: 0-5; default 3 (matches :meth:`count_pairs`).
                Includes relationship categories whose registry degree is
                <= this cutoff: 0=MZ only, 1=parent-offspring/full-sib,
                2 adds half-sibs/grandparent/avuncular, 3 adds 1st
                cousins and other degree-3 categories, and 5 reaches 2nd
                cousins.
            scope: ``"full"`` (default) or ``"subsample"``.  On
                non-subsample graphs the two are equivalent.  On
                ``from_subsample`` graphs, ``"subsample"`` raises
                ``NotImplementedError``.
        """
        if scope not in ("subsample", "full"):
            raise ValueError(f"scope must be 'subsample' or 'full', got {scope!r}")
        max_degree = _validate_max_degree(max_degree)
        if scope == "subsample" and self._sample_mask is not None:
            raise NotImplementedError(
                "count_pairs_streaming(scope='subsample') is not supported on "
                "graphs constructed via from_subsample; the scalar path is "
                "full-graph only.  Use count_pairs() for subsample-restricted "
                "counts, or call count_pairs_streaming(scope='full') for the "
                "underlying full-pedigree counts.",
            )

        key = ("streaming", int(max_degree), 0.0)
        entry = self._pair_count_cache.get(key)
        if entry is not None:
            raw, _sub = entry
            return dict(raw)

        counts = StreamingPairCounter(self).count(max_degree)
        # Streaming is full-graph only, so raw and subsample slots coincide.
        self._pair_count_cache[key] = (dict(counts), dict(counts))
        # Release the transient adjacency powers the streaming counter built
        # (_A…_A5), exactly as extract_pairs does — the counts are cached and
        # nothing reads the matrices after this point.  Without it they stay
        # resident for the graph's lifetime and inflate any later
        # inbreeding/Ne work on the same graph.  See issue #4.
        self._release_pair_matrices()
        return dict(counts)

    # ------------------------------------------------------------------
    # Alternative constructor
    # ------------------------------------------------------------------

    @classmethod
    # 0.8.0-DELETE: renamed to from_frame in 0.8.
    def from_dataframe(cls, df: FrameLike) -> PedigreeGraph:
        """Construct from a DataFrame (any :class:`FrameLike` table).

        Explicit DataFrame entry point — pandas and polars frames both
        satisfy the structural protocol.  ``__init__`` also accepts frames
        directly via type dispatch; this classmethod is provided (and kept
        as a compatibility name) for callers that want the intent in the
        call site.
        """
        return cls(df)

    @classmethod
    # 0.8.0-DELETE: the positional 0.7.1 signature, the generation fallback, and
    # the all-female sex default; 0.8's from_arrays is keyword-only.
    def from_arrays(
        cls,
        ids: np.ndarray,
        mothers: np.ndarray,
        fathers: np.ndarray,
        twins: np.ndarray | None = None,
        generation: np.ndarray | None = None,
        birth_year: np.ndarray | None = None,
        sex: np.ndarray | None = None,
    ) -> PedigreeGraph:
        """Construct a PedigreeGraph directly from numpy arrays.

        Used by hot-loop callers (PA-FGRS, external-tool exports) that
        don't have a ``pedigree.parquet`` DataFrame handy.  When
        *generation* is None, it is derived from the parent graph via a
        fixed-point sweep (founders = 0, offspring = max(parent_gen)+1).

        *birth_year* is optional; sentinel ``-1`` marks unknown.  When
        supplied, parent-child edges with both endpoints known are
        validated to satisfy ``child.birth_year >= parent.birth_year``.

        *sex* is optional and defaults to zeros (``int8``).  **Foot-gun
        warning**: the default makes every individual female (sex=0),
        which silently degenerates sex-aware estimators
        (:func:`ne_sex_ratio`, :func:`ne_variance_family_size`) to
        ``ne=None``.  Always pass ``sex=`` when constructing graphs that
        will feed into ``compute_all_ne``.  Kinship-only callers
        (relationship-pair extraction, GRMs, PA-FGRS) can ignore this —
        their consumers do not read ``pg.sex``.
        """
        ids_arr = np.asarray(ids)
        n = len(ids_arr)
        data: dict[str, np.ndarray] = {
            "id": ids_arr,
            "mother": np.asarray(mothers),
            "father": np.asarray(fathers),
            "twin": np.full(n, -1, dtype=np.int64) if twins is None else np.asarray(twins),
        }
        for name, column in (("sex", sex), ("generation", generation), ("birth_year", birth_year)):
            if column is not None:
                data[name] = np.asarray(column)
        return cls(data)

    @classmethod
    # 0.8.0-DELETE: replaced by full.view(ids=...) (ADR 0006).
    def from_subsample(
        cls,
        full_pedigree: dict[str, np.ndarray] | FrameLike,
        df: dict[str, np.ndarray] | FrameLike,
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
        if "id" not in df_arrays:
            raise PedigreeValidationError(
                "missing_field",
                "from_subsample: df is missing the required 'id' field",
                field="id",
            )
        # Same id-column contract as the constructor (unique, nonnegative).
        df_ids = validate_id_field(df_arrays["id"])

        full_arrays = _coerce_to_array_dict(full_pedigree)
        full_ids = np.asarray(full_arrays["id"])
        if len(df_ids) > 0:
            in_full = np.isin(df_ids, full_ids)
            if not in_full.all():
                positions = np.flatnonzero(~in_full)
                first = int(positions[0])
                raise PedigreeValidationError(
                    "unknown_view_id",
                    f"from_subsample: {positions.size} id(s) in df are not present in full_pedigree",
                    id=int(df_ids[first]),
                    position=first,
                    missing_count=int(positions.size),
                )

        # Constructing the full graph validates the full pedigree's id column.
        pg = cls(full_arrays)
        full_ids = pg._ids  # validated int64 copy

        if len(df_ids) == 0:
            # Empty subsample → mask filters everything; remap unused.
            pg._sample_mask = np.zeros(len(full_ids), dtype=bool)
            pg._subsample_remap = np.full(len(full_ids), -1, dtype=np.intp)
            return pg

        pg._sample_mask = np.isin(full_ids, df_ids)

        # Build full-graph-row → df-row table via the same searchsorted remap
        # used for parent ids: target = df ids, query = full ids.
        pg._subsample_remap = _map_ids_to_rows(df_ids, full_ids, np.intp)

        # Build the inverse df-row → full-graph-row table so consumers that
        # need graph coordinates (e.g. compute_pair_kinship indexing the
        # full kinship matrix) can map caller-coordinate pairs back.
        graph_rows_in_sub = np.flatnonzero(pg._subsample_remap >= 0)
        inverse = np.full(len(df_ids), -1, dtype=np.intp)
        inverse[pg._subsample_remap[graph_rows_in_sub]] = graph_rows_in_sub
        pg._subsample_inverse = inverse

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

        Streams the DP row storage through :func:`_compute_theta_per_gen`;
        no CSC matrix is ever materialized.  Result is cached under
        ``min_kinship`` in ``self._theta_per_gen_cache``.  If the full
        kinship matrix is already cached at the same threshold, θ̄ is
        derived from that matrix instead of re-running the DP.

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
        K = self._kinship_cache.get(key)
        if K is not None:
            theta = _per_gen_mean_kinship(
                K,
                np.asarray(self.generation),
                np.asarray(self.twin),
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
        decomposition of the numerator relationship matrix, over the
        genome-node pedigree in which MZ co-twins share one node (ADR 0008).
        MZ-aware: equals ``2 * phi(i, i) - 1`` from :meth:`compute_pair_kinship`
        and from the :meth:`kinship_matrix` diagonal.  Result is cached on the
        graph (``self._inbreeding``).

        Raises:
            PedigreeValidationError: ``mz_nonreciprocal`` or
                ``mz_parent_mismatch`` if a represented MZ reference is not
                reciprocal or the co-twins do not share both parent rows.  An
                absent co-twin (``twin == -1``, e.g. outside a subsample) is
                not an MZ pair and is fine.
        """
        if self._inbreeding is None:
            self._check_mz_invariant()
            if self._depth is None:
                self._depth = _compute_depth(self.mother, self.father, self.n)
            self._inbreeding = _compute_F_meuwissen_luo(self.mother, self.father, self.twin, self._depth, self.n)
        return self._inbreeding

    def _check_mz_invariant(self) -> None:
        """Reject MZ references that are not reciprocal, two-member, parent-identical."""
        rows = np.flatnonzero(self.twin >= 0)
        if rows.size == 0:
            return
        partner = self.twin[rows]
        nonreciprocal = self.twin[partner] != rows
        mismatched = {
            "mother": self.mother[partner] != self.mother[rows],
            "father": self.father[partner] != self.father[rows],
        }
        bad = nonreciprocal | mismatched["mother"] | mismatched["father"]
        if not bad.any():
            return
        first = int(np.flatnonzero(bad)[0])
        row = int(rows[first])
        twin_row = int(partner[first])
        located = {"row": row, "id": int(self._ids[row]), "twin_id": int(self._ids[twin_row])}
        if nonreciprocal[first]:
            raise PedigreeValidationError(
                "mz_nonreciprocal",
                f"MZ reference at row {row} is not reciprocal; compute_inbreeding requires "
                "MZ pairs to be reciprocal, two-member, and parent-identical",
                **located,
            )
        raise PedigreeValidationError(
            "mz_parent_mismatch",
            f"MZ co-twins at rows {row} and {twin_row} do not share both parents",
            parent_roles=tuple(role for role, flags in mismatched.items() if flags[first]),
            **located,
        )

    def compute_n_descendants(self) -> np.ndarray:
        """Per-individual descendant count, **path-count semantics**.

        ``n_desc[v]`` counts (v, w) walks down the DAG, not unique
        descendants.  Equivalent to unique counts in non-inbred
        pedigrees; over-counts by the inbreeding rate where marriage
        loops give a descendant multiple ancestor paths to v.  Matches
        the convention used for GP / Av / 1C pair counts.

        Returns an ``int32`` array of length ``self.n``, cached on the
        graph (``self._n_descendants``).

        Raises:
            ResourceError: ``arithmetic_overflow`` if any per-individual path
                count exceeds ``np.iinfo(np.int32).max``.  The kernel accumulates in
                ``int64``; the cast to ``int32`` happens here after a
                bounds check so deeply inbred / loop-heavy pedigrees
                cannot silently wrap.
        """
        if self._n_descendants is None:
            n_desc64 = _compute_n_descendants(
                self.mother,
                self.father,
                self.n,
            )
            if n_desc64.size and int(n_desc64.max()) > np.iinfo(np.int32).max:
                raise ResourceError(
                    "arithmetic_overflow",
                    "compute_n_descendants: at least one path count exceeds "
                    f"int32 max ({np.iinfo(np.int32).max:,}); the pedigree is "
                    "too inbred / loop-heavy for the int32-cached output.  "
                    "Inspect the int64 kernel output via "
                    "pedigree_graph._lineage_kernel._compute_n_descendants "
                    "if larger values are required.",
                    operation="compute_n_descendants",
                    dtype="int32",
                )
            self._n_descendants = n_desc64.astype(np.int32)
        return self._n_descendants

    def compute_n_ancestors(self) -> np.ndarray:
        """Per-individual distinct strict-ancestor count.

        ``n_anc[v]`` is the number of unique ancestors of v — an
        ancestor reachable through multiple paths (loops introduced by
        inbreeding) is counted once, contrasting with the path-count
        semantics of :meth:`compute_n_descendants`.

        Returns an ``int32`` array of length ``self.n``, cached on the
        graph (``self._n_ancestors``).  Backed by a sparse boolean
        transitive closure of the parent graph; memory scales with
        ``sum_i n_anc[i]``, so very deep / very large pedigrees may
        need a future retirement-style DP variant.
        """
        if self._n_ancestors is None:
            self._n_ancestors = _compute_n_ancestors(
                self.mother,
                self.father,
                self.n,
            )
        return self._n_ancestors

    def compute_pair_kinship(
        self,
        pairs: dict[str, tuple[np.ndarray, np.ndarray]],
    ) -> dict[str, np.ndarray]:
        """Exact kinship for each requested pair.

        Returns ``{code: float64 array}`` positionally aligned to the input
        ``pairs[code]`` (orientation preserved).  Always exact — there is **no**
        nominal-lookup fast path, because the nominal ``PAIR_KINSHIP[code]``
        value is wrong for any pair related through *multiple* lineages even
        without inbreeding (e.g. double first cousins have ``phi = 0.125``, not
        the single-path ``1C`` value ``0.0625``).  Exact pairwise kinship can
        therefore exceed the nominal code value because of inbreeding, MZ
        co-coalescence, or multiple relationship paths.

        Computation:

        * if the exact full matrix ``kinship_matrix(0.0)`` is already cached,
          sample it directly (it is symmetric, so input orientation is
          irrelevant) — no ``.tocsr()`` duplication;
        * otherwise compute exact kinship for *only* the requested pairs via the
          direct memoized recurrence in :mod:`pedigree_graph._kinship_pairwise`,
          never materializing the ``n x n`` matrix.

        ``compute_inbreeding`` is **not** consulted: the recurrence derives each
        ``F`` as ``phi(mother, father)`` exactly as the matrix DP does, and all
        three agree (ADR 0008).  Accepts arbitrary code keys, not only ``REL_REGISTRY`` ones.

        Call *after* :meth:`extract_pairs`.
        """
        # extract_pairs returns caller-input coordinates on from_subsample
        # graphs, but kinship is computed in full-graph coordinates.  Map back
        # through the inverse remap before indexing/recurring.
        inverse = self._subsample_inverse
        result: dict[str, np.ndarray] = {}
        graph_pairs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for code, (idx1, idx2) in pairs.items():
            if len(idx1) == 0:
                result[code] = np.array([], dtype=np.float64)
                continue
            a = np.asarray(idx1)
            b = np.asarray(idx2)
            if inverse is not None:
                a = inverse[a]
                b = inverse[b]
            graph_pairs[code] = (a, b)

        if not graph_pairs:
            return result

        # Reuse the exact full matrix if it already exists (avoid recompute and
        # avoid duplicating it via .tocsr()).  K is symmetric.
        cached_k = self._kinship_cache.get(0.0)
        if cached_k is not None:
            for code, (a, b) in graph_pairs.items():
                result[code] = np.asarray(cached_k[a, b], dtype=np.float64).ravel()
            return result

        # Otherwise compute exact kinship for only the requested pairs.  Flatten
        # all codes into a single kernel call so overlapping ancestor-pairs are
        # memoized once across codes.
        codes = list(graph_pairs.keys())
        flat_a = np.concatenate([graph_pairs[c][0] for c in codes])
        flat_b = np.concatenate([graph_pairs[c][1] for c in codes])
        flat = pairwise_kinship(self.mother, self.father, self.twin, flat_a, flat_b)
        offset = 0
        for c in codes:
            count = len(graph_pairs[c][0])
            result[c] = flat[offset : offset + count]
            offset += count
        return result
