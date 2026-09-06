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
    "FrameLike",
    "PedigreeGraph",
]

import logging
import time
from functools import cached_property
from typing import TYPE_CHECKING, Literal, overload

import numpy as np
import scipy.sparse as sp

from pedigree_graph._compat import from_subsample as _from_subsample
from pedigree_graph._compat import legacy_count_pairs as _legacy_count_pairs
from pedigree_graph._compat import legacy_count_pairs_streaming as _legacy_count_pairs_streaming
from pedigree_graph._compat import legacy_extract_pairs as _legacy_extract_pairs
from pedigree_graph._effective_size import _per_gen_mean_kinship
from pedigree_graph._errors import PedigreeValidationError, ResourceError
from pedigree_graph._frames import FrameLike
from pedigree_graph._input import (
    parse_pedigree_arrays,
    parse_pedigree_input,
)
from pedigree_graph._kinship_kernel import (
    _compute_F_meuwissen_luo,
    _compute_theta_per_gen,
)
from pedigree_graph._kinship_matrix import PedigreeMatrixMethods
from pedigree_graph._kinship_pairwise import graph_pair_kinship, view_pair_kinship
from pedigree_graph._lineage_kernel import (
    _compute_n_ancestors,
    _compute_n_descendants,
)
from pedigree_graph._ne_common import _require_complete_generation_labels
from pedigree_graph._pair_extractor import relationship_pairs as _relationship_pairs
from pedigree_graph._pair_utils import pairs_from_groups, subtract_pairs
from pedigree_graph._properties import PedigreeProperties
from pedigree_graph._streaming_counter import estimate_relationship_counts as _estimate_relationship_counts
from pedigree_graph._topology import build_topology
from pedigree_graph._view import CoordinateToken, _build_view
from pedigree_graph.relationships import RelationshipCountResult

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from pedigree_graph._input import PedigreeInput
    from pedigree_graph._streaming_counter import CachedEstimate
    from pedigree_graph._topology import Topology
    from pedigree_graph._view import PedigreeView
    from pedigree_graph.relationships import RelationshipPairBlock, RelationshipPairs

logger = logging.getLogger(__name__)

# 0.8.0-DELETE: the 0.7.1 from_arrays positional order.
_LEGACY_ARRAY_ORDER = ("ids", "mothers", "fathers", "twins", "generation", "birth_year", "sex")


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


class PedigreeGraph(PedigreeProperties, PedigreeMatrixMethods):
    """Parent→child DAG for efficient relationship queries.

    Each individual is a vertex whose index equals its row index in the
    input.  Sparse CSR matrices encode parent-child edges for O(nnz)
    relationship extraction via matrix products.

    Build one with :meth:`from_frame` or :meth:`from_arrays`.  Both validate
    through :mod:`pedigree_graph._input` and hand the parsed result to
    :meth:`_from_input`, so every graph reaches the engine the same way.
    Neither invents a value: an absent optional column reads as absent.

    Args:
        data: 0.8.0-DELETE — the loose constructor's argument.  Either a
            ``dict[str, np.ndarray]`` or any :class:`FrameLike` table; pandas
            and polars DataFrames both satisfy the structural protocol, and
            neither library is imported by this package.  ``id``, ``mother``,
            and ``father`` are required; ``twin``, ``sex``, ``generation``,
            and ``birth_year`` are optional and other keys are ignored.
    """

    # 0.8.0-DELETE: the loose dict-or-frame constructor; 0.8 constructs through
    # from_frame / from_arrays only.
    def __init__(self, data: dict[str, np.ndarray] | FrameLike) -> None:
        self._initialize(parse_pedigree_input(data), legacy_defaults=True)

    @classmethod
    # 0.8.0-DELETE: the legacy_defaults keyword; 0.8 has no defaults to switch on.
    def _from_input(cls, parsed: PedigreeInput, *, legacy_defaults: bool) -> PedigreeGraph:
        """Build a graph over already-parsed input, the one path all constructors share.

        Args:
            parsed: Validated, owned input from :mod:`pedigree_graph._input`.
            legacy_defaults: Whether absent metadata takes its 0.7.1 default
                rather than reading as absent.  Only the compatibility entry
                points set it.

        Returns:
            The constructed graph.
        """
        graph = cls.__new__(cls)
        graph._initialize(parsed, legacy_defaults=legacy_defaults)
        return graph

    # 0.8.0-DELETE: the legacy_defaults keyword.
    def _initialize(self, parsed: PedigreeInput, *, legacy_defaults: bool) -> None:
        """Populate the graph's storage, caches, and parent matrices from *parsed*."""
        self._input = parsed
        self._legacy_defaults = legacy_defaults  # 0.8.0-DELETE
        n = parsed.n_individuals
        self.n = n  # 0.8.0-DELETE: renamed n_individuals.
        self._coordinate_token = CoordinateToken()

        self._legacy_view: PedigreeView | None = None  # 0.8.0-DELETE: set only by from_subsample.

        # Matrix caches are separated by operation and selector: complete,
        # closest-category, and propagation-pruned support are distinct
        # contracts even when two calls happen to produce the same structure.
        self._complete_kinship_cache: sp.csc_matrix | None = None
        self._relationship_kinship_cache: dict[tuple[str, object], sp.csc_matrix] = {}
        self._approximate_kinship_cache: dict[float, sp.csc_matrix] = {}
        # 0.8.0-DELETE: the generation-summary adapter consults the old
        # threshold-keyed cache.  Only the complete matrix is entered here;
        # corrected approximate-support values must not silently change the
        # old per_gen_mean_kinship(min_kinship>0) calculation.
        self._kinship_cache: dict[float, sp.csc_matrix] = {}
        # Lazy per-generation mean kinship cache — populated by
        # per_gen_mean_kinship(); keyed by min_kinship so callers using a
        # non-default threshold do not get a stale θ̄.
        self._theta_per_gen_cache: dict[float, np.ndarray] = {}
        # Matrix-engine counts written by extract_pairs(), keyed on
        # ("matrix", max_degree, min_kinship).  Value is a (raw_counts,
        # subsample_counts) pair so the scope='full' and scope='subsample'
        # fast paths of count_pairs read the same entry.
        self._pair_count_cache: dict[tuple[str, int, float], tuple[dict[str, int], dict[str, int]]] = {}
        # Keyed by max_degree; a hit is silent even if the entry clamped.
        self._estimate_cache: dict[int, CachedEstimate] = {}
        self._inbreeding: np.ndarray | None = None
        # Lazy lineage caches populated by compute_n_ancestors() and
        # compute_n_descendants().
        self._n_ancestors: np.ndarray | None = None
        self._n_descendants: np.ndarray | None = None
        # Lazy cache of known-parent edge filters, keyed "mother"/"father"
        # (see _known_parent_edges_for); shared by the overlapping-generation
        # diagnostics so the full-pedigree edge scan runs once per side.
        self._known_parent_edges_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        # 0.8.0-DELETE: the whole block, renamed to the ids / *_ids / *_rows
        # properties; the kernels and 0.7.1 consumers still read these names.
        self._ids = parsed.ids
        self._orig_mother = parsed.mother_ids
        self._orig_father = parsed.father_ids
        self.mother = parsed.mother_rows
        self.father = parsed.father_rows
        self.twin = parsed.twin_rows

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

    @cached_property
    def _topology(self) -> Topology:
        """Structural depth plus the private stable depth-major row order.

        Public coordinates are input rows in any acyclic order; the kernels
        that need parents to precede children run in this order and their
        outputs are mapped back.  Supplied generation labels never enter it.
        """
        return build_topology(self.depth)

    @property
    def _rows_are_topological(self) -> bool:
        """True when every parent row precedes its child row in graph space.

        Integer kernels whose only requirement is parents-before-children can
        then sweep the graph arrays directly, with no permutation and no
        scatter back.  The depth-major order is still used wherever pair and
        matrix kinship must peel in the same coordinates.
        """
        return self._input.rows_topological

    @cached_property
    def _topological_parents(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(mother, father, twin)`` rewritten into the private topological order."""
        topo = self._topology
        return (
            topo.to_topological(self.mother),
            topo.to_topological(self.father),
            topo.to_topological(self.twin),
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

    # 0.8.0-DELETE: public name; callers request FS / MHS / PHS from relationship_pairs.
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
            F = sp.csr_matrix((ones, (sib1, sib2)), shape=(self.n, self.n))
            self._full_sib_matrix = F + F.T
        else:
            self._full_sib_matrix = sp.csr_matrix((self.n, self.n))

        return full_sib, mat_hs, pat_hs

    # 0.8.0-DELETE: replaced by relationship_pairs (ADR 0006).
    def extract_pairs(
        self,
        max_degree: int = 3,
        min_kinship: float = 0.0,
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Extract all relationship categories with 0.7.1 semantics.

        Returned indices are always in caller-input coordinates:

        - ``__init__(df)``: indices into *df* rows.
        - ``from_subsample(full_pedigree, df)``: indices into *df* rows;
          pairs are filtered to those with both endpoints in *df*.
        - ``from_arrays(ids, ...)``: positions in the input *ids* array.

        Lineal and parent-offspring pairs are ``(descendant, ancestor)``;
        every collateral pair is ``(min row, max row)``.  Use
        :meth:`relationship_pairs` for role-oriented graph-space pairs.

        Args:
            max_degree: Maximum kinship degree to extract (0-5). Includes
                relationship categories whose registry degree is <= this
                cutoff: 0=MZ only, 1=parent-offspring/full-sib, 2 adds
                half-sibs/grandparent/avuncular, 3 adds 1st cousins and
                other degree-3 categories, and 5 reaches 2nd cousins.
            min_kinship: Skip pair types with kinship coefficient below this
                threshold. E.g., 0.125 skips 1st cousins (0.0625) and 2nd
                cousins (0.016), avoiding their expensive sparse products.

        Returns:
            Dict mapping relationship code to (idx1, idx2) row-index arrays.

        See :func:`pedigree_graph._compat.legacy_extract_pairs`.
        """
        return _legacy_extract_pairs(self, max_degree, min_kinship)

    def _release_pair_matrices(self) -> None:
        """Drop the transient adjacency / sibling matrices built for pair work.

        Only the pair arrays and count caches are needed after an
        extraction; the cached sparse matrices can be large, so they are
        released here.  Idempotent — missing attributes are ignored.
        """
        for attr in ("_A", "_A2", "_A3", "_A4", "_A5", "_A2_shared", "_full_sib_matrix", "_half_sib_matrix"):
            self.__dict__.pop(attr, None)

    # 0.8.0-DELETE: replaced by relationship_counts (ADR 0006).
    def count_pairs(self, max_degree: int = 3, scope: Literal["subsample", "full"] = "subsample") -> dict[str, int]:
        """Return the 0.7.1 matrix-engine counts, all 23 codes, ``0`` above the cutoff.

        See :func:`pedigree_graph._compat.legacy_count_pairs`.
        """
        return _legacy_count_pairs(self, max_degree, scope)

    # 0.8.0-DELETE: replaced by estimate_relationship_counts (ADR 0006).
    def count_pairs_streaming(
        self,
        max_degree: int = 3,
        scope: Literal["subsample", "full"] = "full",
    ) -> dict[str, int]:
        """Return the 0.7.1 scalar-estimate dict, all 23 codes, ``0`` above the cutoff.

        See :func:`pedigree_graph._compat.legacy_count_pairs_streaming`.
        """
        return _legacy_count_pairs_streaming(self, max_degree, scope)

    # ------------------------------------------------------------------
    # Alternative constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_frame(
        cls,
        frame: FrameLike | dict[str, np.ndarray],
        *,
        sex_encoding: str = "simace",
    ) -> PedigreeGraph:
        """Construct from a table of columns.

        Args:
            frame: Any :class:`FrameLike` table (pandas and polars frames both
                satisfy the structural protocol) or a ``dict[str, array-like]``.
                ``id``, ``mother``, and ``father`` are required; ``twin``,
                ``sex``, ``generation``, and ``birth_year`` are optional and
                other columns are ignored.
            sex_encoding: ``"simace"`` (``0`` female, ``1`` male, ``-1``
                unknown) or ``"plink"`` (``2`` female, ``1`` male, ``0``
                unknown).  Never inferred from the values.

        Returns:
            A graph whose absent optional columns read as absent: no sex
            default, no generation fallback.

        Raises:
            PedigreeValidationError: For any invalid field, duplicate id,
                shared parent id, cyclic parent reference, or broken MZ pair.
            ResourceError: ``pedigree_too_large`` beyond the int32 row capacity.
        """
        return cls._from_input(parse_pedigree_input(frame, sex_encoding=sex_encoding), legacy_defaults=False)

    @classmethod
    # 0.8.0-DELETE: renamed to from_frame in 0.8.
    def from_dataframe(cls, df: FrameLike) -> PedigreeGraph:
        """Construct from a DataFrame, the 0.7.1 name for :meth:`from_frame`."""
        return cls(df)

    @classmethod
    # 0.8.0-DELETE: replaced by full.view(ids=...) (ADR 0006).
    def from_subsample(
        cls,
        full_pedigree: dict[str, np.ndarray] | FrameLike,
        df: dict[str, np.ndarray] | FrameLike,
    ) -> PedigreeGraph:
        """Construct a graph over *full_pedigree* whose pair output covers only *df*.

        See :func:`pedigree_graph._compat.from_subsample`.
        """
        return _from_subsample(cls, full_pedigree, df)

    @classmethod
    def from_arrays(
        cls,
        # 0.8.0-DELETE: *legacy, the mothers/fathers/twins keywords, and the None
        # default on sex_encoding that detects it; 0.8 keeps the canonical form alone.
        *legacy: object,
        ids: object | None = None,
        mother_ids: object | None = None,
        father_ids: object | None = None,
        twin_ids: object | None = None,
        sex: object | None = None,
        generation: object | None = None,
        birth_year: object | None = None,
        sex_encoding: str | None = None,
        mothers: object | None = None,
        fathers: object | None = None,
        twins: object | None = None,
    ) -> PedigreeGraph:
        """Construct from separate columns, for callers with no table to hand.

        The canonical form is keyword-only and applies no defaults::

            PedigreeGraph.from_arrays(ids=ids, mother_ids=m, father_ids=f)

        The 0.7.1 form, recognised by ``mothers``/``fathers`` positionally or
        by keyword, keeps its generation fallback and its all-female sex
        default.  Naming both forms in one call, or neither, is a
        :exc:`TypeError`.

        Args:
            *legacy: The 0.7.1 positional columns, in the order ``ids``,
                ``mothers``, ``fathers``, ``twins``, ``generation``,
                ``birth_year``, ``sex``.
            ids: Row ids.
            mother_ids: Mother ids, ``-1`` or a host null when missing.
            father_ids: Father ids, as *mother_ids*.
            twin_ids: MZ co-twin ids, as *mother_ids*.
            sex: Sex codes in *sex_encoding*.
            generation: Generation labels, ``-1`` when unknown.
            birth_year: Birth years, ``-1`` when unknown.
            sex_encoding: ``"simace"`` (the default) or ``"plink"``.
                Canonical form only.
            mothers: 0.7.1 name for *mother_ids*.
            fathers: 0.7.1 name for *father_ids*.
            twins: 0.7.1 name for *twin_ids*.

        Returns:
            The constructed graph.

        Raises:
            TypeError: For a call that mixes the two forms, names neither, or
                omits a required column.
            PedigreeValidationError: As :meth:`from_frame`.
        """
        # 0.8.0-DELETE: everything from here to the end of the `if legacy_form`
        # block is 0.7.1 call-form dispatch; 0.8 keeps only the two lines after it.
        supplied: dict[str, object | None] = {
            "ids": ids,
            "mothers": mothers,
            "fathers": fathers,
            "twins": twins,
            "generation": generation,
            "birth_year": birth_year,
            "sex": sex,
        }
        if len(legacy) > len(_LEGACY_ARRAY_ORDER):
            raise TypeError(
                f"from_arrays() takes at most {len(_LEGACY_ARRAY_ORDER)} positional arguments, got {len(legacy)}"
            )
        for name, value in zip(_LEGACY_ARRAY_ORDER, legacy, strict=False):
            if supplied[name] is not None:
                raise TypeError(f"from_arrays() got multiple values for argument {name!r}")
            supplied[name] = value

        canonical = mother_ids is not None or father_ids is not None
        legacy_form = bool(legacy) or any(supplied[name] is not None for name in ("mothers", "fathers", "twins"))
        if canonical and legacy_form:
            raise TypeError(
                "from_arrays() takes either the canonical mother_ids=/father_ids= keywords or the "
                "0.7.1 mothers/fathers form, not both"
            )
        if not canonical and not legacy_form:
            raise TypeError("from_arrays() requires ids=, mother_ids=, and father_ids=")

        if legacy_form:
            required = ("ids", "mothers", "fathers")
            if any(supplied[name] is None for name in required):
                raise TypeError(f"from_arrays() requires {', '.join(required)}")
            if sex_encoding is not None or twin_ids is not None:
                raise TypeError(
                    "from_arrays() takes sex_encoding= and twin_ids= only with the canonical mother_ids= form; "
                    "the 0.7.1 form spells co-twins twins="
                )
            ids_arr = np.asarray(supplied["ids"])
            data: dict[str, np.ndarray] = {
                "id": ids_arr,
                "mother": np.asarray(supplied["mothers"]),
                "father": np.asarray(supplied["fathers"]),
                "twin": np.full(len(ids_arr), -1, dtype=np.int64)
                if supplied["twins"] is None
                else np.asarray(supplied["twins"]),
            }
            for name in ("sex", "generation", "birth_year"):
                if supplied[name] is not None:
                    data[name] = np.asarray(supplied[name])
            return cls(data)

        if ids is None or mother_ids is None or father_ids is None:
            raise TypeError("from_arrays() requires ids=, mother_ids=, and father_ids=")
        return cls._from_input(
            parse_pedigree_arrays(
                ids=ids,
                mother_ids=mother_ids,
                father_ids=father_ids,
                twin_ids=twin_ids,
                sex=sex,
                generation=generation,
                birth_year=birth_year,
                sex_encoding="simace" if sex_encoding is None else sex_encoding,
            ),
            legacy_defaults=False,
        )

    def view(self, *, ids: object | None = None, rows: object | None = None) -> PedigreeView:
        """Return an ordered :class:`~pedigree_graph._view.PedigreeView` of these rows.

        Args:
            ids: Ids to select, in view order.  Exclusive with *rows*.
            rows: Graph rows to select, in view order.  Exclusive with *ids*.

        Returns:
            The view over that selection, in the order given.

        Raises:
            TypeError: When both keywords are given, or neither.
            PedigreeValidationError: As :func:`pedigree_graph._view._build_view`.
        """
        return _build_view(self, ids=ids, rows=rows)

    def relationship_pairs(
        self,
        *,
        max_degree: int | None = None,
        categories: Iterable[str] | None = None,
    ) -> RelationshipPairs:
        """Return every relationship pair of the selected categories, in graph rows.

        Exactly one selector is given.  Selection is an output filter: the
        engine always resolves the closer categories a selected one depends
        on, so a pair is reported under its closest category (lowest degree,
        then registry order) whichever categories were named.

        Args:
            max_degree: Select every category at or below this degree (0-5).
                Exclusive with *categories*.
            categories: Registry codes to select, any order.  Exclusive with
                *max_degree*.

        Returns:
            A :class:`~pedigree_graph.relationships.RelationshipPairs` over all
            23 codes.  Each block holds owned, read-only int32 graph rows: for
            an asymmetric category ``first_rows`` carries ``first_role``
            (offspring, descendant, niece_nephew, junior_cousin) and
            ``second_rows`` the counterpart; a symmetric category stores
            ``first < second``.  Blocks are sorted by the canonical unordered
            row key.  Unselected categories are empty with ``requested=False``.

        Raises:
            TypeError: Both selectors, neither, or a bare ``str`` for
                *categories*.
            PedigreeValidationError: ``max_degree_out_of_range`` or
                ``unknown_relationship_category``.
        """
        return _relationship_pairs(self, max_degree=max_degree, categories=categories)

    def relationship_counts(
        self,
        *,
        max_degree: int | None = None,
        categories: Iterable[str] | None = None,
    ) -> RelationshipCountResult:
        """Return the exact number of pairs in each selected category.

        Same selectors as :meth:`relationship_pairs`; each count is the length
        of that call's block.

        Returns:
            A :class:`~pedigree_graph.relationships.RelationshipCountResult`
            over all 23 codes, ``None`` for unselected categories.
        """
        return RelationshipCountResult.from_pairs(self.relationship_pairs(max_degree=max_degree, categories=categories))

    def estimate_relationship_counts(self, *, max_degree: int) -> RelationshipCountResult:
        """Estimate the number of pairs in every category up to *max_degree*.

        Memory-bounded scalar arithmetic (per-anchor ``C(k, 2)`` sums and
        lineal-edge ``.nnz`` reads): no pair arrays are built, so peak memory
        is O(N) on pedigrees where :meth:`relationship_counts` would not fit.
        Full-graph only; a view has no estimate.

        Precision (source of truth: ``REL_PLAN.estimate_exact`` in
        ``_registry``, ADR 0011).  MZ, MO, FO, FS, MHS, and PHS equal
        :meth:`relationship_counts` on every input and are in
        ``result.exact``; the half-sib pairs a parent-offspring category
        claims under the precedence fold are subtracted.  Every other
        requested code is in ``result.approximate``.  GP, GGP, GGGP, and
        G3GP are raw ancestor-path counts; they over-count a pair also
        related at a shorter depth, as a half-sib, or as a closer
        collateral.  The cousin / collateral formulas assume a full
        complement of known ancestors and diverge on shallow, inbred, or
        twin-having pedigrees; on deep, lightly inbred pedigrees they are
        within about 1% of exact.  Four of them (``H1C``, ``1C1R``,
        ``1C2R``, ``H1C1R``) are inclusion-exclusion residuals; when one
        underflows it is floored at ``0`` and listed in ``result.clamped``,
        and that ``0`` is not a true absence.

        Warning and cache rule: the result for each cutoff is computed once
        per graph and the same frozen object is returned afterwards.  The
        computation, and only the computation, emits one ``RuntimeWarning``
        naming the clamped codes when ``clamped`` is non-empty, before the
        result is cached; a cached retrieval is silent, and a different
        cutoff computes and warns on its own.  Python's default warning
        filter shows one identical warning per call site; the ``clamped``
        set is the reliable signal.  The 0.7.1 ``count_pairs_streaming``
        adapter shares this cache and returns the raw counts.

        Threads: the call commits the package thread budget
        (:func:`~pedigree_graph.configure_threads`) like every 0.8 operation,
        so reconfiguring to a different value afterwards raises.  The scalar
        counter itself runs single-threaded numpy / scipy, and the integer
        results are the same under any budget.  The transient adjacency
        powers are released on return.

        Args:
            max_degree: Degree cutoff in ``[0, 5]``; categories whose registry
                degree is at or below it are requested.

        Returns:
            A :class:`~pedigree_graph.relationships.RelationshipCountResult`
            over all 23 codes, ``None`` above the cutoff, with ``requested``,
            ``exact``, ``approximate``, and ``clamped`` filled in.

        Raises:
            PedigreeValidationError: ``max_degree_out_of_range``.
        """
        return _estimate_relationship_counts(self, max_degree=max_degree)

    # ------------------------------------------------------------------
    # Sparse kinship, inbreeding, and exact pair kinship
    # ------------------------------------------------------------------

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
        _require_complete_generation_labels(self, "per_gen_mean_kinship")

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
                self.depth,
                min_kinship,
                labels=self.generation,
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
        graph (``self._inbreeding``).  Construction has already established the
        MZ pair contract, so no co-twin reference reaching here is self-directed,
        non-reciprocal, or parent-mismatched.
        """
        if self._inbreeding is None:
            topo = self._topology
            m_idx, f_idx, tw_idx = self._topological_parents
            F = _compute_F_meuwissen_luo(m_idx, f_idx, tw_idx, topo.gather(topo.depth), self.n)
            self._inbreeding = topo.per_row_to_graph(F)
        return self._inbreeding

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
            if self._rows_are_topological:
                n_desc64 = _compute_n_descendants(self.mother, self.father, self.n)
            else:
                m_idx, f_idx, _ = self._topological_parents
                n_desc64 = self._topology.per_row_to_graph(_compute_n_descendants(m_idx, f_idx, self.n))
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

    @overload
    def pair_kinship(self, first: RelationshipPairs, /) -> Mapping[str, np.ndarray]: ...
    @overload
    def pair_kinship(self, first: RelationshipPairBlock, /) -> np.ndarray: ...
    @overload
    def pair_kinship(self, first: object, second: object, /) -> np.ndarray: ...
    def pair_kinship(self, first: object, second: object | None = None, /) -> np.ndarray | Mapping[str, np.ndarray]:
        """Return the pedigree-expected kinship of each requested pair, in graph rows.

        Three call forms: ``pair_kinship(first_rows, second_rows)`` for any
        pairs, self pairs included; ``pair_kinship(block)`` for one
        :class:`~pedigree_graph.relationships.RelationshipPairBlock`; and
        ``pair_kinship(pairs)`` for a whole
        :class:`~pedigree_graph.relationships.RelationshipPairs`, which runs
        one recurrence with one shared memo for every block.

        Each value is the pinned float32 recurrence of ADR 0009: inbreeding,
        MZ genome identity, and every relationship path are included, and the
        value is bit-identical to the ``kinship_matrix`` entry for the same
        pair.  Reversed endpoints give identical bits, a returned ``0`` means
        the exact kinship is ``0``, and no cached matrix is read, so the result
        does not depend on call history.  Two graphs built from the same
        pedigree in different row orders agree within
        ``2 * (depth_a + depth_b + 1) * 2**-25`` on deep inbred pairs.  Widen
        to float64 before comparing against a non-dyadic cutoff.  The call
        runs on one thread and commits the package thread budget like every
        0.8 operation.

        Args:
            first: ``first_rows`` (graph rows, any integer array-like), a
                block, or a pairs collection.
            second: ``second_rows``, same length as ``first_rows``; omitted for
                a block or collection.

        Returns:
            A read-only float32 array positionally aligned to the input pairs,
            or for a collection an immutable mapping over all 23 codes to such
            arrays (empty for unrequested codes).

        Raises:
            TypeError: A block or collection with a second argument, or row
                arrays without one.
            PedigreeValidationError: ``coordinate_space_mismatch`` for a block
                from another receiver; ``invalid_shape``,
                ``invalid_integer_value``, or ``pair_row_out_of_range`` per row
                argument; ``pair_length_mismatch``.
            ResourceError: ``memo_capacity_exceeded`` on a pedigree too inbred
                and deep for the direct recurrence.
        """
        return graph_pair_kinship(self, first, second)

    # 0.8.0-DELETE: replaced by pair_kinship (ADR 0006, ADR 0009).
    def compute_pair_kinship(
        self,
        pairs: dict[str, tuple[np.ndarray, np.ndarray]],
    ) -> dict[str, np.ndarray]:
        """Kinship for each requested pair, keyed like *pairs* (0.7.1 form).

        Returns ``{code: float32 array}`` positionally aligned to the input
        ``pairs[code]``.  The values are those of :meth:`pair_kinship`, an
        explicit 0.8 change from the 0.7.1 float64 recurrence, and like every
        0.8 result the arrays are read-only; the dict form and the caller-space
        rows of a ``from_subsample`` graph are preserved.
        """
        codes = [code for code, (idx1, _) in pairs.items() if len(idx1)]
        empty = np.zeros(0, dtype=np.float32)
        empty.setflags(write=False)
        result: dict[str, np.ndarray] = dict.fromkeys(pairs, empty)
        if not codes:
            return result
        first = np.concatenate([np.asarray(pairs[code][0]) for code in codes])
        second = np.concatenate([np.asarray(pairs[code][1]) for code in codes])
        if self._legacy_view is None:
            flat = graph_pair_kinship(self, first, second, commit_threads=False)
        else:
            flat = view_pair_kinship(self._legacy_view, first, second, commit_threads=False)
        assert isinstance(flat, np.ndarray)
        offset = 0
        for code in codes:
            count = len(pairs[code][0])
            result[code] = flat[offset : offset + count]
            offset += count
        return result
