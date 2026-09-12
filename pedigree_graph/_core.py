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

__all__ = ["PedigreeGraph"]

import logging
import time
from functools import cached_property
from typing import TYPE_CHECKING, Literal, overload

import numpy as np
import scipy.sparse as sp

from pedigree_graph import _native
from pedigree_graph._cohort_utils import generation_interval as _generation_interval
from pedigree_graph._input import host_columns, host_columns_from_arrays
from pedigree_graph._kinship_kernel import (
    _compute_F_meuwissen_luo,
)
from pedigree_graph._kinship_matrix import PedigreeMatrixMethods
from pedigree_graph._kinship_pairwise import _MEMO_RETAIN_LIMIT, graph_pair_kinship
from pedigree_graph._lineage import connected_component_ids as _connected_component_ids
from pedigree_graph._lineage import descendant_path_counts as _descendant_path_counts
from pedigree_graph._lineage import distinct_ancestor_counts as _distinct_ancestor_counts
from pedigree_graph._ne_rates import _generation_kinship_summary
from pedigree_graph._pair_extractor import relationship_pairs as _relationship_pairs
from pedigree_graph._pair_utils import pairs_from_groups, subtract_pairs
from pedigree_graph._properties import PedigreeProperties
from pedigree_graph._relationship_counts import relationship_counts as _relationship_counts
from pedigree_graph._selection import RelationshipSelection
from pedigree_graph._streaming_counter import close_relative_counts as _close_relative_counts
from pedigree_graph._threads import thread_budget
from pedigree_graph._topology import build_topology, readonly
from pedigree_graph._view import CoordinateToken, _build_view

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from pedigree_graph._frames import FrameLike
    from pedigree_graph._kinship_pairwise import _PairMemo
    from pedigree_graph._topology import Topology
    from pedigree_graph._view import PedigreeView
    from pedigree_graph.relationships import RelationshipCountResult, RelationshipPairBlock, RelationshipPairs
    from pedigree_graph.summaries import GenerationKinshipSummary

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
        parent_arr: per-row parent row-index array (``self.mother_rows`` or
            ``self.father_rows``).
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

    Build one with :meth:`from_frame` or :meth:`from_arrays`.  Both coerce
    through :mod:`pedigree_graph._input`, validate and build natively through
    ``pedigree_graph._native.build_pedigree``, and hand the built columns to
    :meth:`_from_built`, so every graph reaches the engine the same way.
    Neither invents a value: an absent optional column reads as absent.
    """

    @classmethod
    def _from_built(cls, built: _native.BuiltPedigree) -> PedigreeGraph:
        """Build a graph over natively validated columns, the one path all constructors share.

        Args:
            built: The validated columns from ``build_pedigree``.

        Returns:
            The constructed graph.
        """
        graph = cls.__new__(cls)
        graph._initialize(built)
        return graph

    def _initialize(self, built: _native.BuiltPedigree) -> None:
        """Populate the graph's storage, caches, and parent matrices from *built*."""
        self._built = built
        for column in (
            built.ids,
            built.mother_ids,
            built.father_ids,
            built.twin_ids,
            built.mother_rows,
            built.father_rows,
            built.twin_rows,
            built.sex,
            built.generation,
            built.birth_year,
        ):
            if column is not None:
                column.setflags(write=False)
        self._coordinate_token = CoordinateToken()

        # Matrix caches are separated by operation and selector: complete,
        # closest-category, and propagation-pruned support are distinct
        # contracts even when two calls happen to produce the same structure.
        self._complete_kinship_cache: sp.csc_matrix | None = None
        self._relationship_kinship_cache: dict[tuple[str, ...], sp.csc_matrix] = {}
        self._approximate_kinship_cache: dict[float, sp.csc_matrix] = {}
        # The pair-recurrence memo the last kernel call left behind, reused as
        # the next call's starting table by pair_kinship and the relationship
        # matrix (both through _kinship_pairwise.memoised_kinship).  Retained
        # only while its tables fit under the limit, in bytes.
        self._pair_memo: _PairMemo | None = None
        self._pair_memo_limit: int = _MEMO_RETAIN_LIMIT
        # mean_kinship_by_generation() is threshold-free: one summary per graph.
        self._generation_kinship_summary: GenerationKinshipSummary | None = None
        self._close_relative_counts_cache: RelationshipCountResult | None = None
        self._inbreeding: np.ndarray | None = None
        # Lineage memos (_lineage.py).
        self._distinct_ancestor_counts: np.ndarray | None = None
        self._descendant_path_counts: np.ndarray | None = None
        self._connected_component_ids: np.ndarray | None = None
        # Lazy cache of known-parent edge filters, keyed "mother"/"father"
        # (see _known_parent_edges_for); shared by the overlapping-generation
        # diagnostics so the full-pedigree edge scan runs once per side.
        self._known_parent_edges_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

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
        return self._built.rows_topological

    @cached_property
    def _topological_parents(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(mother, father, twin)`` rewritten into the private topological order."""
        topo = self._topology
        return (
            topo.to_topological(self.mother_rows),
            topo.to_topological(self.father_rows),
            topo.to_topological(self.twin_rows),
        )

    def _known_parent_edges_for(
        self,
        parent_label: Literal["mother", "father"],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Cached :func:`_known_parent_edges` lookup by parent label.

        ``generation_interval``, ``_cohort_utils.eligible_cohort_range``,
        and ``_hill_age_table`` all call the same edge-filter and
        age-diff computation against the same arrays — at scale that's
        six passes over the full pedigree.  Cached per-graph keyed on
        ``"mother"``/``"father"`` (cache initialised in ``_initialize``).
        Bypassed when ``birth_year is None`` (the underlying helper still
        runs but the result is small).
        """
        hit = self._known_parent_edges_cache.get(parent_label)
        if hit is not None:
            return hit
        parent_arr = self.mother_rows if parent_label == "mother" else self.father_rows
        if self.birth_year is None:
            result = (np.array([], dtype=np.intp), np.array([], dtype=np.int32))
        else:
            result = _known_parent_edges(parent_arr, self.birth_year)
        self._known_parent_edges_cache[parent_label] = result
        return result

    @cached_property
    def generation_interval(self):
        """Sex-split generation interval (Hill 1979 ``L``).

        Returns a :class:`~pedigree_graph.effective_size.GenerationInterval`
        ``(T, T_m, T_f, n_edges)`` over all parent-child edges where both
        endpoints have known ``birth_year``, or ``None`` only when
        ``self.birth_year is None``; see
        :func:`~pedigree_graph._cohort_utils.generation_interval`.  Cached
        on the instance on first read.

        Raises:
            MissingMetadataError: ``insufficient_parent_age_data`` when birth
                years are present but a parent role has no edge with both
                birth years known.
        """
        return _generation_interval(self)

    # ------------------------------------------------------------------
    # Lazy sparse products (computed on first access)
    # ------------------------------------------------------------------

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

    def _release_pair_matrices(self) -> None:
        """Drop the transient adjacency / sibling matrices built for pair work.

        Only the pair arrays and count caches are needed after an
        extraction; the cached sparse matrices can be large, so they are
        released here.  Idempotent — missing attributes are ignored.
        """
        for attr in ("_A", "_A2", "_A3", "_A4", "_A5", "_A2_shared", "_full_sib_matrix", "_half_sib_matrix"):
            self.__dict__.pop(attr, None)

    def _release_kinship_matrices(self) -> None:
        """Drop every cached kinship matrix held by this graph, and the pair memo.

        The three matrix families cache independently and a full-graph CSC can
        run to hundreds of megabytes, so a long-lived graph that called more
        than one of them pins all of them.  Idempotent.
        """
        self._complete_kinship_cache = None
        self._relationship_kinship_cache.clear()
        self._approximate_kinship_cache.clear()
        self._release_pair_memo()

    def _release_pair_memo(self) -> None:
        """Drop the pair-recurrence memo, so the next ``pair_kinship`` starts cold.

        The memo is the ancestor-pair closure of every query so far, 12 bytes a
        slot, kept so a later query pays only for the pairs it newly reaches.
        Releasing it changes no value: a cold call stores the same bits.
        Idempotent.
        """
        self._pair_memo = None

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
            ValueError: For an unknown *sex_encoding*, which is API misuse
                rather than a pedigree-data failure.
            PedigreeValidationError: For any invalid field, duplicate id,
                shared parent id, cyclic parent reference, broken MZ pair, or
                child born before a parent.
            ResourceError: ``pedigree_too_large`` beyond the int32 row capacity.
        """
        columns = host_columns(frame)
        return cls._from_built(_native.build_pedigree(*columns.as_args(), sex_encoding=sex_encoding))

    @classmethod
    def from_arrays(
        cls,
        *,
        ids: object,
        mother_ids: object,
        father_ids: object,
        twin_ids: object | None = None,
        sex: object | None = None,
        generation: object | None = None,
        birth_year: object | None = None,
        sex_encoding: str = "simace",
    ) -> PedigreeGraph:
        """Construct from separate columns, for callers with no table to hand.

        Keyword-only, and applies no defaults::

            PedigreeGraph.from_arrays(ids=ids, mother_ids=m, father_ids=f)

        Args:
            ids: Row ids.
            mother_ids: Mother ids, ``-1`` or a host null when missing.
            father_ids: Father ids, as *mother_ids*.
            twin_ids: MZ co-twin ids, as *mother_ids*.
            sex: Sex codes in *sex_encoding*.
            generation: Generation labels, ``-1`` when unknown.
            birth_year: Birth years, ``-1`` when unknown.
            sex_encoding: ``"simace"`` (the default) or ``"plink"``.

        Returns:
            The constructed graph.

        Raises:
            PedigreeValidationError: As :meth:`from_frame`.
        """
        columns = host_columns_from_arrays(
            ids=ids,
            mother_ids=mother_ids,
            father_ids=father_ids,
            twin_ids=twin_ids,
            sex=sex,
            generation=generation,
            birth_year=birth_year,
        )
        return cls._from_built(_native.build_pedigree(*columns.as_args(), sex_encoding=sex_encoding))

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
        return _relationship_pairs(self, RelationshipSelection.parse(max_degree, categories))

    def relationship_counts(
        self,
        *,
        max_degree: int | None = None,
        categories: Iterable[str] | None = None,
    ) -> RelationshipCountResult:
        """Return the exact number of pairs in each selected category.

        Same selectors and closest-category precedence as
        :meth:`relationship_pairs`; each count equals the length of that
        call's block.  The Rust row-streaming engine classifies every pair
        one row at a time and never builds a pair list, so peak memory is
        O(N) and the call fits pedigrees where :meth:`relationship_pairs`
        would not (ADR 0010).  The counts are the same under every thread
        budget.

        Returns:
            A :class:`~pedigree_graph.relationships.RelationshipCountResult`
            over all 23 codes, ``None`` for unselected categories, every
            requested code in ``exact``.

        Raises:
            TypeError: As :meth:`relationship_pairs`.
            PedigreeValidationError: As :meth:`relationship_pairs`.
        """
        return _relationship_counts(self, RelationshipSelection.parse(max_degree, categories))

    def close_relative_counts(self) -> RelationshipCountResult:
        """Return exact MZ, MO, FO, FS, MHS and PHS pair counts.

        This full-graph-only scalar method counts parent edges and sibling
        groups without building pair lists or adjacency powers. Peak memory
        is O(N). Half-sib pairs claimed by parent-offspring categories are
        subtracted, so every count equals :meth:`relationship_counts` under
        closest-category precedence, including on inbred pedigrees.

        The six codes come from ``REL_PLAN.estimate_exact``. This is not a
        degree cutoff: grandparents and avuncular pairs are not included.
        Use :meth:`relationship_counts` for other categories or view counts.

        The same immutable result is returned on subsequent calls. Each call
        commits the package thread budget, but the scalar calculation is
        single-threaded and its integer results do not depend on that budget.

        Returns:
            A :class:`~pedigree_graph.relationships.RelationshipCountResult`
            over all 23 registry codes. ``requested`` and ``exact`` contain
            the six close categories; every other code maps to ``None``.
        """
        return _close_relative_counts(self)

    # ------------------------------------------------------------------
    # Sparse kinship, inbreeding, and exact pair kinship
    # ------------------------------------------------------------------

    def mean_kinship_by_generation(self) -> GenerationKinshipSummary:
        """Mean pedigree-expected kinship within each observed generation.

        Groups rows by the supplied generation labels, or by structural
        :attr:`depth` when none were supplied, and averages the ADR 0009
        kinship over the unordered pairs of distinct individuals in each
        group.  An MZ twin pair is left out of a group's sum and denominator
        only when both co-twins are in that group; a twin whose partner is
        unlabelled or elsewhere is an ordinary member.  Rows whose label is
        ``-1`` join no group and are reported in
        ``unlabelled_individual_count``, never assigned a depth.  Only labels
        some row carries appear, ascending.

        The kinship is streamed from the retiring DP without materializing
        the kinship matrix, unless the complete matrix is already cached, in
        which case that matrix is walked instead; both routes give the same
        values.  The summary is computed once per graph and the same frozen
        object returned afterwards.  The call commits the package thread
        budget (:func:`~pedigree_graph.configure_threads`) like every 0.8
        operation.

        Returns:
            A :class:`~pedigree_graph.summaries.GenerationKinshipSummary`
            with read-only ``generations``, ``mean_kinship`` (NaN where
            ``pair_counts`` is 0), and ``pair_counts`` arrays.
        """
        thread_budget()
        return _generation_kinship_summary(self)

    def inbreeding(self) -> np.ndarray:
        """Return the inbreeding coefficient *F* of every individual, in graph rows.

        The values are the Meuwissen-Luo ancestor walk of ADR 0008, run over the
        genome-node pedigree: MZ co-twins share the genome node of the lower row, a
        parent step follows the parent's canonical genome node rather than the parent
        row, and a non-canonical twin row copies its node's ``F`` and Mendelian
        sampling variance ``D``.  The walk is therefore MZ-aware, and
        ``F_i = 2 * phi(i, i) - 1`` is a tested invariant against the
        :meth:`pair_kinship` self pair and the :meth:`kinship_matrix` diagonal of the
        same row; the walk itself materialises no kinship.  The array is computed
        once and memoised, so every later call hands back the same frozen object.
        This operation is intentionally full-graph-only: ADR 0006 keeps inbreeding
        off views until a view contract for it is scientifically pinned.  The call
        commits the package thread budget
        (:func:`~pedigree_graph.configure_threads`) like every 0.8 operation.

        Returns:
            A read-only float64 array of length ``n_individuals``, one entry per
            graph row.
        """
        thread_budget()
        return self._inbreeding_values()

    def _inbreeding_values(self) -> np.ndarray:
        """Return the memoised *F* without committing the package thread budget.

        The effective-size estimators read F through here; their own entry
        points commit the budget.  :meth:`inbreeding` is the public entry
        point that commits.
        """
        if self._inbreeding is None:
            topo = self._topology
            m_idx, f_idx, tw_idx = self._topological_parents
            F = _compute_F_meuwissen_luo(m_idx, f_idx, tw_idx, topo.gather(topo.depth), self.n_individuals)
            self._inbreeding = readonly(topo.per_row_to_graph(F))
        return self._inbreeding

    def distinct_ancestor_counts(self) -> np.ndarray:
        """Return the number of distinct strict ancestors of every row.

        An ancestor reachable through several paths, as marriage loops
        create, is counted once.  A missing or external parent contributes
        nothing.  Computed once and memoised; the call commits the package
        thread budget (:func:`~pedigree_graph.configure_threads`) like every
        0.8 operation.

        Returns:
            A read-only int32 array of length ``n_individuals``, in graph rows.
        """
        thread_budget()
        return _distinct_ancestor_counts(self)

    def descendant_path_counts(self) -> np.ndarray:
        """Return the number of descendant *paths* from every row.

        ``counts[v]`` is the number of walks down the pedigree from ``v``:
        its children plus the path counts of those children.  This equals
        the number of distinct descendants in a pedigree without marriage
        loops and exceeds it where a descendant reaches ``v`` through more
        than one child, which is why the name says *paths*; contrast
        :meth:`distinct_ancestor_counts`.  Computed once and memoised; the
        call commits the package thread budget like every 0.8 operation.

        Returns:
            A read-only int64 array of length ``n_individuals``, in graph rows.
        """
        thread_budget()
        return _descendant_path_counts(self)

    def connected_component_ids(self) -> np.ndarray:
        """Return, for every row, the smallest ID in its parent-edge component.

        Two rows share a value exactly when a chain of represented
        parent-child edges joins them.  External or missing parents add no
        edge, so two rows naming the same external parent are in different
        components, and MZ co-twins are joined only through their parents.
        The value is the minimum :attr:`ids` over the component, so it does
        not depend on row order.  Computed once and memoised; the call
        commits the package thread budget like every 0.8 operation.

        Returns:
            A read-only int64 array of length ``n_individuals``, in graph rows.
        """
        thread_budget()
        return _connected_component_ids(self)

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

        The recurrence memo outlives the call: the graph keeps the ancestor
        pairs each query resolved and starts the next query from them, so
        repeated queries on one graph pay for newly reached ancestors only.
        A reused entry is the bit a cold call computes.  The memo costs 12
        bytes per slot for the life of the graph, is retained only while it
        fits the graph's retention limit, and is dropped by
        :meth:`_release_kinship_matrices`.

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
