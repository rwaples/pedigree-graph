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
from functools import cached_property
from typing import TYPE_CHECKING, Literal, overload

import numpy as np

from pedigree_graph import _native
from pedigree_graph._burden import relationship_burden as _relationship_burden
from pedigree_graph._cohort_utils import generation_interval as _generation_interval
from pedigree_graph._input import _own_native, host_columns, host_columns_from_arrays
from pedigree_graph._kinship_matrix import PedigreeMatrixMethods
from pedigree_graph._kinship_pairwise import graph_pair_kinship
from pedigree_graph._lineage import connected_component_ids as _connected_component_ids
from pedigree_graph._lineage import descendant_path_counts as _descendant_path_counts
from pedigree_graph._lineage import distinct_ancestor_counts as _distinct_ancestor_counts
from pedigree_graph._ne_rates import _generation_kinship_summary
from pedigree_graph._properties import PedigreeProperties
from pedigree_graph._relationship_counts import relationship_counts as _relationship_counts
from pedigree_graph._relationship_pairs import check_execution
from pedigree_graph._relationship_pairs import relationship_pairs as _relationship_pairs
from pedigree_graph._selection import RelationshipSelection
from pedigree_graph._streaming_counter import close_relative_counts as _close_relative_counts
from pedigree_graph._threads import thread_budget
from pedigree_graph._view import CoordinateToken, _build_view

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    import scipy.sparse as sp

    from pedigree_graph._burden import RelationshipBurden
    from pedigree_graph._frames import FrameLike
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

    def _release_kinship_matrices(self) -> None:
        """Drop every cached kinship matrix held by this graph, and the pair memo.

        The three matrix families cache independently and a full-graph CSC can
        run to hundreds of megabytes, so a long-lived graph that called more
        than one of them pins all of them.  Idempotent.
        """
        self._complete_kinship_cache = None
        self._relationship_kinship_cache.clear()
        self._approximate_kinship_cache.clear()

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
        execution: str = "speed",
    ) -> RelationshipPairs:
        """Return every relationship pair of the selected categories, in graph rows.

        Exactly one selector is given.  Selection is an output filter: the
        engine always resolves the closer categories a selected one depends
        on, so a pair is reported under its closest category (lowest degree,
        then registry order) whichever categories were named.  Degree and
        category are read from represented parent edges, so they derive from
        structural depth and supplied generation labels never enter the
        classification.  Classification runs on the Rust row-streaming
        engine under the package thread budget
        (:func:`~pedigree_graph.configure_threads`); the result is the same
        for every budget.

        Args:
            max_degree: Select every category at or below this degree (0-5).
                Exclusive with *categories*.
            categories: Registry codes to select, any order.  Exclusive with
                *max_degree*.
            execution: ``"speed"`` (default) for the fastest exact assembly,
                whose peak memory is about 2.3 times the result, or
                ``"memory"`` for the lowest-peak one, the result plus engine
                state, at roughly twice the wall time.  The blocks are
                identical either way.

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
            ValueError: *execution* is not ``"speed"`` or ``"memory"``.
            PedigreeValidationError: ``max_degree_out_of_range`` or
                ``unknown_relationship_category``.
            ResourceError: ``allocation_failed`` when the engine or the
                result cannot be allocated.
        """
        return _relationship_pairs(
            self, RelationshipSelection.parse(max_degree, categories), check_execution(execution)
        )

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
        budget.  Like the pairs they count, they derive from structural depth;
        supplied generation labels never enter them.

        Returns:
            A :class:`~pedigree_graph.relationships.RelationshipCountResult`
            over all 23 codes, ``None`` for unselected categories, every
            requested code in ``exact``.

        Raises:
            TypeError: As :meth:`relationship_pairs`.
            PedigreeValidationError: As :meth:`relationship_pairs`.
        """
        return _relationship_counts(self, RelationshipSelection.parse(max_degree, categories))

    def relationship_burden(self) -> RelationshipBurden:
        """Summarise closest-category pairs without materialising pair lists.

        Counts cover all 23 categories. The read-only graph-row array has one
        column per degree 1 through 5; MZ pairs contribute only to category
        and same-depth counts. Peak output storage is O(N).
        """
        return _relationship_burden(self)

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
        kinship over the unordered pairs of distinct *genomes* in each group.
        MZ co-twins are one genome (ADR 0008), so one row of the pair
        represents it and the other joins no group: the co-twin carrying a
        label, or the earlier label of the two when both carry one, so a
        genome enters at the earliest cohort claimed for it.  That choice is
        made on the labels, never on a row index or an id, so the grouping
        moves with neither.  Rows whose supplied label is ``-1`` join
        no group and are reported in ``unlabelled_individual_count``, never
        assigned a depth; a collapsed co-twin is not tallied there.  Labels
        appear ascending, and only those some *representative* row carries: a
        label borne solely by collapsed co-twins holds no genome and is absent
        from the result, so the labels here can be a strict subset of the
        distinct labels supplied.

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

        The walk follows represented parent edges only, so *F* derives from
        structural depth; supplied generation labels never enter it.

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
            self._inbreeding = _own_native(_native.inbreeding(self._built, self.depth), np.float64)
        return self._inbreeding

    def distinct_ancestor_counts(self) -> np.ndarray:
        """Return the number of distinct strict ancestors of every row.

        An ancestor reachable through several paths, as marriage loops
        create, is counted once.  A missing or external parent contributes
        nothing, so the count derives from structural depth and supplied
        generation labels never enter it.  Computed once and memoised; the call
        commits the package thread budget
        (:func:`~pedigree_graph.configure_threads`) like every 0.8 operation.

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
        :meth:`distinct_ancestor_counts`.  The walk follows represented parent
        edges, so the count derives from structural depth and supplied
        generation labels never enter it.  Computed once and memoised; the
        call commits the package thread budget like every 0.8 operation.

        Returns:
            A read-only int64 array of length ``n_individuals``, in graph rows.

        Raises:
            ResourceError: ``arithmetic_overflow`` when a count exceeds int64,
                which about 63 generations of repeated sib mating reach.
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

        The recurrence follows represented parent edges alone, so every value
        derives from structural depth; supplied generation labels never enter
        it.

        The recurrence runs in the Rust core with one memo per call, shared
        across every pair of the query and freed before the call returns;
        nothing is kept on the graph, so repeated queries pay the full walk
        each time and return identical bits.

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
            ResourceError: ``allocation_failed`` when the memo, the walk's
                stack, or the output cannot be allocated.
        """
        return graph_pair_kinship(self, first, second)
