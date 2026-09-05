"""Focused tests for the decomposed pair engines and shared utilities (PGQ-003).

Covers the newly-isolated pure helpers in ``_pair_utils`` and the read-only
contract of the two engine collaborators (``MatrixPairExtractor``,
``StreamingPairCounter``) established in ADR 0002: the engines compute and
return results but never write the graph's result cache — the public
wrappers do.
"""

import numpy as np
import scipy.sparse as sp

from pedigree_graph import RELATIONSHIPS, PedigreeGraph
from pedigree_graph._pair_extractor import MatrixPairExtractor
from pedigree_graph._pair_utils import (
    dedup_pairs,
    oriented_pairs_from_sparse,
    pairs_from_groups,
    subtract_pairs,
)
from pedigree_graph._streaming_counter import StreamingPairCounter


class TestPairUtils:
    def test_dedup_pairs_canonicalizes_and_dedupes(self):
        # (5,1) and (1,5) are the same unordered pair → one canonical (1,5).
        a_i = np.array([5, 1, 3])
        a_j = np.array([1, 5, 7])
        lo, hi = dedup_pairs(a_i, a_j)
        assert np.all(lo <= hi)
        assert set(zip(lo.tolist(), hi.tolist(), strict=True)) == {(1, 5), (3, 7)}

    def test_dedup_pairs_empty(self):
        lo, hi = dedup_pairs(np.array([], dtype=np.intp), np.array([], dtype=np.intp))
        assert lo.size == 0
        assert hi.size == 0

    def test_pairs_from_groups_enumerates_within_groups(self):
        # rows 0,1 share group 10; rows 2,3 share group 20.
        lo, hi = pairs_from_groups(np.array([0, 1, 2, 3]), np.array([10, 10, 20, 20]))
        assert set(zip(lo.tolist(), hi.tolist(), strict=True)) == {(0, 1), (2, 3)}

    def test_pairs_from_groups_excludes_singletons(self):
        lo, hi = pairs_from_groups(np.array([0, 1, 2]), np.array([10, 20, 30]))
        assert lo.size == 0
        assert hi.size == 0

    def test_oriented_pairs_zero_the_diagonal_and_keep_the_row_role(self):
        dense = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]], dtype=float)
        first, second = oriented_pairs_from_sparse(sp.csr_matrix(dense), row_is_first=True)
        assert list(zip(first.tolist(), second.tolist(), strict=True)) == [(2, 0)]

    def test_oriented_pairs_can_take_the_column_role(self):
        dense = np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0]], dtype=float)
        first, second = oriented_pairs_from_sparse(sp.csr_matrix(dense), row_is_first=False)
        assert list(zip(first.tolist(), second.tolist(), strict=True)) == [(0, 2)]

    def test_oriented_pairs_dual_valid_keeps_the_lower_row_first(self):
        dense = np.array([[0, 1], [1, 0]], dtype=float)
        first, second = oriented_pairs_from_sparse(sp.csr_matrix(dense), row_is_first=True)
        assert list(zip(first.tolist(), second.tolist(), strict=True)) == [(0, 1)]
        first, second = oriented_pairs_from_sparse(sp.csr_matrix(dense), row_is_first=False)
        assert list(zip(first.tolist(), second.tolist(), strict=True)) == [(0, 1)]

    def test_oriented_pairs_subtract_in_either_orientation(self):
        dense = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0]], dtype=float)
        first, second = oriented_pairs_from_sparse(
            sp.csr_matrix(dense), row_is_first=True, subtract=[(np.array([0]), np.array([1]))]
        )
        assert list(zip(first.tolist(), second.tolist(), strict=True)) == [(2, 0)]

    def test_subtract_pairs_preserves_orientation_of_survivors(self):
        keep = (np.array([5, 3, 9]), np.array([1, 4, 2]))
        first, second = subtract_pairs(keep, [(np.array([4]), np.array([3]))])
        assert list(zip(first.tolist(), second.tolist(), strict=True)) == [(5, 1), (9, 2)]


class TestEngineReadOnlyContract:
    """Engines compute results but never persist them (ADR 0002)."""

    def test_matrix_extractor_does_not_write_count_cache(self, small_pedigree):
        pg = PedigreeGraph(small_pedigree)
        assert pg._pair_count_cache == {}
        codes = frozenset(code for code, category in RELATIONSHIPS.items() if category.degree <= 2)
        pairs = MatrixPairExtractor(pg, max_workers=None).extract(codes)
        # The engine must not touch the graph's result cache — that's the wrapper's job.
        assert pg._pair_count_cache == {}
        assert list(pairs) == list(RELATIONSHIPS)
        counts = {code: len(block[0]) for code, block in pairs.items()}
        assert counts["FS"] > 0
        assert all(counts[code] == 0 for code in RELATIONSHIPS if code not in codes)

        # The wrapper, given the same graph, caches exactly what the engine returned.
        pg2 = PedigreeGraph(small_pedigree)
        pg2.extract_pairs(max_degree=2)
        cached_raw, cached_sub = pg2._pair_count_cache[("matrix", 2, 0.0)]
        assert cached_raw == counts
        assert cached_sub == counts

    def test_streaming_counter_does_not_write_count_cache(self, small_pedigree):
        pg = PedigreeGraph(small_pedigree)
        assert pg._pair_count_cache == {}
        counts = StreamingPairCounter(pg).count(2)
        assert pg._pair_count_cache == {}
        assert isinstance(counts, dict)
        assert counts["MZ"] >= 0

        # The wrapper returns the same counts and caches them.
        pg2 = PedigreeGraph(small_pedigree)
        assert pg2.count_pairs_streaming(max_degree=2) == counts
        assert ("streaming", 2, 0.0) in pg2._pair_count_cache
