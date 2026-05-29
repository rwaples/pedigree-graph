"""Focused tests for the decomposed pair engines and shared utilities (PGQ-003).

Covers the newly-isolated pure helpers in ``_pair_utils`` and the read-only
contract of the two engine collaborators (``MatrixPairExtractor``,
``StreamingPairCounter``) established in ADR 0002: the engines compute and
return results but never write the graph's result cache — the public
wrappers do.
"""

import numpy as np
import scipy.sparse as sp

from pedigree_graph import PedigreeGraph
from pedigree_graph._pair_extractor import MatrixPairExtractor
from pedigree_graph._pair_utils import (
    dedup_pairs,
    extract_from_sparse,
    pairs_from_groups,
    remap_pairs_to_caller,
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

    def test_extract_from_sparse_zeroes_diagonal(self):
        # Diagonal entry at (0,0) must be dropped; off-diagonal (0,2) kept.
        dense = np.array([[1, 0, 1, 0], [0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]], dtype=float)
        lo, hi = extract_from_sparse(sp.csr_matrix(dense))
        assert set(zip(lo.tolist(), hi.tolist(), strict=True)) == {(0, 2)}

    def test_extract_from_sparse_subtracts_closer_pairs(self):
        dense = np.array([[0, 1, 1, 0], [1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]], dtype=float)
        # candidate pairs (0,1) and (0,2); subtract (0,1).
        lo, hi = extract_from_sparse(sp.csr_matrix(dense), subtract=[(np.array([0]), np.array([1]))])
        assert set(zip(lo.tolist(), hi.tolist(), strict=True)) == {(0, 2)}

    def test_remap_pairs_to_caller_recanonicalizes(self):
        # graph rows 0,1 map to caller rows 1,0 (reversed); pair must stay lo < hi.
        pairs = {"FS": (np.array([0]), np.array([1]))}
        remap = np.array([1, 0])
        out = remap_pairs_to_caller(pairs, remap)
        lo, hi = out["FS"]
        assert (int(lo[0]), int(hi[0])) == (0, 1)


class TestEngineReadOnlyContract:
    """Engines compute results but never persist them (ADR 0002)."""

    def test_matrix_extractor_does_not_write_count_cache(self, small_pedigree):
        pg = PedigreeGraph(small_pedigree)
        assert pg._pair_count_cache == {}
        pairs, raw, sub = MatrixPairExtractor(pg).extract(2, 0.0)
        # The engine must not touch the graph's result cache — that's the wrapper's job.
        assert pg._pair_count_cache == {}
        assert isinstance(pairs, dict)
        assert "FS" in pairs
        assert set(raw) == set(sub)

        # The wrapper, given the same graph, caches exactly what the engine returned.
        pg2 = PedigreeGraph(small_pedigree)
        pg2.extract_pairs(max_degree=2)
        cached_raw, cached_sub = pg2._pair_count_cache[("matrix", 2, 0.0)]
        assert cached_raw == raw
        assert cached_sub == sub

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
