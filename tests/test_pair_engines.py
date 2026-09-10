"""Focused tests for the decomposed pair engines and shared utilities (PGQ-003).

Covers the newly-isolated pure helpers in ``_pair_utils`` and the read-only
contract of the two engine collaborators (``MatrixPairExtractor``,
``StreamingPairCounter``) established in ADR 0002: the engines compute and
return results but never write the graph's estimate cache — the public
wrappers do.
"""

import warnings

import numpy as np
import pytest
import scipy.sparse as sp

from pedigree_graph import RELATIONSHIPS, PedigreeGraph
from pedigree_graph._pair_extractor import MatrixPairExtractor
from pedigree_graph._pair_utils import (
    oriented_pairs_from_sparse,
    pairs_from_groups,
    subtract_pairs,
)
from pedigree_graph._streaming_counter import StreamingPairCounter


class TestPairUtils:
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

    def test_matrix_extractor_returns_every_code_with_only_the_requested_populated(self, small_pedigree):
        pg = PedigreeGraph.from_frame(small_pedigree)
        codes = frozenset(code for code, category in RELATIONSHIPS.items() if category.degree <= 2)
        pairs = MatrixPairExtractor(pg, max_workers=1).extract(codes)
        assert list(pairs) == list(RELATIONSHIPS)
        counts = {code: len(block[0]) for code, block in pairs.items()}
        assert counts["FS"] > 0
        assert all(counts[code] == 0 for code in RELATIONSHIPS if code not in codes)

    def test_streaming_counter_does_not_write_the_estimate_cache(self, small_pedigree):
        pg = PedigreeGraph.from_frame(small_pedigree)
        raw, overlaps, clamped = StreamingPairCounter(pg).count(2)
        # The engine must not touch the graph's result cache — that's the wrapper's job.
        assert pg._estimate_cache == {}
        assert isinstance(raw, dict)
        assert set(overlaps) == set(raw)
        assert clamped == frozenset()
        assert raw["MZ"] >= 0

        # The wrapper folds the parent-offspring overlap out and caches the result.
        pg2 = PedigreeGraph.from_frame(small_pedigree)
        estimate = pg2.estimate_relationship_counts(max_degree=2)
        assert 2 in pg2._estimate_cache
        for code in estimate.requested:
            assert estimate[code] == raw[code] - overlaps[code], code


RESIDENT_MATRICES = ("_A", "_A2", "_A3", "_A4", "_A5", "_A2_shared", "_full_sib_matrix", "_half_sib_matrix")


def _resident(pg) -> list[str]:
    return sorted(name for name in RESIDENT_MATRICES if name in pg.__dict__)


class TestMatrixReleaseIsExceptionSafe:
    """Both wrappers release the adjacency powers on the failure path too (issue #4)."""

    def test_relationship_pairs_releases_when_extraction_raises(self, small_pedigree, monkeypatch):
        pg = PedigreeGraph.from_frame(small_pedigree)

        def boom(self):
            raise MemoryError("simulated failure inside extract()")

        monkeypatch.setattr(PedigreeGraph, "_mz_twin_pairs", boom)
        with pytest.raises(MemoryError):
            pg.relationship_pairs(max_degree=3)
        assert _resident(pg) == []

    def test_estimate_releases_when_its_own_warning_is_an_error(self, small_pedigree, monkeypatch):
        pg = PedigreeGraph.from_frame(small_pedigree)
        real = StreamingPairCounter.count

        def always_clamped(self, max_degree):
            raw, overlaps, _ = real(self, max_degree)
            return raw, overlaps, frozenset({"H1C"})

        monkeypatch.setattr(StreamingPairCounter, "count", always_clamped)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            with pytest.raises(RuntimeWarning):
                pg.estimate_relationship_counts(max_degree=3)
        assert _resident(pg) == []
