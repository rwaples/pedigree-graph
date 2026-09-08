"""Property-based cross-engine tests for relationship-pair counting.

The matrix engine is the exact reference. Per REL_PLAN: estimate_relationship_counts
is bit-identical only for estimate_exact_codes(); count_pairs_bfs matches the matrix
engine's classification for codes outside bfs_divergent_codes() on any input. BFS
reports a pair under every category that claims it, so the oracle is the extractor's
pre-fold classification rather than the folded relationship_counts. Also checks
pairs<->counts agreement and degree-gating.
"""

from __future__ import annotations

import warnings

from conftest import pedigree_arrays, random_pedigree
from hypothesis import given, settings

from pedigree_graph import RELATIONSHIPS, PedigreeGraph
from pedigree_graph._pair_extractor import MatrixPairExtractor, dependency_closure
from pedigree_graph._registry import bfs_divergent_codes, estimate_exact_codes
from pedigree_graph.experimental import count_pairs_bfs

_SETTINGS = settings(deadline=None, max_examples=40)
_HEAVY = settings(deadline=None, max_examples=25)


def _bfs(pg, max_degree=5):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        return count_pairs_bfs(pg, max_degree=max_degree)


def _unfolded_counts(pg):
    pairs = MatrixPairExtractor(pg, max_workers=1).extract(dependency_closure(frozenset(RELATIONSHIPS)))
    return {code: len(pairs[code][0]) for code in RELATIONSHIPS}


def _estimate(pg, max_degree=5):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return pg.estimate_relationship_counts(max_degree=max_degree)


@_SETTINGS
@given(pg=random_pedigree())
def test_estimate_matches_matrix_on_exact_codes(pg):
    matrix = pg.relationship_counts(max_degree=5)
    estimate = _estimate(pg, 5)
    assert estimate.exact == estimate_exact_codes()
    for code in sorted(estimate.exact):
        assert estimate[code] == matrix[code], code


@_SETTINGS
@given(pg=random_pedigree())
def test_bfs_matches_matrix_on_nondivergent_codes(pg):
    unfolded = _unfolded_counts(pg)
    bfs = _bfs(pg, 5)
    divergent = bfs_divergent_codes()
    for code in RELATIONSHIPS:
        if code not in divergent:
            assert bfs[code] == unfolded[code], code


@_SETTINGS
@given(pg=random_pedigree())
def test_bfs_never_reports_fewer_pairs_than_the_folded_result(pg):
    matrix = pg.relationship_counts(max_degree=5)
    bfs = _bfs(pg, 5)
    divergent = bfs_divergent_codes()
    for code in RELATIONSHIPS:
        if code not in divergent:
            assert bfs[code] >= matrix[code], code


@_SETTINGS
@given(pg=random_pedigree())
def test_pairs_and_counts_agree(pg):
    pairs = pg.relationship_pairs(max_degree=5)
    counts = pg.relationship_counts(max_degree=5)
    for code in RELATIONSHIPS:
        assert len(pairs[code]) == counts[code], code


@_HEAVY
@given(arrays=pedigree_arrays())
def test_degree_gating(arrays):
    ids, mo, fa, sex = arrays
    by_degree = {
        d: PedigreeGraph.from_arrays(ids=ids, mother_ids=mo, father_ids=fa, sex=sex).relationship_counts(max_degree=d)
        for d in range(6)
    }
    for d in range(6):
        counts = by_degree[d]
        assert set(counts) == set(RELATIONSHIPS)  # all 23 keys always present
        for code, category in RELATIONSHIPS.items():
            if category.degree > d:
                assert counts[code] is None, (d, code)
    # Counts for in-degree codes are stable as max_degree grows.
    for d in range(5):
        for code, category in RELATIONSHIPS.items():
            if category.degree <= d:
                assert by_degree[d][code] == by_degree[d + 1][code], (d, code)
