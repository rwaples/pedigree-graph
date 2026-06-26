"""Property-based cross-engine tests for relationship-pair counting.

The matrix engine is the exact reference. Per REL_PLAN: count_pairs_streaming is
bit-identical only for streaming_exact_codes(); count_pairs_bfs matches the matrix
engine for codes outside bfs_divergent_codes() on any input, and for all codes on
non-inbred input. Also checks extract<->count agreement and degree-gating.
"""

from __future__ import annotations

import warnings

from conftest import pedigree_arrays, random_pedigree
from hypothesis import given, settings

from pedigree_graph import REL_REGISTRY, PedigreeGraph
from pedigree_graph._registry import bfs_divergent_codes, streaming_exact_codes
from pedigree_graph.experimental import count_pairs_bfs

_SETTINGS = settings(deadline=None, max_examples=40)
_HEAVY = settings(deadline=None, max_examples=25)


def _bfs(pg, max_degree=5):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        return count_pairs_bfs(pg, max_degree=max_degree)


@_SETTINGS
@given(pg=random_pedigree())
def test_streaming_matches_matrix_on_exact_codes(pg):
    matrix = pg.count_pairs(max_degree=5, scope="full")
    streaming = pg.count_pairs_streaming(max_degree=5, scope="full")
    for code in streaming_exact_codes():
        assert streaming[code] == matrix[code], code


@_SETTINGS
@given(pg=random_pedigree())
def test_bfs_matches_matrix_on_nondivergent_codes(pg):
    matrix = pg.count_pairs(max_degree=5, scope="full")
    bfs = _bfs(pg, 5)
    divergent = bfs_divergent_codes()
    for code in REL_REGISTRY:
        if code not in divergent:
            assert bfs[code] == matrix[code], code


@_SETTINGS
@given(pg=random_pedigree())
def test_extract_and_count_agree(pg):
    pairs = pg.extract_pairs(max_degree=5)
    counts = pg.count_pairs(max_degree=5, scope="full")
    for code in REL_REGISTRY:
        assert len(pairs[code][0]) == counts[code], code


@_HEAVY
@given(arrays=pedigree_arrays())
def test_degree_gating(arrays):
    ids, mo, fa, sex = arrays
    # Fresh graph per max_degree to avoid any pair-count cache cross-talk.
    by_degree = {
        d: PedigreeGraph.from_arrays(ids=ids, mothers=mo, fathers=fa, sex=sex).count_pairs(max_degree=d, scope="full")
        for d in range(6)
    }
    for d in range(6):
        counts = by_degree[d]
        assert set(counts) == set(REL_REGISTRY)  # all 23 keys always present
        for code, rt in REL_REGISTRY.items():
            if rt.degree > d:
                assert counts[code] == 0, (d, code)
    # Counts for in-degree codes are stable as max_degree grows.
    for d in range(5):
        for code, rt in REL_REGISTRY.items():
            if rt.degree <= d:
                assert by_degree[d][code] == by_degree[d + 1][code], (d, code)
