"""Property-based cross-engine tests for relationship-pair counting.

The scalar close-relative counts match the exact reference on all six codes
in estimate_exact_codes(). Also checks pairs<->counts agreement and degree-gating.

crates/core/tests/pair_properties.rs holds the core, which the R binding
shares, to the last two invariants; close_relative_counts and the None
above max_degree are Python, so only this file checks them.
"""

from __future__ import annotations

from _support import property_budget
from conftest import pedigree_arrays, random_pedigree
from hypothesis import given

from pedigree_graph import RELATIONSHIPS, PedigreeGraph
from pedigree_graph._registry import estimate_exact_codes

_SETTINGS = property_budget(40)
_HEAVY = property_budget(25)


@_SETTINGS
@given(pg=random_pedigree())
def test_close_relative_counts_match_exact_counts(pg):
    exact = pg.relationship_counts(max_degree=5)
    close = pg.close_relative_counts()
    assert close.requested == close.exact == estimate_exact_codes()
    for code in RELATIONSHIPS:
        assert close[code] == (exact[code] if code in close.requested else None), code


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
