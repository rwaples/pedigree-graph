"""Tests for ``PedigreeGraph.count_pairs_streaming``.

Pure-scalar memory-bounded count of relationship pairs.  Bit-identical
to ``count_pairs`` on deep non-inbred twin-free pedigrees for the
simple codes; approximate on cousin/collateral codes when the
underlying pedigree breaks one of those conditions.
"""

import numpy as np

from pedigree_graph import PedigreeGraph


def test_streaming_returns_all_23_codes(small_pedigree):
    pg = PedigreeGraph(small_pedigree)
    counts = pg.count_pairs_streaming(max_degree=5)
    from pedigree_graph import REL_REGISTRY
    assert set(counts.keys()) == set(REL_REGISTRY.keys())


def test_streaming_exact_codes_match_matrix(small_pedigree):
    """Lineal + sibling + MZ codes are exact even on inbred fixtures."""
    pg_s = PedigreeGraph(small_pedigree)
    pg_m = PedigreeGraph(small_pedigree)
    s = pg_s.count_pairs_streaming(max_degree=5, scope="full")
    m = pg_m.count_pairs(max_degree=5, scope="full")
    for code in ("MZ", "MO", "FO", "FS", "MHS", "PHS",
                 "GP", "GGP", "GGGP", "G3GP"):
        assert s[code] == m[code], (
            f"streaming {code}={s[code]} but matrix engine returns {m[code]}"
        )


def test_streaming_cousin_codes_approximate(small_pedigree):
    """Cousin codes (1C, H1C, ...) may diverge from matrix engine.

    Documented compromise; this test pins that the divergence is at
    least bounded (not orders-of-magnitude off).  H1C may clamp to 0
    on shallow pedigrees — accept that explicitly.
    """
    pg_s = PedigreeGraph(small_pedigree)
    pg_m = PedigreeGraph(small_pedigree)
    s = pg_s.count_pairs_streaming(max_degree=5, scope="full")
    m = pg_m.count_pairs(max_degree=5, scope="full")
    # 1C: within 2% on this fixture.
    assert abs(s["1C"] - m["1C"]) <= max(50, m["1C"] // 50)
    # HAv: within 2%.
    assert abs(s["HAv"] - m["HAv"]) <= max(50, m["HAv"] // 50)
    # H1C: clamps to 0 on shallow pedigrees with high founder ratio.
    assert s["H1C"] >= 0  # never negative
    assert s["H1C"] <= m["H1C"] + 1  # never over-counts on this fixture


def test_streaming_max_degree_zero_higher_codes(small_pedigree):
    pg = PedigreeGraph(small_pedigree)
    counts = pg.count_pairs_streaming(max_degree=2)
    # Codes above degree 2 should be 0.
    for code in ("GGP", "GGGP", "G3GP", "1C", "H1C", "1C1R", "H1C1R", "1C2R", "2C",
                 "GAv", "GGAv", "G3Av", "HGAv", "HGGAv"):
        assert counts[code] == 0, f"{code} should be 0 at max_degree=2, got {counts[code]}"


def test_cache_does_not_leak_between_engines(small_pedigree):
    """Streaming and matrix engines must not return each other's cached counts.

    Each engine has its own keyed cache entry; an earlier matrix-engine
    call must not short-circuit a later streaming call on the same pg.
    """
    pg_ref = PedigreeGraph(small_pedigree)
    streaming_expected = pg_ref.count_pairs_streaming(max_degree=5, scope="full")

    pg = PedigreeGraph(small_pedigree)
    _ = pg.count_pairs(max_degree=5, scope="full")
    streaming_after = pg.count_pairs_streaming(max_degree=5, scope="full")
    assert streaming_after == streaming_expected


def test_matrix_cache_does_not_leak_across_max_degree(small_pedigree):
    """count_pairs(max_degree=M2) after count_pairs(max_degree=M1<M2)
    must compute the M2 result, not return the cached M1 result."""
    pg_ref = PedigreeGraph(small_pedigree)
    expected = pg_ref.count_pairs(max_degree=5, scope="full")

    pg = PedigreeGraph(small_pedigree)
    shallow = pg.count_pairs(max_degree=2, scope="full")
    deep = pg.count_pairs(max_degree=5, scope="full")
    assert deep == expected
    # Sanity: at least one degree-3+ code must be non-zero in the deep
    # call so the test actually exercises the bug (zero on shallow).
    assert any(deep[c] > 0 for c in ("GGP", "GAv", "1C", "HAv")), (
        "fixture too shallow to exercise cache-leak detection"
    )
    assert any(shallow[c] == 0 for c in ("GGP", "GAv", "1C", "HAv"))


def test_streaming_cache_does_not_leak_across_max_degree(small_pedigree):
    """count_pairs_streaming(max_degree=M2) after streaming(M1<M2)
    must compute the M2 result, not return the cached M1 result."""
    pg_ref = PedigreeGraph(small_pedigree)
    expected = pg_ref.count_pairs_streaming(max_degree=5, scope="full")

    pg = PedigreeGraph(small_pedigree)
    _ = pg.count_pairs_streaming(max_degree=2, scope="full")
    deep = pg.count_pairs_streaming(max_degree=5, scope="full")
    assert deep == expected


def test_matrix_cache_does_not_leak_across_min_kinship(small_pedigree):
    """After extract_pairs(min_kinship=0.125), count_pairs(max_degree=...)
    must NOT reuse the restricted-threshold cache for the default
    min_kinship=0.0.  Cousin codes skipped by the restricted call would
    otherwise stay at zero."""
    pg_ref = PedigreeGraph(small_pedigree)
    expected = pg_ref.count_pairs(max_degree=5, scope="full")

    pg = PedigreeGraph(small_pedigree)
    # 0.125 = degree-2 floor; 1C (1/16), HAv (1/16) and all degree-3+
    # cousin/collateral codes (kinship <= 1/16) are skipped.
    _ = pg.extract_pairs(max_degree=5, min_kinship=0.125)
    fresh = pg.count_pairs(max_degree=5, scope="full")
    assert fresh == expected
    # Sanity: the fixture must have a non-zero code below 0.125 so the
    # cache-leak case would actually return a wrong answer.
    assert any(fresh[c] > 0 for c in ("1C", "HAv", "GGP", "GAv"))


def test_streaming_n1_founder():
    """N=1 founder returns all-zero counts."""
    pg = PedigreeGraph.from_arrays(
        ids=np.array([0]),
        mothers=np.array([-1]),
        fathers=np.array([-1]),
    )
    counts = pg.count_pairs_streaming(max_degree=5)
    for code, c in counts.items():
        assert c == 0, f"{code} = {c} on N=1, expected 0"
