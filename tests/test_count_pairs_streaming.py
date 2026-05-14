"""Tests for ``PedigreeGraph.count_pairs_streaming``.

Pure-scalar memory-bounded count of relationship pairs.  Bit-identical
to ``count_pairs`` on deep non-inbred twin-free pedigrees for the
simple codes; approximate on cousin/collateral codes when the
underlying pedigree breaks one of those conditions.
"""

import numpy as np
import pandas as pd
import pytest

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


def test_streaming_populates_cache(small_pedigree):
    """After streaming, count_pairs() returns the cached dict."""
    pg = PedigreeGraph(small_pedigree)
    s = pg.count_pairs_streaming(max_degree=5, scope="full")
    # Now count_pairs should return the same dict.
    cached = pg.count_pairs(max_degree=5, scope="full")
    assert s == cached


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
