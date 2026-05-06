"""Tests for the experimental BFS pair counter (count_pairs_bfs).

Covers:
- Smoke (kernel imports, JIT compiles).
- Parity vs the matrix engine on the non-inbred small_pedigree fixture.
- Hand-built tiny pedigree (mirror of TestKnownTinyPedigree).
- Full-vs-half 2nd-cousin handling (mirror of TestSecondCousinFullVsHalf).
- Inbred-with-cousins fixture exercising the documented distinct-ancestor
  vs path-multiplicity divergence.
- API contract: NotImplementedError on max_degree != 5 and on subsampled
  graphs; n_threads kwarg accepted; FutureWarning emitted.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from pedigree_graph import PAIR_KINSHIP, PedigreeGraph
from pedigree_graph.experimental import count_pairs_bfs

# Cousin-style codes that may diverge between matrix and BFS engines under
# inbreeding (matrix counts paths; BFS counts distinct shared ancestors).
COUSIN_CODES = {"1C1R", "H1C1R", "1C2R", "2C"}
NON_COUSIN_CODES = set(PAIR_KINSHIP) - COUSIN_CODES


def _bfs(pg: PedigreeGraph) -> dict[str, int]:
    """Run count_pairs_bfs with the FutureWarning silenced (test-side noise control)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        return count_pairs_bfs(pg)


def _tiny_lineage_df() -> pd.DataFrame:
    """3-individual lineage: 0,1 founders → 2 child."""
    return pd.DataFrame(
        {
            "id": np.array([0, 1, 2]),
            "mother": np.array([-1, -1, 0]),
            "father": np.array([-1, -1, 1]),
            "twin": np.array([-1, -1, -1]),
            "sex": np.array([0, 1, 0]),
            "generation": np.array([0, 0, 1]),
        }
    )


def test_kernel_imports_and_jits():
    """Smoke: count_pairs_bfs runs end-to-end on a 3-individual lineage."""
    pg = PedigreeGraph(_tiny_lineage_df())
    out = _bfs(pg)
    assert out["MO"] == 1
    assert out["FO"] == 1
    assert out["FS"] == 0
    assert out["MZ"] == 0
    # All 23 codes present.
    assert set(out) == set(PAIR_KINSHIP)


def test_bfs_matches_matrix_on_small_pedigree(small_pedigree):
    """Non-inbred fixture → BFS counts == matrix counts for all 23 codes."""
    matrix_counts = PedigreeGraph(small_pedigree).count_pairs(max_degree=5)
    bfs_counts = _bfs(PedigreeGraph(small_pedigree))
    for code in PAIR_KINSHIP:
        assert matrix_counts[code] == bfs_counts[code], (
            f"{code}: matrix={matrix_counts[code]} bfs={bfs_counts[code]}"
        )


# ---------------------------------------------------------------------------
# Hand-built tiny pedigree (mirrors TestKnownTinyPedigree)
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_pedigree():
    """3-generation pedigree; see TestKnownTinyPedigree for the layout."""
    n = 11
    return pd.DataFrame(
        {
            "id": np.arange(n),
            "mother": np.array([-1, -1, -1, -1, 0, 0, 2, 2, 4, 5, 4]),
            "father": np.array([-1, -1, -1, -1, 1, 1, 3, 3, 6, 7, 6]),
            "twin": np.full(n, -1),
            "sex": np.array([0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0]),
            "generation": np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2]),
        }
    )


def test_known_tiny_counts(tiny_pedigree):
    """Hand-counted: FS=3, MO=7, FO=7, GP=12, Av=6, 1C=2."""
    out = _bfs(PedigreeGraph(tiny_pedigree))
    assert out["FS"] == 3
    assert out["MO"] == 7
    assert out["FO"] == 7
    assert out["GP"] == 12
    assert out["Av"] == 6
    assert out["1C"] == 2


# ---------------------------------------------------------------------------
# Second-cousin full-vs-half (mirrors TestSecondCousinFullVsHalf)
# ---------------------------------------------------------------------------


@pytest.fixture
def pedigree_with_half_2c():
    """4-generation pedigree with one full-2C pair and one half-2C pair.

    See TestSecondCousinFullVsHalf for the construction; full-2C pair (10,11)
    must be in 2C, half-2C pair (23,24) must be excluded.
    """
    n = 25
    return pd.DataFrame(
        {
            "id": np.arange(n),
            "mother": np.array([
                -1, -1, 0, 0,            # 0-3
                -1, -1, 4, 5,            # 4-7
                -1, -1, 8, 9,            # 8-11
                -1, -1, -1, 12, 12,      # 12-16
                -1, -1, 17, 18,          # 17-20
                -1, -1, 21, 22,          # 21-24
            ]),
            "father": np.array([
                -1, -1, 1, 1,            # 0-3
                -1, -1, 2, 3,            # 4-7
                -1, -1, 6, 7,            # 8-11
                -1, -1, -1, 13, 14,      # 12-16
                -1, -1, 15, 16,          # 17-20
                -1, -1, 19, 20,          # 21-24
            ]),
            "twin": np.full(n, -1),
            "sex": np.array([
                0, 1, 1, 1,
                0, 0, 1, 1,
                0, 0, 0, 0,
                0, 1, 1, 1, 1,
                0, 0, 1, 1,
                0, 0, 0, 0,
            ]),
            "generation": np.array([
                0, 0, 1, 1,
                1, 1, 2, 2,
                2, 2, 3, 3,
                0, 0, 0, 1, 1,
                1, 1, 2, 2,
                2, 2, 3, 3,
            ]),
        }
    )


def test_second_cousin_full_vs_half(pedigree_with_half_2c):
    """Only 1 full 2C pair should exist in this pedigree."""
    out = _bfs(PedigreeGraph(pedigree_with_half_2c))
    assert out["2C"] == 1


# ---------------------------------------------------------------------------
# Inbred-with-cousins divergence
# ---------------------------------------------------------------------------


@pytest.fixture
def inbred_with_cousins_pedigree():
    """Hand-built pedigree exercising path-multiplicity vs distinct-ancestor.

    Layout (13 individuals, 4 generations):

      G0: 0(F), 1(M)                           mated pair
          2(M)                                  alt mate of 0
          3(F)                                  ext mother for 10
          4(M), 5(M)                            ext fathers for 11, 12
      G1: 6=child(0,1), 7=child(0,1)           full sibs (dual descent path to {0,1})
          8=child(0,2)                          half-sib of 6,7 via 0
      G2: 9=child(6,7)                          INCEST: full-sib mating; F=0.25
          10=child(3,8)                         outbred
      G3: 11=child(9,4)                         11's depth-3 = {0,1} via 2 paths each
          12=child(10,5)                        12's depth-3 = {0,2} via 1 path each

    Predicted divergences (worked out by hand from the engines' algorithms):

    * (10, 11): depth-(2,3) shared = X=0 only; matrix path count = 1*2 = 2
                so matrix says full 1C1R; BFS distinct count = 1 so it falls
                into H1C1R bucket.
    * (9, 12):  depth-(2,3) shared = X=0 only; matrix path count = 2*1 = 2
                so matrix says full 1C1R; BFS distinct = 1 → H1C1R.
    * (11, 12): depth-(3,3) shared = X=0 only; matrix path count = 2*1 = 2
                so matrix says 2C; BFS distinct = 1 → not 2C.

    Net counts on this fixture (verified hand-computation):
      matrix: 1C1R=2, H1C1R=0, 2C=1
      bfs:    1C1R=0, H1C1R=2, 2C=0
    """
    n = 13
    return pd.DataFrame(
        {
            "id": np.arange(n),
            "mother": np.array([
                -1, -1, -1, -1, -1, -1,  # 0-5 founders
                 0,  0,  0,              # 6-8 G1
                 6,  3,                  # 9-10 G2
                 9, 10,                  # 11-12 G3
            ]),
            "father": np.array([
                -1, -1, -1, -1, -1, -1,  # 0-5 founders
                 1,  1,  2,              # 6-8 G1
                 7,  8,                  # 9-10 G2
                 4,  5,                  # 11-12 G3
            ]),
            "twin": np.full(n, -1),
            "sex": np.array([
                0, 1, 1, 0, 1, 1,        # 0=F,1=M,2=M,3=F,4=M,5=M
                0, 1, 1,                 # 6=F,7=M,8=M
                0, 0,                    # 9=F,10=F
                0, 0,                    # 11,12 (sex irrelevant)
            ]),
            "generation": np.array([
                0, 0, 0, 0, 0, 0,
                1, 1, 1,
                2, 2,
                3, 3,
            ]),
        }
    )


def test_inbred_with_cousins_non_cousin_codes_match(inbred_with_cousins_pedigree):
    """All non-cousin codes must agree between matrix and BFS engines."""
    matrix_counts = PedigreeGraph(inbred_with_cousins_pedigree).count_pairs(max_degree=5)
    bfs_counts = _bfs(PedigreeGraph(inbred_with_cousins_pedigree))
    for code in NON_COUSIN_CODES:
        assert matrix_counts[code] == bfs_counts[code], (
            f"non-cousin code {code} differs: "
            f"matrix={matrix_counts[code]} bfs={bfs_counts[code]}"
        )


def test_inbred_with_cousins_cousin_codes_diverge(inbred_with_cousins_pedigree):
    """The documented inbred-pedigree quirk on cousin codes.

    Pinned values come from hand-tracing both engines' algorithms over
    this specific fixture. They are the load-bearing assertion that
    BFS's distinct-shared-ancestor counting differs from the matrix
    engine's path-multiplicity counting on inbred cousins.
    """
    matrix_counts = PedigreeGraph(inbred_with_cousins_pedigree).count_pairs(max_degree=5)
    bfs_counts = _bfs(PedigreeGraph(inbred_with_cousins_pedigree))

    # 1C1R: matrix counts (10,11), (9,12) (each via path-multiplicity 2);
    # BFS sees both as count==1 distinct → relegates them to H1C1R.
    assert matrix_counts["1C1R"] == 2
    assert bfs_counts["1C1R"] == 0

    # H1C1R: mirror image — matrix sees no count==1 path (everything is 2);
    # BFS picks up the two pairs that matrix counted as 1C1R.
    assert matrix_counts["H1C1R"] == 0
    assert bfs_counts["H1C1R"] == 2

    # 2C: matrix sees (11,12) via path-multiplicity 2; BFS sees count=1 distinct.
    assert matrix_counts["2C"] == 1
    assert bfs_counts["2C"] == 0

    # 1C2R: no eligible pairs in this fixture → both engines agree at 0.
    assert matrix_counts["1C2R"] == 0
    assert bfs_counts["1C2R"] == 0


# ---------------------------------------------------------------------------
# API contract
# ---------------------------------------------------------------------------


def test_max_degree_lt_5_raises_not_implemented():
    pg = PedigreeGraph(_tiny_lineage_df())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        with pytest.raises(NotImplementedError, match="max_degree=5"):
            count_pairs_bfs(pg, max_degree=3)


def test_subsample_raises_not_implemented(small_pedigree):
    sub = small_pedigree.head(50).copy()
    pg = PedigreeGraph.from_subsample(small_pedigree, sub)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        with pytest.raises(NotImplementedError, match="subsampled"):
            count_pairs_bfs(pg)


def test_n_threads_kwarg_accepts_int():
    """Smoke: n_threads=2 doesn't crash."""
    pg = PedigreeGraph(_tiny_lineage_df())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        out = count_pairs_bfs(pg, n_threads=2)
    assert "FS" in out


def test_future_warning_fires():
    pg = PedigreeGraph(_tiny_lineage_df())
    with pytest.warns(FutureWarning, match="experimental"):
        count_pairs_bfs(pg)


# ---------------------------------------------------------------------------
# _count_after_subtract regression: ensure subtract sets with hi.max() >
# candidate.hi.max() are handled correctly. The old pedsum code had a buggy
# rebuild path that silently dropped accumulated subtract keys.
# ---------------------------------------------------------------------------


def test_count_after_subtract_handles_subtract_with_larger_hi():
    """Subtract list whose hi.max() exceeds the candidate's hi.max().

    Builds a unified max_id base and ensures the subtract is applied
    correctly. Regression test for the latent bug in pedsum's
    _count_after_subtract.
    """
    from pedigree_graph.experimental import _count_after_subtract

    # Candidate: pairs (1,2), (3,4). Subtract list: (1,2) (overlap), and
    # (5, 100) (overlaps the candidate space; hi=100 > candidate hi=4).
    cand_lo = np.array([1, 3], dtype=np.intp)
    cand_hi = np.array([2, 4], dtype=np.intp)
    sub = [
        (np.array([1], dtype=np.intp), np.array([2], dtype=np.intp)),
        (np.array([5], dtype=np.intp), np.array([100], dtype=np.intp)),
    ]
    # Only (1,2) is in the candidate set, so 1 pair is removed → 1 left.
    assert _count_after_subtract(cand_lo, cand_hi, sub) == 1
