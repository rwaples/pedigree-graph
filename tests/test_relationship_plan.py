"""Tests for the relationship plan layer (PGQ-004).

The plan (``REL_PLAN`` + helpers in ``_registry``) is the single source of
truth for per-code engine semantics — streaming exact/approximate and the
BFS distinct-vs-paths divergence — that previously lived only in three
separate docstrings.  These tests pin that source and assert all three
engines agree on the registry key set.
"""

import numpy as np
import pandas as pd
import pytest

from pedigree_graph import PedigreeGraph
from pedigree_graph._registry import (
    PAIR_KINSHIP,
    REL_PLAN,
    REL_REGISTRY,
    bfs_divergent_codes,
    streaming_approximate_codes,
    streaming_exact_codes,
)
from pedigree_graph.experimental import count_pairs_bfs


def test_plan_covers_exactly_the_registry():
    assert set(REL_PLAN) == set(REL_REGISTRY)


def test_streaming_exact_and_approximate_partition_the_registry():
    exact = streaming_exact_codes()
    approx = streaming_approximate_codes()
    assert exact.isdisjoint(approx)
    assert exact | approx == set(REL_REGISTRY)


def test_streaming_exact_codes_are_the_documented_ten():
    # The lineal + sibling + MZ codes, exact even on inbred input.
    assert streaming_exact_codes() == {
        "MZ", "MO", "FO", "FS", "MHS", "PHS", "GP", "GGP", "GGGP", "G3GP",
    }


def test_bfs_divergent_codes_are_the_four_cousin_codes():
    assert bfs_divergent_codes() == {"1C1R", "H1C1R", "1C2R", "2C"}


def test_exact_streaming_codes_never_diverge_in_bfs():
    # A code exact in the scalar engine is path-count-stable, so BFS (which
    # only diverges on path multiplicity) cannot diverge from the matrix
    # engine for it either.
    for code in streaming_exact_codes():
        assert not REL_PLAN[code].bfs_diverges_under_inbreeding, code


class TestAllEnginesReturnRegistryKeySet:
    """Every engine returns exactly the registry codes (criterion 3)."""

    def _pedigree(self):
        # Small 3-generation pedigree with full sibs and cousins; built
        # directly (BFS does not support from_subsample).
        return pd.DataFrame(
            {
                "id": np.arange(10),
                "mother": np.array([-1, -1, -1, -1, 0, 0, 2, 2, 4, 6]),
                "father": np.array([-1, -1, -1, -1, 1, 1, 3, 3, 5, 7]),
                "twin": np.full(10, -1),
                "sex": np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 0]),
                "generation": np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2]),
            }
        )

    def test_matrix_engine(self):
        pg = PedigreeGraph(self._pedigree())
        assert set(pg.count_pairs(max_degree=5)) == set(REL_REGISTRY)

    def test_streaming_engine(self):
        pg = PedigreeGraph(self._pedigree())
        assert set(pg.count_pairs_streaming(max_degree=5)) == set(REL_REGISTRY)

    @pytest.mark.filterwarnings("ignore::FutureWarning")
    def test_bfs_engine(self):
        pg = PedigreeGraph(self._pedigree())
        assert set(count_pairs_bfs(pg, max_degree=5)) == set(REL_REGISTRY)

    def test_registry_and_kinship_keys_agree(self):
        assert set(PAIR_KINSHIP) == set(REL_REGISTRY)
