"""Tests for the relationship plan layer (PGQ-004).

The plan (``REL_PLAN`` + helpers in ``_registry``) is the single source of
truth for per-code engine semantics — the scalar estimate's exactness and the
BFS distinct-vs-paths divergence — that previously lived only in separate
docstrings.  These tests pin that source and assert all three engines agree on
the registry key set.
"""

import numpy as np
import polars as pl
import pytest

from pedigree_graph import PedigreeGraph
from pedigree_graph._registry import (
    PAIR_KINSHIP,
    REL_PLAN,
    RELATIONSHIPS,
    bfs_divergent_codes,
    estimate_exact_codes,
)
from pedigree_graph.experimental import count_pairs_bfs


def test_estimate_exact_codes_are_the_documented_six():
    # ADR 0011: the scalar estimate equals relationship_counts only for
    # MZ, parent-offspring, and the sibling codes.
    assert estimate_exact_codes() == {"MZ", "MO", "FO", "FS", "MHS", "PHS"}
    assert all(REL_PLAN[code].estimate_exact == (code in estimate_exact_codes()) for code in RELATIONSHIPS)


def test_bfs_divergent_codes_are_the_four_cousin_codes():
    assert bfs_divergent_codes() == {"1C1R", "H1C1R", "1C2R", "2C"}


def test_estimate_exact_codes_never_diverge_in_bfs():
    # A code exact in the scalar engine is path-count-stable, so BFS (which
    # only diverges on path multiplicity) cannot diverge from the matrix
    # engine for it either.
    for code in estimate_exact_codes():
        assert not REL_PLAN[code].bfs_diverges_under_inbreeding, code


class TestAllEnginesReturnRegistryKeySet:
    """Every engine returns exactly the registry codes (criterion 3)."""

    def _pedigree(self):
        return pl.DataFrame(
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
        pg = PedigreeGraph.from_frame(self._pedigree())
        assert set(pg.relationship_counts(max_degree=5)) == set(RELATIONSHIPS)

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_estimate_engine(self):
        pg = PedigreeGraph.from_frame(self._pedigree())
        assert set(pg.estimate_relationship_counts(max_degree=5)) == set(RELATIONSHIPS)

    @pytest.mark.filterwarnings("ignore::FutureWarning")
    def test_bfs_engine(self):
        pg = PedigreeGraph.from_frame(self._pedigree())
        assert set(count_pairs_bfs(pg, max_degree=5)) == set(RELATIONSHIPS)

    def test_registry_and_kinship_keys_agree(self):
        assert set(PAIR_KINSHIP) == set(RELATIONSHIPS)
