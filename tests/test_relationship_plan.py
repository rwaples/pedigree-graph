"""Tests for the relationship plan layer (PGQ-004).

The plan (``REL_PLAN`` + helpers in ``_registry``) is the single source of
truth for scalar close-relative coverage. These tests pin the six-code set
and assert the counting methods agree on the registry key set.
"""

import numpy as np
import polars as pl

from pedigree_graph import PedigreeGraph
from pedigree_graph._registry import (
    REL_PLAN,
    RELATIONSHIPS,
    estimate_exact_codes,
)


def test_estimate_exact_codes_are_the_documented_six():
    # Amended ADR 0011: the scalar method computes only MZ, parent-offspring
    # and sibling counts; the internal registry/helper names are retained.
    assert estimate_exact_codes() == {"MZ", "MO", "FO", "FS", "MHS", "PHS"}
    assert all(REL_PLAN[code].estimate_exact == (code in estimate_exact_codes()) for code in RELATIONSHIPS)


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

    def test_close_relative_counts(self):
        pg = PedigreeGraph.from_frame(self._pedigree())
        assert set(pg.close_relative_counts()) == set(RELATIONSHIPS)
