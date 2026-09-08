"""Slice 6c reduces to slice 6b bit for bit on dense ``0..g_max`` labels.

Every fixture in ``tests/data/ne_baseline_6b`` carries dense labels, so each
gap is ``h = 1`` and the gap formula must evaluate exactly the one-step
arithmetic the 6b estimators used.  The golden was written by
``tests/parity/generate_ne_baseline.py`` at ``00a3667``; this test replays
the same fixtures through the ``compute_all_ne`` adapter and compares the
serialized records field by field, with float equality.

One documented migration is allowed through: 6b gave each parentless MZ
co-twin its own Caballero-Toro founder column, 6c gives the pair one
founder-genome column (ADR 0008), so ``n_founders_with_descendants_per_gen``
may drop by at most the number of parentless MZ pairs in the fixture, and
the long-term-contribution sum of squares runs over fewer columns, so its
floating summation order moves in the last bit.  The shipped
``small_pedigree`` carries three such pairs; every other fixture is exact.

The second migration is a fix: 6b accepted any negative regression slope, so
a flat series regressed to an Ne of order 1e16 from least-squares noise (the
``skip_gen`` inbreeding scalar in the golden, slope ``-6e-18``).  6c reports
no estimate for a slope above ``-1e-12``; the golden's noise Ne must become
``None`` and nothing else may change.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from pedigree_graph import PedigreeGraph

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "parity"))

import generate_ne_baseline as golden  # noqa: E402

FIXTURES = sorted(golden.fixtures())


def _parentless_mz_pairs(pg: PedigreeGraph) -> int:
    parentless = (np.asarray(pg.mother_rows) < 0) & (np.asarray(pg.father_rows) < 0)
    return int(np.count_nonzero(parentless & (np.asarray(pg.twin_rows) >= 0)) // 2)


@pytest.mark.parametrize("name", FIXTURES)
def test_compute_all_ne_matches_slice_6b(name: str) -> None:
    df = golden.fixtures()[name]
    expected = json.loads((golden.OUT / f"{name}.json").read_text())
    actual = golden.capture(name, df)
    assert sorted(actual) == sorted(expected)

    merged_pairs = _parentless_mz_pairs(PedigreeGraph(df))
    if merged_pairs:
        ct_key = "n_founders_with_descendants_per_gen"
        ct_drop = np.asarray(expected["ne_caballero_toro"][ct_key]) - np.asarray(actual["ne_caballero_toro"][ct_key])
        assert np.all((ct_drop >= 0) & (ct_drop <= merged_pairs))
        expected["ne_caballero_toro"].pop(ct_key)
        actual["ne_caballero_toro"].pop(ct_key)
        for key in ("sum_c_squared", "max_delta_final", "ne"):
            got = actual["ne_long_term_contributions"].pop(key)
            want = expected["ne_long_term_contributions"].pop(key)
            assert got == pytest.approx(want, rel=1e-12, abs=0.0) if want is not None else got is None
    for estimator in expected:
        want, got = expected[estimator], actual[estimator]
        if want.get("ne") is not None and -1e-12 < (want.get("slope") or -1.0) < 0:
            assert got["ne"] is None, f"{estimator}: noise slope {want['slope']} must give no estimate"
            want, got = {**want, "ne": None}, {**got}
        assert got == want, estimator


def test_small_pedigree_exercises_the_founder_genome_migration() -> None:
    assert _parentless_mz_pairs(PedigreeGraph(golden.fixtures()["small_pedigree"])) == 3
