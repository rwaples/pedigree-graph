"""The 0.8 estimators reduce to the frozen slice-6b golden bit for bit on dense labels.

Every fixture in ``tests/data/ne_baseline_6b`` carries dense ``0..g_max``
labels, so each gap is ``h = 1`` and the gap formula must evaluate exactly
the one-step arithmetic the 6b estimators used.  The golden was written by
``tests/parity/generate_ne_baseline.py`` at ``00a3667``; this test replays
the same fixtures through ``estimate_effective_sizes`` and compares the
serialized records field by field.  Integers, labels, and ``None`` must be
equal; floats must agree to one part in 1e12, because the regression slope
behind every scalar Ne comes from ``np.polyfit`` (LAPACK least squares) and
its last bits move with the BLAS kernel the host CPU selects, which is what
separates a GitHub runner from the machine that wrote the golden.

The golden's records predate the observed-cohort reshape, so each 0.8 record
is projected onto their dense layout first: the label vectors it does not
carry are dropped, a rate estimator's per-transition ``ne_per_gen`` is
scattered onto its target cohort (index 0 has no incoming transition), and
the family-size arrays drop their maximum parent cohort, which 6b did not
report.  Dense labels make that projection exact — the test asserts the
labels really are ``0..k-1`` before relying on it.

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
from pedigree_graph.effective_size import estimate_effective_sizes

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "parity"))

import generate_ne_baseline as golden  # noqa: E402

FIXTURES = sorted(golden.fixtures())

_LABEL_FIELDS = ("generations", "parent_generations", "transition_from", "transition_to", "final_generation")
_RATE_BASED = ("ne_inbreeding", "ne_coancestry", "ne_caballero_toro")


def _parentless_mz_pairs(pg: PedigreeGraph) -> int:
    parentless = (np.asarray(pg.mother_rows) < 0) & (np.asarray(pg.father_rows) < 0)
    return int(np.count_nonzero(parentless & (np.asarray(pg.twin_rows) >= 0)) // 2)


def _projected(name: str, result: object) -> dict:
    """One 0.8 record in the golden's dense per-label layout."""
    payload = result.to_dict()
    labels = payload.get("parent_generations") or payload.get("generations")
    if labels is not None:
        assert labels == list(range(len(labels))), f"{name}: the golden only covers dense 0..g_max labels"
    for field in _LABEL_FIELDS:
        payload.pop(field, None)
    if name in _RATE_BASED:
        payload["ne_per_gen"] = [None, *payload["ne_per_gen"]]
    if name == "ne_variance_family_size":
        payload = {key: value[:-1] if isinstance(value, list) else value for key, value in payload.items()}
    return payload


def _capture(name: str, df) -> dict[str, dict]:
    pg = PedigreeGraph.from_frame(df)
    results = estimate_effective_sizes(pg, hill_vk_scale=name.endswith("birth_years"))
    return {key: _projected(key, result) for key, result in results.items()}


@pytest.mark.parametrize("name", FIXTURES)
def test_the_estimators_match_slice_6b(name: str) -> None:
    df = golden.fixtures()[name]
    expected = json.loads((golden.OUT / f"{name}.json").read_text())
    actual = _capture(name, df)
    assert sorted(actual) == sorted(expected)

    merged_pairs = _parentless_mz_pairs(PedigreeGraph.from_frame(df))
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
        assert _floats_approx(got) == _floats_approx(want), estimator


def _floats_approx(record):
    """Wrap every float in *record* in ``pytest.approx(rel=1e-12)``; leave other values exact."""
    if isinstance(record, dict):
        return {key: _floats_approx(value) for key, value in record.items()}
    if isinstance(record, list):
        return [_floats_approx(value) for value in record]
    if isinstance(record, float):
        return pytest.approx(record, rel=1e-12, abs=0.0)
    return record


def test_small_pedigree_exercises_the_founder_genome_migration() -> None:
    assert _parentless_mz_pairs(PedigreeGraph.from_frame(golden.fixtures()["small_pedigree"])) == 3
