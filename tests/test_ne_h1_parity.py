"""The 0.8 estimators reduce to the frozen slice-6b golden bit for bit on dense labels.

Every fixture in ``tests/data/ne_baseline_6b`` carries dense ``0..g_max``
labels, so each gap is ``h = 1`` and the gap formula must evaluate exactly
the one-step arithmetic the 6b estimators used.  The golden was written by
``tests/parity/generate_ne_baseline.py`` at ``00a3667``; this test replays
the same fixtures through ``estimate_effective_sizes`` and compares the
serialized records field by field.  Integers, labels, and ``None`` must be
equal; floats must agree to one part in 1e9, or to 1e-14 absolutely for a
slope that is itself least-squares noise, because the regression slope behind
every scalar Ne comes from ``np.polyfit`` (LAPACK least squares) and its low
bits move with the BLAS build and kernel the host selects: the PyPI numpy on a
GitHub runner has reproduced relative differences of 3e-12 in a slope and the
Ne derived from it, and a flat series regresses to a slope of order 1e-16 whose
digits are entirely noise.

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
may drop by at most the number of parentless MZ pairs in the fixture.  The
shipped ``small_pedigree`` carries three such pairs; every other fixture is
exact.

The second migration is a fix: 6b accepted any negative regression slope, so
a flat series regressed to an Ne of order 1e16 from least-squares noise (the
``skip_gen`` inbreeding scalar in the golden, slope ``-6e-18``).  6c reports
no estimate for a slope above ``-1e-12``; the golden's noise Ne must become
``None`` and nothing else may change.

The third is a correction the golden has not caught up with, so
``ne_individual_delta_f`` is excluded from the field comparison outright.  It
used ``ΔF_i = 1 − (1 − F_i)^(1/(t−1))`` over rows with ``t > 1``, aggregated by
harmonic mean of the per-cohort Ne; Gutiérrez eq. 2 is ``1/t`` over rows with
``t > 0``, and §2.1 averages ΔF_i over a reference subpopulation (ADR 0012,
issue #15).  Both the values and the record shape changed by design — the
record gained ``standard_error``, ``n_reference``, ``reference_generation``
and ``ne_unrelated_founders`` — so no field of the 6b payload still describes
the estimator.  Its key stays in the key-set assertion, and the exclusion goes
away when ``tests/data/ne_baseline_6b`` is regenerated at the end of the
issue-15 work.

The fourth is the same, so ``ne_long_term_contributions`` is excluded
outright too.  It reported ``1/(2·Σc²)``; Wray & Thompson 1990 eq. 31 is
``Ne ≈ 2N/(μ_r² + σ_r²)``, which at ``μ_r = 1`` gives ``Σr² = N·Σc²`` and so
``Ne = 2/Σc²``, and Caballero & Toro 2000 eq. 19 is
``N_ef = 1/[(1/N²)Σc²_{i(0,t)}]``, i.e. ``N_ef = 1/Σc²``, with their own text
after eq. 20 giving ``N_ef = Ne/2`` (ADR 0012, issue #15).  The record shape
changed with the values: it gained ``n_effective_founders``, and
``n_iterations`` became ``n_cohorts``, which counts observed cohorts rather
than the adjacent comparisons the deleted convergence loop made.  The golden
carries an ``ne`` on ``closed_line_5`` alone, 1.0 against the corrected 4.0;
on the other five ``tol = 1e-6`` was tighter than the contributions' own
fluctuation, so 6b reported no estimate at all, which is the symptom that hid
the error.  Its key stays in the key-set assertion, and this exclusion goes
away with the same regeneration.
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
_REWRITTEN_SINCE_THE_GOLDEN = ("ne_individual_delta_f", "ne_long_term_contributions")


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
    for estimator in expected:
        if estimator in _REWRITTEN_SINCE_THE_GOLDEN:
            continue
        want, got = expected[estimator], actual[estimator]
        if want.get("ne") is not None and -1e-12 < (want.get("slope") or -1.0) < 0:
            assert got["ne"] is None, f"{estimator}: noise slope {want['slope']} must give no estimate"
            want, got = {**want, "ne": None}, {**got}
        assert _floats_approx(got) == _floats_approx(want), estimator


def _floats_approx(record):
    """Wrap every float in *record* in ``pytest.approx(rel=1e-9, abs=1e-14)``; leave other values exact."""
    if isinstance(record, dict):
        return {key: _floats_approx(value) for key, value in record.items()}
    if isinstance(record, list):
        return [_floats_approx(value) for value in record]
    if isinstance(record, float):
        return pytest.approx(record, rel=1e-9, abs=1e-14)
    return record


def test_small_pedigree_exercises_the_founder_genome_migration() -> None:
    assert _parentless_mz_pairs(PedigreeGraph.from_frame(golden.fixtures()["small_pedigree"])) == 3
