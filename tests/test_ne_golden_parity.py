"""Every estimator reproduces the stored 0.9 golden, record for record.

``tests/parity/generate_ne_baseline_0_9.py`` builds the fixtures and writes
``tests/data/ne_baseline_0_9`` through the same ``estimate_effective_sizes``
call this test replays, so the golden pins the numbers and the record shape of
all eight estimators and moves only when someone regenerates it deliberately.

Integers, labels, and ``None`` must be equal; floats must agree to one part in
1e9, or to 1e-14 absolutely for a slope that is itself least-squares noise,
because the regression slope behind every scalar Ne comes from ``np.polyfit``
(LAPACK least squares) and its low bits move with the BLAS build and kernel the
host selects: the PyPI numpy on a GitHub runner has reproduced relative
differences of 3e-12 in a slope and the Ne derived from it, and a flat series
regresses to a slope of order 1e-16 whose digits are entirely noise.
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

import generate_ne_baseline_0_9 as golden  # noqa: E402

FIXTURES = sorted(golden.fixtures())


def _floats_approx(record):
    """Wrap every float in *record* in ``pytest.approx(rel=1e-9, abs=1e-14)``; leave other values exact."""
    if isinstance(record, dict):
        return {key: _floats_approx(value) for key, value in record.items()}
    if isinstance(record, list):
        return [_floats_approx(value) for value in record]
    if isinstance(record, float):
        return pytest.approx(record, rel=1e-9, abs=1e-14)
    return record


def _parentless_mz_pairs(pg: PedigreeGraph) -> int:
    parentless = (np.asarray(pg.mother_rows) < 0) & (np.asarray(pg.father_rows) < 0)
    return int(np.count_nonzero(parentless & (np.asarray(pg.twin_rows) >= 0)) // 2)


@pytest.mark.parametrize("name", FIXTURES)
def test_the_estimators_match_the_golden(name: str) -> None:
    expected = json.loads((golden.OUT / f"{name}.json").read_text())
    actual = golden.capture(name, golden.fixtures()[name])

    assert sorted(actual) == sorted(expected)
    for estimator, want in expected.items():
        assert _floats_approx(actual[estimator]) == _floats_approx(want), estimator


def test_the_corpus_covers_the_founder_genome_collapse() -> None:
    assert _parentless_mz_pairs(PedigreeGraph.from_frame(golden.fixtures()["small_pedigree"])) == 3
