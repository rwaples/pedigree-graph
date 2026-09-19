"""Differential parity against the frozen pedigree-graph 0.7.1 baseline.

Replays ``tests/parity/capture_v08.py::capture``, the 0.8-API reproduction of
the frozen generator's layout, against the installed package and compares
every count and SHA-256 in ``tests/data/parity_v0.7.1/manifest.json``.  The
baseline is the oracle and is never regenerated to make a test pass; a
mismatch is reported down to the first differing element of the stored array.
The manifest's ``streaming_counts`` are the one section left uncompared:
``capture_v08`` explains why the fold-aware 0.8 estimate cannot reproduce them.

Documented 0.8 divergences:

* ADR 0008 made ``compute_inbreeding`` MZ-aware (genome-node Meuwissen-Luo
  walk), landed on ``main`` as ``638b4b4``, after the baseline was frozen.  On
  fixtures with MZ twins the ``inbreeding`` hash is therefore expected to
  differ.  The exemption is held tight: every row that is not an MZ twin or a
  descendant of one must still match 0.7.1 exactly.
* ADR 0009 made pair kinship a pinned float32 recurrence (slice 5a).  Every
  0.7.1 value that float32 can hold is unchanged, which is every fixture but
  ``deep_inbred_60g``; there the values must lie within the ADR 0009
  cross-order envelope of the frozen float64 ones.
* Slice 5b preserves the 0.7.1 propagation-pruned matrix support but replaces
  its approximate propagated values with the pinned recurrence values.  The
  frozen ``approx_values`` hash is therefore exempt; ``approx_support`` stays
  exact, and ``test_kinship_matrices`` checks every retained new value against
  ``pair_kinship`` bits.
* Issue #25 made ``mean_kinship_by_generation`` genome-node: MZ co-twins are
  one genome and contribute one representative row, where 0.7.1 kept both and
  counted that genome's pairs with the rest of the cohort twice.  0.7.1 is a
  frozen record of that convention, not an independent oracle for it, so on
  fixtures with MZ twins the ``per_gen_mean_kinship`` hash is expected to
  differ.  Twelve of the seventeen small fixtures carry no MZ twin; within the
  five that do, the kinship values themselves did not move — only which pairs
  are averaged — so any cohort holding no MZ twin must still match 0.7.1
  exactly; that check is thin where twins reach every cohort (none of
  ``small_pedigree``'s three, one of ``random_1k``'s seven).  What pins the new
  convention is not this gate but the hand-computed expectations in
  ``test_generation_kinship_summary``.

Measured coverage, asserted by ``test_exemptions_are_values_only`` so that
these numbers cannot rot into a general claim of tightness: across the
seventeen small fixtures the manifest holds 883 structural keys — pair blocks
in graph and in view rows, matrix supports, depth, ancestor and descendant
counts — of which none is exempt, and 459 value keys, of which 50 are.  Every
exemption above is a value the package deliberately redefined, so no fixture
compares in full, and the structural oracle is untouched.  What pins the
redefined values is the hand-computed expectations in
``test_generation_kinship_summary``, ``test_inbreeding`` and
``test_kinship_matrices``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from pedigree_graph._kinship_dp import _build_kinship_csc

sys.path.insert(0, str(Path(__file__).resolve().parent / "parity"))

import pedigrees
from capture_v08 import APPROX_THRESHOLD, build, capture

DATA = Path(__file__).resolve().parent / "data" / "parity_v0.7.1"
MANIFEST = json.loads((DATA / "manifest.json").read_text())
FIXTURES = MANIFEST["fixtures"]
SMALL = sorted(name for name, entry in FIXTURES.items() if "file" in entry)

MZ_AWARE_F = "inbreeding"
MZ_AWARE_THETA = "per_gen_mean_kinship"
FLOAT32_PAIR_KINSHIP = frozenset({"deep_inbred_60g"})
CORRECTED_APPROXIMATE_MATRIX_VALUE = "approx_values"
VALUE_HASH_KEYS = frozenset({MZ_AWARE_F, MZ_AWARE_THETA, CORRECTED_APPROXIMATE_MATRIX_VALUE, "complete_values"})

_PAIRED_ARRAYS = {
    "approx_support": ("approx/row", "approx/col"),
    "approx_values": ("approx/val",),
    "complete_support": ("complete/row", "complete/col"),
    "complete_values": ("complete/val",),
}


def _arrays_behind(hash_key: str) -> tuple[str, ...]:
    """Names of the stored arrays a manifest hash key was computed from."""
    if hash_key in _PAIRED_ARRAYS:
        return _PAIRED_ARRAYS[hash_key]
    if hash_key.startswith("pairs/"):
        return (f"{hash_key}/first", f"{hash_key}/second")
    return (hash_key,)


def _first_difference(expected: np.ndarray, actual: np.ndarray) -> str:
    if expected.shape != actual.shape:
        return f"shape {expected.shape} != {actual.shape}"
    unequal = expected != actual
    if np.issubdtype(expected.dtype, np.floating) and np.issubdtype(actual.dtype, np.floating):
        unequal &= ~(np.isnan(expected) & np.isnan(actual))
    positions = np.flatnonzero(unequal)
    if positions.size == 0:
        return "arrays equal but hashes differ (dtype or shape metadata)"
    at = int(positions[0])
    return f"{positions.size} of {expected.size} differ; first at {at}: {expected[at]!r} != {actual[at]!r}"


def _diagnose(hash_key: str, stored: dict[str, np.ndarray], captured: dict[str, np.ndarray]) -> str:
    details = []
    for name in _arrays_behind(hash_key):
        if name not in stored or name not in captured:
            details.append(f"{name}: not stored")
            continue
        details.append(f"{name}: {_first_difference(stored[name], captured[name])}")
    return "; ".join(details)


def _compare_hashes(
    expected: dict[str, str],
    actual: dict[str, str],
    stored: dict[str, np.ndarray],
    captured: dict[str, np.ndarray],
    *,
    exempt: frozenset[str] = frozenset(),
    prefix: str = "",
) -> list[str]:
    assert sorted(actual) == sorted(expected), f"{prefix}hash keys changed"
    return [
        f"{prefix}{key}: {_diagnose(key, stored, captured)}"
        for key, digest in expected.items()
        if key not in exempt and actual[key] != digest
    ]


def _is_value_key(key: str) -> bool:
    """Whether a manifest hash key is a scientific value rather than structure.

    Structure is membership and layout: which pairs exist, which cells the
    matrices support, how deep a row sits, how many ancestors it has.  A value
    is a number the package could decide to compute differently without the
    pedigree changing.
    """
    return key.startswith("pair_kinship/") or key in VALUE_HASH_KEYS


def _exempt_keys(name: str, entry: dict, *, has_twins: bool) -> frozenset[str]:
    """The documented divergences of the module docstring that apply to *name*."""
    exempt = {CORRECTED_APPROXIMATE_MATRIX_VALUE}
    if has_twins:
        exempt |= {MZ_AWARE_F, MZ_AWARE_THETA}
    if name in FLOAT32_PAIR_KINSHIP:
        exempt |= {key for key in entry["hashes"] if key.startswith("pair_kinship/")}
    return frozenset(exempt)


def _input_from_npz(entry: dict) -> dict[str, np.ndarray]:
    with np.load(DATA / entry["file"]) as npz:
        return {name.removeprefix("input/"): npz[name] for name in npz.files if name.startswith("input/")}


def _stored_arrays(entry: dict) -> dict[str, np.ndarray]:
    with np.load(DATA / entry["file"]) as npz:
        return {name: npz[name] for name in npz.files if not name.startswith("input/")}


def _mz_affected_rows(graph) -> np.ndarray:
    """MZ twins and every descendant of one, in graph rows.

    These are exactly the rows ADR 0008's genome-node walk may move away
    from the 0.7.1 MZ-naive value.
    """
    affected = np.asarray(graph.twin_rows) >= 0
    mother, father = np.asarray(graph.mother_rows), np.asarray(graph.father_rows)
    known = [(parents, parents >= 0) for parents in (mother, father)]
    previous = -1
    while affected.sum() != previous:
        previous = int(affected.sum())
        for parents, has_parent in known:
            inherited = np.zeros(graph.n_individuals, dtype=bool)
            inherited[has_parent] = affected[parents[has_parent]]
            affected |= inherited
    return affected


@pytest.mark.parametrize("name", SMALL)
def test_small_fixture_matches_the_frozen_baseline(name):
    entry = FIXTURES[name]
    fx = _input_from_npz(entry)
    assert pedigrees.input_hash(fx) == entry["input_hash"]

    captured, summary = capture(fx, full_arrays=True)
    stored = _stored_arrays(entry)

    assert summary["n"] == entry["n"]
    assert summary["counts"] == entry["counts"]
    assert summary.get("n_descendants_overflow") == entry.get("n_descendants_overflow")
    assert summary["subsample"]["counts"] == entry["subsample"]["counts"]

    has_twins = bool((fx["twin"] >= 0).any())
    problems = _compare_hashes(
        entry["hashes"],
        summary["hashes"],
        stored,
        captured,
        exempt=_exempt_keys(name, entry, has_twins=has_twins),
    )
    problems += _compare_hashes(
        entry["subsample"]["hashes"],
        summary["subsample"]["hashes"],
        {key: array for key, array in stored.items() if key.startswith("subsample/pairs/")},
        {key: array for key, array in captured.items() if key.startswith("subsample/pairs/")},
        prefix="subsample/",
    )
    assert not problems, "0.7.1 parity broken:\n  " + "\n  ".join(problems)

    if has_twins:
        graph = build(fx)
        unaffected = ~_mz_affected_rows(graph)
        np.testing.assert_array_equal(captured["inbreeding"][unaffected], stored["inbreeding"][unaffected])
        # These fixtures carry no supplied labels, so a cohort is a depth, and
        # co-twins share a depth.  Only a depth holding an MZ twin can move.
        depth = np.asarray(graph.depth)
        twinned = np.zeros(captured[MZ_AWARE_THETA].shape[0], dtype=bool)
        twinned[depth[np.asarray(graph.twin_rows) >= 0]] = True
        np.testing.assert_array_equal(captured[MZ_AWARE_THETA][~twinned], stored[MZ_AWARE_THETA][~twinned])
    if name in FLOAT32_PAIR_KINSHIP:
        _assert_pair_kinship_within_envelope(stored, captured)


def _assert_pair_kinship_within_envelope(stored: dict[str, np.ndarray], captured: dict[str, np.ndarray]) -> None:
    """The float32 recurrence stays within ``2 * (depth_a + depth_b + 1) * 2**-25`` of the 0.7.1 float64 value."""
    depth = stored["depth"]
    checked = 0
    for key, frozen in stored.items():
        if not key.startswith("pair_kinship/") or frozen.size == 0:
            continue
        code = key.removeprefix("pair_kinship/")
        first, second = stored[f"pairs/{code}/first"], stored[f"pairs/{code}/second"]
        tolerance = 2.0 * (depth[first] + depth[second] + 1) * 2.0**-25
        deviation = np.abs(frozen - captured[key])
        assert np.all(deviation <= tolerance), (
            f"{key}: max deviation {deviation.max():.3e} beyond the ADR 0009 envelope"
        )
        assert (captured[key] == 0).tolist() == (frozen == 0).tolist(), f"{key}: a zero flipped"
        checked += frozen.size
    assert checked > 0


def _propagated_candidate_for_large_parity(graph):
    """Build old candidate values so this differential gate isolates support."""
    indptr, indices, data = _build_kinship_csc(
        graph.n_individuals,
        graph.mother_rows,
        graph.father_rows,
        graph.twin_rows,
        graph.depth,
        APPROX_THRESHOLD,
    )
    return sp.csc_matrix((data, indices, indptr), shape=(graph.n_individuals, graph.n_individuals))


def test_exemptions_are_values_only():
    """No structural key is ever exempt, and the split is what the docstring claims.

    The counts are asserted, not merely reported, because the claim they back
    is the one that rotted last time: a divergence that reached a pair block, a
    matrix support or a depth would be a regression rather than a decision, and
    a fixture added without revisiting the docstring would leave it overstating
    the coverage again.
    """
    totals = dict.fromkeys(("structural", "value"), 0)
    exempted = dict.fromkeys(("structural", "value"), 0)
    for name in SMALL:
        entry = FIXTURES[name]
        has_twins = bool((_input_from_npz(entry)["twin"] >= 0).any())
        exempt = _exempt_keys(name, entry, has_twins=has_twins)
        assert exempt <= set(entry["hashes"]), name
        for key in entry["hashes"]:
            bucket = "value" if _is_value_key(key) else "structural"
            totals[bucket] += 1
            exempted[bucket] += key in exempt
        # View-space pair blocks, exempt nowhere.
        totals["structural"] += len(entry["subsample"]["hashes"])

    assert exempted["structural"] == 0
    assert totals == {"structural": 883, "value": 459}
    assert exempted["value"] == 50


def test_subsample_hash_key_mapping_covers_every_stored_array():
    entry = FIXTURES["random_1k"]
    stored = set(_stored_arrays(entry))
    covered = {name for key in entry["hashes"] for name in _arrays_behind(key)}
    covered |= {f"subsample/pairs/{code}/{end}" for code in entry["subsample"]["hashes"] for end in ("first", "second")}
    assert stored - covered == set()


@pytest.mark.slow
def test_random_30k_matches_the_frozen_baseline():
    name = "random_30k"
    entry = FIXTURES[name]
    fx = pedigrees.build_random(name, pedigrees.LARGE_FIXTURES[name])
    assert pedigrees.input_hash(fx) == entry["input_hash"]

    _, summary = capture(fx, full_arrays=False, approximate_matrix=_propagated_candidate_for_large_parity)

    assert summary["n"] == entry["n"]
    assert summary["counts"] == entry["counts"]
    assert summary["subsample"]["counts"] == entry["subsample"]["counts"]
    assert summary["subsample"]["hashes"] == entry["subsample"]["hashes"]
    exempt = _exempt_keys(name, entry, has_twins=bool((fx["twin"] >= 0).any()))
    assert {k: v for k, v in summary["hashes"].items() if k not in exempt} == {
        k: v for k, v in entry["hashes"].items() if k not in exempt
    }


def test_random_300k_is_release_only():
    pytest.skip("random_300k is a release/performance gate (plan slice 9), not part of the suite")
