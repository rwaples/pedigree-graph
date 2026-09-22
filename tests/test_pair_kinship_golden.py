"""Golden lock for ``pair_kinship`` float32 bits frozen at ``v0.9.0`` (slice 13).

Replays ``tests/parity/generate_pair_kinship.py::_capture`` and compares every
pair hash, value hash, and stored ``uint32`` bit view in
``tests/data/pair_kinship_v0.9``.  The comparison is over identical pairs by
construction: a pair-hash mismatch fails before any value is looked at.  The
lock is regenerated only as a deliberate contract change, never to make a
test pass; see ``tests/parity/README.md``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

import pedigree_graph

sys.path.insert(0, str(Path(__file__).resolve().parent / "parity"))

import pedigrees
from generate_pair_kinship import _bits, _capture, _columns

DATA = Path(__file__).resolve().parent / "data" / "pair_kinship_v0.9"
MANIFEST = json.loads((DATA / "manifest.json").read_text())
FIXTURES = MANIFEST["fixtures"]
SMALL = sorted(name for name, entry in FIXTURES.items() if "file" in entry)


def _load(entry: dict, prefix: str) -> dict[str, np.ndarray]:
    with np.load(DATA / entry["file"]) as npz:
        return {name.removeprefix(prefix): npz[name] for name in npz.files if name.startswith(prefix)}


def test_manifest_covers_every_registry_code():
    for name, entry in FIXTURES.items():
        assert set(entry["pair_hashes"]) == set(pedigree_graph.RELATIONSHIPS), name
        assert set(entry["value_hashes"]) == set(pedigree_graph.RELATIONSHIPS), name


@pytest.mark.parametrize("name", SMALL)
def test_small_fixture_matches_the_golden_bits(name):
    entry = FIXTURES[name]
    fx = _load(entry, "input/")
    assert pedigrees.input_hash(fx) == entry["input_hash"]

    captured, summary = _capture(pedigree_graph, fx, full_arrays=True)
    assert summary["n"] == entry["n"]
    assert summary["pair_hashes"] == entry["pair_hashes"]
    assert summary["value_hashes"] == entry["value_hashes"]
    assert summary["self_value_hash"] == entry["self_value_hash"]

    stored_values = _load(entry, "values/")
    assert sorted(stored_values) == sorted(pedigree_graph.RELATIONSHIPS)
    for code, expected in stored_values.items():
        assert expected.dtype == np.uint32, code
        np.testing.assert_array_equal(captured[f"values/{code}"], expected, err_msg=code)
    with np.load(DATA / entry["file"]) as npz:
        np.testing.assert_array_equal(captured["self_values"], npz["self_values"])


@pytest.mark.parametrize("name", SMALL)
def test_reversed_endpoints_match_the_golden_bits(name):
    entry = FIXTURES[name]
    graph = pedigree_graph.PedigreeGraph.from_frame(_columns(_load(entry, "input/")))
    stored_pairs = _load(entry, "pairs/")
    stored_values = _load(entry, "values/")
    for code, expected in stored_values.items():
        first, second = stored_pairs[f"{code}/first"], stored_pairs[f"{code}/second"]
        np.testing.assert_array_equal(_bits(graph.pair_kinship(second, first)), expected, err_msg=code)


@pytest.mark.slow
def test_random_30k_matches_the_golden_hashes():
    name = "random_30k"
    entry = FIXTURES[name]
    fx = pedigrees.build_random(name, pedigrees.LARGE_FIXTURES[name])
    assert pedigrees.input_hash(fx) == entry["input_hash"]

    _, summary = _capture(pedigree_graph, fx, full_arrays=False)
    assert summary["n"] == entry["n"]
    assert summary["pair_hashes"] == entry["pair_hashes"]
    assert summary["value_hashes"] == entry["value_hashes"]
    assert summary["self_value_hash"] == entry["self_value_hash"]
