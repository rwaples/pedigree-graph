"""Golden lock for ``relationship_pairs(max_degree=5)`` (slice 4a).

Replays ``tests/parity/generate_relationship_pairs.py::_capture`` and compares
every count, hash, and stored array in ``tests/data/relationship_pairs_v0.8``.
The lock is regenerated only as a deliberate contract change, never to make a
test pass; see ``tests/parity/README.md``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

import pedigree_graph
from pedigree_graph._pair_extractor import check_exclusive

sys.path.insert(0, str(Path(__file__).resolve().parent / "parity"))

import pedigrees
from generate_relationship_pairs import MAX_DEGREE, _capture, _columns

DATA = Path(__file__).resolve().parent / "data" / "relationship_pairs_v0.8"
MANIFEST = json.loads((DATA / "manifest.json").read_text())
FIXTURES = MANIFEST["fixtures"]
SMALL = sorted(name for name, entry in FIXTURES.items() if "file" in entry)


def _input_from_npz(entry: dict) -> dict[str, np.ndarray]:
    with np.load(DATA / entry["file"]) as npz:
        return {name.removeprefix("input/"): npz[name] for name in npz.files if name.startswith("input/")}


def _stored_arrays(entry: dict) -> dict[str, np.ndarray]:
    with np.load(DATA / entry["file"]) as npz:
        return {name: npz[name] for name in npz.files if name.startswith("pairs/")}


def test_manifest_covers_every_registry_code():
    for name, entry in FIXTURES.items():
        assert set(entry["counts"]) == set(pedigree_graph.RELATIONSHIPS), name
        assert set(entry["hashes"]) == set(pedigree_graph.RELATIONSHIPS), name


@pytest.mark.parametrize("name", SMALL)
def test_small_fixture_matches_the_golden_arrays(name):
    entry = FIXTURES[name]
    fx = _input_from_npz(entry)
    assert pedigrees.input_hash(fx) == entry["input_hash"]

    captured, summary = _capture(pedigree_graph, fx, full_arrays=True)
    assert summary["n"] == entry["n"]
    assert summary["counts"] == entry["counts"]
    assert summary["hashes"] == entry["hashes"]

    stored = _stored_arrays(entry)
    assert sorted(stored) == sorted(captured)
    for key, expected in stored.items():
        assert expected.dtype == np.int32, key
        np.testing.assert_array_equal(captured[key], expected, err_msg=key)


@pytest.mark.slow
def test_random_30k_matches_the_golden_hashes():
    name = "random_30k"
    entry = FIXTURES[name]
    fx = pedigrees.build_random(name, pedigrees.LARGE_FIXTURES[name])
    assert pedigrees.input_hash(fx) == entry["input_hash"]

    _, summary = _capture(pedigree_graph, fx, full_arrays=False)
    assert summary["n"] == entry["n"]
    assert summary["counts"] == entry["counts"]
    assert summary["hashes"] == entry["hashes"]

    graph = pedigree_graph.PedigreeGraph.from_frame(_columns(fx))
    check_exclusive(graph.relationship_pairs(max_degree=MAX_DEGREE))
