"""Golden lock for the structural outputs: depth, lineage counts, matrix supports, view pairs.

Replays ``tests/parity/generate_structure.py::_capture`` and compares every
count, hash and stored array in ``tests/data/structure_v0.10``.  The carried
keys reproduced the retired 0.7.1 baseline digest for digest when the lock was
captured (``--verify-against``), and each records that digest as
``v0_7_1_sha256``.  The lock is regenerated only as a deliberate contract
change, never to make a test pass; see ``tests/parity/README.md``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pedigrees
import pytest
from generate_structure import CARRIED_KEYS, _capture

import pedigree_graph

DATA = Path(__file__).resolve().parent / "data" / "structure_v0.10"
MANIFEST = json.loads((DATA / "manifest.json").read_text())
FIXTURES = MANIFEST["fixtures"]
SMALL = sorted(name for name, entry in FIXTURES.items() if "file" in entry)


def _hashes(entry: dict) -> dict[str, str]:
    return {key: record["sha256"] for key, record in entry["keys"].items()}


def test_carried_keys_still_equal_their_0_7_1_digests():
    carried = 0
    for name, entry in FIXTURES.items():
        for key, record in entry["keys"].items():
            if record["v0_7_1_sha256"] is None:
                assert key not in CARRIED_KEYS or key.startswith("complete_"), (name, key)
                continue
            assert key in CARRIED_KEYS, (name, key)
            assert record["sha256"] == record["v0_7_1_sha256"], (name, key)
            carried += 1
    # 17 small fixtures x 7 keys, less deep_inbred_60g's overflowed n_descendants,
    # plus random_30k's depth, lineage counts, approx support and selection.
    assert carried == 123


@pytest.mark.parametrize("name", SMALL)
def test_small_fixture_matches_the_golden_arrays(name):
    entry = FIXTURES[name]
    with np.load(DATA / entry["file"]) as npz:
        fx = {key.removeprefix("input/"): npz[key] for key in npz.files if key.startswith("input/")}
        stored = {key: npz[key] for key in npz.files if not key.startswith("input/")}
    assert pedigrees.input_hash(fx) == entry["input_hash"]

    captured, summary = _capture(pedigree_graph, fx, full_arrays=True)
    assert summary["n"] == entry["n"]
    assert summary["subsample_n"] == entry["subsample_n"]
    assert summary.get("n_descendants_overflow") == entry.get("n_descendants_overflow")
    assert summary["counts"] == entry["counts"]
    assert sorted(stored) == sorted(captured)
    for key, expected in stored.items():
        assert captured[key].dtype == expected.dtype, key
        np.testing.assert_array_equal(captured[key], expected, err_msg=key)
    assert summary["hashes"] == _hashes(entry)


@pytest.mark.slow
def test_random_30k_matches_the_golden_hashes():
    name = "random_30k"
    entry = FIXTURES[name]
    fx = pedigrees.build_random(name, pedigrees.LARGE_FIXTURES[name])
    assert pedigrees.input_hash(fx) == entry["input_hash"]

    _, summary = _capture(pedigree_graph, fx, full_arrays=False)
    assert summary["n"] == entry["n"]
    assert summary["subsample_n"] == entry["subsample_n"]
    assert summary["counts"] == entry["counts"]
    assert summary["hashes"] == _hashes(entry)
