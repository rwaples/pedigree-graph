"""Every arm of ``bench_relatives_per_person.py`` on the ``random_1k`` parity pedigree.

The engine and pairs arms of one configuration must fold to the same int32
arrays, which is the benchmark's correctness check; this runs it in seconds
instead of on the pedsum pedigrees.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "benchmarks"))

import bench_relatives_per_person as bench  # noqa: E402
from _harness import Prepared  # noqa: E402


@pytest.fixture(scope="module")
def measured():
    graph = bench.SUITE.fixture("random_1k").build()
    out = {}
    for arm in bench.ARMS:
        prepared = arm.setup(graph) if arm.setup is not None else Prepared()
        out[arm.name] = arm.run(graph, prepared.payload).resolved()
    return out


@pytest.mark.parametrize("configuration", sorted(bench.CONFIGURATIONS))
def test_arms_of_a_configuration_print_one_checksum(measured, configuration):
    arms = [arm.name for arm in bench.CONFIGURATIONS[configuration]]
    checksums = {name: measured[name].checksum for name in arms}
    assert len(set(checksums.values())) == 1, checksums


@pytest.mark.parametrize("configuration", sorted(bench.CONFIGURATIONS))
def test_arms_of_a_configuration_credit_the_same_relatives(measured, configuration):
    credited = {arm.name: measured[arm.name].facts["credited"] for arm in bench.CONFIGURATIONS[configuration]}
    assert len(set(credited.values())) == 1, credited
    assert next(iter(credited.values())) > 0


def test_the_trait_columns_count_a_strict_nonempty_subset_of_relatives():
    graph = bench.SUITE.fixture("random_1k").build()
    inputs = bench._trait_inputs(graph).payload
    result = graph.relatives_per_person(
        categories=bench.CODES,
        thresholds={
            trait: (np.where(inputs.affected[trait], inputs.onset[trait], np.float64(np.nan)), inputs.cutoff)
            for trait in bench.TRAITS
        },
    )
    for trait in bench.TRAITS:
        passed = int(result.counts[:, :, result.columns.index(trait)].sum(dtype=np.int64))
        assert 0 < passed < int(result.counts[:, :, 0].sum(dtype=np.int64))
