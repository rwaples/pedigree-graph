"""The ``execution`` keyword of ``relationship_pairs`` and the native hand-over behind it (slice 12).

ADR 0006 as amended: ``"speed"`` and ``"memory"`` change resource use only,
so both must return element-for-element identical blocks, on graphs and on
views, equal to the matrix oracle in ``tests/oracle``.  ADR 0007 as
amended: the blocks are the core's own arrays, frozen without a copy, and
an allocation the core cannot make is ``ResourceError("allocation_failed")``
rather than an abort.
"""

from __future__ import annotations

import dataclasses
import os
import subprocess
import sys
import textwrap
from typing import TYPE_CHECKING

import numpy as np
import pytest
from conftest import parity_columns, parity_fixtures
from oracle.relationship_pairs import check_exclusive, oracle_pairs, oracle_view_pairs

from pedigree_graph import RELATIONSHIPS, PedigreeGraph

if TYPE_CHECKING:
    from pedigree_graph import RelationshipPairs

FIXTURES = parity_fixtures("random_1k", "deep_inbred_60g")
FIXTURE_NAMES = sorted(FIXTURES)
EXECUTIONS = ("speed", "memory")
SELECTORS = (
    {"max_degree": 0},
    {"max_degree": 2},
    {"max_degree": 5},
    {"categories": ["FS"]},
    {"categories": ["2C", "MO", "H1C1R"]},
    {"categories": []},
)


def _graph(name: str) -> PedigreeGraph:
    return PedigreeGraph.from_frame(parity_columns(FIXTURES[name]))


def _assert_equal_blocks(got: RelationshipPairs, want: dict, requested: frozenset[str]) -> None:
    assert tuple(got) == tuple(RELATIONSHIPS)
    for code, block in got.items():
        assert block.requested is (code in requested)
        np.testing.assert_array_equal(block.first_rows, want[code][0], err_msg=code)
        np.testing.assert_array_equal(block.second_rows, want[code][1], err_msg=code)
    check_exclusive(got)


@pytest.mark.parametrize("name", FIXTURE_NAMES)
@pytest.mark.parametrize("selector", SELECTORS, ids=str)
@pytest.mark.parametrize("execution", EXECUTIONS)
def test_graph_blocks_equal_the_oracle(name, selector, execution):
    graph = _graph(name)
    got = graph.relationship_pairs(**selector, execution=execution)
    requested = frozenset(code for code, block in got.items() if block.requested)
    _assert_equal_blocks(got, oracle_pairs(graph, **selector), requested)


@pytest.mark.parametrize("name", FIXTURE_NAMES)
@pytest.mark.parametrize("shape", ["reversed", "half", "singleton", "empty"])
@pytest.mark.parametrize("execution", EXECUTIONS)
def test_view_blocks_equal_the_oracle(name, shape, execution):
    graph = _graph(name)
    n = graph.n_individuals
    rows = {
        "reversed": np.arange(n)[::-1],
        "half": np.random.default_rng(5).permutation(n)[: n // 2],
        "singleton": np.array([n // 2]),
        "empty": np.array([], dtype=np.int64),
    }[shape]
    view = graph.view(rows=rows)
    for selector in SELECTORS:
        got = view.relationship_pairs(**selector, execution=execution)
        requested = frozenset(code for code, block in got.items() if block.requested)
        _assert_equal_blocks(got, oracle_view_pairs(view, **selector), requested)


class TestKeyword:
    def test_the_default_is_speed_and_both_modes_agree(self, small_pedigree):
        graph = PedigreeGraph.from_frame(small_pedigree)
        default = graph.relationship_pairs(max_degree=5)
        speed = graph.relationship_pairs(max_degree=5, execution="speed")
        memory = graph.relationship_pairs(max_degree=5, execution="memory")
        for code in RELATIONSHIPS:
            for result in (speed, memory):
                np.testing.assert_array_equal(result[code].first_rows, default[code].first_rows)
                np.testing.assert_array_equal(result[code].second_rows, default[code].second_rows)

    @pytest.mark.parametrize("bad", ["fast", "buffered", "two_pass", "", None, 1])
    def test_other_values_are_rejected(self, small_pedigree, bad):
        graph = PedigreeGraph.from_frame(small_pedigree)
        with pytest.raises(ValueError, match="execution must be one of"):
            graph.relationship_pairs(max_degree=1, execution=bad)
        with pytest.raises(ValueError, match="execution must be one of"):
            graph.view(rows=np.arange(3)).relationship_pairs(max_degree=1, execution=bad)

    def test_execution_is_keyword_only(self, small_pedigree):
        graph = PedigreeGraph.from_frame(small_pedigree)
        with pytest.raises(TypeError):
            graph.relationship_pairs(1, "memory")  # type: ignore[misc]


class TestHandOver:
    @pytest.mark.parametrize("execution", EXECUTIONS)
    def test_blocks_are_the_core_arrays_frozen_without_a_copy(self, small_pedigree, execution):
        graph = PedigreeGraph.from_frame(small_pedigree)
        for block in graph.relationship_pairs(max_degree=3, execution=execution).values():
            for array in block:
                assert array.dtype == np.int32
                assert array.flags.c_contiguous
                assert not array.flags.writeable
                assert not isinstance(array.base, np.ndarray)
                with pytest.raises(ValueError, match="read-only"):
                    array[:0] = 0

    def test_blocks_unpack_and_replace(self, small_pedigree):
        graph = PedigreeGraph.from_frame(small_pedigree)
        result = graph.relationship_pairs(categories=["MO"], execution="memory")
        first, second = result["MO"]
        assert len(first) == len(result["MO"]) > 0
        swapped = dataclasses.replace(result["MO"], first_rows=second, second_rows=first)
        assert swapped.first_rows is second

    def test_counts_and_pair_kinship_line_up_with_the_blocks(self, small_pedigree):
        graph = PedigreeGraph.from_frame(small_pedigree)
        pairs = graph.relationship_pairs(max_degree=5, execution="memory")
        assert {code: len(block) for code, block in pairs.items()} == dict(graph.relationship_counts(max_degree=5))
        kinship = graph.pair_kinship(pairs)
        for code, block in pairs.items():
            assert len(kinship[code]) == len(block)
            if len(block) and RELATIONSHIPS[code].degree > 0:
                assert np.all(kinship[code] > 0)

    def test_a_refused_allocation_is_a_resource_error_through_the_public_call(self):
        script = textwrap.dedent(
            """
            import numpy as np
            from pedigree_graph import PedigreeGraph, ResourceError, _native
            graph = PedigreeGraph.from_frame({"id": np.arange(4), "mother": [-1, -1, 0, 0], "father": [-1, -1, 1, 1]})
            for execution in ("speed", "memory"):
                _native.fail_next_allocation("pair_block")
                try:
                    graph.relationship_pairs(max_degree=1, execution=execution)
                except ResourceError as e:
                    print(execution, e.code, e.fields["operation"], e.fields["dtype"])
                _native.fail_next_allocation("view_sort_scratch")
                try:
                    graph.view(rows=np.array([3, 2, 1, 0])).relationship_pairs(max_degree=1, execution=execution)
                except ResourceError as e:
                    print(execution, e.code, e.fields["operation"], e.fields["dtype"])
            print(len(graph.relationship_pairs(max_degree=1)["FS"]))
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env={**os.environ, "PEDIGREE_GRAPH_ALLOW_TEST_SEAM": "1"},
            check=False,
        )
        assert result.returncode == 0, result.stderr
        # The view family's first reservation is the map's permutation check,
        # which precedes the packed sort keys.
        assert result.stdout.splitlines() == [
            "speed allocation_failed pair_block int32",
            "speed allocation_failed view_sort_scratch bool",
            "memory allocation_failed pair_block int32",
            "memory allocation_failed view_sort_scratch bool",
            "1",
        ]
