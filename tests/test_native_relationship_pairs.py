"""``_native.relationship_pairs`` equals the matrix engine's ``relationship_pairs`` block for block.

The binding is the seam slice 12 moves pair extraction across (ADR 0006 and
0007 as amended): the Rust row-streaming engine classifies, orients, and
assembles; Python keeps the selector and the result type.  These tests hold
the raw binding against the in-tree matrix oracle on every parity fixture,
selector, receiver, and execution, and pin the boundary contract: owned
int32 arrays, registry-ordered keys, both executions element for element
equal, and structured errors instead of aborts.  Anything that needs a
different package pool runs in a fresh interpreter, because the pool is
built once per process.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import numpy as np
import pytest
from conftest import parity_columns, parity_fixtures

from pedigree_graph import RELATIONSHIPS, PedigreeGraph, PedigreeValidationError, _native
from pedigree_graph._threads import thread_budget

FIXTURES = parity_fixtures("random_1k", "deep_inbred_60g")
FIXTURE_NAMES = sorted(FIXTURES)
EXECUTIONS = ("speed", "memory")
SELECTORS = (
    {"max_degree": 0},
    {"max_degree": 1},
    {"max_degree": 3},
    {"max_degree": 5},
    {"categories": ["FS"]},
    {"categories": ["1C", "Av"]},
    {"categories": ["H1C1R", "MO", "2C"]},
)


def _native_pairs(receiver, execution: str, **selector):
    """The binding's result for *receiver* and *selector*, as the facade will call it."""
    from pedigree_graph._selection import RelationshipSelection

    selection = RelationshipSelection.parse(selector.get("max_degree"), selector.get("categories"))
    graph = receiver if isinstance(receiver, PedigreeGraph) else receiver._graph
    view_rows = None if isinstance(receiver, PedigreeGraph) else receiver._graph_to_view()
    return _native.relationship_pairs(
        graph._built,
        max_degree=selection.top_degree or 0,
        requested=list(selection.ordered),
        threads=thread_budget(),
        execution=execution,
        view_rows=view_rows,
    )


def _assert_matches_oracle(receiver, **selector) -> None:
    expected = receiver.relationship_pairs(**selector)
    for execution in EXECUTIONS:
        got = _native_pairs(receiver, execution, **selector)
        assert tuple(got) == tuple(RELATIONSHIPS)
        for code, block in expected.items():
            first, second = got[code]
            assert first.dtype == np.int32
            assert second.dtype == np.int32
            if block.requested:
                np.testing.assert_array_equal(first, block.first_rows, err_msg=f"{code} first {execution}")
                np.testing.assert_array_equal(second, block.second_rows, err_msg=f"{code} second {execution}")
            else:
                assert len(first) == 0
                assert len(second) == 0


@pytest.mark.parametrize("name", FIXTURE_NAMES)
@pytest.mark.parametrize("selector", SELECTORS, ids=str)
def test_graph_blocks_equal_the_matrix_engine(name, selector):
    graph = PedigreeGraph.from_frame(parity_columns(FIXTURES[name]))
    _assert_matches_oracle(graph, **selector)


@pytest.mark.parametrize("name", FIXTURE_NAMES)
@pytest.mark.parametrize("seed", [3, 4])
def test_view_blocks_equal_the_matrix_engine(name, seed):
    graph = PedigreeGraph.from_frame(parity_columns(FIXTURES[name]))
    n = graph.n_individuals
    rows = np.random.default_rng(seed).permutation(n)[: max(n // 2, min(n, 2))]
    view = graph.view(rows=rows)
    for selector in SELECTORS:
        _assert_matches_oracle(view, **selector)


def test_reversed_and_tiny_views_match(small_pedigree):
    graph = PedigreeGraph.from_frame(small_pedigree)
    n = graph.n_individuals
    _assert_matches_oracle(graph.view(rows=np.arange(n)[::-1]), max_degree=5)
    _assert_matches_oracle(graph.view(rows=np.array([n - 1, 0])), max_degree=5)


class TestBoundary:
    def test_arrays_are_owned_contiguous_int32_with_a_native_base(self, small_pedigree):
        graph = PedigreeGraph.from_frame(small_pedigree)
        first, second = _native_pairs(graph, "speed", max_degree=2)["MO"]
        for array in (first, second):
            assert array.dtype == np.int32
            assert array.ndim == 1
            assert array.flags.c_contiguous
            assert not isinstance(array.base, np.ndarray)
        assert len(first) == len(second) > 0

    def test_the_two_executions_agree_element_for_element(self, small_pedigree):
        graph = PedigreeGraph.from_frame(small_pedigree)
        speed = _native_pairs(graph, "speed", max_degree=5)
        memory = _native_pairs(graph, "memory", max_degree=5)
        for code in RELATIONSHIPS:
            np.testing.assert_array_equal(speed[code][0], memory[code][0])
            np.testing.assert_array_equal(speed[code][1], memory[code][1])

    def test_block_lengths_equal_the_counts(self, small_pedigree):
        graph = PedigreeGraph.from_frame(small_pedigree)
        pairs = _native_pairs(graph, "memory", max_degree=5)
        counts = graph.relationship_counts(max_degree=5)
        assert {code: len(pairs[code][0]) for code in RELATIONSHIPS} == dict(counts)

    def test_malformed_arguments_raise_before_any_work(self, small_pedigree):
        graph = PedigreeGraph.from_frame(small_pedigree)
        call = {"max_degree": 5, "requested": ["FS"], "threads": thread_budget(), "execution": "speed"}
        with pytest.raises(ValueError, match="execution"):
            _native.relationship_pairs(graph._built, **{**call, "execution": "buffered"})
        with pytest.raises(ValueError, match="unknown relationship code"):
            _native.relationship_pairs(graph._built, **{**call, "requested": ["cousin"]})
        with pytest.raises(ValueError, match="view_rows"):
            _native.relationship_pairs(graph._built, **call, view_rows=np.array([0], dtype=np.int32))
        with pytest.raises(ValueError, match="threads"):
            _native.relationship_pairs(graph._built, **{**call, "threads": 0})
        with pytest.raises(PedigreeValidationError, match="max_degree"):
            _native.relationship_pairs(graph._built, **{**call, "max_degree": 6})
        with pytest.raises(ValueError, match="unknown allocation family"):
            _native.fail_next_allocation("heap")


def _run_child(*parts: str, **env: str) -> str:
    """Run the dedented *parts* as one script in a fresh interpreter, returning its stdout."""
    import os

    code = "\n".join(textwrap.dedent(part) for part in parts)
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={**os.environ, **env},
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


CHILD_PRELUDE = """
    import hashlib, sys
    import numpy as np
    from pedigree_graph import PedigreeGraph, _native
    from pedigree_graph._threads import thread_budget
    n = 400
    rng = np.random.default_rng(7)
    mother = np.full(n, -1); father = np.full(n, -1)
    for i in range(20, n):
        lo = max(0, i - 60)
        mother[i], father[i] = rng.integers(lo, i), rng.integers(lo, i)
        if father[i] == mother[i]:
            father[i] = -1
    graph = PedigreeGraph.from_frame({"id": np.arange(n), "mother": mother, "father": father})
    view = np.where(np.arange(n) % 3 == 0, -1, np.arange(n) // 3 * 2 + np.arange(n) % 3 - 1).astype(np.int32)
"""


class TestProcessWidePool:
    def test_blocks_are_identical_under_every_thread_budget(self):
        body = """
            codes = list(graph.relationship_counts(max_degree=5))
            pairs = _native.relationship_pairs(
                graph._built, max_degree=5, requested=codes, threads=thread_budget(), execution="speed", view_rows=view
            )
            digest = hashlib.sha256()
            for code, (first, second) in pairs.items():
                digest.update(first.tobytes()); digest.update(second.tobytes())
            print(thread_budget(), digest.hexdigest())
        """
        one = _run_child(CHILD_PRELUDE, body, PEDIGREE_GRAPH_THREADS="1").split()
        four = _run_child(CHILD_PRELUDE, body, PEDIGREE_GRAPH_THREADS="4").split()
        assert one[0] == "1"
        assert four[0] == "4"
        assert one[1] == four[1]

    def test_the_pool_keeps_its_first_size(self):
        body = """
            print(_native.configure_pool(2), _native.configure_pool(2))
            try:
                _native.configure_pool(3)
            except ValueError as e:
                print("conflict:", e)
            _native.relationship_counts(graph._built, max_degree=2, threads=2)
            try:
                _native.relationship_counts(graph._built, max_degree=2, threads=5)
            except ValueError as e:
                print("conflict:", e)
        """
        out = _run_child(CHILD_PRELUDE, body)
        assert out.startswith("2 2")
        assert out.count("conflict: the thread pool is already configured with 2 threads") == 2


FAMILIES = (
    "parent_edges",
    "csr",
    "sibling_index",
    "accumulator",
    "row_set",
    "task_chunk",
    "task_table",
    "pair_block",
    "view_sort_scratch",
)
COUNT_FAMILIES = FAMILIES[:5]


@pytest.mark.parametrize("family", FAMILIES)
def test_a_refused_allocation_raises_a_resource_error(family):
    """Every allocation family surfaces as ``ResourceError("allocation_failed")`` in a fresh process.

    The plant fires on the family's first reservation, which for a collected
    iterator is its zero lower size bound, so ``requested_elements`` is only
    required to be a count.
    """
    body = f"""
        from pedigree_graph import ResourceError
        family = {family!r}
        def pairs():
            return _native.relationship_pairs(
                graph._built, max_degree=5, requested=["FS", "2C"], threads=1, execution="speed", view_rows=view
            )
        def counts():
            return _native.relationship_counts(graph._built, max_degree=5, threads=1)
        calls = [("pairs", pairs)] + ([("counts", counts)] if family in {COUNT_FAMILIES!r} else [])
        for label, call in calls:
            _native.fail_next_allocation(family)
            try:
                call()
            except ResourceError as e:
                print(label, e.code, e.fields["operation"], e.fields["dtype"], e.fields["requested_elements"] >= 0)
            else:
                print(label, "no error")
        _native.fail_next_allocation(None)
        call()
        print("recovered")
    """
    lines = _run_child(CHILD_PRELUDE, body).strip().splitlines()
    assert lines[-1] == "recovered"
    for line in lines[:-1]:
        _label, code, operation, dtype, counted = line.split()
        assert (code, operation, counted) == ("allocation_failed", family, "True"), line
        assert dtype in {"int32", "int64", "intp", "uint8", "uint64", "object"}
