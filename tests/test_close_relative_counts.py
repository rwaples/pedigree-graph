"""Exact six-code scalar counts, cache ownership and the issue #17 API break."""

from __future__ import annotations

import tracemalloc
import warnings
from pathlib import Path

import numpy as np
import pedigrees
import pytest
from _support import _run_child
from conftest import FIXTURE_NAMES, parity_columns, parity_graph

from pedigree_graph import RELATIONSHIPS, PedigreeGraph, RelationshipCountResult, _streaming_counter
from pedigree_graph._registry import estimate_exact_codes
from pedigree_graph._threads import _reset_thread_state, configure_threads, thread_budget

CODES = tuple(RELATIONSHIPS)


@pytest.fixture
def small_graph(small_pedigree) -> PedigreeGraph:
    return PedigreeGraph.from_frame(small_pedigree)


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_exact_values_and_metadata(name):
    result = parity_graph(name).close_relative_counts()
    exact = parity_graph(name).relationship_counts(max_degree=5)
    assert isinstance(result, RelationshipCountResult)
    assert tuple(result) == CODES
    assert result.requested == result.exact == estimate_exact_codes()
    assert not hasattr(result, "approximate")
    assert not hasattr(result, "clamped")
    for code in CODES:
        if code in result.requested:
            assert isinstance(result[code], int), code
            assert result[code] >= 0, code
            assert result[code] == exact[code], code
        else:
            assert result[code] is None, code


def test_result_is_immutable(small_graph):
    result = small_graph.close_relative_counts()
    with pytest.raises((TypeError, AttributeError)):
        result["FS"] = 0  # type: ignore[index]
    with pytest.raises((AttributeError, TypeError)):
        result.exact = frozenset()  # type: ignore[misc]


@pytest.mark.parametrize("n", [0, 1])
def test_founders_have_zero_close_counts(n):
    graph = PedigreeGraph.from_arrays(ids=np.arange(n), mother_ids=np.full(n, -1), father_ids=np.full(n, -1))
    result = graph.close_relative_counts()
    assert result.requested == estimate_exact_codes()
    assert all(result[code] == 0 for code in result.requested)
    assert all(result[code] is None for code in CODES if code not in result.requested)


@pytest.mark.parametrize("swap_parents", [False, True], ids=["paternal", "maternal"])
def test_half_sib_pairs_claimed_by_parent_offspring(swap_parents):
    # 4 and its mother 3 share father 1. Parent-offspring takes precedence
    # over PHS; swapping parent roles exercises the MHS correction too.
    mother = np.array([-1, -1, 2, 3])
    father = np.array([-1, -1, 1, 1])
    if swap_parents:
        mother, father = father, mother
    graph = PedigreeGraph.from_arrays(ids=np.arange(1, 5), mother_ids=mother, father_ids=father)
    result = graph.close_relative_counts()
    exact = graph.relationship_counts(max_degree=5)
    for code in result.requested:
        assert result[code] == exact[code], code
    assert result["MHS" if swap_parents else "PHS"] == 0
    assert result["GP"] is None


class TestCache:
    def test_first_and_cached_calls_do_not_warn(self, small_graph, monkeypatch):
        real = _streaming_counter._count_close_relatives
        calls = 0

        def count(graph):
            nonlocal calls
            calls += 1
            return real(graph)

        monkeypatch.setattr(_streaming_counter, "_count_close_relatives", count)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            first = small_graph.close_relative_counts()
            second = small_graph.close_relative_counts()
        assert first is second is small_graph._close_relative_counts_cache
        assert calls == 1

    def test_failure_does_not_cache_a_partial_result(self, small_graph, monkeypatch):
        real = _streaming_counter._count_close_relatives

        def fail(graph):
            raise MemoryError("simulated counting failure")

        monkeypatch.setattr(_streaming_counter, "_count_close_relatives", fail)
        with pytest.raises(MemoryError):
            small_graph.close_relative_counts()
        assert small_graph._close_relative_counts_cache is None
        monkeypatch.setattr(_streaming_counter, "_count_close_relatives", real)
        assert small_graph.close_relative_counts().requested == estimate_exact_codes()


def test_does_not_build_pairs(small_graph, monkeypatch):
    reference = small_graph.relationship_counts(max_degree=5)

    def forbidden(*args, **kwargs):
        raise AssertionError("close counts must use only parent/twin arrays")

    for name in ("relationship_pairs", "relationship_counts"):
        monkeypatch.setattr(PedigreeGraph, name, forbidden)
    result = small_graph.close_relative_counts()
    assert all(result[code] == reference[code] for code in result.requested)


def test_takes_no_selector_and_has_no_view_form(small_graph):
    assert not hasattr(small_graph.view(rows=np.array([0])), "close_relative_counts")
    with pytest.raises(TypeError):
        small_graph.close_relative_counts(max_degree=2)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        small_graph.close_relative_counts(2)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        small_graph.close_relative_counts(categories=["FS"])  # type: ignore[call-arg]


@pytest.mark.usefixtures("fresh_thread_state")
class TestThreads:
    """The Python budget commits on the first public call and the native pool follows it.

    The pool is built once per process (ADR 0007), so a comparison across
    budgets runs each budget in a fresh interpreter, and the in-process test
    reconfigures only to the value this process already runs.
    """

    def test_budget_of_four_matches_one_thread(self):
        body = f"""
            import sys
            sys.path.insert(0, {str(Path(__file__).parent)!r})
            from conftest import parity_graph
            print(dict(parity_graph("random_1k").close_relative_counts()))
        """
        one = _run_child(body, PEDIGREE_GRAPH_THREADS="1")
        four = _run_child(body, PEDIGREE_GRAPH_THREADS="4")
        assert one == four
        assert "'FS'" in one

    @pytest.mark.parametrize("cached", [False, True])
    def test_public_call_commits_budget(self, cached):
        budget = thread_budget()
        _reset_thread_state()
        graph = parity_graph("random_1k")
        if cached:
            graph.close_relative_counts()
            _reset_thread_state()
        configure_threads(budget)
        graph.close_relative_counts()
        configure_threads(budget)
        with pytest.raises(RuntimeError):
            configure_threads(budget + 1)


@pytest.mark.slow
def test_random_30k_matches_exact_counts():
    fx = pedigrees.build_random("random_30k", pedigrees.LARGE_FIXTURES["random_30k"])
    graph = PedigreeGraph.from_frame(parity_columns(fx))
    result = graph.close_relative_counts()
    exact = graph.relationship_counts(max_degree=5)
    assert result.exact == estimate_exact_codes()
    for code in result.exact:
        assert result[code] == exact[code], code


class TestMemoryIsIndependentOfIdMagnitude:
    """Group by dense parent index, never allocate arrays indexed by original ID."""

    @staticmethod
    def _pedigree(base: int) -> dict[str, np.ndarray]:
        ids = np.arange(20, dtype=np.int64) + base
        mother = np.array([-1] * 6 + [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7])
        father = np.array([-1] * 6 + [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 0, 0, 8, 9])
        return {
            "id": ids,
            "mother": np.where(mother >= 0, ids[np.maximum(mother, 0)], -1),
            "father": np.where(father >= 0, ids[np.maximum(father, 0)], -1),
        }

    def _peak_bytes(self, base: int) -> int:
        graph = PedigreeGraph.from_frame(self._pedigree(base))
        tracemalloc.start()
        try:
            graph.close_relative_counts()
            return tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()

    def test_ten_digit_ids_allocate_no_more_than_small_ids(self):
        small = self._peak_bytes(0)
        large = self._peak_bytes(10_000_000)
        assert large < 4 * small + 1_000_000, (small, large)

    def test_counts_do_not_depend_on_id_magnitude(self):
        small = dict(PedigreeGraph.from_frame(self._pedigree(0)).close_relative_counts())
        large = dict(PedigreeGraph.from_frame(self._pedigree(10_000_000)).close_relative_counts())
        assert small == large
