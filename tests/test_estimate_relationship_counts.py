"""Tests for ``PedigreeGraph.estimate_relationship_counts`` (ADR 0006, ADR 0011, slice 4c).

The scalar formulas are those of the 0.7.1 ``count_pairs_streaming``; what is
new is the typed result that says per code whether the value is exact,
approximate, or clamped, and the one ``RuntimeWarning`` per clamped
computation.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
from test_relationship_pairs import FIXTURE_NAMES, _columns, _graph

from pedigree_graph import RELATIONSHIPS, PedigreeGraph, PedigreeValidationError, RelationshipCountResult
from pedigree_graph._registry import estimate_exact_codes
from pedigree_graph._threads import _reset_thread_state, configure_threads, thread_budget

sys.path.insert(0, str(Path(__file__).resolve().parent / "parity"))

import pedigrees

CODES = tuple(RELATIONSHIPS)
MANIFEST = json.loads((Path(__file__).resolve().parent / "data" / "parity_v0.7.1" / "manifest.json").read_text())
BASELINE = MANIFEST["fixtures"]
CLAMPING_FIXTURE = "small_pedigree"
CLEAN_FIXTURE = "random_1k"
LINEAL_CODES = frozenset({"GP", "GGP", "GGGP", "G3GP"})
# Fixtures on which the precedence fold files a lineal pair under a closer
# code, with the lineal codes affected; the raw path counts exceed the exact
# counts there, which is why the lineal codes are approximate (ADR 0011).
FOLDED_LINEAL = {
    "backcross_and_selfing_like": frozenset({"GP", "GGP", "GGGP", "G3GP"}),
    "one_parent_known": frozenset({"GGP"}),
    "random_1k": frozenset({"GGP", "GGGP", "G3GP"}),
}


def _graphs(name: str) -> tuple[PedigreeGraph, PedigreeGraph]:
    """Two independent graphs of the same fixture, so caches do not cross paths."""
    return _graph(name), _graph(name)


def _requested(max_degree: int) -> frozenset[str]:
    return frozenset(code for code in CODES if RELATIONSHIPS[code].degree <= max_degree)


def _as_legacy_dict(result: RelationshipCountResult) -> dict[str, int]:
    return {code: 0 if value is None else value for code, value in result.items()}


@pytest.fixture
def small_graph(small_pedigree) -> PedigreeGraph:
    return PedigreeGraph(small_pedigree)


def _estimate_quietly(graph: PedigreeGraph, max_degree: int) -> RelationshipCountResult:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return graph.estimate_relationship_counts(max_degree=max_degree)


class TestResultMetadata:
    @pytest.mark.parametrize("name", FIXTURE_NAMES)
    @pytest.mark.parametrize("max_degree", [0, 2, 5])
    def test_partition_of_requested(self, name, max_degree):
        result = _estimate_quietly(_graph(name), max_degree)
        assert isinstance(result, RelationshipCountResult)
        assert tuple(result) == CODES
        assert result.requested == _requested(max_degree)
        assert result.exact == result.requested & estimate_exact_codes()
        assert result.approximate == result.requested - result.exact
        assert result.clamped <= result.approximate
        assert (LINEAL_CODES & result.requested) <= result.approximate
        for code in CODES:
            if code in result.requested:
                assert isinstance(result[code], int), code
                assert result[code] >= 0, code
            else:
                assert result[code] is None, code

    def test_small_pedigree_metadata(self, small_graph):
        result = _estimate_quietly(small_graph, 5)
        assert result.requested == frozenset(CODES)
        assert "H1C" in result.clamped
        assert result.clamped <= result.approximate
        assert result[next(iter(result.clamped))] == 0

    def test_result_is_immutable(self, small_graph):
        result = _estimate_quietly(small_graph, 2)
        with pytest.raises((TypeError, AttributeError)):
            result["FS"] = 0  # type: ignore[index]
        with pytest.raises((AttributeError, TypeError)):
            result.clamped = frozenset()  # type: ignore[misc]


class TestValues:
    @pytest.mark.parametrize("name", FIXTURE_NAMES)
    def test_exact_codes_agree_with_the_exact_path(self, name):
        estimate_graph, exact_graph = _graphs(name)
        estimate = _estimate_quietly(estimate_graph, 5)
        exact = exact_graph.relationship_counts(max_degree=5)
        assert estimate.exact == frozenset(estimate_exact_codes())
        for code in sorted(estimate.exact):
            assert estimate[code] == exact[code], code

    @pytest.mark.parametrize("name", FIXTURE_NAMES)
    def test_lineal_codes_are_raw_path_counts(self, name):
        graph = _graph(name)
        estimate = _estimate_quietly(graph, 5)
        raw = graph.count_pairs_streaming(max_degree=5)
        exact = graph.relationship_counts(max_degree=5)
        for code in sorted(LINEAL_CODES):
            assert estimate[code] == raw[code], code
            if code in FOLDED_LINEAL.get(name, frozenset()):
                assert estimate[code] > exact[code], code
            else:
                assert estimate[code] == exact[code], code

    @pytest.mark.parametrize("name", sorted(name for name in FIXTURE_NAMES if "streaming_counts" in BASELINE[name]))
    def test_adapter_equals_the_frozen_streaming_counts(self, name):
        """The 0.7.1 baseline holds the unfolded counts, which the adapter keeps returning."""
        graph = _graph(name)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            legacy = graph.count_pairs_streaming(max_degree=MANIFEST["max_degree"])
        assert legacy == BASELINE[name]["streaming_counts"]
        estimate = graph.estimate_relationship_counts(max_degree=MANIFEST["max_degree"])
        for code in estimate.approximate:
            assert estimate[code] == legacy[code], code

    def test_small_pedigree_adapter_equals_the_frozen_streaming_counts(self, small_graph):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            legacy = small_graph.count_pairs_streaming(max_degree=MANIFEST["max_degree"])
        assert legacy == BASELINE[CLAMPING_FIXTURE]["streaming_counts"]


def _pedigree(*rows: tuple[int, int, int]) -> PedigreeGraph:
    """Build a graph from ``(id, mother, father)`` rows; ``0`` is unknown."""
    ids = np.array([row[0] for row in rows])
    mother = np.array([row[1] if row[1] else -1 for row in rows])
    father = np.array([row[2] if row[2] else -1 for row in rows])
    return PedigreeGraph({"id": ids, "mother": mother, "father": father})


class TestHalfSibFold:
    def test_backcross_phs_claimed_by_mo_and_gp_stays_raw(self):
        # Father 1 mates with his daughter 3; 4 shares father 1 with its own
        # mother 3 (PHS folded into MO) and 1 is both father and grandfather
        # of 4 (GP stays the raw path count).
        graph = _pedigree((1, 0, 0), (2, 0, 0), (3, 2, 1), (4, 3, 1))
        estimate = _estimate_quietly(graph, 5)
        raw = graph.count_pairs_streaming(max_degree=5)
        exact = graph.relationship_counts(max_degree=5)
        for code in sorted(estimate.exact):
            assert estimate[code] == exact[code], code
        assert (raw["PHS"], estimate["PHS"], exact["PHS"]) == (1, 0, 0)
        assert "GP" in estimate.approximate
        assert (raw["GP"], estimate["GP"], exact["GP"]) == (2, 2, 1)


class TestWarningAndCache:
    def test_first_clamping_computation_warns_once_in_registry_order(self, small_graph):
        with pytest.warns(RuntimeWarning) as record:
            result = small_graph.estimate_relationship_counts(max_degree=5)
        assert len(record) == 1
        message = str(record[0].message)
        assert "H1C" in message
        assert "max_degree=5" in message
        names = [code for code in CODES if code in result.clamped]
        assert ", ".join(names) in message
        assert result.clamped == frozenset(names)
        assert record[0].filename == __file__

    def test_cached_retrieval_is_silent_and_identical(self, small_graph):
        first = _estimate_quietly(small_graph, 5)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            second = small_graph.estimate_relationship_counts(max_degree=5)
        assert second is first

    def test_warning_precedes_the_cache_write(self, small_graph):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(RuntimeWarning):
                small_graph.estimate_relationship_counts(max_degree=5)
            assert small_graph._estimate_cache == {}
            with pytest.raises(RuntimeWarning):
                small_graph.estimate_relationship_counts(max_degree=5)
            assert small_graph._estimate_cache == {}
        result = _estimate_quietly(small_graph, 5)
        assert "H1C" in result.clamped
        assert 5 in small_graph._estimate_cache

    def test_non_clamping_cutoff_computes_without_warning(self, small_graph):
        _estimate_quietly(small_graph, 5)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            shallow = small_graph.estimate_relationship_counts(max_degree=2)
        assert shallow.clamped == frozenset()
        assert shallow.requested == _requested(2)
        assert shallow["1C"] is None

    def test_clamping_cutoff_after_a_clean_one_warns_exactly_once(self, small_graph):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            small_graph.estimate_relationship_counts(max_degree=2)
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            small_graph.estimate_relationship_counts(max_degree=5)
            small_graph.estimate_relationship_counts(max_degree=5)
        assert [type(w.message) for w in record] == [RuntimeWarning]

    def test_each_clamping_cutoff_warns_independently(self, small_graph):
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            small_graph.estimate_relationship_counts(max_degree=4)
            small_graph.estimate_relationship_counts(max_degree=5)
        assert [type(w.message) for w in record] == [RuntimeWarning, RuntimeWarning]

    def test_pedigree_without_clamp_never_warns(self):
        graph = _graph(CLEAN_FIXTURE)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            for max_degree in range(6):
                assert graph.estimate_relationship_counts(max_degree=max_degree).clamped == frozenset()

    def test_adapter_shares_the_cache_so_a_later_estimate_is_silent(self, small_graph):
        with pytest.warns(RuntimeWarning):
            legacy = small_graph.count_pairs_streaming(max_degree=5)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            estimate = small_graph.estimate_relationship_counts(max_degree=5)
        for code in estimate.approximate:
            assert estimate[code] == legacy[code], code
        assert small_graph._pair_count_cache == {}


class TestErrors:
    @pytest.mark.parametrize("bad", [-1, 6, 100])
    def test_max_degree_out_of_range(self, small_graph, bad):
        with pytest.raises(PedigreeValidationError) as info:
            small_graph.estimate_relationship_counts(max_degree=bad)
        assert info.value.code == "max_degree_out_of_range"
        assert info.value.fields["value"] == bad
        assert (info.value.fields["minimum"], info.value.fields["maximum"]) == (0, 5)
        assert small_graph._estimate_cache == {}

    def test_positional_max_degree_is_a_type_error(self, small_graph):
        with pytest.raises(TypeError):
            small_graph.estimate_relationship_counts(3)  # type: ignore[misc]


class TestThreads:
    @pytest.fixture(autouse=True)
    def reset_thread_state(self):
        _reset_thread_state()
        yield
        _reset_thread_state()

    def test_budget_of_four_matches_the_one_thread_result(self, monkeypatch):
        monkeypatch.delenv("PEDIGREE_GRAPH_THREADS", raising=False)
        reference = _estimate_quietly(_graph(CLEAN_FIXTURE), 5)
        _reset_thread_state()
        configure_threads(4)
        result = _estimate_quietly(_graph(CLEAN_FIXTURE), 5)
        assert dict(result) == dict(reference)
        assert (result.exact, result.approximate, result.clamped) == (
            reference.exact,
            reference.approximate,
            reference.clamped,
        )

    def test_public_call_commits_the_budget(self, monkeypatch):
        monkeypatch.delenv("PEDIGREE_GRAPH_THREADS", raising=False)
        configure_threads(2)
        _estimate_quietly(_graph(CLEAN_FIXTURE), 1)
        configure_threads(2)
        with pytest.raises(RuntimeError):
            configure_threads(3)

    def test_adapter_leaves_the_budget_uncommitted(self, monkeypatch):
        monkeypatch.delenv("PEDIGREE_GRAPH_THREADS", raising=False)
        graph = _graph(CLEAN_FIXTURE)
        graph.count_pairs_streaming(max_degree=5)
        configure_threads(4)
        assert thread_budget() == 4


def test_transient_matrices_are_released(small_graph):
    transient = ("_A", "_A2", "_A3", "_A4", "_A5")
    _estimate_quietly(small_graph, 5)
    assert not [attr for attr in transient if attr in small_graph.__dict__]
    assert 5 in small_graph._estimate_cache


@pytest.mark.slow
def test_random_30k_matches_the_frozen_streaming_counts():
    name = "random_30k"
    entry = BASELINE[name]
    fx = pedigrees.build_random(name, pedigrees.LARGE_FIXTURES[name])
    assert pedigrees.input_hash(fx) == entry["input_hash"]
    graph = PedigreeGraph(_columns(fx))
    estimate = _estimate_quietly(graph, MANIFEST["max_degree"])
    assert graph.count_pairs_streaming(max_degree=MANIFEST["max_degree"]) == entry["streaming_counts"]
    exact = graph.relationship_counts(max_degree=MANIFEST["max_degree"])
    for code in sorted(estimate.exact):
        assert estimate[code] == exact[code], code
    assert estimate.exact == frozenset(estimate_exact_codes())
