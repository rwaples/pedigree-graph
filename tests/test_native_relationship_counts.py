"""``relationship_counts`` through the Rust engine equals the block lengths of ``relationship_pairs``.

The matrix engine in ``_pair_extractor.py`` is the live oracle: it builds the
pair lists and folds precedence in Python.  The Rust engine (ADR 0010, as
amended) must give the same counts on every fixture, every selector, every
row order, and every view, without building a pair list.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import parity_columns, parity_fixtures
from hypothesis import given, settings
from hypothesis import strategies as st

from pedigree_graph import RELATIONSHIPS, PedigreeGraph, PedigreeValidationError, _native
from pedigree_graph._threads import _reset_thread_state, configure_threads

FIXTURES = parity_fixtures("random_1k", "deep_inbred_60g")
FIXTURE_NAMES = sorted(FIXTURES)
SELECTORS = (
    {"max_degree": 0},
    {"max_degree": 1},
    {"max_degree": 2},
    {"max_degree": 3},
    {"max_degree": 5},
    {"categories": ["FS"]},
    {"categories": ["1C", "Av"]},
    {"categories": ["2C"]},
    {"categories": ["H1C1R", "MO"]},
)


def _pair_lengths(receiver, **selector) -> dict[str, int | None]:
    pairs = receiver.relationship_pairs(**selector)
    return {code: len(block) if block.requested else None for code, block in pairs.items()}


def _assert_counts_match(receiver, **selector) -> None:
    counts = receiver.relationship_counts(**selector)
    assert dict(counts) == _pair_lengths(receiver, **selector)
    assert counts.requested == frozenset(code for code, value in counts.items() if value is not None)
    assert counts.exact == counts.requested
    assert counts.approximate == frozenset()
    assert counts.clamped == frozenset()


@pytest.mark.parametrize("name", FIXTURE_NAMES)
@pytest.mark.parametrize("selector", SELECTORS, ids=str)
def test_graph_counts_equal_pair_block_lengths(name, selector):
    graph = PedigreeGraph.from_frame(parity_columns(FIXTURES[name]))
    _assert_counts_match(graph, **selector)


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_reordered_rows_give_the_same_counts(name):
    columns = parity_columns(FIXTURES[name])
    n = len(columns["id"])
    reference = PedigreeGraph.from_frame(columns).relationship_counts(max_degree=5)
    for seed in (11, 12):
        perm = np.random.default_rng(seed).permutation(n)
        shuffled = {key: np.asarray(value)[perm] for key, value in columns.items()}
        graph = PedigreeGraph.from_frame(shuffled)
        assert dict(graph.relationship_counts(max_degree=5)) == dict(reference)
        _assert_counts_match(graph, max_degree=5)


@pytest.mark.parametrize("name", FIXTURE_NAMES)
@pytest.mark.parametrize("seed", [3, 4])
def test_view_counts_equal_view_pair_block_lengths(name, seed):
    graph = PedigreeGraph.from_frame(parity_columns(FIXTURES[name]))
    n = graph.n_individuals
    rng = np.random.default_rng(seed)
    rows = rng.permutation(n)[: max(n // 2, min(n, 2))]
    view = graph.view(rows=rows)
    for selector in SELECTORS:
        _assert_counts_match(view, **selector)


def test_a_view_of_one_row_counts_nothing(small_pedigree):
    graph = PedigreeGraph.from_frame(small_pedigree)
    counts = graph.view(rows=np.array([3])).relationship_counts(max_degree=5)
    assert set(counts.values()) == {0}


@st.composite
def random_pedigree(draw):
    """A valid pedigree in shuffled row order with external parents, missing parents, twins, and loops."""
    n = draw(st.integers(min_value=0, max_value=40))
    mother = [-1] * n
    father = [-1] * n
    for i in range(1, n):
        for parents in (mother, father):
            kind = draw(st.sampled_from(["missing", "row", "row", "external"]))
            if kind == "row":
                parents[i] = draw(st.integers(min_value=0, max_value=i - 1))
            elif kind == "external":
                parents[i] = -2 - draw(st.integers(min_value=0, max_value=2))
        if mother[i] == father[i] and mother[i] != -1:
            father[i] = -1
    twin = [-1] * n
    for i in range(n):
        if twin[i] != -1:
            continue
        for j in range(i + 1, n):
            if twin[j] == -1 and mother[j] == mother[i] and father[j] == father[i] and draw(st.booleans()):
                twin[i], twin[j] = j, i
                break
    perm = draw(st.permutations(range(n)))
    inverse = [0] * n
    for position, row in enumerate(perm):
        inverse[row] = position
    ids = [row * 5 + 2 for row in range(n)]

    def to_id(ref):
        if ref == -1:
            return -1
        if ref < -1:
            return 9_000 + (-ref)
        return ids[ref]

    return {
        "id": np.array([ids[perm[k]] for k in range(n)], dtype=np.int64),
        "mother": np.array([to_id(mother[perm[k]]) for k in range(n)], dtype=np.int64),
        "father": np.array([to_id(father[perm[k]]) for k in range(n)], dtype=np.int64),
        "twin": np.array([to_id(twin[perm[k]]) for k in range(n)], dtype=np.int64),
    }


@settings(max_examples=300, deadline=None)
@given(random_pedigree(), st.integers(min_value=0, max_value=5))
def test_random_pedigrees_match_the_matrix_engine(columns, max_degree):
    graph = PedigreeGraph.from_frame(columns)
    _assert_counts_match(graph, max_degree=max_degree)
    if graph.n_individuals >= 2:
        view = graph.view(rows=np.arange(graph.n_individuals)[::2])
        _assert_counts_match(view, max_degree=max_degree)


class TestSelectorsAndErrors:
    def test_the_selectors_are_validated_as_for_pairs(self, small_pedigree):
        graph = PedigreeGraph.from_frame(small_pedigree)
        with pytest.raises(TypeError):
            graph.relationship_counts()
        with pytest.raises(TypeError):
            graph.relationship_counts(max_degree=2, categories=["FS"])
        with pytest.raises(TypeError):
            graph.relationship_counts(categories="FS")
        with pytest.raises(PedigreeValidationError, match="max_degree"):
            graph.relationship_counts(max_degree=6)
        with pytest.raises(PedigreeValidationError, match="unknown"):
            graph.relationship_counts(categories=["cousin"])

    def test_the_result_iterates_every_code_in_registry_order(self, small_pedigree):
        counts = PedigreeGraph.from_frame(small_pedigree).relationship_counts(categories=["FS"])
        assert tuple(counts) == tuple(RELATIONSHIPS)
        assert counts["FS"] is not None
        assert counts["MO"] is None

    def test_counts_are_the_same_under_every_thread_budget(self, small_pedigree):
        graph = PedigreeGraph.from_frame(small_pedigree)
        args = (graph.mother_rows, graph.father_rows, graph.twin_rows, graph.mother_ids, graph.father_ids)
        one = _native.relationship_counts(*args, max_degree=5, threads=1)
        four = _native.relationship_counts(*args, max_degree=5, threads=4)
        assert one.tolist() == four.tolist()
        _reset_thread_state()
        try:
            configure_threads(3)
            assert dict(graph.relationship_counts(max_degree=5)) == dict(zip(RELATIONSHIPS, one.tolist(), strict=True))
        finally:
            _reset_thread_state()

    def test_the_binding_rejects_malformed_rows_with_value_error(self):
        rows = np.array([-1, 0], dtype=np.int32)
        ids = np.array([-1, 7], dtype=np.int64)
        good = (rows, rows.copy(), np.full(2, -1, dtype=np.int32), ids, ids)
        assert _native.relationship_counts(*good, max_degree=5, threads=1).tolist() == [0, 1] + [0] * 21
        with pytest.raises(ValueError, match="twin_rows"):
            _native.relationship_counts(
                *good[:2], np.array([-1, 5], dtype=np.int32), *good[3:], max_degree=5, threads=1
            )
        with pytest.raises(ValueError, match="mother_ids"):
            _native.relationship_counts(*good[:3], ids[:1], ids, max_degree=5, threads=1)
        with pytest.raises(ValueError, match="selected"):
            _native.relationship_counts(*good, max_degree=5, threads=1, selected=np.array([True]))
        with pytest.raises(ValueError, match="threads"):
            _native.relationship_counts(*good, max_degree=5, threads=0)
