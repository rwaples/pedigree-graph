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
from pedigree_graph.relationships import RelationshipCountResult

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


def test_count_result_copies_and_freezes_its_mapping(small_pedigree):
    graph = PedigreeGraph.from_frame(small_pedigree)
    original = graph.relationship_counts(max_degree=3)
    counts = dict(original)
    result = RelationshipCountResult(
        counts,
        original.requested,
        original.exact,
        original.approximate,
        original.clamped,
    )
    counts["MZ"] = 999
    assert result["MZ"] == original["MZ"]
    with pytest.raises(TypeError):
        result._counts["MZ"] = 999  # ty: ignore[invalid-assignment]


@pytest.mark.parametrize("malformed", ["missing", "reordered"])
def test_count_result_rejects_invalid_registry_shape(small_pedigree, malformed):
    original = PedigreeGraph.from_frame(small_pedigree).relationship_counts(max_degree=3)
    counts = dict(original)
    if malformed == "missing":
        counts.pop("MZ")
    else:
        counts = dict(reversed(tuple(counts.items())))
    with pytest.raises(ValueError, match="registry code in registry order"):
        RelationshipCountResult(
            counts,
            original.requested,
            original.exact,
            original.approximate,
            original.clamped,
        )


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
        with pytest.raises(TypeError, match="exactly one of max_degree") as selector_error:
            graph.relationship_counts()
        assert "relationship_pairs" not in str(selector_error.value)
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
        one = _native.relationship_counts(graph._built, max_degree=5, threads=1)
        four = _native.relationship_counts(graph._built, max_degree=5, threads=4)
        assert one == four
        _reset_thread_state()
        try:
            configure_threads(3)
            assert dict(graph.relationship_counts(max_degree=5)) == one
        finally:
            _reset_thread_state()

    def test_the_binding_keys_its_counts_by_code_in_registry_order(self, small_pedigree):
        """A reordering of the Rust ``Category::ALL`` must not silently repermute the counts.

        The positional return this replaced could only be checked for length.
        """
        graph = PedigreeGraph.from_frame(small_pedigree)
        counted = _native.relationship_counts(graph._built, max_degree=5, threads=1)
        assert tuple(counted) == tuple(RELATIONSHIPS)

    def test_a_built_pedigree_cannot_be_constructed_from_python(self):
        """``build_pedigree`` is the only source, which is why the binding no longer revalidates."""
        with pytest.raises(TypeError):
            _native.BuiltPedigree()

    def test_the_binding_rejects_a_malformed_mask_or_thread_count(self, small_pedigree):
        """The loose arguments are the only ones a caller can still malform.

        The column invariants moved into the core's checked constructor and are
        covered by the Rust unit tests of ``Pedigree::try_new``.
        """
        graph = PedigreeGraph.from_frame(small_pedigree)
        with pytest.raises(ValueError, match="selected"):
            _native.relationship_counts(graph._built, max_degree=5, threads=1, selected=np.array([True]))
        with pytest.raises(ValueError, match="threads"):
            _native.relationship_counts(graph._built, max_degree=5, threads=0)

    @pytest.mark.parametrize("max_degree", [6, 9, 255])
    def test_the_binding_rejects_an_out_of_range_max_degree(self, small_pedigree, max_degree):
        """The core clamped instead of rejecting, so degree 9 returned degree-5 counts (issue #20).

        The pure-Python path already raises this exact error from
        ``_validate_max_degree``, so both surfaces report one code.
        """
        graph = PedigreeGraph.from_frame(small_pedigree)
        with pytest.raises(PedigreeValidationError) as info:
            _native.relationship_counts(graph._built, max_degree=max_degree, threads=1)
        assert info.value.code == "max_degree_out_of_range"
        assert info.value.fields == {"value": max_degree, "minimum": 0, "maximum": 5}

    def test_a_mutated_built_pedigree_raises_rather_than_panicking(self, small_pedigree):
        """The core rechecks its preconditions, so the one remaining way to forge bad columns is safe.

        A graph's columns are read-only (``_initialize``), but a caller holding
        a raw ``build_pedigree`` result can still write to them.
        """
        graph = PedigreeGraph.from_frame(small_pedigree)
        built = _native.build_pedigree(
            np.asarray(graph.ids),
            np.asarray(graph.mother_ids),
            np.asarray(graph.father_ids),
            sex_encoding="simace",
        )
        built.mother_rows[0] = graph.n_individuals
        with pytest.raises(PedigreeValidationError):
            _native.relationship_counts(built, max_degree=5, threads=1)
