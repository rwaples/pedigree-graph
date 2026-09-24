"""``relationship_counts`` through the Rust engine equals the block lengths of ``relationship_pairs``.

``relationship_pairs`` runs on the same Rust engine, so this
is a self-consistency check between its two consumers: the counts must equal
the block lengths on every fixture, every selector, every row order, and
every view, without building a pair list.  The independent oracle for both
is the matrix engine in ``tests/oracle/relationship_pairs.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from _support import _run_child
from conftest import FIXTURE_NAMES, FIXTURES, parity_columns
from hypothesis import given, settings
from hypothesis import strategies as st

from pedigree_graph import RELATIONSHIPS, PedigreeGraph, PedigreeValidationError, _native
from pedigree_graph._threads import thread_budget
from pedigree_graph.relationships import RelationshipCountResult

SMALL_PEDIGREE = Path(__file__).parent / "data" / "small_pedigree.parquet"
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
    assert not hasattr(counts, "approximate")
    assert not hasattr(counts, "clamped")


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
        with pytest.raises(PedigreeValidationError, match="unknown"):
            PedigreeGraph.from_frame(small_pedigree).relationship_counts(categories=["cousin"])

    def test_the_result_iterates_every_code_in_registry_order(self, small_pedigree):
        counts = PedigreeGraph.from_frame(small_pedigree).relationship_counts(categories=["FS"])
        assert tuple(counts) == tuple(RELATIONSHIPS)
        assert counts["FS"] is not None
        assert counts["MO"] is None

    def test_counts_are_the_same_under_every_thread_budget(self):
        """The package pool is built once per process, so each budget runs in its own interpreter."""
        body = f"""
            import polars as pl
            from pedigree_graph import PedigreeGraph, _native
            from pedigree_graph._threads import thread_budget
            graph = PedigreeGraph.from_frame(pl.read_parquet({str(SMALL_PEDIGREE)!r}))
            print(thread_budget(), _native.relationship_counts(graph._built, max_degree=5, threads=thread_budget()))
        """
        outputs = {}
        for threads in ("1", "4"):
            budget, counts = _run_child(body, PEDIGREE_GRAPH_THREADS=threads).split(" ", 1)
            assert budget == threads
            outputs[threads] = counts
        assert outputs["1"] == outputs["4"]

    def test_the_binding_configures_the_package_pool_from_its_threads_argument(self, small_pedigree):
        graph = PedigreeGraph.from_frame(small_pedigree)
        budget = thread_budget()
        assert _native.configure_pool(budget) == budget
        _native.relationship_counts(graph._built, max_degree=2, threads=budget)
        # The same class `configure_threads` raises for the same reason.
        with pytest.raises(RuntimeError, match="already configured"):
            _native.relationship_counts(graph._built, max_degree=2, threads=budget + 1)

    def test_the_binding_keys_its_counts_by_code_in_registry_order(self, small_pedigree):
        """A reordering of the Rust ``Category::ALL`` must not silently repermute the counts.

        The positional return this replaced could only be checked for length.
        """
        graph = PedigreeGraph.from_frame(small_pedigree)
        counted = _native.relationship_counts(graph._built, max_degree=5, threads=thread_budget())
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
            _native.relationship_counts(graph._built, max_degree=5, threads=thread_budget(), selected=np.array([True]))
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
            _native.relationship_counts(graph._built, max_degree=max_degree, threads=thread_budget())
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
            _native.relationship_counts(built, max_degree=5, threads=thread_budget())
