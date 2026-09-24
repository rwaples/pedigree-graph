"""``_native.pair_kinship`` and ``_native.kinship_support_values`` against the pure-Python oracle.

The binding is the seam the pairwise recurrence crosses (ADR 0007, 0009):
the Rust walk evaluates in graph space with structural depth as
the peel input, and Python keeps the query resolution.  These tests hold the
raw binding against ``tests/oracle/pair_kinship.py`` on every parity fixture
in both endpoint orders, and pin the boundary
contract: an owned float32 array, structured errors, and every kinship
allocation family surfacing as ``allocation_failed``.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from _support import _PAIRWISE_FIXTURES, CHILD_PRELUDE, _run_child
from conftest import parity_columns, parity_fixtures
from oracle.pair_kinship import pair_kinship as oracle_pair_kinship

from pedigree_graph import PedigreeGraph, PedigreeValidationError, _native

FIXTURES = parity_fixtures("random_1k", "deep_inbred_60g")
FIXTURE_NAMES = sorted(FIXTURES)
KINSHIP_FAMILIES = ("kinship_memo", "kinship_stack", "kinship_output")
ENVELOPE_UNIT = 2.0**-25


def _graph(name: str) -> PedigreeGraph:
    return PedigreeGraph.from_frame(parity_columns(FIXTURES[name]))


def _native_kinship(graph: PedigreeGraph, first, second) -> np.ndarray:
    return _native.pair_kinship(
        graph._built, graph.depth, np.asarray(first, dtype=np.int32), np.asarray(second, dtype=np.int32)
    )


def _oracle(graph: PedigreeGraph, first, second) -> np.ndarray:
    return oracle_pair_kinship(graph.mother_rows, graph.father_rows, graph.twin_rows, graph.depth, first, second)


def _all_pairs(n: int) -> tuple[np.ndarray, np.ndarray]:
    return np.triu_indices(n)


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_every_pair_matches_the_oracle_in_both_orders(name):
    graph = _graph(name)
    first, second = _all_pairs(graph.n_individuals)
    expected = _oracle(graph, first, second)
    assert _native_kinship(graph, first, second).tobytes() == expected.tobytes()
    assert _native_kinship(graph, second, first).tobytes() == expected.tobytes()


@pytest.mark.parametrize("build", _PAIRWISE_FIXTURES, ids=lambda b: b.__name__)
def test_mz_and_inbred_constructions_match_the_oracle(build):
    graph = PedigreeGraph.from_frame(build())
    first, second = _all_pairs(graph.n_individuals)
    assert _native_kinship(graph, first, second).tobytes() == _oracle(graph, first, second).tobytes()


def _random_pedigree(rng: np.random.Generator, p_twin: float = 0.3) -> pl.DataFrame:
    """Generate a small valid (topologically ordered) random pedigree.

    Parent-index-only; no simace dependency.  Mixes inbreeding (mates drawn
    from the same cohort) and occasional MZ twins so the fuzz exercises both
    correction paths.  Children always get a higher id than their parents, so
    the topological invariant holds.
    """
    n_founders = int(rng.integers(2, 5))
    n_gen = int(rng.integers(1, 5))
    per_gen = int(rng.integers(1, 4))
    ids = list(range(n_founders))
    mother = [-1] * n_founders
    father = [-1] * n_founders
    twin = [-1] * n_founders
    gen = [0] * n_founders
    sex = [int(rng.integers(0, 2)) for _ in range(n_founders)]
    cur = list(range(n_founders))
    next_id = n_founders
    for g in range(1, n_gen + 1):
        new_gen: list[int] = []
        females = [i for i in cur if sex[i] == 0] or cur
        males = [i for i in cur if sex[i] == 1] or cur
        for _ in range(per_gen):
            m = int(rng.choice(females))
            # A child cannot name one individual in both parent roles.
            mates = [i for i in males if i != m] or [i for i in cur if i != m]
            f = int(rng.choice(mates)) if mates else -1
            ids.append(next_id)
            mother.append(m)
            father.append(f)
            twin.append(-1)
            gen.append(g)
            sex.append(int(rng.integers(0, 2)))
            new_gen.append(next_id)
            next_id += 1
        # Occasionally turn the last two new individuals into MZ twins.
        if len(new_gen) >= 2 and rng.random() < p_twin:
            a, b = new_gen[-1], new_gen[-2]
            mother[b] = mother[a]
            father[b] = father[a]
            twin[a] = b
            twin[b] = a
            sex[b] = sex[a]
        cur = new_gen
    return pl.DataFrame({"id": ids, "mother": mother, "father": father, "twin": twin, "sex": sex, "generation": gen})


def test_fuzz_native_equals_the_oracle_and_the_matrix():
    rng = np.random.default_rng(20240609)
    checked = 0
    for _ in range(200):
        pg = PedigreeGraph.from_frame(_random_pedigree(rng))
        if pg.n_individuals < 2:
            continue
        K = pg.kinship_matrix().toarray()
        ii, jj = np.triu_indices(pg.n_individuals)
        py = _oracle(pg, ii, jj)
        nb = _native_kinship(pg, ii, jj)
        np.testing.assert_array_equal(nb, py)
        np.testing.assert_array_equal(nb, K[ii, jj])
        checked += 1
    assert checked > 100  # the generator should mostly yield n >= 2


def test_self_pairs_encode_inbreeding():
    graph = _graph("deep_inbred_60g")
    rows = np.arange(graph.n_individuals)
    values = _native_kinship(graph, rows, rows).astype(np.float64)
    assert np.abs(2.0 * values - 1.0 - graph.inbreeding()).max() <= 2.0**-22


def test_permuted_graphs_stay_inside_the_envelope(capsys):
    fixture = FIXTURES["deep_inbred_60g"]
    reference = _graph("deep_inbred_60g")
    n = reference.n_individuals
    first, second = _all_pairs(n)
    want = _native_kinship(reference, first, second)
    tolerance = 2.0 * (reference.depth[first] + reference.depth[second] + 1) * ENVELOPE_UNIT
    worst = 0
    for seed in (11, 12):
        perm = np.random.default_rng(seed).permutation(n)
        permuted = PedigreeGraph.from_frame({key: value[perm] for key, value in parity_columns(fixture).items()})
        inverse = np.empty(n, dtype=np.intp)
        inverse[perm] = np.arange(n)
        got = _native_kinship(permuted, inverse[first], inverse[second])
        assert got.tobytes() == _oracle(permuted, inverse[first], inverse[second]).tobytes()
        assert np.all(np.abs(want.astype(np.float64) - got.astype(np.float64)) <= tolerance)
        ulp = np.abs(want.view(np.int32).astype(np.int64) - got.view(np.int32).astype(np.int64))
        worst = max(worst, int(ulp.max()))
    with capsys.disabled():
        print(f"native cross-order deep_inbred_60g: max {worst} ulp")


class TestBoundary:
    def test_the_result_is_an_owned_contiguous_float32(self):
        graph = _graph("random_1k")
        values = _native_kinship(graph, [0, 1], [1, 2])
        assert values.dtype == np.float32
        assert values.ndim == 1
        assert values.flags.c_contiguous
        assert not isinstance(values.base, np.ndarray)

    def test_empty_pairs_give_an_empty_float32(self):
        graph = _graph("random_1k")
        empty = np.zeros(0, dtype=np.int32)
        assert _native_kinship(graph, empty, empty).shape == (0,)

    def test_malformed_arguments_raise_structured_errors(self):
        graph = _graph("nuclear_full_sibs")
        n = graph.n_individuals
        with pytest.raises(PedigreeValidationError) as info:
            _native_kinship(graph, [0, n], [0, 0])
        assert info.value.code == "value_out_of_range"
        assert info.value.fields["field"] == "first"
        with pytest.raises(PedigreeValidationError) as info:
            _native_kinship(graph, [0], [0, 1])
        assert info.value.code == "length_mismatch"
        with pytest.raises(PedigreeValidationError) as info:
            _native.pair_kinship(graph._built, graph.depth[:-1], np.zeros(1, np.int32), np.zeros(1, np.int32))
        assert info.value.code == "length_mismatch"
        assert info.value.fields["field"] == "depth"

    def test_the_stub_names_both_entries(self):
        from pathlib import Path

        import pedigree_graph

        stub = (Path(pedigree_graph.__file__).parent / "_native.pyi").read_text()
        for name in ("pair_kinship", "kinship_support_values"):
            assert hasattr(_native, name)
            assert f"def {name}(" in stub


class TestSupportValues:
    @staticmethod
    def _support(graph: PedigreeGraph, matrix):
        return _native.kinship_support_values(
            graph._built,
            graph.depth,
            np.ascontiguousarray(matrix.indptr, dtype=np.int64),
            np.ascontiguousarray(matrix.indices, dtype=np.int32),
        )

    @pytest.mark.parametrize("name", ["double_first_cousins", "deep_inbred_60g", "random_1k"])
    def test_complete_support_reproduces_the_matrix(self, name):
        graph = _graph(name)
        matrix = graph.kinship_matrix()
        values = _native.kinship_support_values(
            graph._built, graph.depth, matrix.indptr.astype(np.int64), matrix.indices
        )
        assert values.dtype == np.float32
        assert values.tobytes() == matrix.data.tobytes()

    def test_relationship_support_matches_pair_kinship(self):
        graph = _graph("random_1k")
        matrix = graph.relationship_kinship_matrix(max_degree=3)
        values = self._support(graph, matrix)
        coo = matrix.tocoo()
        assert values.tobytes() == _native_kinship(graph, coo.row, coo.col).tobytes()

    def test_a_missing_mirror_is_asymmetric(self):
        graph = _graph("nuclear_full_sibs")
        indptr = np.array([0, 1, 2, 3, 4, 6], dtype=np.int64)
        indices = np.array([0, 1, 2, 3, 0, 4], dtype=np.int32)
        with pytest.raises(PedigreeValidationError) as info:
            _native.kinship_support_values(graph._built, graph.depth, indptr, indices)
        assert info.value.code == "kinship_support_asymmetric"
        assert dict(info.value.fields) == {"row": 0, "column": 4}

    def test_a_lower_entry_without_an_upper_mirror_is_asymmetric(self):
        graph = _graph("nuclear_full_sibs")
        indptr = np.array([0, 2, 3, 4, 5, 6], dtype=np.int64)
        indices = np.array([0, 4, 1, 2, 3, 4], dtype=np.int32)
        with pytest.raises(PedigreeValidationError) as info:
            _native.kinship_support_values(graph._built, graph.depth, indptr, indices)
        assert info.value.code == "kinship_support_asymmetric"
        assert dict(info.value.fields) == {"row": 4, "column": 0}

    def test_an_indptr_that_stops_short_of_nnz_is_rejected(self):
        graph = _graph("nuclear_full_sibs")
        indptr = np.array([0, 1, 2, 3, 4, 4], dtype=np.int64)
        indices = np.array([0, 1, 2, 3, 4], dtype=np.int32)
        with pytest.raises(PedigreeValidationError) as info:
            _native.kinship_support_values(graph._built, graph.depth, indptr, indices)
        assert info.value.code == "value_out_of_range"
        assert info.value.fields["field"] == "indptr"
        assert info.value.fields["position"] == 5

    def test_an_unsorted_column_is_rejected(self):
        graph = _graph("nuclear_full_sibs")
        indptr = np.array([0, 2, 3, 4, 5, 7], dtype=np.int64)
        indices = np.array([4, 0, 1, 2, 3, 0, 4], dtype=np.int32)
        with pytest.raises(PedigreeValidationError) as info:
            _native.kinship_support_values(graph._built, graph.depth, indptr, indices)
        assert info.value.code == "kinship_support_unsorted"
        assert info.value.fields["column"] == 0


@pytest.mark.parametrize("family", KINSHIP_FAMILIES)
def test_a_refused_allocation_raises_a_resource_error(family):
    """Both entries surface every kinship family as ``ResourceError("allocation_failed")`` in a fresh process."""
    body = f"""
        from pedigree_graph import ResourceError
        family = {family!r}
        rows = np.arange(n, dtype=np.int32)
        matrix = graph.relationship_kinship_matrix(max_degree=2)
        def pairs():
            return _native.pair_kinship(graph._built, graph.depth, rows, rows)
        def support():
            return _native.kinship_support_values(
                graph._built, graph.depth, matrix.indptr.astype(np.int64), matrix.indices
            )
        for label, call in (("pairs", pairs), ("support", support)):
            _native.fail_next_allocation(family, 1)
            try:
                call()
            except ResourceError as e:
                print(label, e.code, e.fields["operation"], e.fields["requested_elements"] >= 1)
            else:
                print(label, "no error")
            _native.fail_next_allocation(None)
            call()
        print("recovered")
    """
    lines = _run_child(CHILD_PRELUDE, body).strip().splitlines()
    assert lines[-1] == "recovered"
    assert lines[:-1] == [f"pairs allocation_failed {family} True", f"support allocation_failed {family} True"]
