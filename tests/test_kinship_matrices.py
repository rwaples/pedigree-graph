"""The three explicit kinship-matrix support contracts (ADR 0006, slice 5b)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
from conftest import parity_columns, parity_fixtures

from pedigree_graph import PedigreeGraph
from pedigree_graph._kinship_matrix import _exactify_support
from pedigree_graph._threads import _reset_thread_state, configure_threads

sys.path.insert(0, str(Path(__file__).resolve().parent / "parity"))

import pedigrees

FIXTURES = parity_fixtures("random_1k", "deep_inbred_60g")


def _graph(name: str) -> PedigreeGraph:
    return PedigreeGraph(parity_columns(FIXTURES[name]))


def _upper(matrix: sp.csc_matrix) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coo = matrix.tocoo()
    keep = coo.row <= coo.col
    row = coo.row[keep].astype(np.int32)
    col = coo.col[keep].astype(np.int32)
    values = coo.data[keep].astype(np.float32)
    order = np.lexsort((col, row))
    return row[order], col[order], values[order]


def _assert_csc_contract(matrix: sp.csc_matrix) -> None:
    assert isinstance(matrix, sp.csc_matrix)
    assert matrix.data.dtype == np.float32
    assert matrix.indices.dtype == np.int32
    assert matrix.indptr.dtype == np.int32
    assert matrix.has_sorted_indices
    assert not matrix.data.flags.writeable
    assert not matrix.indices.flags.writeable
    assert not matrix.indptr.flags.writeable


class TestCompleteMatrix:
    def test_complete_support_is_every_nonzero_pair_plus_diagonal(self):
        graph = _graph("deep_inbred_60g")
        matrix = graph.kinship_matrix()
        first, second = np.triu_indices(graph.n_individuals)
        values = graph.pair_kinship(first, second)
        got = np.asarray(matrix[first, second], dtype=np.float32).ravel()
        assert got.tobytes() == values.tobytes()
        np.testing.assert_array_equal(got != 0, values != 0)
        _assert_csc_contract(matrix)

    def test_complete_matrix_is_cached(self):
        graph = _graph("double_first_cousins")
        assert graph.kinship_matrix() is graph.kinship_matrix()


class TestRelationshipMatrix:
    def test_support_is_exactly_selected_closest_categories_plus_diagonal(self):
        graph = _graph("double_first_cousins")
        pairs = graph.relationship_pairs(categories=["1C", "GP"])
        matrix = graph.relationship_kinship_matrix(categories=["1C", "GP"])
        row, col, _ = _upper(matrix)
        off_diagonal = row != col
        got = set(zip(row[off_diagonal].tolist(), col[off_diagonal].tolist(), strict=True))
        expected = {
            tuple(sorted((int(first), int(second))))
            for code in ("1C", "GP")
            for first, second in zip(pairs[code].first_rows, pairs[code].second_rows, strict=True)
        }
        assert got == expected
        np.testing.assert_array_equal(row[~off_diagonal], np.arange(graph.n_individuals, dtype=np.int32))
        _assert_csc_contract(matrix)

    @pytest.mark.parametrize(
        "name",
        ["founder_mz_twins", "backcross_and_selfing_like", "double_first_cousins", "deep_inbred_60g"],
    )
    def test_retained_values_are_pair_kinship_identical(self, name):
        graph = _graph(name)
        matrix = graph.relationship_kinship_matrix(max_degree=5)
        row, col, values = _upper(matrix)
        assert values.tobytes() == graph.pair_kinship(row, col).tobytes()

    def test_empty_category_selection_is_diagonal_only(self):
        graph = _graph("double_first_cousins")
        matrix = graph.relationship_kinship_matrix(categories=[])
        np.testing.assert_array_equal(matrix.indices, np.arange(graph.n_individuals, dtype=np.int32))
        assert matrix.nnz == graph.n_individuals

    def test_selector_specific_caches_do_not_alias(self):
        graph = _graph("single_individual")
        by_degree = graph.relationship_kinship_matrix(max_degree=0)
        by_categories = graph.relationship_kinship_matrix(categories=["MZ"])
        assert by_degree is graph.relationship_kinship_matrix(max_degree=0)
        assert by_categories is graph.relationship_kinship_matrix(categories=["MZ"])
        assert by_degree is not by_categories
        assert by_degree is not graph.kinship_matrix()

    def test_one_shot_category_iterable_is_consumed_once(self):
        graph = _graph("double_first_cousins")
        matrix = graph.relationship_kinship_matrix(categories=(code for code in ["1C"]))
        assert matrix.nnz == graph.n_individuals + 2 * len(graph.relationship_pairs(categories=["1C"])["1C"])


class TestApproximateSupportMatrix:
    def test_support_matches_the_frozen_071_propagated_candidate_set(self):
        graph = _graph("random_1k")
        matrix = graph.approximate_kinship_matrix(min_propagated_kinship=0.001)
        row, col, _ = _upper(matrix)
        with np.load(Path(__file__).parent / "data" / "parity_v0.7.1" / "random_1k.npz") as frozen:
            np.testing.assert_array_equal(row, frozen["approx/row"])
            np.testing.assert_array_equal(col, frozen["approx/col"])
        _assert_csc_contract(matrix)

    @pytest.mark.parametrize(
        "name",
        [
            "founder_mz_twins",
            "mz_twins_with_children",
            "one_parent_known",
            "disconnected_components",
            "backcross_and_selfing_like",
            "double_first_cousins",
            "deep_inbred_60g",
        ],
    )
    def test_retained_values_are_recomputed_pair_kinship_bits(self, name):
        graph = _graph(name)
        matrix = graph.approximate_kinship_matrix(min_propagated_kinship=0.001)
        row, col, values = _upper(matrix)
        assert values.tobytes() == graph.pair_kinship(row, col).tobytes()

    def test_non_dyadic_threshold_is_compared_without_float32_narrowing(self):
        # The lineage's (0, 2) coefficient is exactly 0.125.  The adjacent
        # float64 thresholds straddle it but both round to 0.125 in float32.
        graph = PedigreeGraph.from_frame(
            {
                "id": np.arange(3),
                "mother": np.array([-1, 0, 1]),
                "father": np.full(3, -1),
            }
        )
        below = np.nextafter(0.125, 0.0)
        above = np.nextafter(0.125, 1.0)
        assert graph.approximate_kinship_matrix(min_propagated_kinship=below)[0, 2] == 0.125
        assert graph.approximate_kinship_matrix(min_propagated_kinship=above)[0, 2] == 0.0

    def test_zero_delegates_to_complete(self):
        graph = _graph("double_first_cousins")
        assert graph.approximate_kinship_matrix(min_propagated_kinship=0) is graph.kinship_matrix()

    def test_positive_threshold_cache_is_isolated_from_other_matrix_families(self):
        graph = _graph("single_individual")
        approximate = graph.approximate_kinship_matrix()
        assert approximate is graph.approximate_kinship_matrix()
        assert approximate is not graph.kinship_matrix()
        assert approximate is not graph.relationship_kinship_matrix(categories=[])

    @pytest.mark.parametrize("threshold", [-1, np.inf, -np.inf, np.nan, 1.000001, "not-a-number"])
    def test_invalid_threshold_is_an_ordinary_value_error(self, threshold):
        with pytest.raises(ValueError, match="min_propagated_kinship"):
            _graph("single_individual").approximate_kinship_matrix(
                min_propagated_kinship=threshold  # type: ignore[arg-type]
            )

    def test_legacy_positive_threshold_dispatches_to_the_explicit_method(self):
        graph = _graph("double_first_cousins")
        assert graph.kinship_matrix(min_kinship=0.001) is graph.approximate_kinship_matrix(min_propagated_kinship=0.001)

    def test_legacy_max_degree_preserves_propagated_support_semantics(self):
        graph = _graph("double_first_cousins")
        threshold = 0.5 ** (3 + 1) - 1e-9
        assert graph.kinship_matrix(max_degree=3) is graph.approximate_kinship_matrix(min_propagated_kinship=threshold)


class TestChunking:
    def test_chunk_boundaries_do_not_change_values(self):
        graph = _graph("double_first_cousins")
        template = graph.kinship_matrix()
        matrices = []
        for chunk_size in (1, 7, 1 << 20):
            candidate = template.copy()
            candidate.data.setflags(write=True)
            candidate.data.fill(np.nan)
            matrices.append(_exactify_support(graph, candidate, chunk_size=chunk_size))
        for actual in matrices[1:]:
            np.testing.assert_array_equal(actual.indptr, matrices[0].indptr)
            np.testing.assert_array_equal(actual.indices, matrices[0].indices)
            assert actual.data.tobytes() == matrices[0].data.tobytes()

    def test_invalid_internal_chunk_size_is_rejected(self):
        graph = _graph("single_individual")
        candidate = graph.kinship_matrix().copy()
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            _exactify_support(graph, candidate, chunk_size=0)


class TestRowOrder:
    def test_all_matrix_families_match_pair_values_after_reordering(self):
        fixture = FIXTURES["deep_inbred_60g"]
        permutation = np.random.default_rng(22).permutation(len(fixture["ids"]))
        graph = PedigreeGraph({key: value[permutation] for key, value in parity_columns(fixture).items()})
        matrices = (
            graph.kinship_matrix(),
            graph.relationship_kinship_matrix(max_degree=5),
            graph.approximate_kinship_matrix(),
        )
        for matrix in matrices:
            row, col, values = _upper(matrix)
            assert values.tobytes() == graph.pair_kinship(row, col).tobytes()


class TestThreads:
    @pytest.fixture(autouse=True)
    def reset_thread_state(self, monkeypatch):
        monkeypatch.delenv("PEDIGREE_GRAPH_THREADS", raising=False)
        _reset_thread_state()
        yield
        _reset_thread_state()

    @pytest.mark.parametrize("method", ["complete", "relationship", "approximate"])
    def test_new_matrix_call_commits_the_budget(self, method):
        graph = _graph("single_individual")
        if method == "complete":
            graph.kinship_matrix()
        elif method == "relationship":
            graph.relationship_kinship_matrix(categories=[])
        else:
            graph.approximate_kinship_matrix()
        with pytest.raises(RuntimeError):
            configure_threads(3)


@pytest.mark.slow
def test_random_30k_approximate_matrix_runs_full_exact_value_path():
    fixture = pedigrees.build_random("random_30k", pedigrees.LARGE_FIXTURES["random_30k"])
    graph = PedigreeGraph(parity_columns(fixture))

    matrix = graph.approximate_kinship_matrix(min_propagated_kinship=0.001)

    _assert_csc_contract(matrix)
    assert matrix.nnz == 53_817_918
    assert np.isfinite(matrix.data).all()
    assert np.all(matrix.diagonal() >= np.float32(0.5))
