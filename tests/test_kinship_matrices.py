"""The three explicit kinship-matrix support contracts (ADR 0006)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pedigrees
import pytest
import scipy.sparse as sp
from conftest import parity_columns, parity_graph

from pedigree_graph import RELATIONSHIPS, PedigreeGraph
from pedigree_graph._kinship_matrix import _exactify_support
from pedigree_graph._threads import configure_threads


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
        graph = parity_graph("deep_inbred_60g")
        matrix = graph.kinship_matrix()
        first, second = np.triu_indices(graph.n_individuals)
        values = graph.pair_kinship(first, second)
        got = np.asarray(matrix[first, second], dtype=np.float32).ravel()
        assert got.tobytes() == values.tobytes()
        np.testing.assert_array_equal(got != 0, values != 0)
        _assert_csc_contract(matrix)

    def test_complete_matrix_is_cached(self):
        graph = parity_graph("double_first_cousins")
        assert graph.kinship_matrix() is graph.kinship_matrix()


class TestRelationshipMatrix:
    def test_support_is_exactly_selected_closest_categories_plus_diagonal(self):
        graph = parity_graph("double_first_cousins")
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
        graph = parity_graph(name)
        matrix = graph.relationship_kinship_matrix(max_degree=5)
        row, col, values = _upper(matrix)
        assert values.tobytes() == graph.pair_kinship(row, col).tobytes()

    def test_empty_category_selection_is_diagonal_only(self):
        graph = parity_graph("double_first_cousins")
        matrix = graph.relationship_kinship_matrix(categories=[])
        np.testing.assert_array_equal(matrix.indices, np.arange(graph.n_individuals, dtype=np.int32))
        assert matrix.nnz == graph.n_individuals

    def test_the_cache_is_per_selection_and_never_aliases_another_family(self):
        """Repeat calls hit the cache; the relationship family stays its own contract.

        ``max_degree=0`` and ``categories=["MZ"]`` name the same code, so since
        the cache became selection-keyed they are one entry — covered by
        :func:`test_equivalent_selectors_share_one_relationship_cache_entry`.
        What must not alias is a *different* family: complete, closest-category,
        and propagation-pruned support are distinct contracts even when a
        pedigree makes them structurally identical.
        """
        graph = parity_graph("single_individual")
        by_degree = graph.relationship_kinship_matrix(max_degree=0)
        assert by_degree is graph.relationship_kinship_matrix(max_degree=0)
        assert graph.relationship_kinship_matrix(categories=["FS"]) is not by_degree
        assert by_degree is not graph.kinship_matrix()
        assert by_degree is not graph.approximate_kinship_matrix(min_propagated_kinship=0.001)

    def test_one_shot_category_iterable_is_consumed_once(self):
        graph = parity_graph("double_first_cousins")
        matrix = graph.relationship_kinship_matrix(categories=(code for code in ["1C"]))
        assert matrix.nnz == graph.n_individuals + 2 * len(graph.relationship_pairs(categories=["1C"])["1C"])


class TestApproximateSupportMatrix:
    def test_support_matches_the_frozen_071_propagated_candidate_set(self):
        graph = parity_graph("random_1k")
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
        graph = parity_graph(name)
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
        graph = parity_graph("double_first_cousins")
        assert graph.approximate_kinship_matrix(min_propagated_kinship=0) is graph.kinship_matrix()

    def test_positive_threshold_cache_is_isolated_from_other_matrix_families(self):
        graph = parity_graph("single_individual")
        approximate = graph.approximate_kinship_matrix()
        assert approximate is graph.approximate_kinship_matrix()
        assert approximate is not graph.kinship_matrix()
        assert approximate is not graph.relationship_kinship_matrix(categories=[])

    @pytest.mark.parametrize("threshold", [-1, np.inf, -np.inf, np.nan, 1.000001, "not-a-number"])
    def test_invalid_threshold_is_an_ordinary_value_error(self, threshold):
        with pytest.raises(ValueError, match="min_propagated_kinship"):
            parity_graph("single_individual").approximate_kinship_matrix(
                min_propagated_kinship=threshold  # type: ignore[arg-type]
            )


class TestSupportWalk:
    def test_exactifying_the_complete_support_reproduces_the_matrix(self):
        graph = parity_graph("double_first_cousins")
        template = graph.kinship_matrix()
        candidate = template.copy()
        candidate.data.setflags(write=True)
        candidate.data.fill(np.nan)
        actual = _exactify_support(graph, candidate)
        np.testing.assert_array_equal(actual.indptr, template.indptr)
        np.testing.assert_array_equal(actual.indices, template.indices)
        assert actual.data.tobytes() == template.data.tobytes()
        assert actual.data.dtype == np.float32
        assert not actual.data.flags.writeable


class TestRowOrder:
    def test_all_matrix_families_match_pair_values_after_reordering(self):
        graph = parity_graph("deep_inbred_60g", seed=22)
        matrices = (
            graph.kinship_matrix(),
            graph.relationship_kinship_matrix(max_degree=5),
            graph.approximate_kinship_matrix(),
        )
        for matrix in matrices:
            row, col, values = _upper(matrix)
            assert values.tobytes() == graph.pair_kinship(row, col).tobytes()


@pytest.mark.usefixtures("fresh_thread_state")
class TestThreads:
    @pytest.mark.parametrize("method", ["complete", "relationship", "approximate"])
    def test_new_matrix_call_commits_the_budget(self, method):
        graph = parity_graph("single_individual")
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
    graph = PedigreeGraph.from_frame(parity_columns(fixture))

    matrix = graph.approximate_kinship_matrix(min_propagated_kinship=0.001)

    _assert_csc_contract(matrix)
    assert matrix.nnz == 53_817_918
    assert np.isfinite(matrix.data).all()
    assert np.all(matrix.diagonal() >= np.float32(0.5))


def test_equivalent_selectors_share_one_relationship_cache_entry():
    """``max_degree=2`` and the codes it names are one selection, so one entry.

    Both selectors resolve to the same code set, hence the same support and the
    same matrix.  Keying the cache by the resolved codes rather than by the
    selector the caller happened to write collapses them; keying by selector
    shape recomputes a matrix the graph already holds.
    """
    graph = parity_graph("random_1k")
    codes = tuple(code for code, category in RELATIONSHIPS.items() if category.degree <= 2)

    by_degree = graph.relationship_kinship_matrix(max_degree=2)
    by_codes = graph.relationship_kinship_matrix(categories=codes)
    shuffled = graph.relationship_kinship_matrix(categories=tuple(reversed(codes)))

    assert len(graph._relationship_kinship_cache) == 1
    assert by_codes is by_degree
    assert shuffled is by_degree


def test_release_kinship_matrices_drops_every_family_and_is_idempotent():
    graph = parity_graph("random_1k")
    complete = graph.kinship_matrix()
    graph.approximate_kinship_matrix(min_propagated_kinship=0.001)
    graph.relationship_kinship_matrix(max_degree=2)
    assert graph._complete_kinship_cache is not None
    assert graph._approximate_kinship_cache
    assert graph._relationship_kinship_cache

    graph._release_kinship_matrices()
    graph._release_kinship_matrices()

    assert graph._complete_kinship_cache is None
    assert not graph._approximate_kinship_cache
    assert not graph._relationship_kinship_cache
    rebuilt = graph.kinship_matrix()
    assert rebuilt is not complete
    assert np.array_equal(rebuilt.data, complete.data)
