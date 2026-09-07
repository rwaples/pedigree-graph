"""``mean_kinship_by_generation`` groups by observed labels and pins the MZ rule.

Slice 6a of the 0.8 plan.  Every expected value here is computed by hand on
a small pedigree; the two large fixtures only check that the streamed DP and
the cached-matrix walk agree and that row order does not matter.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import parity_columns, parity_fixtures

from pedigree_graph import MissingMetadataError, PedigreeGraph, ne_coancestry
from pedigree_graph.summaries import GenerationKinshipSummary

# Two founder couples, each with two children, then one grandchild couple:
#   0 x 1 -> 2, 3        4 x 5 -> 6, 7        3 x 6 -> 8, 9   (3 is a mother, so 2 and 3 are both sex 0)
# Row-aligned structural depth: [0 0 1 1 0 0 1 1 2 2].
_IDS = np.arange(10)
_MOTHER = np.array([-1, -1, 0, 0, -1, -1, 4, 4, 3, 3])
_FATHER = np.array([-1, -1, 1, 1, -1, -1, 5, 5, 6, 6])
_SEX = np.array([0, 1, 0, 0, 0, 1, 0, 1, 0, 1])


def _graph(generation=None, twin=None) -> PedigreeGraph:
    columns = {"id": _IDS, "mother": _MOTHER, "father": _FATHER, "sex": _SEX}
    if generation is not None:
        columns["generation"] = np.asarray(generation)
    if twin is not None:
        columns["twin"] = np.asarray(twin)
    return PedigreeGraph.from_frame(columns)


def _assert_summary(summary: GenerationKinshipSummary, generations, mean_kinship, pair_counts, unlabelled):
    np.testing.assert_array_equal(summary.generations, np.asarray(generations, dtype=np.int32))
    np.testing.assert_allclose(summary.mean_kinship, mean_kinship, rtol=0, atol=1e-12, equal_nan=True)
    np.testing.assert_array_equal(summary.pair_counts, np.asarray(pair_counts, dtype=np.int64))
    assert summary.unlabelled_individual_count == unlabelled


# Depth-0 founders are unrelated: 6 pairs, all 0.
# Depth 1 holds 2, 3, 6, 7: two full-sib pairs at 0.25 and four unrelated pairs.
# Depth 2 holds the sibs 8, 9 at 0.25.
_DEPTH_MEANS = [0.0, 0.5 / 6, 0.25]
_DEPTH_PAIRS = [6, 6, 1]


def test_absent_labels_fall_back_to_structural_depth():
    pg = _graph()
    assert pg.generation_labels is None
    _assert_summary(pg.mean_kinship_by_generation(), [0, 1, 2], _DEPTH_MEANS, _DEPTH_PAIRS, 0)


def test_partial_labels_exclude_and_count_the_unlabelled_rows():
    # The second founder couple and their children carry no label.
    pg = _graph(generation=[0, 0, 1, 1, -1, -1, -1, -1, 2, 2])
    _assert_summary(pg.mean_kinship_by_generation(), [0, 1, 2], [0.0, 0.25, 0.25], [1, 1, 1], 4)


def test_partial_labels_still_reject_the_estimators_and_the_adapter():
    pg = _graph(generation=[0, 0, 1, 1, -1, -1, -1, -1, 2, 2])
    pg.mean_kinship_by_generation()
    with pytest.raises(MissingMetadataError):
        ne_coancestry(pg)
    with pytest.raises(MissingMetadataError) as info:
        pg.per_gen_mean_kinship()
    assert info.value.fields["operation"] == "per_gen_mean_kinship"


def test_sparse_labels_return_only_observed_groups():
    pg = _graph(generation=[5, 5, 9, 9, 5, 5, 9, 9, 30, 30])
    _assert_summary(pg.mean_kinship_by_generation(), [5, 9, 30], _DEPTH_MEANS, _DEPTH_PAIRS, 0)


def test_rebased_labels_change_only_the_generations_vector():
    base = _graph(generation=[0, 0, 1, 1, 0, 0, 1, 1, 2, 2]).mean_kinship_by_generation()
    shifted = _graph(generation=[3, 3, 4, 4, 3, 3, 4, 4, 5, 5]).mean_kinship_by_generation()
    np.testing.assert_array_equal(shifted.generations, base.generations + 3)
    np.testing.assert_array_equal(shifted.mean_kinship, base.mean_kinship)
    np.testing.assert_array_equal(shifted.pair_counts, base.pair_counts)


def test_labels_that_merge_depths_group_across_them():
    # Everyone in one cohort: 45 pairs.  Related pairs: four full-sib pairs
    # (2-3, 6-7, 8-9 at 0.25 each; 8-9 are also inbred through nothing, so
    # 0.25), four parent-child pairs per family (0-2, 0-3, 1-2, 1-3, 4-6, 4-7,
    # 5-6, 5-7, 3-8, 3-9, 6-8, 6-9 at 0.25), four grandparent pairs (0-8,
    # 0-9, 1-8, 1-9, 4-8, 4-9, 5-8, 5-9 at 0.125), and avuncular pairs (2-8,
    # 2-9, 7-8, 7-9 at 0.125).
    pg = _graph(generation=np.zeros(10, dtype=np.int64))
    expected = (3 * 0.25 + 12 * 0.25 + 8 * 0.125 + 4 * 0.125) / 45
    _assert_summary(pg.mean_kinship_by_generation(), [0], [expected], [45], 0)


def test_mz_pair_in_the_same_group_leaves_sum_and_denominator():
    # 2 and 3 are MZ co-twins in depth 1.  Depth 1 keeps 6 - 1 = 5 pairs:
    # sibs 6-7 at 0.25, and the four cross-family pairs at 0.
    pg = _graph(twin=[-1, -1, 3, 2, -1, -1, -1, -1, -1, -1])
    _assert_summary(pg.mean_kinship_by_generation(), [0, 1, 2], [0.0, 0.25 / 5, 0.25], [6, 5, 1], 0)


def test_mz_pair_split_across_groups_counts_both_twins_as_ordinary_members():
    # Same twins, but the label puts 2 in cohort 1 and 3 in cohort 7 with 6, 7.
    labels = [0, 0, 1, 7, 0, 0, 7, 7, 2, 2]
    pg = _graph(generation=labels, twin=[-1, -1, 3, 2, -1, -1, -1, -1, -1, -1])
    # Cohort 7 = {3, 6, 7}: sibs 6-7 at 0.25 and two unrelated pairs.
    _assert_summary(pg.mean_kinship_by_generation(), [0, 1, 2, 7], [0.0, np.nan, 0.25, 0.25 / 3], [6, 0, 1, 3], 0)


def test_mz_twin_whose_co_twin_is_unlabelled_is_an_ordinary_member():
    labels = [0, 0, 1, -1, 0, 0, 1, 1, 2, 2]
    pg = _graph(generation=labels, twin=[-1, -1, 3, 2, -1, -1, -1, -1, -1, -1])
    # Cohort 1 = {2, 6, 7}: sibs 6-7 at 0.25 and two unrelated pairs.
    _assert_summary(pg.mean_kinship_by_generation(), [0, 1, 2], [0.0, 0.25 / 3, 0.25], [6, 3, 1], 1)


def test_single_member_group_has_no_pairs_and_nan_mean():
    pg = _graph(generation=[0, 0, 1, 1, 0, 0, 1, 1, 2, 3])
    _assert_summary(pg.mean_kinship_by_generation(), [0, 1, 2, 3], [0.0, 0.5 / 6, np.nan, np.nan], [6, 6, 0, 0], 0)


def test_result_is_frozen_and_its_arrays_read_only():
    summary = _graph().mean_kinship_by_generation()
    assert isinstance(summary, GenerationKinshipSummary)
    assert len(summary) == 3
    for name in ("generations", "mean_kinship", "pair_counts"):
        array = getattr(summary, name)
        assert not array.flags.writeable
        with pytest.raises(ValueError, match="read-only"):
            array[0] = 0
    with pytest.raises(AttributeError):
        summary.unlabelled_individual_count = 1  # type: ignore[misc]
    assert summary.generations.dtype == np.int32
    assert summary.mean_kinship.dtype == np.float64
    assert summary.pair_counts.dtype == np.int64


def test_summary_is_computed_once_per_graph():
    pg = _graph()
    first = pg.mean_kinship_by_generation()
    assert pg.mean_kinship_by_generation() is first


def test_cached_matrix_path_and_streamed_path_agree_on_partial_labels():
    labels = [0, 0, 1, -1, 0, 0, 1, 1, 2, 2]
    twin = [-1, -1, 3, 2, -1, -1, -1, -1, -1, -1]
    streamed = _graph(generation=labels, twin=twin).mean_kinship_by_generation()
    with_matrix = _graph(generation=labels, twin=twin)
    with_matrix.kinship_matrix(0.0)
    walked = with_matrix.mean_kinship_by_generation()
    _assert_summary(walked, streamed.generations, streamed.mean_kinship, streamed.pair_counts, 1)


def test_adapter_scatters_sparse_labels_with_nan_gaps():
    theta = _graph(generation=[5, 5, 9, 9, 5, 5, 9, 9, 30, 30]).per_gen_mean_kinship()
    assert theta.shape == (31,)
    observed = np.array([5, 9, 30])
    np.testing.assert_allclose(theta[observed], _DEPTH_MEANS, rtol=0, atol=1e-12)
    gaps = np.setdiff1d(np.arange(31), observed)
    assert np.isnan(theta[gaps]).all()


FIXTURES = parity_fixtures("random_1k", "deep_inbred_60g")


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_matrix_path_parity_on_safe_sizes(name):
    fixture = FIXTURES[name]
    depth = np.asarray(PedigreeGraph(parity_columns(fixture)).depth, dtype=np.int64)
    # Merge adjacent depths, blank one row in three, and rebase, so the
    # cohorts are sparse, partial, and span structural depths at once.
    labels = 10 * (depth // 2) + 100
    labels[np.arange(len(labels)) % 3 == 0] = -1
    columns = {**parity_columns(fixture), "generation": labels}

    graph = PedigreeGraph(columns)
    streamed = graph.mean_kinship_by_generation()
    with_matrix = PedigreeGraph(columns)
    with_matrix.kinship_matrix(0.0)
    walked = with_matrix.mean_kinship_by_generation()

    # A column that is -1 everywhere parses as no labels at all, so the tiny
    # motifs fall back to depth; the summary follows the graph's own labels.
    if graph.generation_labels is None:
        labels = depth
    np.testing.assert_array_equal(streamed.generations, np.unique(labels[labels >= 0]))
    np.testing.assert_array_equal(streamed.pair_counts, walked.pair_counts)
    np.testing.assert_allclose(streamed.mean_kinship, walked.mean_kinship, rtol=0, atol=1e-12, equal_nan=True)
    assert streamed.unlabelled_individual_count == int(np.count_nonzero(labels < 0))


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_summary_is_invariant_to_input_row_order(name):
    fixture = FIXTURES[name]
    columns = parity_columns(fixture)
    depth = np.asarray(PedigreeGraph(columns).depth, dtype=np.int64)
    labels = depth // 2
    labels[np.arange(len(labels)) % 5 == 0] = -1
    reference = PedigreeGraph({**columns, "generation": labels}).mean_kinship_by_generation()

    order = np.random.default_rng(6).permutation(len(labels))
    shuffled = PedigreeGraph(
        {key: np.asarray(value)[order] for key, value in {**columns, "generation": labels}.items()}
    )
    permuted = shuffled.mean_kinship_by_generation()

    np.testing.assert_array_equal(permuted.generations, reference.generations)
    np.testing.assert_array_equal(permuted.pair_counts, reference.pair_counts)
    # Row order may move floating bits inside the ADR 0009 envelope.
    np.testing.assert_allclose(permuted.mean_kinship, reference.mean_kinship, rtol=0, atol=1e-6, equal_nan=True)
    assert permuted.unlabelled_individual_count == reference.unlabelled_individual_count
