"""Cohort densification: one bucket per observed label, a sentinel for ``-1``.

Generation labels arrive sparse, rebased, or partly unknown, and every
cohort-indexed estimator groups by :class:`ObservedCohorts`.  Pinning the
bucket mapping, the membership lists, and the fallback to structural depth
here keeps each estimator's own tests off the grouping contract.
"""

import numpy as np
import pytest

from pedigree_graph import MissingMetadataError, PedigreeGraph
from pedigree_graph._cohorts import ObservedCohorts

FIELD_DTYPES = {"generations": np.int32, "dense": np.int32, "counts": np.int64}

_SPARSE = np.array([0, 0, 2, 2, 5])
_REBASED = np.array([10, 10, 12, 12, 15])
_EMPTY = np.empty(0, dtype=np.int32)
_UNORDERED = np.array([5, 0, 2, 0, 5, -1])

_IDS = np.arange(8)
_MOTHER = np.array([-1, -1, 0, 0, 2, 2, 4, 4])
_FATHER = np.array([-1, -1, 1, 1, 3, 3, 5, 5])
_SEX = np.array([0, 1, 0, 1, 0, 1, 0, 1])


def _graph(generation=None):
    return PedigreeGraph.from_arrays(ids=_IDS, mothers=_MOTHER, fathers=_FATHER, sex=_SEX, generation=generation)


def _two_rows_per_cohort(k):
    return np.repeat(np.arange(k, dtype=np.int32), 2)


@pytest.mark.parametrize("labels", [_SPARSE, _EMPTY], ids=["sparse", "empty"])
@pytest.mark.parametrize(("field", "dtype"), list(FIELD_DTYPES.items()), ids=list(FIELD_DTYPES))
def test_field_carries_its_declared_dtype(labels, field, dtype):
    assert getattr(ObservedCohorts.from_labels(labels), field).dtype == dtype


def test_sparse_labels_densify_to_consecutive_buckets():
    oc = ObservedCohorts.from_labels(_SPARSE)
    np.testing.assert_array_equal(oc.generations, [0, 2, 5])
    np.testing.assert_array_equal(oc.dense, [0, 0, 1, 1, 2])
    np.testing.assert_array_equal(oc.counts, [2, 2, 1])
    assert oc.k == len(oc) == 3
    assert oc.unlabelled_individual_count == 0


def test_rebasing_labels_changes_only_the_labels():
    sparse = ObservedCohorts.from_labels(_SPARSE)
    rebased = ObservedCohorts.from_labels(_REBASED)
    np.testing.assert_array_equal(rebased.generations, [10, 12, 15])
    np.testing.assert_array_equal(rebased.dense, sparse.dense)
    np.testing.assert_array_equal(rebased.counts, sparse.counts)


def test_unlabelled_rows_take_the_sentinel_bucket_and_join_no_cohort():
    labels = np.array([0, -1, 2, -1, 5])
    oc = ObservedCohorts.from_labels(labels)
    unlabelled = np.flatnonzero(labels < 0)
    np.testing.assert_array_equal(oc.dense[unlabelled], oc.k)
    assert oc.unlabelled_individual_count == unlabelled.size
    assert int(oc.counts.sum()) == labels.size - unlabelled.size
    assert not np.isin(unlabelled, np.concatenate(oc.members())).any()


def test_wholly_unlabelled_input_observes_no_cohort():
    oc = ObservedCohorts.from_labels(np.full(4, -1))
    assert oc.k == 0
    assert oc.generations.size == oc.counts.size == 0
    assert oc.members() == []
    assert oc.unlabelled_individual_count == 4


def test_empty_input_observes_no_cohort():
    oc = ObservedCohorts.from_labels(_EMPTY)
    assert oc.k == 0
    assert oc.generations.size == oc.counts.size == oc.dense.size == 0
    assert oc.members() == []


def test_members_partition_labelled_rows_in_ascending_order():
    oc = ObservedCohorts.from_labels(_UNORDERED)
    members = oc.members()
    assert [len(rows) for rows in members] == oc.counts.tolist()
    for bucket, rows in enumerate(members):
        np.testing.assert_array_equal(rows, np.sort(rows))
        np.testing.assert_array_equal(_UNORDERED[rows], oc.generations[bucket])


def test_for_graph_falls_back_to_structural_depth_without_labels():
    pg = _graph()
    assert pg.generation_labels is None
    expected = ObservedCohorts.from_labels(pg.depth)
    oc = ObservedCohorts.for_graph(pg, "cohort_probe")
    for field in FIELD_DTYPES:
        np.testing.assert_array_equal(getattr(oc, field), getattr(expected, field), err_msg=field)
    assert oc.unlabelled_individual_count == expected.unlabelled_individual_count


def test_for_graph_groups_by_supplied_labels_not_by_depth():
    pg = _graph([0, 0, 2, 2, 5, 5, 9, 9])
    oc = ObservedCohorts.for_graph(pg, "cohort_probe")
    np.testing.assert_array_equal(oc.generations, [0, 2, 5, 9])
    np.testing.assert_array_equal(np.unique(pg.depth), [0, 1, 2, 3])


def test_for_graph_rejects_partly_unknown_labels():
    pg = _graph([0, 0, 1, 1, 2, 2, -1, -1])
    with pytest.raises(MissingMetadataError) as info:
        ObservedCohorts.for_graph(pg, "cohort_probe")
    assert info.value.code == "missing_generation_labels"
    assert info.value.fields["operation"] == "cohort_probe"
    assert info.value.fields["status"] == "partial"
    assert info.value.fields["missing_count"] == 2


@pytest.mark.parametrize(("k", "expected_from", "expected_to"), [(0, [], []), (1, [], []), (2, [0], [1])])
def test_transitions_pair_adjacent_observed_cohorts(k, expected_from, expected_to):
    oc = ObservedCohorts.from_labels(_two_rows_per_cohort(k))
    assert oc.k == k
    np.testing.assert_array_equal(oc.transition_from(), expected_from)
    np.testing.assert_array_equal(oc.transition_to(), expected_to)
    np.testing.assert_array_equal(oc.transition_from(), oc.generations[:-1])
    np.testing.assert_array_equal(oc.transition_to(), oc.generations[1:])
    assert oc.transition_from().size == oc.transition_to().size == max(k - 1, 0)
