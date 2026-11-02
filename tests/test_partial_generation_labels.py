"""Partly known generation labels are a structured error, never a wrapped bucket.

A ``-1`` label would index the last cohort of every label-indexed
accumulator (the kinship DP theta sums, the group-coancestry cohort sums,
the per-cohort inbreeding means) and silently bias the estimate.  Absent
labels are fine: the estimators fall back to structural depth.
"""

import numpy as np

from pedigree_graph import PedigreeGraph

_IDS = np.arange(8)
_MOTHER = np.array([-1, -1, 0, 0, 2, 2, 4, 4])
_FATHER = np.array([-1, -1, 1, 1, 3, 3, 5, 5])
_SEX = np.array([0, 1, 0, 1, 0, 1, 0, 1])

_PARTIAL = [0, 0, 1, 1, 2, 2, -1, -1]


def _graph(generation):
    return PedigreeGraph.from_frame(
        {"id": _IDS, "mother": _MOTHER, "father": _FATHER, "sex": _SEX, "generation": generation}
    )


def test_generation_kinship_summary_groups_only_the_labelled_rows():
    summary = _graph(_PARTIAL).mean_kinship_by_generation()
    np.testing.assert_array_equal(summary.generations, [0, 1, 2])
    assert summary.unlabelled_individual_count == 2


def test_generation_kinship_summary_agrees_on_the_cached_matrix_path():
    streamed = _graph(_PARTIAL).mean_kinship_by_generation()
    cached = _graph(_PARTIAL)
    cached.kinship_matrix()
    assert cached.mean_kinship_by_generation() == streamed


def test_complete_labels_and_absent_labels_both_run():
    labelled = _graph([0, 0, 1, 1, 2, 2, 3, 3])
    unlabelled = PedigreeGraph.from_arrays(ids=_IDS, mother_ids=_MOTHER, father_ids=_FATHER, sex=_SEX)
    assert unlabelled.generation_labels is None
    assert labelled.mean_kinship_by_generation() == unlabelled.mean_kinship_by_generation()
