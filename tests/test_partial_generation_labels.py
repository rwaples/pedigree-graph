"""Partly known generation labels are a structured error, never a wrapped bucket.

A ``-1`` label would index the last cohort of every label-indexed
accumulator (the kinship DP theta sums, the Caballero-Toro founder sweep, the
per-cohort inbreeding means) and silently bias the estimate.  Absent labels
are fine: the 0.7.1 estimators fall back to structural depth.
"""

import numpy as np
import pytest

from pedigree_graph import (
    MissingMetadataError,
    PedigreeGraph,
    compute_all_ne,
    ne_caballero_toro,
    ne_coancestry,
    ne_inbreeding,
    ne_individual_delta_f,
    ne_long_term_contributions,
    ne_sex_ratio,
    ne_variance_family_size,
)

_IDS = np.arange(8)
_MOTHER = np.array([-1, -1, 0, 0, 2, 2, 4, 4])
_FATHER = np.array([-1, -1, 1, 1, 3, 3, 5, 5])
_SEX = np.array([0, 1, 0, 1, 0, 1, 0, 1])


def _graph(generation):
    return PedigreeGraph.from_frame(
        {"id": _IDS, "mother": _MOTHER, "father": _FATHER, "sex": _SEX, "generation": generation}
    )


ESTIMATORS = [
    ne_inbreeding,
    ne_coancestry,
    ne_individual_delta_f,
    ne_variance_family_size,
    ne_sex_ratio,
    ne_long_term_contributions,
    ne_caballero_toro,
]


@pytest.mark.parametrize("estimator", ESTIMATORS, ids=lambda f: f.__name__)
def test_estimators_reject_partial_labels(estimator):
    pg = _graph([0, 0, 1, 1, 2, 2, -1, -1])
    with pytest.raises(MissingMetadataError) as info:
        estimator(pg)
    assert info.value.code == "missing_generation_labels"
    assert info.value.fields["status"] == "partial"
    assert info.value.fields["missing_count"] == 2
    assert isinstance(info.value.fields["operation"], str)


def test_per_gen_mean_kinship_rejects_partial_labels():
    pg = _graph([0, 0, 1, 1, 2, 2, -1, -1])
    with pytest.raises(MissingMetadataError) as info:
        pg.per_gen_mean_kinship()
    assert info.value.code == "missing_generation_labels"
    assert info.value.fields["operation"] == "per_gen_mean_kinship"


def test_per_gen_mean_kinship_rejects_partial_labels_on_the_cached_matrix_path():
    pg = _graph([0, 0, 1, 1, 2, 2, -1, -1])
    pg.kinship_matrix(0.0)
    with pytest.raises(MissingMetadataError):
        pg.per_gen_mean_kinship()


def test_compute_all_ne_rejects_partial_labels():
    with pytest.raises(MissingMetadataError):
        compute_all_ne(_graph([0, 0, 1, 1, 2, 2, -1, -1]))


def test_complete_labels_and_absent_labels_both_run():
    labelled = _graph([0, 0, 1, 1, 2, 2, 3, 3])
    unlabelled = PedigreeGraph.from_arrays(_IDS, _MOTHER, _FATHER, sex=_SEX)
    assert unlabelled.generation_labels is None
    np.testing.assert_array_equal(labelled.per_gen_mean_kinship(), unlabelled.per_gen_mean_kinship())
    assert ne_caballero_toro(labelled).ne == ne_caballero_toro(unlabelled).ne
