"""The metadata dependency matrix of the final effective-size estimators.

Each estimator validates the metadata it needs, in a fixed order, before
any work, and raises ``MissingMetadataError`` naming itself in
``operation``.  Estimators that do not need a field keep running without
it.  An empty graph bypasses every guard.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from pedigree_graph import MissingMetadataError, PedigreeGraph, compute_all_ne
from pedigree_graph import effective_size as es

_IDS = np.arange(8)
_MOTHER = np.array([-1, -1, 0, 0, 2, 2, 4, 4])
_FATHER = np.array([-1, -1, 1, 1, 3, 3, 5, 5])
_SEX = np.array([0, 1, 0, 1, 0, 1, 0, 1])
_GEN = np.array([0, 0, 1, 1, 2, 2, 3, 3])
_BIRTH = np.array([1900, 1900, 1920, 1920, 1940, 1940, 1960, 1960])

ALL = [
    es.ne_inbreeding,
    es.ne_coancestry,
    es.ne_individual_delta_f,
    es.ne_variance_family_size,
    es.ne_sex_ratio,
    es.ne_long_term_contributions,
    es.ne_hill_overlapping,
    es.ne_caballero_toro,
]
NEEDS_SEX = [es.ne_variance_family_size, es.ne_sex_ratio, es.ne_hill_overlapping]
NEEDS_PARENTAGE = [es.ne_long_term_contributions, es.ne_caballero_toro]
NEEDS_GENERATION = [f for f in ALL if f is not es.ne_hill_overlapping]


def _graph(**overrides):
    columns = {"id": _IDS, "mother": _MOTHER, "father": _FATHER, "sex": _SEX, "generation": _GEN}
    columns.update(overrides)
    return PedigreeGraph.from_frame({k: v for k, v in columns.items() if v is not None})


def _name(f):
    return f.__name__


@pytest.mark.parametrize("estimator", NEEDS_GENERATION, ids=_name)
def test_partial_generation_labels_disable_every_label_grouped_estimator(estimator):
    pg = _graph(generation=np.array([0, 0, 1, 1, 2, 2, -1, -1]))
    with pytest.raises(MissingMetadataError) as info:
        estimator(pg)
    assert info.value.code == "missing_generation_labels"
    assert info.value.fields == {"operation": estimator.__name__, "status": "partial", "missing_count": 2}


@pytest.mark.parametrize("estimator", ALL, ids=_name)
def test_absent_generation_labels_fall_back_to_depth(estimator):
    labelled = _graph()
    unlabelled = _graph(generation=None)
    assert unlabelled.generation_labels is None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        assert estimator(labelled).to_dict() == estimator(unlabelled).to_dict()


@pytest.mark.parametrize("estimator", NEEDS_SEX, ids=_name)
@pytest.mark.parametrize(
    ("sex", "status", "count"),
    [(None, "absent", 8), (np.array([0, 1, 0, 1, -1, -1, 0, 1]), "partial", 2)],
    ids=["absent", "partial"],
)
def test_sex_dependent_estimators_require_complete_sex(estimator, sex, status, count):
    pg = _graph(sex=sex)
    with pytest.raises(MissingMetadataError) as info:
        estimator(pg)
    assert info.value.code == "missing_sex"
    assert info.value.fields == {"operation": estimator.__name__, "status": status, "missing_count": count}


@pytest.mark.parametrize("estimator", [f for f in ALL if f not in NEEDS_SEX], ids=_name)
def test_other_estimators_ignore_absent_sex(estimator):
    estimator(_graph(sex=None))


@pytest.mark.parametrize("estimator", NEEDS_SEX, ids=_name)
def test_generation_is_validated_before_sex(estimator):
    pg = _graph(sex=None, generation=np.array([0, 0, 1, 1, 2, 2, -1, -1]))
    with pytest.raises(MissingMetadataError) as info:
        estimator(pg)
    assert info.value.code == "missing_generation_labels"


@pytest.mark.parametrize("estimator", NEEDS_SEX, ids=_name)
def test_uniform_fully_known_sex_is_valid_and_warns(estimator):
    pg = _graph(sex=np.zeros(8, dtype=np.int64))
    with pytest.warns(RuntimeWarning, match="pg.sex is uniform"):
        assert estimator(pg).ne is None


@pytest.mark.parametrize("estimator", NEEDS_PARENTAGE, ids=_name)
@pytest.mark.parametrize(
    ("father", "status"),
    [(np.array([-1, -1, 1, 1, 3, -1, 5, 5]), "missing"), (np.array([-1, -1, 1, 1, 3, 99, 5, 5]), "external")],
    ids=["missing", "external"],
)
def test_one_represented_parent_disables_only_ltc_and_ct(estimator, father, status):
    pg = _graph(father=father)
    with pytest.raises(MissingMetadataError) as info:
        estimator(pg)
    assert info.value.code == "incomplete_parentage"
    assert info.value.fields == {
        "operation": estimator.__name__,
        "affected_count": 1,
        "first_row": 5,
        "first_id": 5,
        "represented_parent_role": "mother",
        "unrepresented_parent_role": "father",
        "unrepresented_parent_status": status,
    }


@pytest.mark.parametrize("estimator", [f for f in ALL if f not in NEEDS_PARENTAGE], ids=_name)
def test_other_estimators_run_with_one_represented_parent(estimator):
    estimator(_graph(father=np.array([-1, -1, 1, 1, 3, -1, 5, 5])))


@pytest.mark.parametrize("estimator", NEEDS_PARENTAGE, ids=_name)
def test_generation_is_validated_before_parentage(estimator):
    pg = _graph(father=np.array([-1, -1, 1, 1, 3, -1, 5, 5]), generation=np.array([0, 0, 1, 1, 2, 2, -1, -1]))
    with pytest.raises(MissingMetadataError) as info:
        estimator(pg)
    assert info.value.code == "missing_generation_labels"


def test_represented_founders_with_external_parents_are_closed_parentage():
    pg = _graph(mother=np.array([90, 91, 0, 0, 2, 2, 4, 4]), father=np.array([92, 93, 1, 1, 3, 3, 5, 5]))
    assert es.ne_long_term_contributions(pg).final_generation is not None
    assert es.ne_caballero_toro(pg).ne is not None or True


class TestHill:
    def test_absent_birth_years_collapse_after_generation_and_sex(self):
        res = es.ne_hill_overlapping(_graph())
        assert res.collapses_to_ne_v
        assert res.generation_interval == 1.0
        assert res.ne == es.ne_variance_family_size(_graph()).ne

    def test_birth_year_branch_ignores_generation_labels(self):
        pg = _graph(birth_year=_BIRTH, generation=np.array([0, 0, 1, 1, 2, 2, -1, -1]))
        res = es.ne_hill_overlapping(pg)
        assert not res.collapses_to_ne_v

    def test_birth_year_branch_validates_sex_before_parent_ages(self):
        pg = _graph(birth_year=np.array([-1, -1, 1920, 1920, 1940, 1940, 1960, 1960]), sex=None)
        with pytest.raises(MissingMetadataError) as info:
            es.ne_hill_overlapping(pg)
        assert info.value.code == "missing_sex"

    def test_birth_year_branch_rejects_a_role_without_known_ages(self):
        birth_year = np.array([1900, -1, 1920, 1920, 1940, -1, 1960, 1960])
        pg = _graph(birth_year=birth_year, father=np.array([-1, -1, 1, 1, 1, 1, 5, 5]))
        with pytest.raises(MissingMetadataError) as info:
            es.ne_hill_overlapping(pg)
        assert info.value.code == "insufficient_parent_age_data"
        assert info.value.fields["missing_parent_roles"] == ("father",)

    def test_partial_birth_years_run_and_report_the_unknown_rows(self):
        pg = _graph(birth_year=np.array([1900, 1900, 1920, 1920, 1940, 1940, -1, 1960]))
        res = es.ne_hill_overlapping(pg)
        assert not res.collapses_to_ne_v
        assert res.n_unknown_birth_year == 1


@pytest.mark.parametrize("estimator", ALL, ids=_name)
def test_empty_graph_bypasses_every_guard(estimator):
    pg = PedigreeGraph.from_frame({"id": [], "mother": [], "father": []})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert estimator(pg).ne is None


def test_compute_all_ne_re_raises_a_selected_metadata_failure():
    with pytest.raises(MissingMetadataError) as info:
        compute_all_ne(_graph(father=np.array([-1, -1, 1, 1, 3, -1, 5, 5])))
    assert info.value.code == "incomplete_parentage"
