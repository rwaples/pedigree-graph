"""Tests for the ``sex`` kwarg on ``PedigreeGraph.from_arrays``.

Covers the round trip through the constructor and the quiet path of the
sex-aware Ne estimators: with both sexes present neither one warns.
"""

import warnings

import numpy as np

from pedigree_graph import PedigreeGraph
from pedigree_graph.effective_size import ne_sex_ratio, ne_variance_family_size


def test_sex_round_trips():
    pg = PedigreeGraph.from_arrays(
        ids=np.array([0, 1, 2]),
        mother_ids=np.array([-1, -1, 0]),
        father_ids=np.array([-1, -1, 1]),
        sex=np.array([0, 1, 0], dtype=np.int8),
    )
    assert pg.sex.dtype == np.int8
    np.testing.assert_array_equal(pg.sex, [0, 1, 0])


def test_sex_accepts_python_list():
    pg = PedigreeGraph.from_arrays(
        ids=np.array([0, 1]),
        mother_ids=np.array([-1, -1]),
        father_ids=np.array([-1, -1]),
        sex=[1, 0],
    )
    np.testing.assert_array_equal(pg.sex, [1, 0])


def test_sex_coexists_with_other_optional_args():
    pg = PedigreeGraph.from_arrays(
        ids=np.array([0, 1, 2]),
        mother_ids=np.array([-1, -1, 0]),
        father_ids=np.array([-1, -1, 1]),
        sex=np.array([0, 1, 0], dtype=np.int8),
        generation=np.array([0, 0, 1], dtype=np.int32),
        birth_year=np.array([1980, 1980, 2010], dtype=np.int32),
    )
    np.testing.assert_array_equal(pg.sex, [0, 1, 0])
    np.testing.assert_array_equal(pg.generation_labels, [0, 0, 1])


def test_no_warning_when_both_sexes_are_present():
    pg = PedigreeGraph.from_arrays(
        ids=np.array([0, 1, 2, 3]),
        mother_ids=np.array([-1, -1, 0, 0]),
        father_ids=np.array([-1, -1, 1, 1]),
        sex=np.array([0, 1, 0, 1], dtype=np.int8),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        ne_sex_ratio(pg)
        ne_variance_family_size(pg)
