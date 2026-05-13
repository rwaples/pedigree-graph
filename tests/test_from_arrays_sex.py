"""Tests for the ``sex`` kwarg on ``PedigreeGraph.from_arrays``.

Includes a regression for the silent-degeneracy foot-gun: when sex
defaults to all-zero, sex-aware Ne estimators must emit a
``RuntimeWarning`` so callers learn they're consuming bad results.
"""

import warnings

import numpy as np
import pytest

from pedigree_graph import (
    PedigreeGraph,
    ne_sex_ratio,
    ne_variance_family_size,
)


def test_sex_round_trips():
    pg = PedigreeGraph.from_arrays(
        ids=np.array([0, 1, 2]),
        mothers=np.array([-1, -1, 0]),
        fathers=np.array([-1, -1, 1]),
        sex=np.array([0, 1, 0], dtype=np.int8),
    )
    assert pg.sex.dtype == np.int8
    np.testing.assert_array_equal(pg.sex, [0, 1, 0])


def test_sex_default_is_zeros_backcompat():
    pg = PedigreeGraph.from_arrays(
        ids=np.array([0, 1, 2]),
        mothers=np.array([-1, -1, 0]),
        fathers=np.array([-1, -1, 1]),
    )
    assert pg.sex.dtype == np.int8
    np.testing.assert_array_equal(pg.sex, np.zeros(3, dtype=np.int8))


def test_sex_accepts_python_list():
    pg = PedigreeGraph.from_arrays(
        ids=np.array([0, 1]),
        mothers=np.array([-1, -1]),
        fathers=np.array([-1, -1]),
        sex=[1, 0],
    )
    np.testing.assert_array_equal(pg.sex, [1, 0])


def test_sex_coexists_with_other_optional_args():
    pg = PedigreeGraph.from_arrays(
        ids=np.array([0, 1, 2]),
        mothers=np.array([-1, -1, 0]),
        fathers=np.array([-1, -1, 1]),
        sex=np.array([0, 1, 0], dtype=np.int8),
        generation=np.array([0, 0, 1], dtype=np.int32),
        birth_year=np.array([1980, 1980, 2010], dtype=np.int32),
    )
    np.testing.assert_array_equal(pg.sex, [0, 1, 0])
    np.testing.assert_array_equal(pg.generation, [0, 0, 1])


# -- Foot-gun guard: silent degeneracy when sex defaults to zeros -------------


def _two_gen_pg(sex=None):
    """A trivial 4-row pedigree: two founders + two children."""
    return PedigreeGraph.from_arrays(
        ids=np.array([0, 1, 2, 3]),
        mothers=np.array([-1, -1, 0, 0]),
        fathers=np.array([-1, -1, 1, 1]),
        sex=sex,
    )


def test_ne_sex_ratio_warns_when_sex_defaulted():
    pg = _two_gen_pg(sex=None)  # all-zero default
    with pytest.warns(RuntimeWarning, match="pg.sex is uniform"):
        result = ne_sex_ratio(pg)
    assert result.ne is None


def test_ne_variance_family_size_warns_when_sex_defaulted():
    pg = _two_gen_pg(sex=None)
    with pytest.warns(RuntimeWarning, match="pg.sex is uniform"):
        result = ne_variance_family_size(pg)
    assert result.ne is None


def test_no_warning_when_sex_is_supplied():
    pg = _two_gen_pg(sex=np.array([0, 1, 0, 1], dtype=np.int8))
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        # Neither estimator should warn — both sexes are present.
        ne_sex_ratio(pg)
        ne_variance_family_size(pg)
