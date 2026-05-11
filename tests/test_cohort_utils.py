"""Tests for pedigree_graph._cohort_utils."""

import numpy as np
import pytest

from pedigree_graph import CohortWindow, PedigreeGraph, eligible_cohort_range


def _three_gen_pedigree(birth_year: np.ndarray) -> PedigreeGraph:
    # Two founders + one child:
    #   id 0: mother of 2
    #   id 1: father of 2
    #   id 2: child
    return PedigreeGraph.from_arrays(
        ids=np.array([0, 1, 2]),
        mothers=np.array([-1, -1, 0]),
        fathers=np.array([-1, -1, 1]),
        birth_year=birth_year,
    )


class TestCohortWindowDataclass:
    def test_namedtuple_fields(self):
        w = CohortWindow(c_min=1990, c_max=2005, reproductive_age_p95=10.0)
        assert w.c_min == 1990
        assert w.c_max == 2005
        assert w.reproductive_age_p95 == 10.0


class TestEligibleCohortRange:
    def test_raises_when_birth_year_missing(self):
        pg = PedigreeGraph.from_arrays(
            ids=np.array([0, 1, 2]),
            mothers=np.array([-1, -1, 0]),
            fathers=np.array([-1, -1, 1]),
        )
        with pytest.raises(ValueError, match=r"requires pg\.birth_year"):
            eligible_cohort_range(pg)

    def test_raises_when_no_known_birth_years(self):
        pg = _three_gen_pedigree(np.array([-1, -1, -1]))
        with pytest.raises(ValueError, match="no individuals have known birth_year"):
            eligible_cohort_range(pg)

    def test_raises_when_no_edges_with_both_birth_years(self):
        # All children have at least one parent with unknown birth_year.
        pg = _three_gen_pedigree(np.array([-1, -1, 2010]))
        with pytest.raises(ValueError, match="no parent-child edges"):
            eligible_cohort_range(pg)

    def test_basic_window(self):
        # Two parents born 1990 and 1992; child born 2010.
        # Mother-edge Δ=20; father-edge Δ=18.
        # p95 of {20, 18} = 19.9.
        # y_min=1990, y_max=2010; default c_max = floor(2010 - 19.9) = 1990.
        pg = _three_gen_pedigree(np.array([1990, 1992, 2010]))
        w = eligible_cohort_range(pg)
        assert w.c_min == 1990
        assert w.reproductive_age_p95 == pytest.approx(19.9)
        assert w.c_max == 1990  # floor(2010 - 19.9)

    def test_user_override_c_min(self):
        pg = _three_gen_pedigree(np.array([1990, 1992, 2010]))
        w = eligible_cohort_range(pg, c_min=1995)
        assert w.c_min == 1995  # overridden
        assert w.c_max == 1990  # heuristic kept

    def test_user_override_c_max(self):
        pg = _three_gen_pedigree(np.array([1990, 1992, 2010]))
        w = eligible_cohort_range(pg, c_max=2000)
        assert w.c_min == 1990  # heuristic kept
        assert w.c_max == 2000  # overridden

    def test_user_override_both(self):
        pg = _three_gen_pedigree(np.array([1990, 1992, 2010]))
        w = eligible_cohort_range(pg, c_min=1995, c_max=2005)
        assert w.c_min == 1995
        assert w.c_max == 2005

    def test_percentile_parameter_changes_cutoff(self):
        # Build a longer pedigree where the percentile actually matters.
        # Five children all of the same two parents, spread across years.
        pg = PedigreeGraph.from_arrays(
            ids=np.arange(7),
            mothers=np.array([-1, -1, 0, 0, 0, 0, 0]),
            fathers=np.array([-1, -1, 1, 1, 1, 1, 1]),
            birth_year=np.array([1990, 1990, 2000, 2005, 2010, 2015, 2020]),
        )
        w50 = eligible_cohort_range(pg, percentile=50.0)
        w95 = eligible_cohort_range(pg, percentile=95.0)
        # p95 reaches further into the tail → bigger cutoff → smaller c_max.
        assert w95.reproductive_age_p95 > w50.reproductive_age_p95
        assert w95.c_max <= w50.c_max

    def test_unknown_endpoints_excluded_from_percentile(self):
        # Mother-edge has unknown mother birth_year; only father edge counts.
        pg = PedigreeGraph.from_arrays(
            ids=np.array([0, 1, 2]),
            mothers=np.array([-1, -1, 0]),
            fathers=np.array([-1, -1, 1]),
            birth_year=np.array([-1, 1992, 2010]),
        )
        w = eligible_cohort_range(pg)
        # Only the father edge (Δ=18) contributes → p95 of {18} = 18.
        assert w.reproductive_age_p95 == pytest.approx(18.0)
