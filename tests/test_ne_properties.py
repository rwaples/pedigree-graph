"""Property-based tests for the closed-form effective-size estimators.

ne_sex_ratio has an exact per-generation closed form (4*Nm*Nf/(Nm+Nf)) and is
invariant to swapping the sexes; ne_inbreeding's mean-F-per-generation must match
compute_inbreeding; and every compute_all_ne estimator returns None or a positive
finite Ne.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import non_inbred_pedigree, pedigree_arrays, random_pedigree
from hypothesis import given, settings

from pedigree_graph import PedigreeGraph, compute_all_ne, ne_inbreeding, ne_sex_ratio

_SETTINGS = settings(deadline=None, max_examples=50)

# A randomly-generated pedigree can have uniform sex in/across a cohort; the
# sex-aware estimators then legitimately return ne=None after a RuntimeWarning.
_UNIFORM_SEX_OK = pytest.mark.filterwarnings("ignore:.*is uniform.*:RuntimeWarning")


@_UNIFORM_SEX_OK
@_SETTINGS
@given(pg=random_pedigree())
def test_ne_sex_ratio_closed_form(pg):
    res = ne_sex_ratio(pg)
    gen, sex = pg.generation, pg.sex
    for g in range(int(gen.max()) + 1):
        nm = int(((sex == 1) & (gen == g)).sum())
        nf = int(((sex == 0) & (gen == g)).sum())
        if nm > 0 and nf > 0:
            assert res.ne_per_gen[g] == pytest.approx(4.0 * nm * nf / (nm + nf))
            assert 0.0 <= res.ne_per_gen[g] <= nm + nf + 1e-9
        else:
            assert np.isnan(res.ne_per_gen[g])


@_UNIFORM_SEX_OK
@_SETTINGS
@given(arrays=pedigree_arrays())
def test_ne_sex_ratio_sex_swap_invariant(arrays):
    ids, mo, fa, sex = arrays
    a = ne_sex_ratio(PedigreeGraph.from_arrays(ids=ids, mothers=mo, fathers=fa, sex=sex)).ne
    b = ne_sex_ratio(PedigreeGraph.from_arrays(ids=ids, mothers=mo, fathers=fa, sex=1 - sex)).ne
    if a is None:
        assert b is None
    else:
        assert b == pytest.approx(a)


@_SETTINGS
@given(pg=random_pedigree())
def test_ne_inbreeding_mean_f_consistency(pg):
    res = ne_inbreeding(pg)
    F, gen = pg.compute_inbreeding(), pg.generation
    for g in range(int(gen.max()) + 1):
        mask = gen == g
        if mask.any():
            assert res.mean_f_per_gen[g] == pytest.approx(float(F[mask].mean()))
    assert res.mean_f_per_gen[0] == pytest.approx(0.0)  # founders carry no inbreeding
    finite = np.isfinite(res.ne_per_gen)
    assert np.all(res.ne_per_gen[finite] > 0.0)


@_SETTINGS
@given(pg=non_inbred_pedigree())
def test_ne_inbreeding_none_when_non_inbred(pg):
    assert ne_inbreeding(pg).ne is None


@_UNIFORM_SEX_OK
@_SETTINGS
@given(pg=random_pedigree())
def test_compute_all_ne_none_or_positive(pg):
    for name, result in compute_all_ne(pg).items():
        ne = result.ne
        assert ne is None or (np.isfinite(ne) and ne > 0.0), name
