"""Property-based tests for the closed-form effective-size estimators.

ne_sex_ratio has an exact per-cohort closed form (4*Nm*Nf/(Nm+Nf)) and is
invariant to swapping the sexes; ne_inbreeding's mean-F-per-cohort must match
inbreeding; and every estimator of a batch returns None or a positive finite
Ne.  Three further cross-cutting properties generalise the example suite over
random pedigrees: ne_coancestry agrees whether theta is streamed from the DP
or walked from a cached kinship matrix; one-parent rows disable the two
founder estimators and nothing else; and founder contributions are conserved
(sum to 1 per cohort) under complete parentage.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import (
    complete_parentage_pedigree,
    non_inbred_pedigree,
    pedigree_arrays,
    random_pedigree,
)
from hypothesis import given, settings

from pedigree_graph import MissingMetadataError, PedigreeGraph
from pedigree_graph._ne_founders import _per_gen_founder_means
from pedigree_graph.effective_size import (
    UnavailableEffectiveSize,
    estimate_effective_sizes,
    ne_coancestry,
    ne_inbreeding,
    ne_long_term_contributions,
    ne_sex_ratio,
)

_SETTINGS = settings(deadline=None, max_examples=50)
_HEAVY = settings(deadline=None, max_examples=30)

# A randomly-generated pedigree can have uniform sex in/across a cohort; the
# sex-aware estimators then legitimately return ne=None after a RuntimeWarning.
_UNIFORM_SEX_OK = pytest.mark.filterwarnings("ignore:.*is uniform.*:RuntimeWarning")

_FOUNDER_BASED = ("ne_long_term_contributions", "ne_caballero_toro")


@_UNIFORM_SEX_OK
@_SETTINGS
@given(pg=random_pedigree())
def test_ne_sex_ratio_closed_form(pg):
    res = ne_sex_ratio(pg)
    depth, sex = np.asarray(pg.depth), pg.sex
    for bucket, g in enumerate(res.generations):
        nm = int(((sex == 1) & (depth == g)).sum())
        nf = int(((sex == 0) & (depth == g)).sum())
        if nm > 0 and nf > 0:
            assert res.ne_per_gen[bucket] == pytest.approx(4.0 * nm * nf / (nm + nf))
            assert 0.0 <= res.ne_per_gen[bucket] <= nm + nf + 1e-9
        else:
            assert np.isnan(res.ne_per_gen[bucket])


@_UNIFORM_SEX_OK
@_SETTINGS
@given(arrays=pedigree_arrays())
def test_ne_sex_ratio_sex_swap_invariant(arrays):
    ids, mo, fa, sex = arrays
    a = ne_sex_ratio(PedigreeGraph.from_arrays(ids=ids, mother_ids=mo, father_ids=fa, sex=sex)).ne
    b = ne_sex_ratio(PedigreeGraph.from_arrays(ids=ids, mother_ids=mo, father_ids=fa, sex=1 - sex)).ne
    if a is None:
        assert b is None
    else:
        assert b == pytest.approx(a)


@_SETTINGS
@given(pg=random_pedigree())
def test_ne_inbreeding_mean_f_consistency(pg):
    res = ne_inbreeding(pg)
    F, depth = pg.inbreeding(), np.asarray(pg.depth)
    for bucket, g in enumerate(res.generations):
        assert res.mean_f_per_gen[bucket] == pytest.approx(float(F[depth == g].mean()))
    assert res.mean_f_per_gen[0] == pytest.approx(0.0)  # founders carry no inbreeding
    finite = np.isfinite(res.ne_per_gen)
    assert np.all(res.ne_per_gen[finite] > 0.0)


@_SETTINGS
@given(pg=non_inbred_pedigree())
def test_ne_inbreeding_none_when_non_inbred(pg):
    assert ne_inbreeding(pg).ne is None


@_UNIFORM_SEX_OK
@_SETTINGS
@given(pg=complete_parentage_pedigree())
def test_every_estimator_reports_none_or_a_positive_ne(pg):
    for name, result in estimate_effective_sizes(pg).items():
        assert not isinstance(result, UnavailableEffectiveSize), name
        ne = result.ne
        assert ne is None or (np.isfinite(ne) and ne > 0.0), name


@_UNIFORM_SEX_OK
@_SETTINGS
@given(pg=random_pedigree())
def test_one_parent_rows_disable_only_the_founder_estimators(pg):
    # The founder-based estimators need closed represented parentage; the
    # batch reports missing metadata for them and keeps the other six.
    one_parent = (np.asarray(pg.mother_rows) < 0) != (np.asarray(pg.father_rows) < 0)
    results = estimate_effective_sizes(pg)
    assert len(results) == 8
    for name, result in results.items():
        refused = isinstance(result, UnavailableEffectiveSize)
        assert refused == (bool(one_parent.any()) and name in _FOUNDER_BASED), name
        if refused:
            assert result.code == "incomplete_parentage", name
            assert result.fields["affected_count"] == int(one_parent.sum()), name
    if one_parent.any():
        with pytest.raises(MissingMetadataError) as info:
            ne_long_term_contributions(pg)
        assert info.value.fields["affected_count"] == int(one_parent.sum())


def _ne_close(a, b):
    if a is None or b is None:
        return a is None and b is None
    return a == pytest.approx(b, rel=1e-9, abs=1e-12)


@_HEAVY
@given(arrays=pedigree_arrays())
def test_ne_coancestry_streamed_and_matrix_theta_agree(arrays):
    # The default streams theta from the retiring DP; a graph that already
    # caches the complete kinship matrix walks that instead.  Both routes must
    # produce the same coancestry-rate Ne and the same per-cohort theta.
    ids, mother, father, sex = arrays

    def build():
        return PedigreeGraph.from_arrays(ids=ids, mother_ids=mother, father_ids=father, sex=sex)

    streamed = ne_coancestry(build())
    with_matrix = build()
    with_matrix.kinship_matrix()
    walked = ne_coancestry(with_matrix)
    assert _ne_close(streamed.ne, walked.ne)
    assert np.allclose(streamed.mean_theta_per_gen, walked.mean_theta_per_gen, equal_nan=True)


@_HEAVY
@given(pg=complete_parentage_pedigree())
def test_founder_contributions_conserved(pg):
    # With complete parentage (no half-missing parent), every individual's
    # genome is fully partitioned among the founders, so the per-cohort mean
    # founder contributions sum to 1 in each observed cohort.
    m_g, founder_idx = _per_gen_founder_means(pg)
    for bucket in m_g:
        if np.isfinite(bucket).all():
            assert bucket.sum() == pytest.approx(1.0, abs=1e-9)
    # Corollary: Σ_f c_f² ∈ [1/n_founders, 1] ⇒ Ne_LTC ∈ [0.5, n_founders/2].
    res = ne_long_term_contributions(pg)
    if res.asymptote_reached and res.ne is not None:
        assert 0.5 - 1e-9 <= res.ne <= len(founder_idx) / 2.0 + 1e-9
