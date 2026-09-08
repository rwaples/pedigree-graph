"""Property-based tests for the closed-form effective-size estimators.

ne_sex_ratio has an exact per-generation closed form (4*Nm*Nf/(Nm+Nf)) and is
invariant to swapping the sexes; ne_inbreeding's mean-F-per-generation must match
inbreeding; and every compute_all_ne estimator returns None or a positive
finite Ne.  Three further cross-cutting properties generalise the example suite
over random pedigrees: ne_coancestry agrees across its three code paths
(default-stream / K / pre-computed θ̄); compute_all_ne is thread-count
invariant; and founder contributions are conserved (sum to 1 per cohort) under
complete parentage.
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

from pedigree_graph import (
    MissingMetadataError,
    PedigreeGraph,
    compute_all_ne,
    ne_coancestry,
    ne_inbreeding,
    ne_long_term_contributions,
    ne_sex_ratio,
)
from pedigree_graph._ne_founders import _per_gen_founder_means

_SETTINGS = settings(deadline=None, max_examples=50)
_HEAVY = settings(deadline=None, max_examples=30)

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
    F, gen = pg.inbreeding(), pg.generation
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
@given(pg=complete_parentage_pedigree())
def test_compute_all_ne_none_or_positive(pg):
    for name, result in compute_all_ne(pg).items():
        ne = result.ne
        assert ne is None or (np.isfinite(ne) and ne > 0.0), name


@_UNIFORM_SEX_OK
@_SETTINGS
@given(pg=random_pedigree())
def test_compute_all_ne_disables_only_the_founder_estimators_on_one_parent_rows(pg):
    # The founder-based estimators need closed represented parentage; the
    # adapter reports no estimate for them and keeps the other six.
    one_parent = (np.asarray(pg.mother_rows) < 0) != (np.asarray(pg.father_rows) < 0)
    results = compute_all_ne(pg)
    assert len(results) == 8
    if one_parent.any():
        assert results["ne_long_term_contributions"].ne is None
        assert results["ne_caballero_toro"].ne is None
        with pytest.raises(MissingMetadataError) as info:
            ne_long_term_contributions(pg)
        assert info.value.fields["affected_count"] == int(one_parent.sum())


def _ne_close(a, b):
    if a is None or b is None:
        return a is None and b is None
    return a == pytest.approx(b, rel=1e-9, abs=1e-12)


@_HEAVY
@given(pg=random_pedigree())
def test_ne_coancestry_three_paths_agree(pg):
    # The default streams θ̄ from the DP (the scale path that avoids
    # materialising K); the other two pass K or a pre-computed θ̄.  All three
    # must produce the same coancestry-rate Ne and the same per-cohort θ̄.
    default = ne_coancestry(pg)
    k_path = ne_coancestry(pg, K=pg.kinship_matrix(0.0))
    theta_path = ne_coancestry(pg, theta_per_gen=pg.per_gen_mean_kinship())
    assert _ne_close(default.ne, k_path.ne)
    assert _ne_close(default.ne, theta_path.ne)
    assert np.allclose(default.mean_theta_per_gen, k_path.mean_theta_per_gen, equal_nan=True)
    assert np.allclose(default.mean_theta_per_gen, theta_path.mean_theta_per_gen, equal_nan=True)


@_UNIFORM_SEX_OK
@_HEAVY
@given(pg=complete_parentage_pedigree())
def test_compute_all_ne_thread_count_invariant(pg):
    # Independent estimators dispatched to worker threads must give bit-identical
    # results to the serial path (a cache race would surface as a mismatch).
    serial = compute_all_ne(pg, n_threads=1)
    threaded = compute_all_ne(pg, n_threads=4)
    assert serial.keys() == threaded.keys()
    for name in serial:
        assert serial[name].to_dict() == threaded[name].to_dict(), name


@_HEAVY
@given(pg=complete_parentage_pedigree())
def test_founder_contributions_conserved(pg):
    # With complete parentage (no half-missing parent), every individual's
    # genome is fully partitioned among the founders, so the per-cohort mean
    # founder contributions sum to 1 in each non-empty generation.
    m_g, founder_idx = _per_gen_founder_means(pg)
    gen = np.asarray(pg.generation)
    for g in range(int(gen.max()) + 1):
        if (gen == g).any() and np.isfinite(m_g[g]).all():
            assert m_g[g].sum() == pytest.approx(1.0, abs=1e-9)
    # Corollary: Σ_f c_f² ∈ [1/n_founders, 1] ⇒ Ne_LTC ∈ [0.5, n_founders/2].
    res = ne_long_term_contributions(pg)
    if res.asymptote_reached and res.ne is not None:
        assert 0.5 - 1e-9 <= res.ne <= len(founder_idx) / 2.0 + 1e-9
