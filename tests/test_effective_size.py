"""Toy-pedigree validation for effective population size estimators.

Hand-derived F, θ, EqG, Ne_V, Ne_sr on small pedigrees with closed-form
expectations.  Finite-sample tolerances are loose (~0.01) and analytic
cases use 1e-9.
"""

import numpy as np
import polars as pl
import pytest
from _support import _build_closed_line, _df, _random_mating

from pedigree_graph import PedigreeGraph
from pedigree_graph._cohorts import ObservedCohorts
from pedigree_graph._ne_common import _harmonic_mean
from pedigree_graph._ne_family_size import (
    _sex_specific_family_table,
    _sigma2_from_quadrants,
)
from pedigree_graph._ne_rates import _equivalent_generations
from pedigree_graph.effective_size import (
    ALL_EFFECTIVE_SIZE_ESTIMATORS,
    estimate_effective_sizes,
    ne_coancestry,
    ne_group_coancestry,
    ne_hill_overlapping,
    ne_inbreeding,
    ne_individual_delta_f,
    ne_long_term_contributions,
    ne_sex_ratio,
    ne_variance_family_size,
)


def _toy_birth_year_pedigree(
    *,
    cohort_a: int = 1900,
    cohort_b: int = 1910,
    paternity: str = "balanced",
) -> pl.DataFrame:
    """Build a 2-cohort pedigree with controlled σ²_m, σ²_f for eq. (10) tests.

    Cohort A: 2 males (ids 0,1) + 2 females (ids 2,3).  Cohort B: 4 offspring.

    * ``paternity='balanced'``: each parent has 1 son + 1 daughter →
      σ²_m = σ²_f = 0.
    * ``paternity='skewed'``: father 0 + mother 2 produce all 4 offspring →
      σ²_m = σ²_f = 8.
    * ``paternity='male_skewed_only'``: father 0 has all 4 sons + daughters
      but mothers are split (mother 2 → 2 kids, mother 3 → 2 kids) →
      σ²_m = 8, σ²_f = 0.
    """
    records = [
        {"id": 0, "sex": 1, "generation": 0, "birth_year": cohort_a},
        {"id": 1, "sex": 1, "generation": 0, "birth_year": cohort_a},
        {"id": 2, "sex": 0, "generation": 0, "birth_year": cohort_a},
        {"id": 3, "sex": 0, "generation": 0, "birth_year": cohort_a},
    ]
    if paternity == "balanced":
        parents = [(0, 2), (1, 3), (0, 2), (1, 3)]
    elif paternity == "skewed":
        parents = [(0, 2), (0, 2), (0, 2), (0, 2)]
    elif paternity == "male_skewed_only":
        # All 4 children sired by father 0, but mothers split → σ²_m=8, σ²_f=0
        parents = [(0, 2), (0, 2), (0, 3), (0, 3)]
    elif paternity == "female_skewed_only":
        # All 4 children mothered by mother 2; fathers split → σ²_m=0, σ²_f=8
        parents = [(0, 2), (0, 2), (1, 2), (1, 2)]
    else:
        raise ValueError(f"unknown paternity={paternity!r}")
    for i, (f, m) in enumerate(parents):
        records.append(
            {
                "id": 4 + i,
                "sex": 1 if i < 2 else 0,
                "generation": 1,
                "birth_year": cohort_b,
                "father": f,
                "mother": m,
            }
        )
    return _df(records)


# ---------------------------------------------------------------------------
# Toy 1 — 2-founder full-sib mating: F=0.25, θ_sibs=0.25, EqG=1
# ---------------------------------------------------------------------------


def test_toy1_full_sib_mating_F_theta_eqg():
    df = _df(
        [
            {"id": 0, "sex": 1, "generation": 0},  # founder M
            {"id": 1, "sex": 0, "generation": 0},  # founder F
            # Two full sibs of (0, 1)
            {"id": 2, "sex": 1, "generation": 1, "mother": 1, "father": 0},
            {"id": 3, "sex": 0, "generation": 1, "mother": 1, "father": 0},
            # Inbred offspring of full sibs
            {"id": 4, "sex": 1, "generation": 2, "mother": 3, "father": 2},
        ]
    )
    pg = PedigreeGraph.from_frame(df)
    F = pg.inbreeding()

    # F[founders] = 0; F[full sibs of unrelated parents] = 0; F[inbred] = 0.25
    assert F[0] == 0.0
    assert F[1] == 0.0
    assert F[2] == 0.0
    assert F[3] == 0.0
    assert F[4] == pytest.approx(0.25, abs=1e-12)

    # θ(2, 3) = 0.25 (full sibs of non-inbred parents)
    K = pg.kinship_matrix().toarray()
    assert K[2, 3] == pytest.approx(0.25, abs=1e-12)

    # EqG: founders 0; gen-1 with both founder parents → 1; gen-2 with two
    # gen-1 parents (each with EqG=1) → 1 + 0.5*(1+1) = 2.
    eqg = _equivalent_generations(pg)
    assert eqg[0] == 0.0
    assert eqg[1] == 0.0
    assert eqg[2] == pytest.approx(1.0, abs=1e-12)
    assert eqg[3] == pytest.approx(1.0, abs=1e-12)
    assert eqg[4] == pytest.approx(2.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Toy 2 — Nm=Nf=10, balanced multinomial families ≈ Poisson(2): Ne_V ≈ Ne_sr ≈ 20
# ---------------------------------------------------------------------------


def _build_random_mating_pedigree(
    rng: np.random.Generator,
    n_male: int,
    n_female: int,
    n_offspring: int,
) -> pl.DataFrame:
    """Two-generation random-mating pedigree with multinomial parent picks.

    Each offspring picks one father uniformly, one mother uniformly,
    independently.  Sex of each offspring is Bernoulli(0.5).  Family
    size per parent is therefore Binomial(n_offspring, 1/n_parent_sex)
    ≈ Poisson(n_offspring / n_parent_sex) for small p.
    """
    male_ids = list(range(n_male))
    female_ids = list(range(n_male, n_male + n_female))
    records: list[dict] = [{"id": mid, "sex": 1, "generation": 0} for mid in male_ids]
    records.extend({"id": fid, "sex": 0, "generation": 0} for fid in female_ids)
    next_id = n_male + n_female
    for _ in range(n_offspring):
        f = int(rng.choice(male_ids))
        m = int(rng.choice(female_ids))
        sex = int(rng.integers(0, 2))
        records.append(
            {
                "id": next_id,
                "sex": sex,
                "generation": 1,
                "mother": m,
                "father": f,
            }
        )
        next_id += 1
    return _df(records)


def test_toy2_random_mating_NeV_NeSr_match_N():
    rng = np.random.default_rng(0)
    df = _build_random_mating_pedigree(rng, n_male=10, n_female=10, n_offspring=20)
    pg = PedigreeGraph.from_frame(df)

    sr = ne_sex_ratio(pg)
    # Gen 0: 10 M, 10 F → Ne_sr_0 = 4*10*10/20 = 20.
    assert sr.ne_per_gen[0] == pytest.approx(20.0, abs=1e-9)
    # Gen 1 cohort sex ratio drift around 20.
    assert sr.ne is not None
    assert 5.0 < sr.ne < 50.0

    v = ne_variance_family_size(pg)
    # Single transition (gen 0→1).  Random multinomial with N_m=N_f=10
    # and 20 offspring → V(k_total)/k̄ ≈ 1, so Ne_V should land near
    # Ne_sr = 20.  Loose check — small-sample variance is high.
    assert v.ne is not None
    assert 5.0 < v.ne < 80.0


# ---------------------------------------------------------------------------
# Toy 3 — Skewed family size: one male sires all
# ---------------------------------------------------------------------------


def test_toy3_skewed_male_NeV_below_sex_ratio():
    """Single dominant male; N_m=N_f=4, one male sires every offspring.

    Hand: V(k_mm)+V(k_mf)+2·Cov_m for males with k_total = (8,0,0,0):
        kbar_m = 2; V(k_total)/kbar = 16/2 = 8.
    Females evenly distributed (each has 2 kids ⇒ V_f = 0):
        V(k_total)/kbar_f = 0.
    ΔF = 8/(4·4·2) + 0/(4·4·2) = 8/32 = 0.25 ⇒ Ne_V = 2.
    Ne_sr at gen 0 = 4·4·4/8 = 8.  Ne_V should be ≪ Ne_sr.
    """
    male_ids = [0, 1, 2, 3]
    female_ids = [4, 5, 6, 7]
    records: list[dict] = [{"id": mid, "sex": 1, "generation": 0} for mid in male_ids]
    records.extend({"id": fid, "sex": 0, "generation": 0} for fid in female_ids)
    # Male 0 sires all 8 offspring — 2 per female.  Sex 50/50.
    next_id = 8
    sex_pattern = [0, 1] * 4  # 4F + 4M
    for fem, sx in zip(np.repeat(female_ids, 2), sex_pattern, strict=True):
        records.append(
            {
                "id": next_id,
                "sex": int(sx),
                "generation": 1,
                "mother": int(fem),
                "father": 0,
            }
        )
        next_id += 1
    df = _df(records)
    pg = PedigreeGraph.from_frame(df)

    sr = ne_sex_ratio(pg)
    assert sr.ne_per_gen[0] == pytest.approx(8.0, abs=1e-9)

    v = ne_variance_family_size(pg)
    # V(k_total) for males: counts (8, 0, 0, 0) → mean 2, var = (36+4+4+4)/3 = 16.
    assert v.ne is not None
    assert v.ne == pytest.approx(2.0, abs=1e-9)
    # Sanity: Ne_V ≪ Ne_sr because variance dominates.
    assert v.ne < sr.ne_per_gen[0]


# ---------------------------------------------------------------------------
# Toy 4 — Closed line, Nm=Nf=1 per gen, full-sib mating, 5 generations
# ---------------------------------------------------------------------------


def test_toy4_closed_line_F_recursion():
    """Full-sib mating chain: F follows F_{t+1} = (1+2F_t+F_{t-1})/4.

    F values: F_0=F_1=0, F_2=0.25, F_3=0.375, F_4=0.5, F_5=0.59375.
    Asymptotic Ne ≈ 2.62 (eigenvalue (1+√5)/4 ≈ 0.809).
    """
    pg = PedigreeGraph.from_frame(_build_closed_line(n_gens=5))

    res = ne_inbreeding(pg)
    expected = [0.0, 0.0, 0.25, 0.375, 0.5, 0.59375]
    np.testing.assert_allclose(res.mean_f_per_gen, expected, atol=1e-12)
    # Ne should be in [2, 3] for sib-mating chain (asymptotic ≈ 2.62 with finite-sample bias).
    assert res.ne is not None
    assert 1.5 < res.ne < 4.0


# ---------------------------------------------------------------------------
# Cross-cutting sanity: ne_coancestry on toy 1 and toy 4
# ---------------------------------------------------------------------------


def test_ne_coancestry_toy1_smoke():
    """Toy 1 has too few cohorts for a meaningful slope, but the per-gen θ̄ should be exact.

    Gen 0 has 2 founders → 1 unordered pair, θ = 0 → mean θ_0 = 0.
    Gen 1 has 2 full sibs → θ(2,3) = 0.25 → mean θ_1 = 0.25.
    Gen 2 has 1 individual → mean θ_2 = NaN (no pair).
    """
    df = _df(
        [
            {"id": 0, "sex": 1, "generation": 0},
            {"id": 1, "sex": 0, "generation": 0},
            {"id": 2, "sex": 1, "generation": 1, "mother": 1, "father": 0},
            {"id": 3, "sex": 0, "generation": 1, "mother": 1, "father": 0},
            {"id": 4, "sex": 1, "generation": 2, "mother": 3, "father": 2},
        ]
    )
    pg = PedigreeGraph.from_frame(df)
    res = ne_coancestry(pg)
    assert res.mean_theta_per_gen[0] == pytest.approx(0.0, abs=1e-12)
    assert res.mean_theta_per_gen[1] == pytest.approx(0.25, abs=1e-12)
    assert np.isnan(res.mean_theta_per_gen[2])


# ---------------------------------------------------------------------------
# Ne_iΔF (Gutiérrez): closed-line F recursion drives ΔF_i
# ---------------------------------------------------------------------------


def test_ne_individual_delta_f_closed_line():
    """Gutiérrez eq. 2 on the closed-line full-sib chain.

    Equivalent complete generations are 0, 1, 2, 3, 4, 5 by cohort and F is
    0, 0, 0.25, 0.375, 0.5, 0.59375, so ``ΔF_i = 1 − (1 − F_i)^(1/t_i)`` is::

        cohort 1   1 − (1 − 0)^(1/1)       = 0
        cohort 2   1 − (1 − 0.25)^(1/2)    = 0.13397460…
        cohort 3   1 − (1 − 0.375)^(1/3)   = 0.14501203…
        cohort 4   1 − (1 − 0.5)^(1/4)     = 0.15910358…
        cohort 5   1 − (1 − 0.59375)^(1/5) = 0.16486062…

    The founders have ``t = 0`` and no rate at all.  Cohort 1 has ``t = 1``
    and ``F = 0``, so it is eligible with ``ΔF_i = 0`` exactly — counted, but
    still no Ne.  Both members of a later cohort share one pedigree, so ΔF̄_g
    is that cohort's single value and ``Ne_g = 1/(2·ΔF̄_g)``.

    The reference subpopulation defaults to the last observed cohort, whose
    two members are identical, so the scalar is its ``Ne_g``, σ_ΔF is zero and
    the standard error with it.

    ``ne_unrelated_founders`` is this package's diagnostic and not the paper's,
    dividing by ``t − 1`` instead, so the last cohort's ΔF′ is
    ``1 − (1 − 0.59375)^(1/4)``.  It is the smaller of the two here, and on
    any pedigree whose ``t ≤ 1`` rows carry ``F = 0``: ``1/(t − 1) > 1/t``
    raises every ΔF′_i to at least its ΔF_i, and dropping those rows then only
    removes zeros.  A child of two MZ co-twins carries ``F = 0.5`` at ``t = 1``
    and breaks both halves of that, which ``test_effective_size_api.py`` pins.
    """
    pg = PedigreeGraph.from_frame(_build_closed_line(n_gens=5))
    res = ne_individual_delta_f(pg)

    np.testing.assert_array_equal(res.n_used_per_gen, [0, 2, 2, 2, 2, 2])
    assert np.isnan(res.mean_eqg_per_gen[0])
    np.testing.assert_allclose(res.mean_eqg_per_gen[1:], [1.0, 2.0, 3.0, 4.0, 5.0], atol=1e-12)

    assert np.isnan(res.ne_per_gen[0])
    assert np.isnan(res.ne_per_gen[1])
    f_per_gen = np.array([0.25, 0.375, 0.5, 0.59375])
    eqg_per_gen = np.array([2.0, 3.0, 4.0, 5.0])
    delta_f = 1.0 - (1.0 - f_per_gen) ** (1.0 / eqg_per_gen)
    np.testing.assert_allclose(res.ne_per_gen[2:], 1.0 / (2.0 * delta_f), atol=1e-12)

    assert res.ne == pytest.approx(1.0 / (2.0 * delta_f[-1]), abs=1e-12)
    assert res.standard_error == 0.0
    assert res.n_reference == 2
    assert res.reference_generation == 5

    lagged_df = 1.0 - (1.0 - 0.59375) ** (1.0 / 4.0)
    assert res.ne_unrelated_founders == pytest.approx(1.0 / (2.0 * lagged_df), abs=1e-12)
    assert res.ne_unrelated_founders == pytest.approx(2.4796571025831384, abs=1e-9)
    assert res.ne_unrelated_founders < res.ne


def _unequal_cohort_pedigree() -> pl.DataFrame:
    """Cohorts of 2, 2, 3 and 4 rows, with two pedigree depths in the last one.

    Generation 2 holds the full-sib pair (4, 5) and a third founder, 6, so
    generation 3 mixes 7 and 10 (children of 4 × 5, ``F = 0.375`` at ``t = 3``)
    with their paternal half-sibs 8 and 9 (children of 4 × 6, ``F = 0`` at
    ``t = 2``).
    """
    return _df(
        [
            {"id": 0, "sex": 1, "generation": 0},
            {"id": 1, "sex": 0, "generation": 0},
            {"id": 2, "sex": 1, "generation": 1, "mother": 1, "father": 0},
            {"id": 3, "sex": 0, "generation": 1, "mother": 1, "father": 0},
            {"id": 4, "sex": 1, "generation": 2, "mother": 3, "father": 2},
            {"id": 5, "sex": 0, "generation": 2, "mother": 3, "father": 2},
            {"id": 6, "sex": 0, "generation": 2},
            {"id": 7, "sex": 1, "generation": 3, "mother": 5, "father": 4},
            {"id": 8, "sex": 1, "generation": 3, "mother": 6, "father": 4},
            {"id": 9, "sex": 0, "generation": 3, "mother": 6, "father": 4},
            {"id": 10, "sex": 0, "generation": 3, "mother": 5, "father": 4},
        ]
    )


def test_ne_individual_delta_f_averages_delta_f_over_the_reference_subpopulation():
    """On unequal cohorts the scalar is ``1/(2ΔF̄)``, not a harmonic mean of cohort Ne.

    Gutiérrez §2.1 averages ΔF_i over a reference subpopulation.  Aggregating
    the per-cohort Ne instead gives every cohort the same weight whatever its
    size and whatever spread of pedigree depth it holds, and here the two
    answers are two units apart (issue #15, ADR 0012).
    """
    pg = PedigreeGraph.from_frame(_unequal_cohort_pedigree())
    res = ne_individual_delta_f(pg)

    np.testing.assert_array_equal(res.n_used_per_gen, [0, 2, 2, 4])
    assert res.n_reference == 4
    assert res.reference_generation == 3

    f_last = np.array([0.375, 0.0, 0.0, 0.375])
    eqg_last = np.array([3.0, 2.0, 2.0, 3.0])
    delta_f = 1.0 - (1.0 - f_last) ** (1.0 / eqg_last)
    assert res.ne == pytest.approx(1.0 / (2.0 * delta_f.mean()), abs=1e-12)
    assert res.standard_error == pytest.approx(2.0 / np.sqrt(delta_f.size) * res.ne**2 * delta_f.std(ddof=1), rel=1e-12)

    assert res.ne == pytest.approx(6.895979754377508, abs=1e-9)
    assert res.standard_error == pytest.approx(3.981395767516064, abs=1e-9)
    assert _harmonic_mean(res.ne_per_gen) == pytest.approx(4.843069778788811, abs=1e-9)


@pytest.mark.parametrize(("n_per_gen", "n_gens", "seed"), [(60, 12, 7), (200, 16, 3)])
def test_ne_individual_delta_f_tracks_the_census_size_under_random_mating(n_per_gen, n_gens, seed):
    """Last-cohort reference tracks N on an idealised population, carrying its own upward bias.

    Gutiérrez §2.1 derives ΔF_i by equating F_i to the inbreeding of a
    hypothetical idealised population of size Ne with that individual's
    pedigree structure, so on a pedigree that really is idealised the estimate
    should track the census size.  It does, and it runs high while doing so,
    by roughly ``t/(t − 1)``.  Eq. 1 inverts ``F_t = 1 − (1 − ΔF)^t``, so eq. 2
    recovers N only where the pedigree has accumulated ``t`` generations of
    drift.  These founders are unrelated and non-inbred by construction, so
    generation 1 has ``F = 0`` exactly and inbreeding runs one generation
    behind the generation count: measured mean F matches the idealised curve
    at ``g − 1``, not ``g``, at every generation out to 16.  Replicated over
    20 Wright-Fisher pedigrees at N=200 the mean lands +19.1% at ``t = 5``,
    +10.5% at 10, +5.6% at 16 and +3.8% at 24, against a lag prediction of
    +25.0/+11.1/+6.7/+4.3% less a 1.8/0.8/0.5/0.3% convexity term, because
    ``F ↦ ΔF_i`` is convex and within-cohort variance in F raises ΔF̄.  The
    bias is therefore a property of a shallow pedigree with a hard founder
    boundary, and it decays as ``1/t``.  The seeds pinned here give +10.7% at
    N=60 and +0.0% at N=200, and the 15% band holds both without asserting an
    accuracy the estimator does not have.

    What the band does separate is the reduction.  Averaging over the whole
    genealogy feeds ΔF̄ every generation's N rows with ``t = 1`` and ``F = 0``,
    which lands +40% out at N=60 (ADR 0012).
    """
    pg = PedigreeGraph.from_frame(_random_mating(n_per_gen, n_gens, seed))
    res = ne_individual_delta_f(pg)

    assert res.n_reference == n_per_gen
    assert res.reference_generation == n_gens
    assert res.ne == pytest.approx(n_per_gen, rel=0.15)
    assert 0.0 < res.standard_error < res.ne


def test_the_unrelated_founder_diagnostic_beats_the_estimator_on_a_wright_fisher_pedigree():
    """Removing the one-generation lag lands nearer N on a pedigree whose founders are unrelated.

    ``ne_unrelated_founders`` is this package's diagnostic, not Gutiérrez's.
    Eq. 2 recovers the census size only where the pedigree has accumulated
    ``t`` generations of drift, and a Wright-Fisher pedigree's founders are
    unrelated and non-inbred by construction, so ``t`` generations of pedigree
    carry ``t − 1`` generations of drift and ``ne`` runs high by ``t/(t − 1)``.
    Dividing by ``t − 1`` over the rows with ``t > 1`` removes that lag, and
    here it moves +10.65% down to +1.47%.

    One pinned seed is not evidence.  The ``(200, 16, 3)`` parametrization of
    the test above is a counter-case: its ``ne`` lands at +0.01% and the
    diagnostic at −6.23%, so the closer-to-N claim is deliberately not asserted
    there.  What the claim rests on is the 20-replicate-per-cell Wright-Fisher
    table in ADR 0012, where the correction cuts a bias running from +5.57% to
    +24.66% to a residual no worse than −4.57%.

    The assumption is true of a simulated pedigree and false of a real one,
    whose founders are merely where record-keeping stopped and are generally
    related.  There is no lag to remove there, and the field biases downward.
    """
    pg = PedigreeGraph.from_frame(_random_mating(60, 12, 7))
    res = ne_individual_delta_f(pg)

    assert res.n_reference == 60
    assert res.ne == pytest.approx(66.3906634390028, rel=1e-9)
    assert res.ne_unrelated_founders == pytest.approx(60.881503258688106, rel=1e-9)
    assert abs(res.ne_unrelated_founders - 60) < abs(res.ne - 60)


# ---------------------------------------------------------------------------
# Ne_LTC: W&T eq. 31 / C&T eq. 19 at the last observed cohort
# ---------------------------------------------------------------------------


def test_ne_long_term_contributions_closed_line():
    """Closed line, 2 founder genomes → c stable at (0.5, 0.5) ⇒ Σc² = 0.5.

    Caballero & Toro 2000 (Genet. Res. 75(3):331-343) eq. 19 is
    ``N_ef = 1/[(1/N²)Σc²_{i(0,t)}]``, which in normalised contributions is
    ``1/Σc² = 2``.  Wray & Thompson 1990 (Genet. Res. 55(1):41-54) eq. 31 is
    ``Ne ≈ 2N/(μ_r² + σ_r²)``, which at ``μ_r = 1`` gives ``Σr² = N·Σc²`` and
    so ``Ne = 2/Σc² = 4``; C&T's own text after their eq. 20 states the same
    link as ``N_ef = Ne/2``.  The version this replaces reported
    ``1/(2Σc²) = 1`` (ADR 0012, issue #15).

    Forced full-sib mating is not W&T's regular random mating, so what is
    pinned here is the formula and not the interpretation: nothing about this
    pedigree makes 4 its effective size.  The interpretation is pinned by
    ``test_ne_long_term_contributions_tracks_the_census_size_under_random_mating``.
    """
    pg = PedigreeGraph.from_frame(_build_closed_line(n_gens=5))
    res = ne_long_term_contributions(pg)
    assert res.sum_c_squared == pytest.approx(0.5, abs=1e-12)
    assert res.n_effective_founders == pytest.approx(2.0, abs=1e-12)
    assert res.ne == pytest.approx(4.0, abs=1e-12)
    assert res.max_delta_final == pytest.approx(0.0, abs=1e-12)
    assert res.asymptote_reached
    assert res.n_cohorts == 6
    assert res.final_generation == 5


def test_ne_long_term_contributions_4_founders_symmetric():
    """4 founders → 4 gen-1 individuals with each founder seen by exactly 2.

    Each gen-1 individual has a c-vector with two 0.5s and two 0s; the
    cohort mean is uniform 0.25, so Σc² = 4·0.25² = 0.25, C&T eq. 19 gives
    ``N_ef = 1/Σc² = 4`` and W&T eq. 31 gives ``Ne = 2/Σc² = 8``.  As in the
    closed line above, this mating scheme is not W&T's regular random mating,
    so it pins the formula and not the interpretation.
    """
    df = _df(
        [
            {"id": 0, "sex": 1, "generation": 0},
            {"id": 1, "sex": 1, "generation": 0},
            {"id": 2, "sex": 0, "generation": 0},
            {"id": 3, "sex": 0, "generation": 0},
            {"id": 4, "sex": 1, "generation": 1, "mother": 2, "father": 0},
            {"id": 5, "sex": 1, "generation": 1, "mother": 3, "father": 1},
            {"id": 6, "sex": 0, "generation": 1, "mother": 3, "father": 0},
            {"id": 7, "sex": 0, "generation": 1, "mother": 2, "father": 1},
        ]
    )
    pg = PedigreeGraph.from_frame(df)
    res = ne_long_term_contributions(pg)
    assert res.sum_c_squared == pytest.approx(0.25, abs=1e-12)
    assert res.n_effective_founders == pytest.approx(4.0, abs=1e-12)
    assert res.ne == pytest.approx(8.0, abs=1e-12)
    assert res.max_delta_final == pytest.approx(0.0, abs=1e-12)
    assert res.asymptote_reached
    assert res.n_cohorts == 2
    assert res.final_generation == 1


def test_ne_long_term_contributions_tracks_the_census_size_under_random_mating():
    """``Ne = 2/Σc²`` lands on N on the pedigree its assumptions describe.

    Wray & Thompson 1990 eq. 31 holds under regular random mating with
    long-term contribution variance at its asymptote, so on a Wright-Fisher
    pedigree that really is idealised the estimate should track the census
    size.  Over 30 replicates at N=200, g=10 (seeds 2026-2055) the mean lands
    at 202.30, +1.15% of N, sd 11.90.  The residual is O(1/N) and shrinks
    with N — +5.02% at N=60 (g=8), +3.30% at N=120 (g=10), +1.15% at N=200
    (g=10), all over the same 30 seeds.  A single replicate carries far more
    spread than the mean does: these 30 run from 172.09 (−13.95%) to 226.95
    (+13.48%), so the band is asserted on the mean and no single pedigree is
    asserted to land near N.

    The 5% band sits about 3.5 sem from the measured mean (sem 2.17) and
    still separates ``2/Σc²`` from the ``1/(2Σc²)`` it replaces, from a
    factor of 2, and from a sign flip (ADR 0012, issue #15).

    ``asymptote_reached`` is False on every one of the 30 — ``max_delta_final``
    runs 3.6e-4 to 1.2e-3 against a 1e-6 tolerance — which is why the version
    that gated ``ne`` on that flag reported no estimate on any realistic
    pedigree.
    """
    results = [
        ne_long_term_contributions(PedigreeGraph.from_frame(_random_mating(200, 10, seed)))
        for seed in range(2026, 2056)
    ]

    assert np.mean([res.ne for res in results]) == pytest.approx(200.0, rel=0.05)
    assert results[0].ne == pytest.approx(2.0 * results[0].n_effective_founders, abs=1e-12)
    for res in results:
        assert res.ne is not None
        assert not res.asymptote_reached


# ---------------------------------------------------------------------------
# Ne_H (Hill 1979): regression sentinel for L=1 collapse to Ne_V
# ---------------------------------------------------------------------------


def test_ne_hill_collapses_to_ne_v():
    """Discrete-generation sentinel: Hill 1979 reduces to Ne_V at L=1."""
    pg = PedigreeGraph.from_frame(_build_closed_line(n_gens=5))
    h = ne_hill_overlapping(pg)
    v = ne_variance_family_size(pg)
    assert h.collapses_to_ne_v
    assert h.generation_interval == 1.0
    if v.ne is None:
        assert h.ne is None
    else:
        assert h.ne == pytest.approx(v.ne, abs=1e-12)


# ---------------------------------------------------------------------------
# Ne_H birth-year branch — Hill 1979 eq. (10) via pg.birth_year
# ---------------------------------------------------------------------------


def test_ne_hill_uses_birth_year_handcomputed_balanced():
    """Hand-computed toy: σ²_m = σ²_f = 0, N1 = 4, T = 10 → Ne = 80.

    Per cohort 1900: 2 males + 2 females, each producing 1 son + 1
    daughter at year 1910.  All sex-of-offspring quadrants k_mm = k_mf
    = k_fm = k_ff = 1 for every parent → σ²_m = σ²_f = 0.  Hill 1979
    eq. (10): Ne = 8·4·10 / (0 + 0 + 4) = 80.
    """
    df = _toy_birth_year_pedigree(paternity="balanced")
    pg = PedigreeGraph.from_frame(df)
    res = ne_hill_overlapping(pg)
    assert res.collapses_to_ne_v is False
    assert res.T_m == pytest.approx(10.0)
    assert res.T_f == pytest.approx(10.0)
    assert res.generation_interval == pytest.approx(10.0)
    assert res.Vk_m == pytest.approx(0.0, abs=1e-12)
    assert res.Vk_f == pytest.approx(0.0, abs=1e-12)
    assert res.ne == pytest.approx(80.0, rel=1e-9)
    assert res.n_eligible_cohorts == 1


def test_ne_hill_uses_birth_year_handcomputed_skewed():
    """Hand-computed toy: σ²_m = σ²_f = 8, N1 = 4, T = 10 → Ne = 16.

    Father 0 + mother 2 produce all 4 offspring; father 1 + mother 3
    produce nothing.  V(k_mm)=V(k_mf)=2, Cov(k_mm,k_mf)=2 →
    σ²_m = 2+2+2·2 = 8 (symmetric for females).  Hill 1979 eq. (10):
    Ne = 8·4·10 / (8+8+4) = 16.
    """
    df = _toy_birth_year_pedigree(paternity="skewed")
    pg = PedigreeGraph.from_frame(df)
    res = ne_hill_overlapping(pg)
    assert res.Vk_m == pytest.approx(8.0, abs=1e-9)
    assert res.Vk_f == pytest.approx(8.0, abs=1e-9)
    assert res.ne == pytest.approx(16.0, rel=1e-9)


def test_ne_hill_monotonic_in_sigma_m():
    """Ne_H decreases as σ²_m increases (all else equal)."""
    pg_lo = PedigreeGraph.from_frame(_toy_birth_year_pedigree(paternity="balanced"))
    pg_hi = PedigreeGraph.from_frame(_toy_birth_year_pedigree(paternity="skewed"))
    ne_lo = ne_hill_overlapping(pg_lo).ne
    ne_hi = ne_hill_overlapping(pg_hi).ne
    assert ne_lo is not None
    assert ne_hi is not None
    assert ne_lo > ne_hi


def test_ne_hill_linear_in_T():
    """Ne_H scales linearly in T when σ²_m, σ²_f, N1 are fixed."""
    pg_t10 = PedigreeGraph.from_frame(_toy_birth_year_pedigree(cohort_a=1900, cohort_b=1910))
    pg_t20 = PedigreeGraph.from_frame(_toy_birth_year_pedigree(cohort_a=1900, cohort_b=1920))
    ne_t10 = ne_hill_overlapping(pg_t10).ne
    ne_t20 = ne_hill_overlapping(pg_t20).ne
    assert ne_t10 is not None
    assert ne_t20 is not None
    assert ne_t20 == pytest.approx(2.0 * ne_t10, rel=1e-9)


def test_ne_hill_sex_symmetry():
    """Swapping (N_m, σ²_m) ↔ (N_f, σ²_f) leaves Ne unchanged."""
    pg_m = PedigreeGraph.from_frame(_toy_birth_year_pedigree(paternity="male_skewed_only"))
    pg_f = PedigreeGraph.from_frame(_toy_birth_year_pedigree(paternity="female_skewed_only"))
    res_m = ne_hill_overlapping(pg_m)
    res_f = ne_hill_overlapping(pg_f)
    # σ²_m and σ²_f should swap; Ne is invariant.
    assert res_m.Vk_m == pytest.approx(res_f.Vk_f, abs=1e-9)
    assert res_m.Vk_f == pytest.approx(res_f.Vk_m, abs=1e-9)
    assert res_m.ne == pytest.approx(res_f.ne, rel=1e-9)


def test_ne_hill_eligible_cohort_filtering():
    """Right-censored cohorts excluded; cohort_window reflects this."""
    # Three cohorts: 1900 (founders), 1910 (kids of 1900), 1920 (kids of 1910).
    # Eligible window cutoff = y_max - p95(Δ) = 1920 - 10 = 1910.
    # So cohort 1900 alone is eligible (1910 is at the boundary,
    # included; 1920 excluded).
    records = [
        # Cohort 1900 founders: 2 males, 2 females
        {"id": 0, "sex": 1, "generation": 0, "birth_year": 1900},
        {"id": 1, "sex": 1, "generation": 0, "birth_year": 1900},
        {"id": 2, "sex": 0, "generation": 0, "birth_year": 1900},
        {"id": 3, "sex": 0, "generation": 0, "birth_year": 1900},
    ]
    # Cohort 1910 — 4 kids of cohort-1900 parents
    pairs_1910 = [(0, 2), (1, 3), (0, 2), (1, 3)]
    records.extend(
        {"id": 4 + i, "sex": 1 if i < 2 else 0, "generation": 1, "birth_year": 1910, "father": f, "mother": m}
        for i, (f, m) in enumerate(pairs_1910)
    )
    # Cohort 1920 — 4 kids of cohort-1910 parents
    pairs_1920 = [(4, 6), (5, 7), (4, 6), (5, 7)]
    records.extend(
        {"id": 8 + i, "sex": 1 if i < 2 else 0, "generation": 2, "birth_year": 1920, "father": f, "mother": m}
        for i, (f, m) in enumerate(pairs_1920)
    )
    df = _df(records)
    pg = PedigreeGraph.from_frame(df)
    res = ne_hill_overlapping(pg)
    assert res.cohort_window is not None
    # p95 of edge Δs = 10 (all edges are 10y); c_max = 1920 - 10 = 1910.
    assert res.cohort_window.c_max == 1910
    # Cohorts 1900 and 1910 both fall in [c_min=1900, c_max=1910] and
    # both have observed offspring (in 1910 and 1920 respectively), so
    # both contribute a per-cohort Ne.
    assert res.n_eligible_cohorts == 2
    # Cohort 1920 individuals (4 of them) are right-censored.
    assert res.n_excluded_right_censored == 4


def test_ne_hill_age_table_descriptive():
    """age_table is populated at the simulated parental ages."""
    pg = PedigreeGraph.from_frame(_toy_birth_year_pedigree(cohort_a=1900, cohort_b=1910))
    res = ne_hill_overlapping(pg)
    assert res.age_table is not None
    # All father-child edges are 10y; same for mothers.
    np.testing.assert_array_equal(res.age_table["ages_m"], [10])
    np.testing.assert_array_equal(res.age_table["ages_f"], [10])
    # 4 offspring → 4 father edges and 4 mother edges.
    np.testing.assert_array_equal(res.age_table["offspring_count_m"], [4])
    np.testing.assert_array_equal(res.age_table["offspring_count_f"], [4])
    assert res.n_offspring_pairs == 8  # 4 mother + 4 father edges


def test_ne_hill_birth_year_branch_survives_the_batch():
    """The orchestrated Hill record is the direct one, birth-year branch included."""
    pg = PedigreeGraph.from_frame(_toy_birth_year_pedigree(paternity="skewed"))
    assert estimate_effective_sizes(pg)["ne_hill_overlapping"] == ne_hill_overlapping(pg)


def test_ne_hill_serializes_to_dict():
    """to_dict() yields a YAML-ready dict including new fields."""
    pg = PedigreeGraph.from_frame(_toy_birth_year_pedigree(paternity="balanced"))
    res = ne_hill_overlapping(pg)
    d = res.to_dict()
    assert d["collapses_to_ne_v"] is False
    assert d["ne"] == pytest.approx(80.0)
    assert d["T_m"] == 10.0
    assert d["cohort_window"]["c_min"] == 1900
    assert d["age_table"]["ages_m"] == [10]
    # n_unknown_birth_year is 0 in this toy (every individual has a known year).
    assert d["n_unknown_birth_year"] == 0


# ---------------------------------------------------------------------------
# _sex_specific_family_table refactor: cohort_mode parameter
# ---------------------------------------------------------------------------


def _family_table(pg: PedigreeGraph, cohorts: ObservedCohorts | None = None):
    return _sex_specific_family_table(
        np.asarray(pg.mother_rows),
        np.asarray(pg.father_rows),
        np.asarray(pg.sex),
        ObservedCohorts.for_graph(pg, "test") if cohorts is None else cohorts,
    )


def test_sex_specific_family_table_has_every_observed_label():
    """Keys are the observed labels, the maximum included, with all-zero counts there."""
    pg = PedigreeGraph.from_frame(_build_closed_line(n_gens=5))
    table = _family_table(pg)
    g_max = int(np.asarray(pg.generation_labels).max())
    assert list(table) == list(range(g_max + 1))
    for p in range(g_max + 1):
        assert len(table[p].males_in_parent_gen) == 1
        assert len(table[p].females_in_parent_gen) == 1
    np.testing.assert_array_equal(table[g_max].k_mm, [0])
    np.testing.assert_array_equal(table[g_max].k_ff, [0])


def test_sex_specific_family_table_keys_on_rebased_labels():
    """Birth years (or any rebased labels) key the table by their own values."""
    pg = PedigreeGraph.from_frame(_build_closed_line(n_gens=5))
    fake_birth_year = (np.asarray(pg.generation_labels) + 1970).astype(np.int32)
    table = _family_table(pg, ObservedCohorts.from_labels(fake_birth_year))
    assert set(table) == {1970, 1971, 1972, 1973, 1974, 1975}
    np.testing.assert_array_equal(table[1975].k_mm, [0])
    np.testing.assert_array_equal(table[1975].k_mf, [0])
    np.testing.assert_array_equal(table[1975].k_fm, [0])
    np.testing.assert_array_equal(table[1975].k_ff, [0])


def test_sex_specific_family_table_unlabelled_rows_belong_to_no_cohort():
    """A ``-1`` row has no entry but its offspring still count for its parents."""
    pg = PedigreeGraph.from_frame(_build_closed_line(n_gens=3))
    cohort = np.asarray(pg.generation_labels).copy()
    cohort[0] = -1
    table = _family_table(pg, ObservedCohorts.from_labels(cohort))
    assert -1 not in table
    assert len(table[0].males_in_parent_gen) == 0
    assert len(table[0].females_in_parent_gen) == 1
    np.testing.assert_array_equal(table[0].k_fm, [1])
    np.testing.assert_array_equal(table[0].k_ff, [1])


def test_group_coancestry_equals_diluted_theta_plus_self_coancestry():
    """Closed line: ``f̄_g = θ̄_g·(n_g − 1)/n_g + s̄_g/n_g`` holds exactly.

    Eq. 3 sums the ``n_g`` self-coancestries ``s_i = (1 + F_i)/2`` alongside
    the ``n_g(n_g − 1)`` ordered pairs, so group coancestry is the
    within-cohort θ̄ diluted by ``(n_g − 1)/n_g`` plus a self term worth
    ``s̄_g/n_g``.  That second term is the algebra behind ``census_ratio``:
    it is the part of f̄ that tracks cohort size rather than drift, so a
    changing census shifts ``ln(1 − f̄)`` and the regression reads the shift
    as drift.

    The closed line has two rows per cohort and no MZ twins, so ``n_g == 2``
    throughout and ``θ̄_g`` is the single within-cohort pair.
    """
    pg = PedigreeGraph.from_frame(_build_closed_line(n_gens=5))
    res = ne_group_coancestry(pg)
    np.testing.assert_array_equal(res.n_genomes_per_gen, [2, 2, 2, 2, 2, 2])
    theta = ne_coancestry(pg).mean_theta_per_gen
    s_bar = (1.0 + ne_inbreeding(pg).mean_f_per_gen) / 2.0
    n_g = res.n_genomes_per_gen.astype(np.float64)
    np.testing.assert_allclose(
        res.mean_group_coancestry_per_gen,
        theta * (n_g - 1.0) / n_g + s_bar / n_g,
        atol=1e-12,
    )


# ---------------------------------------------------------------------------
# estimate_effective_sizes entry point
# ---------------------------------------------------------------------------


def test_estimate_effective_sizes_returns_eight_serializable_records():
    """Every estimator runs over one prerequisite memo and serializes."""
    pg = PedigreeGraph.from_frame(_build_closed_line(n_gens=5))
    results = estimate_effective_sizes(pg)
    assert set(results) == set(ALL_EFFECTIVE_SIZE_ESTIMATORS)
    for name, result in results.items():
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "ne" in d, f"{name} missing 'ne' field"


def test_sigma2_from_quadrants_returns_none_below_two_per_sex():
    pg = PedigreeGraph.from_frame(_build_closed_line(n_gens=4))
    table = _family_table(pg)
    # _build_closed_line has one male + one female per cohort, so
    # every cohort lacks two of a sex → decomposition is None.
    for entry in table.values():
        assert _sigma2_from_quadrants(entry) is None
