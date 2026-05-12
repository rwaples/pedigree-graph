"""Toy-pedigree validation for effective population size estimators (step 1).

Hand-derived F, θ, EqG, Ne_V, Ne_sr on small pedigrees with closed-form
expectations.  Per the master plan, finite-sample tolerances are loose
(~0.01) and analytic cases use 1e-9.
"""

import numpy as np
import pandas as pd
import pytest

from pedigree_graph import (
    PedigreeGraph,
    compute_all_ne,
    ne_caballero_toro,
    ne_coancestry,
    ne_hill_overlapping,
    ne_inbreeding,
    ne_individual_delta_f,
    ne_long_term_contributions,
    ne_sex_ratio,
    ne_variance_family_size,
)
from pedigree_graph._effective_size import _per_gen_mean_kinship
from pedigree_graph._kinship_kernel import (
    _compute_eqg,
    _compute_theta_per_gen,
    _per_gen_mean_kinship_from_dp,
)


def _df(records: list[dict]) -> pd.DataFrame:
    """Build a pedigree DataFrame from per-row dicts (defaults filled)."""
    rows = [
        {
            "id": r["id"],
            "mother": r.get("mother", -1),
            "father": r.get("father", -1),
            "twin": r.get("twin", -1),
            "sex": r["sex"],
            "generation": r["generation"],
        }
        for r in records
    ]
    return pd.DataFrame(rows)


def _df_by(records: list[dict]) -> pd.DataFrame:
    """Build a pedigree DataFrame including a ``birth_year`` column."""
    rows = [
        {
            "id": r["id"],
            "mother": r.get("mother", -1),
            "father": r.get("father", -1),
            "twin": r.get("twin", -1),
            "sex": r["sex"],
            "generation": r["generation"],
            "birth_year": r["birth_year"],
        }
        for r in records
    ]
    return pd.DataFrame(rows)


def _toy_birth_year_pedigree(
    *,
    cohort_a: int = 1900,
    cohort_b: int = 1910,
    paternity: str = "balanced",
) -> pd.DataFrame:
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
    return _df_by(records)


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
    pg = PedigreeGraph(df)
    F = pg.compute_inbreeding()

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
    eqg = _compute_eqg(np.asarray(pg.mother), np.asarray(pg.father), pg.n)
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
) -> pd.DataFrame:
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
    pg = PedigreeGraph(df)

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
    pg = PedigreeGraph(df)

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


def _build_closed_line(n_gens: int = 5) -> pd.DataFrame:
    """Closed-line full-sib mating: 2 founders, 1 male + 1 female per gen for ``n_gens``."""
    records = [
        {"id": 0, "sex": 1, "generation": 0},
        {"id": 1, "sex": 0, "generation": 0},
    ]
    next_id = 2
    prev_m, prev_f = 0, 1
    for g in range(1, n_gens + 1):
        m = next_id
        records.append({"id": m, "sex": 1, "generation": g, "mother": prev_f, "father": prev_m})
        f = next_id + 1
        records.append({"id": f, "sex": 0, "generation": g, "mother": prev_f, "father": prev_m})
        prev_m, prev_f = m, f
        next_id += 2
    return _df(records)


def test_toy4_closed_line_F_recursion():
    """Full-sib mating chain: F follows F_{t+1} = (1+2F_t+F_{t-1})/4.

    F values: F_0=F_1=0, F_2=0.25, F_3=0.375, F_4=0.5, F_5=0.59375.
    Asymptotic Ne ≈ 2.62 (eigenvalue (1+√5)/4 ≈ 0.809).
    """
    pg = PedigreeGraph(_build_closed_line(n_gens=5))

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
    pg = PedigreeGraph(df)
    res = ne_coancestry(pg)
    assert res.mean_theta_per_gen[0] == pytest.approx(0.0, abs=1e-12)
    assert res.mean_theta_per_gen[1] == pytest.approx(0.25, abs=1e-12)
    assert np.isnan(res.mean_theta_per_gen[2])


# ---------------------------------------------------------------------------
# Step 2 — Ne_iΔF (Gutiérrez): closed-line F recursion drives ΔF_i
# ---------------------------------------------------------------------------


def test_ne_individual_delta_f_closed_line():
    """Per-individual ΔF on the closed-line full-sib chain.

    EqG values: gen 1 → 1 (excluded; ΔF undefined for EqG=1),
    gen 2 → 2, gen 3 → 3, gen 4 → 4, gen 5 → 5.
    Hand:
      ΔF_2 = 1 − (1 − 0.25)^(1/1) = 0.25
      ΔF_3 = 1 − (0.625)^(1/2)    = 0.20943058…
      ΔF_4 = 1 − (0.5)^(1/3)      = 0.20629947…
      ΔF_5 = 1 − (0.40625)^(1/4)  = 0.20144841…
    Per-gen Ne_g = 1/(2·ΔF̄_g).  Aggregate is harmonic mean.
    """
    pg = PedigreeGraph(_build_closed_line(n_gens=5))
    res = ne_individual_delta_f(pg)

    # Gen 0 (founders, F=0, EqG=0) and gen 1 (EqG=1) excluded — n_used=0.
    assert res.n_used_per_gen[0] == 0
    assert res.n_used_per_gen[1] == 0
    # Gens 2–5: 2 individuals each.
    np.testing.assert_array_equal(res.n_used_per_gen[2:], [2, 2, 2, 2])

    # Per-gen Ne values (closed-form).
    expected_df = np.array(
        [
            0.25,
            1.0 - 0.625 ** (1.0 / 2.0),
            1.0 - 0.5 ** (1.0 / 3.0),
            1.0 - 0.40625 ** (1.0 / 4.0),
        ]
    )
    np.testing.assert_allclose(res.ne_per_gen[2:6], 1.0 / (2.0 * expected_df), atol=1e-12)
    # Aggregate harmonic mean of (2, 2.387, 2.423, 2.482) ≈ 2.31.
    assert res.ne is not None
    assert 1.5 < res.ne < 3.5


# ---------------------------------------------------------------------------
# Step 2 — Ne_LTC (Wray–Thompson): asymptote on small closed pedigrees
# ---------------------------------------------------------------------------


def test_ne_long_term_contributions_closed_line():
    """Closed line, N_founders = 2 → c stable at (0.5, 0.5) ⇒ Ne_LTC = 1.

    The master plan's "Ne ≈ 2·N_founders" rule of thumb does not hold for
    the formula ``Ne = 1/(2·Σ c²)``; the analytic value here is 1.
    """
    pg = PedigreeGraph(_build_closed_line(n_gens=5))
    res = ne_long_term_contributions(pg)
    assert res.asymptote_reached
    # Stabilizes at gen 1 (c_per_gen[0] == c_per_gen[1] = (0.5, 0.5)).
    assert res.n_iterations == 1
    assert res.sum_c_squared == pytest.approx(0.5, abs=1e-12)
    assert res.ne == pytest.approx(1.0, abs=1e-12)
    assert res.max_delta_final == pytest.approx(0.0, abs=1e-12)


def test_ne_long_term_contributions_4_founders_symmetric():
    """4 founders → 4 gen-1 individuals with each founder seen by exactly 2.

    Each gen-1 individual has c-vector with two 0.5s and two 0s; the
    cohort mean is uniform 0.25.  Σ c² = 4·0.25² = 0.25 ⇒ Ne_LTC = 2.
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
    pg = PedigreeGraph(df)
    res = ne_long_term_contributions(pg)
    assert res.asymptote_reached
    assert res.sum_c_squared == pytest.approx(0.25, abs=1e-12)
    assert res.ne == pytest.approx(2.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Step 2 — Ne_H (Hill 1979): regression sentinel for L=1 collapse to Ne_V
# ---------------------------------------------------------------------------


def test_ne_hill_collapses_to_ne_v():
    """Discrete-generation sentinel: Hill 1979 reduces to Ne_V at L=1."""
    pg = PedigreeGraph(_build_closed_line(n_gens=5))
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
    pg = PedigreeGraph(df)
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
    pg = PedigreeGraph(df)
    res = ne_hill_overlapping(pg)
    assert res.Vk_m == pytest.approx(8.0, abs=1e-9)
    assert res.Vk_f == pytest.approx(8.0, abs=1e-9)
    assert res.ne == pytest.approx(16.0, rel=1e-9)


def test_ne_hill_monotonic_in_sigma_m():
    """Ne_H decreases as σ²_m increases (all else equal)."""
    pg_lo = PedigreeGraph(_toy_birth_year_pedigree(paternity="balanced"))
    pg_hi = PedigreeGraph(_toy_birth_year_pedigree(paternity="skewed"))
    ne_lo = ne_hill_overlapping(pg_lo).ne
    ne_hi = ne_hill_overlapping(pg_hi).ne
    assert ne_lo is not None
    assert ne_hi is not None
    assert ne_lo > ne_hi


def test_ne_hill_linear_in_T():
    """Ne_H scales linearly in T when σ²_m, σ²_f, N1 are fixed."""
    pg_t10 = PedigreeGraph(_toy_birth_year_pedigree(cohort_a=1900, cohort_b=1910))
    pg_t20 = PedigreeGraph(_toy_birth_year_pedigree(cohort_a=1900, cohort_b=1920))
    ne_t10 = ne_hill_overlapping(pg_t10).ne
    ne_t20 = ne_hill_overlapping(pg_t20).ne
    assert ne_t10 is not None
    assert ne_t20 is not None
    assert ne_t20 == pytest.approx(2.0 * ne_t10, rel=1e-9)


def test_ne_hill_sex_symmetry():
    """Swapping (N_m, σ²_m) ↔ (N_f, σ²_f) leaves Ne unchanged."""
    pg_m = PedigreeGraph(_toy_birth_year_pedigree(paternity="male_skewed_only"))
    pg_f = PedigreeGraph(_toy_birth_year_pedigree(paternity="female_skewed_only"))
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
        {"id": 4 + i, "sex": 1 if i < 2 else 0, "generation": 1, "birth_year": 1910,
         "father": f, "mother": m}
        for i, (f, m) in enumerate(pairs_1910)
    )
    # Cohort 1920 — 4 kids of cohort-1910 parents
    pairs_1920 = [(4, 6), (5, 7), (4, 6), (5, 7)]
    records.extend(
        {"id": 8 + i, "sex": 1 if i < 2 else 0, "generation": 2, "birth_year": 1920,
         "father": f, "mother": m}
        for i, (f, m) in enumerate(pairs_1920)
    )
    df = _df_by(records)
    pg = PedigreeGraph(df)
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
    pg = PedigreeGraph(_toy_birth_year_pedigree(cohort_a=1900, cohort_b=1910))
    res = ne_hill_overlapping(pg)
    assert res.age_table is not None
    # All father-child edges are 10y; same for mothers.
    np.testing.assert_array_equal(res.age_table["ages_m"], [10])
    np.testing.assert_array_equal(res.age_table["ages_f"], [10])
    # 4 offspring → 4 father edges and 4 mother edges.
    np.testing.assert_array_equal(res.age_table["offspring_count_m"], [4])
    np.testing.assert_array_equal(res.age_table["offspring_count_f"], [4])
    assert res.n_offspring_pairs == 8  # 4 mother + 4 father edges


def test_ne_hill_threaded_matches_serial_with_birth_year():
    """compute_all_ne n_threads=2 matches n_threads=1 for birth-year branch.

    Targeted regression for the hill_from_variance bypass risk that
    existed before Step 7.
    """
    pg = PedigreeGraph(_toy_birth_year_pedigree(paternity="skewed"))
    r1 = compute_all_ne(pg, n_threads=1)["ne_hill_overlapping"]
    r2 = compute_all_ne(pg, n_threads=2)["ne_hill_overlapping"]
    assert r1.collapses_to_ne_v == r2.collapses_to_ne_v is False
    assert r1.ne == pytest.approx(r2.ne, rel=1e-12)
    assert r1.Vk_m == pytest.approx(r2.Vk_m, rel=1e-12)


def test_ne_hill_serializes_to_dict():
    """to_dict() yields a YAML-ready dict including new fields."""
    pg = PedigreeGraph(_toy_birth_year_pedigree(paternity="balanced"))
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


def test_sex_specific_family_table_compact_mode_unchanged():
    """Default invocation reproduces pre-refactor behavior exactly."""
    from pedigree_graph._effective_size import _sex_specific_family_table

    df = _build_closed_line(n_gens=5)
    pg = PedigreeGraph(df)
    table = _sex_specific_family_table(
        np.asarray(pg.mother),
        np.asarray(pg.father),
        np.asarray(pg.sex),
        np.asarray(pg.generation),
    )
    # Keys are 0..g_max-1 (g_max excluded).
    g_max = int(np.asarray(pg.generation).max())
    assert set(table.keys()) == set(range(g_max))
    # Each cohort has the expected sex counts (1 male + 1 female per gen).
    for p in range(g_max):
        assert len(table[p]["males_in_parent_gen"]) == 1
        assert len(table[p]["females_in_parent_gen"]) == 1


def test_sex_specific_family_table_arbitrary_mode_includes_max_label():
    """Arbitrary mode keys on actual labels and includes the maximum."""
    from pedigree_graph._effective_size import _sex_specific_family_table

    df = _build_closed_line(n_gens=5)
    pg = PedigreeGraph(df)
    # Use a fake "birth_year" derived from generation + a constant offset
    # so cohort labels are non-contiguous from zero.
    fake_birth_year = (np.asarray(pg.generation) + 1970).astype(np.int32)
    table = _sex_specific_family_table(
        np.asarray(pg.mother),
        np.asarray(pg.father),
        np.asarray(pg.sex),
        np.asarray(pg.generation),
        cohort=fake_birth_year,
        cohort_mode="arbitrary",
    )
    # Keys are 1970..1975 (g_max INCLUDED).
    assert set(table.keys()) == {1970, 1971, 1972, 1973, 1974, 1975}
    # The youngest cohort (1975 = gen 5) has no offspring → all-zero k's.
    np.testing.assert_array_equal(table[1975]["k_mm"], [0])
    np.testing.assert_array_equal(table[1975]["k_mf"], [0])
    np.testing.assert_array_equal(table[1975]["k_fm"], [0])
    np.testing.assert_array_equal(table[1975]["k_ff"], [0])


def test_sex_specific_family_table_arbitrary_mode_filters_sentinels():
    """Cohort labels of ``-1`` are excluded from the output dict."""
    from pedigree_graph._effective_size import _sex_specific_family_table

    df = _build_closed_line(n_gens=3)
    pg = PedigreeGraph(df)
    cohort = np.asarray(pg.generation).copy()
    cohort[0] = -1  # mark one founder as unknown cohort
    table = _sex_specific_family_table(
        np.asarray(pg.mother),
        np.asarray(pg.father),
        np.asarray(pg.sex),
        np.asarray(pg.generation),
        cohort=cohort,
        cohort_mode="arbitrary",
    )
    assert -1 not in table
    # The id=0 founder (male) is no longer in cohort 0, so cohort 0 has
    # only the female founder.
    assert len(table[0]["males_in_parent_gen"]) == 0
    assert len(table[0]["females_in_parent_gen"]) == 1


def test_sex_specific_family_table_arbitrary_mode_requires_cohort():
    """Missing cohort argument in arbitrary mode raises ValueError."""
    from pedigree_graph._effective_size import _sex_specific_family_table

    pg = PedigreeGraph(_build_closed_line(n_gens=3))
    with pytest.raises(ValueError, match="cohort argument"):
        _sex_specific_family_table(
            np.asarray(pg.mother),
            np.asarray(pg.father),
            np.asarray(pg.sex),
            np.asarray(pg.generation),
            cohort_mode="arbitrary",
        )


def test_sex_specific_family_table_invalid_mode_raises():
    from pedigree_graph._effective_size import _sex_specific_family_table

    pg = PedigreeGraph(_build_closed_line(n_gens=3))
    with pytest.raises(ValueError, match="cohort_mode must be"):
        _sex_specific_family_table(
            np.asarray(pg.mother),
            np.asarray(pg.father),
            np.asarray(pg.sex),
            np.asarray(pg.generation),
            cohort_mode="bogus",
        )


# ---------------------------------------------------------------------------
# Step 2 — Ne_CT (Caballero & Toro 2002): closed-line round-trip
# ---------------------------------------------------------------------------


def test_ne_caballero_toro_closed_line_self_coancestry():
    """Closed line: per-gen mean self-coancestry follows (1 + F_g)/2.

    Both founders descend from every gen ≥ 1 (closed line), so
    f̄_s,g = (1 + F_g)/2.  Hand values:
      gen 1: F=0     ⇒ 0.5
      gen 2: F=0.25  ⇒ 0.625
      gen 3: F=0.375 ⇒ 0.6875
      gen 4: F=0.5   ⇒ 0.75
      gen 5: F=0.59375 ⇒ 0.796875
    """
    pg = PedigreeGraph(_build_closed_line(n_gens=5))
    res = ne_caballero_toro(pg)
    expected_fs = np.array([0.5, 0.625, 0.6875, 0.75, 0.796875])
    np.testing.assert_allclose(res.mean_self_coancestry_per_gen[1:], expected_fs, atol=1e-12)
    np.testing.assert_array_equal(res.n_founders_with_descendants_per_gen[1:], [2, 2, 2, 2, 2])
    # Aggregate slope-derived Ne should be in the same ballpark as Ne_I (~2–3).
    assert res.ne is not None
    assert 1.5 < res.ne < 4.0


# ---------------------------------------------------------------------------
# Step 2 — compute_all_ne entry point
# ---------------------------------------------------------------------------


def test_compute_all_ne_returns_eight_keys():
    """compute_all_ne dispatches every estimator with the K and contribution caches."""
    pg = PedigreeGraph(_build_closed_line(n_gens=5))
    results = compute_all_ne(pg)
    expected_keys = {
        "ne_inbreeding",
        "ne_coancestry",
        "ne_variance_family_size",
        "ne_sex_ratio",
        "ne_individual_delta_f",
        "ne_long_term_contributions",
        "ne_hill_overlapping",
        "ne_caballero_toro",
    }
    assert set(results.keys()) == expected_keys
    # Every entry serializes to a YAML-ready dict.
    for k, r in results.items():
        d = r.to_dict()
        assert isinstance(d, dict)
        assert "ne" in d, f"{k} missing 'ne' field"


def test_compute_all_ne_threaded_matches_serial():
    """Threaded estimator dispatch preserves the serial result payloads."""
    rng = np.random.default_rng(2027)
    df = _build_random_mating_pedigree(rng, n_male=12, n_female=12, n_offspring=48)

    serial = {k: v.to_dict() for k, v in compute_all_ne(PedigreeGraph(df), n_threads=1).items()}
    threaded = {k: v.to_dict() for k, v in compute_all_ne(PedigreeGraph(df), n_threads=4).items()}

    assert threaded == serial


def _streaming_theta(pg: PedigreeGraph) -> np.ndarray:
    """Helper: compute θ̄_g via the streaming path (no K materialization)."""
    return _compute_theta_per_gen(
        pg.n,
        np.asarray(pg.mother, dtype=np.int32),
        np.asarray(pg.father, dtype=np.int32),
        np.asarray(pg.twin, dtype=np.int32),
        np.asarray(pg.generation, dtype=np.int32),
        0.0,
    )


def test_streaming_theta_matches_K_path_toy1():
    """Streaming θ̄_g must equal the K-path on toy 1 within float tolerance.

    Different summation orders (row-major streaming vs. col-major COO)
    preclude bit-identical results in general; ``rtol=0, atol=1e-12`` is
    the tightest tolerance the float64 accumulator can guarantee.
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
    pg = PedigreeGraph(df)
    theta_k = _per_gen_mean_kinship(
        pg.kinship_matrix(),
        np.asarray(pg.generation),
        np.asarray(pg.twin),
    )
    theta_stream = _streaming_theta(pg)
    # NaN positions must match.
    assert np.array_equal(np.isnan(theta_k), np.isnan(theta_stream))
    # Non-NaN values must agree.
    mask = ~np.isnan(theta_k)
    assert np.allclose(theta_k[mask], theta_stream[mask], rtol=0, atol=1e-12)


def test_streaming_theta_matches_K_path_random_mating():
    """Streaming θ̄_g must equal K-path on a multi-gen random-mating pedigree."""
    rng = np.random.default_rng(2026)
    df = _build_random_mating_pedigree(rng, n_male=12, n_female=12, n_offspring=48)
    pg = PedigreeGraph(df)
    theta_k = _per_gen_mean_kinship(
        pg.kinship_matrix(),
        np.asarray(pg.generation),
        np.asarray(pg.twin),
    )
    theta_stream = _streaming_theta(pg)
    assert np.array_equal(np.isnan(theta_k), np.isnan(theta_stream))
    mask = ~np.isnan(theta_k)
    assert np.allclose(theta_k[mask], theta_stream[mask], rtol=0, atol=1e-12)


def test_per_gen_mean_kinship_reuses_cached_K():
    """When K is already cached, per_gen_mean_kinship avoids a fresh DP."""
    rng = np.random.default_rng(2032)
    df = _build_random_mating_pedigree(rng, n_male=8, n_female=8, n_offspring=32)
    pg = PedigreeGraph(df)

    # Force K build first; this populates pg._kinship_cache[0.0].
    _ = pg.kinship_matrix()
    # Now θ̄ via the public API must come from the cached K, not a fresh DP.
    theta_via_cache = pg.per_gen_mean_kinship()
    # Compare against the streaming path directly.
    theta_streamed = _streaming_theta(pg)
    assert np.array_equal(np.isnan(theta_via_cache), np.isnan(theta_streamed))
    mask = ~np.isnan(theta_via_cache)
    assert np.allclose(theta_via_cache[mask], theta_streamed[mask], rtol=0, atol=1e-12)


def test_per_gen_mean_kinship_cached_per_threshold():
    """Cache is keyed by min_kinship — different thresholds get fresh results."""
    rng = np.random.default_rng(2031)
    df = _build_random_mating_pedigree(rng, n_male=8, n_female=8, n_offspring=32)
    pg = PedigreeGraph(df)

    theta_0 = pg.per_gen_mean_kinship()  # min_kinship=0.0
    theta_0_cached = pg.per_gen_mean_kinship()  # cache hit
    # Same object returned (cache hit, not recomputed).
    assert theta_0 is theta_0_cached

    # A different threshold must produce a different array (different
    # cache slot).  Use a threshold high enough to prune some entries.
    theta_pruned = pg.per_gen_mean_kinship(min_kinship=0.05)
    assert theta_pruned is not theta_0
    # Cache stores both.
    assert 0.0 in pg._theta_per_gen_cache
    assert 0.05 in pg._theta_per_gen_cache


def test_streaming_theta_helper_directly_bit_identical_to_self():
    """_per_gen_mean_kinship_from_dp must be deterministic across calls."""
    rng = np.random.default_rng(2030)
    df = _build_random_mating_pedigree(rng, n_male=8, n_female=8, n_offspring=32)
    pg = PedigreeGraph(df)
    a = _streaming_theta(pg)
    b = _streaming_theta(pg)
    # Same inputs in same order — bit-identical accumulator state.
    assert np.array_equal(np.isnan(a), np.isnan(b))
    mask = ~np.isnan(a)
    assert (a[mask] == b[mask]).all()


def test_compute_all_ne_threaded_matches_serial_when_coancestry_skipped():
    """Threaded dispatch also preserves the large-pedigree skip branch."""
    rng = np.random.default_rng(2028)
    df = _build_random_mating_pedigree(rng, n_male=12, n_female=12, n_offspring=48)

    serial = {
        k: v.to_dict()
        for k, v in compute_all_ne(PedigreeGraph(df), skip_ne_coancestry=True, n_threads=1).items()
    }
    threaded = {
        k: v.to_dict()
        for k, v in compute_all_ne(PedigreeGraph(df), skip_ne_coancestry=True, n_threads=4).items()
    }

    assert threaded == serial
