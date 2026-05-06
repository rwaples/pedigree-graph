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
from pedigree_graph._kinship_kernel import _compute_eqg


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
