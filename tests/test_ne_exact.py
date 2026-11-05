"""Ne estimators against hand arithmetic, cited formulas, and exact transformations.

Every expected value is either derived by hand on a constructed pedigree, in
the test, from a formula ADR 0012 lists as read (Gutiérrez et al. 2008,
Caballero & Toro 2000, Wray & Thompson 1990), or a factor derived here from
how a transformation acts on those formulas.  Nothing is computed by calling
the implementation's own formula.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest
from _support import _build_closed_line, _df, _random_mating, property_budget
from hypothesis import example, given
from hypothesis import strategies as st

from pedigree_graph import PedigreeGraph
from pedigree_graph.effective_size import (
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

#: Closed-line full-sib mating, F_{t+1} = (1 + 2F_t + F_{t-1}) / 4 from F_0 = F_1 = 0.
_CLOSED_LINE_F = [0.0, 0.0, 0.25, 0.375, 0.5, 0.59375]


def test_the_closed_line_rates_are_the_hand_values():
    # Gutiérrez Eq. 1 at a one-label gap: Δ = (F_b − F_a)/(1 − F_a), Ne = 1/(2Δ).
    # Δ = 0, 1/4, 1/6, 1/5, 3/16, so Ne = NaN (no rate), 2, 3, 5/2, 8/3.
    # In float64 each quotient rounds back to that rational, 8/3 to its
    # nearest double.  Rejects Δ reported for 1/(2Δ) and a first transition
    # given a rate.
    pg = PedigreeGraph.from_frame(_build_closed_line(n_gens=5))
    result = ne_inbreeding(pg)
    np.testing.assert_array_equal(result.mean_f_per_gen, _CLOSED_LINE_F)
    np.testing.assert_array_equal(result.ne_per_gen, [np.nan, 2.0, 3.0, 2.5, 8 / 3])

    # The scalar is −1/(2·slope), slope the OLS slope of ln(1 − F̄) on
    # t = label − first label over the five cohorts after the baseline.  With
    # t = 1..5, Σ(t − 3) = 0 and Σ(t − 3)² = 10, so the slope is
    # Σ(t − 3)·ln(1 − F_t) / 10.  Both it and np.polyfit are within a few
    # float64 ulps of the true slope: five faithfully rounded logs of
    # magnitude < 1, summed with weights |t − 3| <= 2, and a backward-stable
    # 5-by-2 least squares; 1e-14 is above either error and far below the
    # 0.2207 slope.  Rejects fitting the baseline cohort and Ne = −1/slope.
    t = np.arange(1, 6, dtype=np.float64)
    y = np.log1p(-np.array(_CLOSED_LINE_F[1:]))
    slope = math.fsum((t - 3.0) * y) / 10.0
    assert result.n_generations_used == 5
    assert result.slope == pytest.approx(slope, rel=0, abs=1e-14)
    assert result.ne == pytest.approx(-1.0 / (2.0 * slope), rel=1e-13)


def test_coancestry_leads_inbreeding_by_one_cohort_on_the_closed_line():
    # Each cohort is one mated pair, so its θ̄ is that pair's kinship, which
    # is the F of their children: θ̄_g = F_{g+1}, with θ̄_5 = F_6 =
    # (1 + 2·0.59375 + 0.5)/4 = 0.671875.  Rejects a θ̄ shifted onto the
    # children's cohort.
    pg = PedigreeGraph.from_frame(_build_closed_line(n_gens=5))
    theta = ne_coancestry(pg).mean_theta_per_gen
    np.testing.assert_array_equal(theta, [*_CLOSED_LINE_F[1:], 0.671875])


def test_an_mz_selfing_chain_has_the_exact_rate():
    # Co-twins mated to each other each generation: F_t = (1 + F_{t−1})/2 from
    # F_0 = 0, so F_t = 1 − 2**−t, Δ = 1/2 at every step, Ne = 1 exactly, and
    # ln(1 − F_t) = −t·ln 2 is exactly linear with slope −ln 2.  Rejects a
    # co-twin pair treated as two genomes, which gives full-sib mating instead.
    ids, mother, father, twin, generation = [], [], [], [], []
    for g in range(7):
        parents = (-1, -1) if g == 0 else (2 * g - 2, 2 * g - 1)
        ids += [2 * g, 2 * g + 1]
        mother += [parents[0]] * 2
        father += [parents[1]] * 2
        twin += [2 * g + 1, 2 * g]
        generation += [g, g]
    pg = PedigreeGraph.from_frame(
        {"id": ids, "mother": mother, "father": father, "twin": twin, "generation": generation}
    )
    result = ne_inbreeding(pg)
    np.testing.assert_array_equal(result.mean_f_per_gen, [1.0 - 2.0**-g for g in range(7)])
    np.testing.assert_array_equal(result.ne_per_gen, np.ones(6))
    assert result.slope == pytest.approx(-math.log(2.0), rel=0, abs=1e-15)
    assert result.ne == pytest.approx(1.0 / (2.0 * math.log(2.0)), rel=1e-14)


def test_a_late_founder_counts_in_long_term_contributions():
    # Row 4 (generation 2) has mother 3, a founder first seen in generation 1,
    # and father 2, the child of founders 0 and 1: c = (¼, ¼, ½) over
    # founders 0, 1, 3, so Σc² = 3/8, N_ef = 1/Σc² = 8/3 (Caballero & Toro
    # 2000 Eq. 19) and Ne = 2/Σc² = 16/3 (Wray & Thompson 1990 Eq. 31).
    # Rejects founders taken to be generation 0.
    pg = PedigreeGraph.from_frame(
        _df(
            [
                {"id": 0, "sex": 1, "generation": 0},
                {"id": 1, "sex": 0, "generation": 0},
                {"id": 2, "sex": 1, "generation": 1, "mother": 1, "father": 0},
                {"id": 3, "sex": 0, "generation": 1},
                {"id": 4, "sex": 1, "generation": 2, "mother": 3, "father": 2},
            ]
        )
    )
    result = ne_long_term_contributions(pg)
    assert result.final_generation == 2
    assert result.sum_c_squared == 0.375
    assert result.n_effective_founders == 8 / 3
    assert result.ne == 16 / 3


def _dated(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns((1900 + 25 * pl.col("generation")).alias("birth_year"))


def _replicated(frame: pl.DataFrame) -> pl.DataFrame:
    shift = int(frame["id"].max()) + 1
    shifted = frame.with_columns(
        [pl.col("id") + shift]
        + [
            pl.when(pl.col(c) >= 0).then(pl.col(c) + shift).otherwise(pl.col(c)).alias(c)
            for c in ("mother", "father", "twin")
        ]
    )
    return pl.concat([frame, shifted])


@property_budget(50)
@given(
    half=st.integers(min_value=2, max_value=5),
    n_gens=st.integers(min_value=1, max_value=5),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
@example(half=5, n_gens=5, seed=1428)  # f̄ halves only to 1 ulp here
@example(half=3, n_gens=4, seed=11)  # the fixture this test replaced
def test_a_disjoint_replicate_scales_each_estimate_by_its_derived_factor(half, n_gens, seed):
    # Two unrelated copies with the same labels, of a random-mating pedigree
    # of 2·half rows (half of each sex) in each of n_gens + 1 cohorts.  Per
    # cohort: every count doubles and no cross-copy pair is related.
    # * F̄ and each individual's ΔF are unchanged, so Ne_F and Ne_ΔF are.
    # * Ne_sr = 4·Nm·Nf/(Nm + Nf) doubles, exactly (every step scales by 2).
    # * Founder contributions halve over twice as many founders, so Σc²
    #   halves and N_ef = 1/Σc², Ne_LTC = 2/Σc² double.  Each c is a cohort
    #   mean, halved exactly, but Σc² over F founders and over 2F sums in a
    #   different order: recursive summation of nonnegative terms is within
    #   (terms − 1)·2**−53 relative, so the two differ by under 3F·2**−53.
    # * Group coancestry f̄ = Σ_{i,j} φ_ij / n² over all ordered genome pairs:
    #   the sum doubles and n² quadruples, so f̄ halves, but not exactly:
    #   the estimator rebuilds the pair sum as round(round(S/P)·P) from the
    #   kinship summary's mean (``_ne_group_coancestry.py``), adds Σs over n
    #   genomes, and divides.  Each side is within (n + 4)·2**−53 relative
    #   of its rational (two roundings for the pair sum, n − 1 for Σs, one
    #   each for the sum and the quotient), so they differ by under
    #   (2n_max + 8)·2**−53, n_max the replicate's largest cohort.
    # * θ̄ averages the n(n−1)/2 distinct pairs: the kinship sum doubles over
    #   2n(2n−1)/2 pairs, so θ̄·pairs doubles.
    # Rejects wrong denominators: pairs for n², rows for genomes.
    frame = _random_mating(n_per_gen=2 * half, n_gens=n_gens, seed=seed)
    one, two = PedigreeGraph.from_frame(frame), PedigreeGraph.from_frame(_replicated(frame))

    f1, f2 = ne_inbreeding(one), ne_inbreeding(two)
    np.testing.assert_array_equal(f2.mean_f_per_gen, f1.mean_f_per_gen)
    np.testing.assert_array_equal(f2.ne_per_gen, f1.ne_per_gen)
    d1, d2 = ne_individual_delta_f(one), ne_individual_delta_f(two)
    np.testing.assert_array_equal(d2.mean_eqg_per_gen, d1.mean_eqg_per_gen)
    np.testing.assert_array_equal(d2.n_used_per_gen, 2 * d1.n_used_per_gen)

    s1, s2 = ne_sex_ratio(one), ne_sex_ratio(two)
    np.testing.assert_array_equal(s2.ne_per_gen, 2 * s1.ne_per_gen)

    l1, l2 = ne_long_term_contributions(one), ne_long_term_contributions(two)
    founders = int(frame.filter((pl.col("mother") < 0) & (pl.col("father") < 0)).height)
    rtol = 3 * founders * 2.0**-53
    assert l2.sum_c_squared == pytest.approx(l1.sum_c_squared / 2, rel=rtol)
    assert l2.n_effective_founders == pytest.approx(2 * l1.n_effective_founders, rel=rtol + 2.0**-52)
    assert l2.ne == pytest.approx(2 * l1.ne, rel=rtol + 2.0**-52)

    g1, g2 = ne_group_coancestry(one), ne_group_coancestry(two)
    np.testing.assert_array_equal(g2.n_genomes_per_gen, 2 * g1.n_genomes_per_gen)
    gc_rtol = (2 * int(g2.n_genomes_per_gen.max()) + 8) * 2.0**-53
    np.testing.assert_allclose(
        g2.mean_group_coancestry_per_gen, g1.mean_group_coancestry_per_gen / 2, rtol=gc_rtol, atol=0
    )

    theta1, theta2 = one.mean_kinship_by_generation(), two.mean_kinship_by_generation()
    genomes = g1.n_genomes_per_gen
    np.testing.assert_array_equal(theta1.pair_counts, genomes * (genomes - 1) // 2)
    np.testing.assert_array_equal(theta2.pair_counts, (2 * genomes) * (2 * genomes - 1) // 2)
    # Both sums are exact (dyadic kinships, few pairs); each mean is one
    # correctly rounded division, so the products differ by at most 2 ulps.
    np.testing.assert_allclose(
        theta2.mean_kinship * theta2.pair_counts, 2 * theta1.mean_kinship * theta1.pair_counts, rtol=4 * 2.0**-52
    )

    # Ne_V's family-size moments, a regression pin on the ddof=1 variance the
    # implementation uses (Caballero 1994's S² carries a Gaussian correction
    # but sets no per-sex ddof).  Duplicating N parents of each sex keeps
    # every mean and scales each sample (co)variance by 2(N − 1)/(2N − 1).
    # Ne_V itself mixes that with the unscaled segregation term, so it has no
    # single factor; the exact Ne_V tests below cover the formula.
    counts = frame.group_by("generation", "sex").len()
    assert counts["len"].n_unique() == 1
    n = int(counts["len"][0])
    v1, v2 = ne_variance_family_size(one), ne_variance_family_size(two)
    for name in ("v_mm", "v_mf", "v_fm", "v_ff", "cov_m", "cov_f"):
        np.testing.assert_allclose(
            getattr(v2, name),
            2 * (n - 1) / (2 * n - 1) * getattr(v1, name),
            rtol=8 * 2.0**-52,
            atol=1e-15,
            err_msg=name,
        )


def _mirrored(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns(
        pl.col("father").alias("mother"), pl.col("mother").alias("father"), (1 - pl.col("sex")).alias("sex")
    )


def test_swapping_the_sexes_swaps_every_per_sex_field_and_no_estimate():
    # Ne_V sums one term per sex, Ne_sr = 4·Nm·Nf/(Nm + Nf), and Hill combines
    # per-sex sizes as 4·Ne_m·Ne_f/(Ne_m + Ne_f) with the same per-sex
    # formula (``_ne_hill.py``): each is symmetric under the swap, and float
    # addition and multiplication commute, so each is bit-identical.  The
    # per-sex fields trade places, which catches a sex-coding defect the
    # symmetric Ne hides.  The fixture has unequal sex counts per cohort.
    frame = _dated(_random_mating(n_per_gen=7, n_gens=4, seed=5))
    base, mirror = PedigreeGraph.from_frame(frame), PedigreeGraph.from_frame(_mirrored(frame))

    s1, s2 = ne_sex_ratio(base), ne_sex_ratio(mirror)
    assert not np.array_equal(s1.n_male_per_gen, s1.n_female_per_gen)
    np.testing.assert_array_equal(s2.n_male_per_gen, s1.n_female_per_gen)
    np.testing.assert_array_equal(s2.n_female_per_gen, s1.n_male_per_gen)
    np.testing.assert_array_equal(s2.ne_per_gen, s1.ne_per_gen)

    v1, v2 = ne_variance_family_size(base), ne_variance_family_size(mirror)
    for a, b in (("v_mm", "v_ff"), ("v_mf", "v_fm"), ("cov_m", "cov_f")):
        np.testing.assert_array_equal(getattr(v2, a), getattr(v1, b), err_msg=a)
        np.testing.assert_array_equal(getattr(v2, b), getattr(v1, a), err_msg=b)
    np.testing.assert_array_equal(v2.ne_per_transition, v1.ne_per_transition)

    h1, h2 = ne_hill_overlapping(base), ne_hill_overlapping(mirror)
    assert h1.ne is not None
    assert (h2.kbar_m, h2.kbar_f) == (h1.kbar_f, h1.kbar_m)
    assert h2.ne == h1.ne


def _plain_equal(a: object, b: object) -> bool:
    """Equality of ``to_dict`` output, NaN equal to NaN."""
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_plain_equal(a[k], b[k]) for k in a)
    if isinstance(a, list | tuple) and isinstance(b, list | tuple):
        return len(a) == len(b) and all(_plain_equal(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


@pytest.mark.parametrize("mapping", ["above_int32", "reversed"])
def test_every_estimate_is_independent_of_id_values(mapping):
    # Row order and labels stay; only the id values move.  Rejects an
    # estimator that groups, sorts or dedups by id.
    frame = _dated(_random_mating(n_per_gen=6, n_gens=4, seed=3))
    top = int(frame["id"].max())
    new = (lambda c: 2**40 + 7 * c) if mapping == "above_int32" else (lambda c: top - c)
    relabelled = frame.with_columns(
        [new(pl.col("id")).alias("id")]
        + [
            pl.when(pl.col(c) >= 0).then(new(pl.col(c))).otherwise(pl.col(c)).alias(c)
            for c in ("mother", "father", "twin")
        ]
    )
    before = estimate_effective_sizes(PedigreeGraph.from_frame(frame)).to_dict()
    after = estimate_effective_sizes(PedigreeGraph.from_frame(relabelled)).to_dict()
    assert _plain_equal(after, before)


def _replacing_population(n_couples: int, n_gens: int, shape: float, seed: int) -> pl.DataFrame:
    """Monogamous couples whose cohorts replace themselves exactly.

    Every cohort has ``n_couples`` males and ``n_couples`` females, paired
    once.  Couple family sizes are multinomial over gamma weights, so V(k)
    exceeds the Wright-Fisher 2 for small ``shape``.  Each cohort has exactly
    ``n_couples`` sons and ``n_couples`` daughters, so every pathway mean
    (sons per father, daughters per father, ...) is 1: Hill 1979's stable case.
    """
    rng = np.random.default_rng(seed)
    males = list(range(n_couples))
    females = list(range(n_couples, 2 * n_couples))
    rows: list[dict] = [{"id": i, "sex": 1, "generation": 0} for i in males]
    rows += [{"id": i, "sex": 0, "generation": 0} for i in females]
    next_id = 2 * n_couples
    for g in range(1, n_gens + 1):
        sizes = rng.multinomial(2 * n_couples, rng.dirichlet(np.full(n_couples, shape)))
        sexes = rng.permutation([1] * n_couples + [0] * n_couples)
        new_m: list[int] = []
        new_f: list[int] = []
        for c, sx in zip(np.repeat(np.arange(n_couples), sizes), sexes, strict=True):
            rows.append({"id": next_id, "sex": int(sx), "generation": g, "father": males[c], "mother": females[c]})
            (new_m if sx == 1 else new_f).append(next_id)
            next_id += 1
        males, females = new_m, new_f
    return _df(rows)


@pytest.mark.parametrize(("shape", "seed"), [(0.3, 1), (1.0, 2), (50.0, 3)])
def test_ne_v_is_hill_1979_eq_10_where_cohorts_replace_themselves(shape, seed):
    # Hill 1979 eq. 10 at L = 1: Ne = 8N/(σ²_m + σ²_f + 4), N the cohort size.
    # The fixture is the stable case it assumes (M = F = N/2, every pathway
    # mean 1), and monogamy makes σ²_m = σ²_f = V(k) of the couples.  Rejects
    # 2N/V(k), which agrees only at V(k) = 2, and a bare "+2" without the
    # coefficient change, which halves Ne.
    n_couples = 40
    frame = _replacing_population(n_couples, n_gens=4, shape=shape, seed=seed)
    result = ne_variance_family_size(PedigreeGraph.from_frame(frame))
    for g in range(4):
        children = frame.filter(pl.col("generation") == g + 1)
        fathers = frame.filter((pl.col("generation") == g) & (pl.col("sex") == 1))["id"]
        k = np.array([(children["father"] == f).sum() for f in fathers], dtype=np.float64)
        expected = 8 * (2 * n_couples) / (2 * k.var(ddof=1) + 4)
        assert result.ne_per_transition[g] == pytest.approx(expected, rel=1e-12)
    assert np.isnan(result.ne_per_transition[4])


def test_ne_v_weights_the_pathways_by_the_sex_ratio_as_caballero_1994_eq_19():
    # Two males and four females, replaced by two sons and four daughters.
    # Fathers 0 and 1 each have one son and two daughters, so every male
    # moment is 0.  Mothers: k_fm = (1, 0, 1, 0), k_ff = (1, 1, 1, 1), so
    # S²_fm = 1/3 (ddof=1), S²_ff = 0, S_fm,ff = 0.  Caballero 1994 eq. 19
    # (= Hill 1979 eq. 9 at L = 1) with N_m = 2, N_f = 4:
    #   1/Ne = 2/(16·2) + [2 + (4/2)²·(1/3)]/(16·4) = 1/16 + 5/96 = 11/96.
    # Rejects the unweighted σ²_f, which gives 1/16 + (7/3)/64.
    rows: list[dict] = [{"id": i, "sex": 1, "generation": 0} for i in (0, 1)]
    rows += [{"id": i, "sex": 0, "generation": 0} for i in (2, 3, 4, 5)]
    children = [(0, 2, 1), (0, 2, 0), (0, 3, 0), (1, 4, 1), (1, 4, 0), (1, 5, 0)]
    rows += [
        {"id": 6 + j, "sex": sx, "generation": 1, "father": f, "mother": m} for j, (f, m, sx) in enumerate(children)
    ]
    result = ne_variance_family_size(PedigreeGraph.from_frame(_df(rows)))
    assert result.ne_per_transition[0] == pytest.approx(96 / 11, rel=1e-12)
    assert result.ne == pytest.approx(96 / 11, rel=1e-12)


def test_hill_branches_agree_on_the_same_edges_at_unit_generation_interval():
    # Hill 1979 at L = 1 is the discrete-generation Ne, so dating each cohort
    # one year after its parents must reproduce Ne_V cohort by cohort.  V(k)
    # is far from 2 here, where Ne_V's old 2N/V(k) and the birth-year
    # branch's σ² + 2 disagreed.
    frame = _replacing_population(30, n_gens=5, shape=0.3, seed=7)
    v = ne_variance_family_size(PedigreeGraph.from_frame(frame))
    sentinel = ne_hill_overlapping(PedigreeGraph.from_frame(frame))
    dated = ne_hill_overlapping(PedigreeGraph.from_frame(frame.with_columns(pl.col("generation").alias("birth_year"))))
    assert sentinel.collapses_to_ne_v
    assert sentinel.ne == v.ne
    assert dated.generation_interval == 1.0
    assert dated.cohort_years is not None
    assert dated.ne_per_cohort is not None
    assert dated.n_eligible_cohorts >= 4
    for year, ne_c in zip(dated.cohort_years, dated.ne_per_cohort, strict=True):
        assert ne_c == pytest.approx(v.ne_per_transition[int(year)], rel=1e-12)
