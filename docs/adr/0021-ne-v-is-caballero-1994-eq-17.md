# ADR 0021: Ne_V is Caballero 1994 eq. 17, segregation term included

**Status:** accepted
**Date:** 2026-11-05
**Context:** issue #36 (Ne_V omits Hill's segregation term); simACE#72, whose
drift-only pedigrees measured the defect. Amends ADR 0012, whose policy this
applies to an estimator ADR 0012 did not check, and whose consequence bullet on
`ne_long_term_contributions` described `Ne_V` as "Crow-Kimura-form `2N/V(k)`".

## Context

`ne_variance_family_size` cited "Caballero 1994 eq. 6". Nobody had read it.
Two primary sources have now been read in the original:

* **Hill 1979**, Genetics 92(1):317-322, doi 10.1093/genetics/92.1.317.
  Eq. 8 (monoecious) is `Ne = 4NL/(σ_n² + 2)`. Eq. 9 is the two-sex form for
  `M` males and `F` females born per cohort:

      1/Ne = [2 + σ²_mm + 2(M/F)·cov(mm,mf) + (M/F)²·σ²_mf] / (16ML)
           + [2 + (F/M)²·σ²_fm + 2(F/M)·cov(fm,ff) + σ²_ff] / (16FL)

  At `M = F = N/2` it is eq. 10, `Ne = 8NL/(σ²_m + σ²_f + 4)`. The `2` is
  Mendelian segregation.
* **Caballero 1994**, Heredity 73(6):657-679, doi 10.1038/hdy.1994.174.
  **Eq. 6 is the effective size without selfing (`Ne ≈ N + ½`)**, not a
  family-size formula, so the citation was wrong at the equation number.
  §5.1.2 has the formula the code half-implements. Eq. 17, credited there to
  Crow & Denniston (1988) except for `2α_O`, is

      N_es = 4N_s / { (1/μ_sm + 1/μ_sf)(1 − α_I,s)
                      + (S²_sm/μ²_sm + 2S_sm,sf/(μ_sm μ_sf) + S²_sf/μ²_sf)(1 + α_I,s + 2α_O) }

  with `μ_sm = N_m/N_s` the mean sons per parent of sex `s`. Eq. 18 combines
  the sexes, `1/Ne = 1/(4N_em) + 1/(4N_ef)`. Eq. 19 is its large-population form
  at `α = 0`, "as obtained by Hill (1972b, 1979)". Its sum over the sexes is
  Hill's eq. 9; the per-sex brackets group segregation differently when
  `M ≠ F`.

The code computed `ΔF_s = V_s/(4·N_s·k̄_s²)`, which is eq. 17 without its
`1/μ_sm + 1/μ_sf` segregation term and with one pooled `k̄_s` in place of the
two pathway means. At `k̄ = 2` and equal sexes that is `Ne = 2N/V(k)`, which
equals Hill's `4N/(2 + V(k))` only at `V(k) = 2`. A Wright-Fisher check could
not catch it. Adding only `+2` would also be wrong, because the coefficient
changes too: `V_s/(16N_s)` becomes `(V_s + 2)/(32N_s)` per sex in `ΔF`.

`ne_hill_overlapping`'s two branches disagreed for the same reason. The
sentinel passed Ne_V through and claimed it was Hill at `L = 1`. The birth-year
branch used `σ²_s + 2`, which is eq. 10 per sex, without eq. 9's sex-ratio
weights. Under the old code, they disagreed wherever `V(k) ≠ 2`, and
`tests/test_ne_exact.py` now asserts their agreement.

## Decision

**Ne_V is Caballero 1994 eq. 17 at `α_I = α_O = 0`, combined by eq. 18:**

    D_s = 1/μ_sm + 1/μ_sf + S²_sm/μ_sm² + 2·S_sm,sf/(μ_sm·μ_sf) + S²_sf/μ_sf²
    1/Ne = D_m/(16·N_m) + D_f/(16·N_f)

per parent cohort, with `S²` the ddof=1 sample moments the record already
reports. The aggregate stays the harmonic mean over parent cohorts.

* **Non-constant size (`k̄ ≠ 2`) is a package-defined extension.** Hill
  assumes each cohort replaces itself, and so does Caballero's §5.1.2: it
  adopts constant size just before eq. 17 (p. 663). Eq. 17's pathway means
  `μ` still differ from 1 there when `M ≠ F`. This package plugs in each
  cohort's *observed* `μ` whatever the size change, which scales each
  pathway's offspring to the offspring cohort and makes Ne_V the *variance*
  effective size. The sourced anchor is Caballero's monoecious eq. 10, which
  allows a changing size: at `α = 0` it is `2N_t/(1 + S²/μ_k)`, and the
  monoecious analogue of eq. 17 with observed `μ` reduces to exactly that. Caballero's eqs. 10 and 14 contrast
  that with the *inbreeding* effective size: the variance size follows the
  number of offspring, the inbreeding size the number of parents. That is the
  reference the old code already used, since its `V/k̄²` is the same pathway
  scaling pooled over the two offspring sexes, so this decision keeps
  the reference and adds the missing term. Under Poisson family sizes with `M`
  sons and `F` daughters, the new formula gives the offspring cohort's
  `4MF/(M + F)` and the old one gave `M + F`. They agree at `M = F`. Where each
  cohort replaces itself (`μ_mm = μ_ff = 1`, `μ_mf = F/M`), eq. 17 summed
  over the sexes is Hill eq. 9 exactly, and at `M = F` it is eq. 10.
* **The `α` terms are dropped.** Eq. 19 drops them too. `α_I = −1/(2N_s − 1)`
  under random mating, an `O(1/N)` term, and `α_O` needs the mates' kinship,
  which this estimator does not read.
* **The Hill birth-year branch uses the same `D_s`**, as
  `Ne_s = 4·N1_s·T/D_s`. That is eq. 9 with `L = T`, and on the same edges at
  `T = 1` it agrees with the sentinel cohort by cohort (asserted to 1e-12 in
  `tests/test_ne_exact.py`).
* **`vk_scale=True` is unchanged.** It rescales `σ²_s` to `k̄ = 2` with Waples
  2002 eq. 5 and uses eq. 10's `σ²_s + 2`. Algebraically, in the monoecious
  analogue, Waples' rescaling plus 2 is `(4/k̄)(k̄ − 1 + V/k̄)`, the denominator
  of Caballero's inbreeding size, eq. 14. So the flag selects the
  parent-referenced size, and the default selects the offspring-referenced one.
  The two agree only where each cohort replaces itself with `M = F`. On
  `tests/test_ne_exact.py`'s `M = 2, F = 4` fixture at `T = 1` they give
  `96/11` (default) and `432/49` (scaled). The sentinel ignores the
  flag, as before.
* **A cohort with a zero pathway mean has no Ne_V.** If a sex's parents had no
  sons or no daughters, eq. 17 is undefined. The no-estimate code
  `no_family_size_variance` is renamed to `single_sex_offspring`. The old code
  could no longer fire: with the segregation term, zero family-size variance
  gives `Ne = 2N`, Hill's value for equal family sizes, not "no estimate".
* **The citation moves to Caballero 1994 eqs. 17-18**, with Hill 1979 eqs.
  9-10 as the constant-size special case, in every docstring that said "eq. 6".
  The PDF is in simACE's `external/pdfs/`.

## Consequences

* **Ne_V and Ne_H change numerically.** Persisted results from earlier
  versions are not comparable. The CHANGELOG marks this breaking.
* **Measured on simACE's drift-only ZTP pedigrees**: N = 2000, 4 generations,
  12 replicates, 36 transitions per setting, run through simACE's own fixture
  (`tests/analysis/test_effective_size.py::_drift_only_pedigree`). The
  realized-drift column is simACE#72's pair-sharing measurement.

  | ZTP λ | `2N/V(k)` | Hill `4N/(2+V(k))` | Ne_V before | Ne_V after | realized drift |
  |---|---:|---:|---:|---:|---:|
  | 0.5 | 1469.7 | 1694.3 | 1438 | 1668.5 | 1668 |
  | 2 | 1254.7 | 1542.0 | 1228 | 1532.5 | 1543 |

* **The golden is regenerated.** `tests/data/ne_baseline_0_9` keeps its six
  byte-identical fixtures. Only the `ne_variance_family_size` and
  `ne_hill_overlapping` blocks change, and `closed_line_5` changes in neither,
  because it has no estimate:

  | fixture | old Ne_V | new Ne_V | old Ne_H | new Ne_H |
  |---|---:|---:|---:|---:|
  | `skip_gen` | 18.000000 | 4.571429 | 18.000000 | 4.571429 |
  | `small_pedigree` | 734.088538 | 845.298191 | 734.088538 | 845.298191 |
  | `wf_n20_g4` | 19.459459 | 19.726027 | 19.459459 | 19.726027 |
  | `wf_n40_g5_birth_years` | 38.000000 | 38.974359 | 116.923077 | 116.923077 |
  | `wf_n60_g6` | 66.496815 | 63.081571 | 66.496815 | 63.081571 |

  `skip_gen` moves the most. Its generation-0 fathers have one son between
  them, so `μ_mm = ½`, `D_m = 2 + 1 + ½/¼ = 5`, `D_f = 2`, and `1/Ne = 7/32`.
  The old 18 was more than four times the census of 4.
  `wf_n40_g5_birth_years` runs Hill with `vk_scale=True`, which is unchanged.
* **ADR 0012's `ne_long_term_contributions` expectation simplifies.** That
  bullet derived `2/Ne_LTC = 1/N + 1/Ne_V` from W&T's `4N/(2 + V(k))` and the
  old `2N/V(k)`. With Ne_V now `4N/(2 + V(k))` at constant size, the
  expectation for Ne_LTC under random mating is Ne_V itself. The measurement
  recorded there still stands. The harmonic mean of `N` and `2N/V(k)` is
  `4N/(2 + V(k))`, so it was already agreement with the corrected Ne_V's
  constant-size value.
* **Consumers.** simACE withholds its Ne_V and Ne_H expectations under ZTP
  mating and keeps a canary,
  `tests/analysis/test_effective_size.py::test_pg_ne_v_still_returns_2n_over_vk`,
  that fails against this build. pedsum maps a null Ne_V record to the old
  `no_family_size_variance` code (`pedsum/sections.py`, README, and
  `tests/test_effective_size_cmd.py` "outbred"). Its outbred fixture now gets
  an estimate.

## Rejected

* **Add `+2` to the old formula.** That halves Ne and breaks `V = 2 → N`.
* **Hill eq. 10 with raw `σ²_s` and no `k̄` scaling.** It is faithful only at
  constant size. Under growth it reads the larger mean family size as drift.
* **Waples rescaling as the default.** It answers a different question, the
  inbreeding size, and Ne_V has always been the variance size. The flag keeps
  it available on the birth-year branch.
* **Keep `no_family_size_variance` as the name.** Zero family-size variance is
  now an estimate (`2N`). A code that names a state that gives an estimate
  would lie.
