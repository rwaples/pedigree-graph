# ADR 0012: every named Ne estimator cites a formula someone has read

**Status:** accepted
**Date:** 2026-09-11
**Context:** issue #15, resolved before the 0.9 scientific contract is frozen;
refines the effective-size surface of ADR 0006 and the genome-node pedigree of
ADR 0008. Plan: simACE `plans/pedigree-graph-issue-15-estimator-fidelity.md`.

## Context

The 0.8 effective-size API names eight estimators after published methods.
Primary-source verification found that three of them did not match the papers
they were attributed to, and the slice-6c plan had explicitly deferred that
check ("Reverify this baseline against the primary Caballero–Toro source before
documenting it").

All four cited sources were then obtained and read in full, rather than recalled:

* **Gutiérrez et al. 2008**, Genet Sel Evol 40(4):359–378 — CC-BY, read via PMC2674907.
* **Caballero & Toro 2000**, Genet Res 75(3):331–343.
* **Wray & Thompson 1990**, Genet Res 55(1):41–54.
* **Caballero & Toro 2002**, Cons Genet 3(3):289–299.

Three findings, each cross-checked against a second equation in the same paper:

1. `ne_individual_delta_f` used `ΔF_i = 1 − (1−F_i)^(1/(t−1))` and excluded
   `t ≤ 1`. Gutiérrez Eq. 2 is `1/t`. It also aggregated by harmonic mean of
   per-cohort `Ne`; §2.1 specifies `Ne = 1/(2ΔF̄)` over a reference
   subpopulation. `_compute_eqg` already matched the paper's `t`, so there was
   no compensating index shift.

2. `ne_caballero_toro` averaged descendant self-coancestry `(1+F)/2` within each
   founder's reachable set, averaged that across founders, and regressed against
   a `0.5` baseline. Caballero & Toro 2002 contains no such statistic: its `s`
   feeds the diversity partition, and its effective size (Eq. 14) is a
   contribution-variance formula. Caballero & Toro 2000 verifies the identity
   `f_N = ½(1+F)` but its coancestry-rate baseline is `f̄₀ = 1/(2N)` on *group*
   coancestry (Eq. 3), a different quantity. Measured: where every represented
   founder reaches every cohort member the statistic is **bitwise** `(1+F̄)/2`,
   and `ne_caballero_toro` reproduces `ne_inbreeding` to 15 significant figures
   (`2.321011526332312` vs `2.3210115263323123`). Where founder reach is
   incomplete it differs by a reweighting that decays to zero as reach
   saturates — largest in the early cohorts, where the regression has the most
   leverage, and motivated by no line in any cited paper.

3. `ne_long_term_contributions` reported `1/(2Σc²)`. Wray & Thompson Eq. 31 is
   `Ne ≈ 2N/(μ_r²+σ_r²)` with `μ_r = 1`, i.e. `Ne ≈ 2/Σc²` normalised;
   Caballero & Toro Eq. 19 is `N_ef = 1/Σc²`, and their text states
   `N_ef = Ne/2`. The reported value was 4× below `Ne` and 2× below `N_ef`.
   Measured on random-mating pedigrees: `V₀,ₜ → 1` as W&T predict, and `2/Σc²`
   lands on `N` (123.7 for N=120, 195.1 for N=200). The error survived because
   `tol=1e-6` is tighter than the ~1e-4 fluctuation of contributions, so
   `asymptote_reached` was `False` and `ne` was `None` on every realistic
   pedigree — corroborated by the 0.8 migration record, which notes the
   estimator "reported no estimate under both versions".

## Decision

**A named estimator cites a formula someone has actually read.** An estimator
whose formula cannot be verified against its primary source is renamed to a
package-defined statistic or removed — never shipped under an attribution the
source does not support.

Applied:

* **`ne_individual_delta_f`** follows Gutiérrez Eq. 2 exactly: exponent `1/t`,
  eligibility `t > 0`, scalar `1/(2ΔF̄)` over a reference subpopulation
  defaulting to the last observed cohort and overridable by graph rows, plus
  the paper's standard error. Its result record also carries
  `ne_unrelated_founders`, a **package-defined** companion statistic that the
  policy sentence above authorises rather than contradicts: it ships under no
  attribution, because no line of Gutiérrez contains it. It is `1/(2ΔF̄′)` over
  `ΔF′_i = 1 − (1−F_i)^(1/(t_i−1))`, eligible at `t_i > 1`. It is a
  diagnostic; `ne` remains the estimator.
* **`ne_caballero_toro` is replaced by `ne_group_coancestry`** — C&T Eq. 3
  group coancestry, evaluated on the genome-node pedigree, with a *computed*
  baseline, reduced by the same `ln(1−x)` regression the other rate estimators
  use. Its record carries `n_genomes_per_gen`, the `n` behind the baseline's
  `1/(2n)`, so a reader can check that number against the record rather than
  the pedigree, and `census_ratio`, which is how the estimator states the one
  assumption its reduction cannot verify. See the consequence below.
* **`ne_long_term_contributions`** reports `n_effective_founders = 1/Σc²` as
  the assumption-free quantity and `ne = 2·n_effective_founders` as the derived
  one, at the last observed cohort, with the achieved `max_delta` reported as
  evidence. Reading the last cohort unconditionally retires the convergence
  loop, and with it two pieces of API named for that loop. The `tol` keyword is
  gone: it would no longer change any reported number, only the descriptive
  `asymptote_reached` boolean, so a caller passing `tol=1e-3` would reasonably
  believe they had tuned the estimate and would not have. `n_iterations`
  becomes `n_cohorts` — the same integer, under a name that describes what the
  record counts rather than iterations it no longer performs.

## Consequences

* Three of eight estimators change numerically. Persisted 0.8.x results are not
  comparable.
* **The 6b golden is retired rather than regenerated.** `tests/data/ne_baseline_6b`
  was gated by a generator frozen at a pre-0.8 API: `compute_all_ne` no longer
  exists and the positional `PedigreeGraph(df)` raises `TypeError`, and the
  file's own docstring forbade migrating it forward. `tests/data/ne_baseline_0_9`
  replaces it, written by `tests/parity/generate_ne_baseline_0_9.py` against the
  current API, with the six fixture builders byte-identical so the two goldens
  differ only by the estimator changes. Everything that existed to bridge the 6b
  record shapes goes with it: the dense-label projection, the MZ founder-column
  migration allowance, the noise-slope exception, and the two outright
  exclusions the old test promised would "go away when the baseline is
  regenerated". The parity module falls from 152 lines to 63, and the generator
  it reads is the generator that wrote it.
* One canonical key is renamed, so pedsum and simACE both need edits. The three
  repos move as a coordinated set, but **not in the same change**. Both
  consumers pin `pedigree-graph>=0.8,<0.9` and resolve 0.8.3 from PyPI, so
  nothing that calls the library sees the rename until 0.9 ships. Pure
  expectation constants moved early; every site that reads a key out of a live
  result dict moves at the relock, with no both-keys compatibility shim in
  between. The relock must also amend simACE's `CONTEXT.md`, which currently
  records these estimator names as not to be renamed and protects
  `mean_self_coancestry` and `Ne_caballero_toro` as fixed caption identifiers.
* **The Caballero-Toro numba ancestor-set arena is deleted, not relocated.**
  Issue #1 names `_caballero_toro_accumulators` as "the exact retirement-style
  ancestor-set DP pattern" it wants for `_compute_n_ancestors`, but it asks for
  that pattern adapted inline and excludes a shared kernel outright, so keeping
  the three kernels alive would ship an importer-free module the issue declined.
  Git history is the durable record, and the issue carries a pointer to the path
  and commit that hold it.
* `ne_group_coancestry` streams from the existing DP (the genome-node collapse
  is a row mask, the diagonal needs only `F`), so unlike `ne_coancestry` it
  carries no OOM exposure and needs no `skip_` flag downstream.
* **`ne_group_coancestry`'s scalar is not a new independent number, and this
  ADR does not claim it is.** Against `ne_coancestry` over 10 random-mating
  seeds per cell it runs `−0.16%` at `N=6`, `−0.64%` at `N=10`, `+0.05%` at
  `N=20` and under `0.03%` from `N=40` up; sweeping the MZ fraction of every
  cohort from 0 to 0.9 at `N=60` never separates them by more than `0.41%`.
  The algebra says why: `f̄_g = θ̄_g·(n_g−1)/n_g + s̄_g/n_g`, whose extra term is
  `O(1/n)` and near-constant across cohorts, so it is nearly absent from the
  slope both scalars are read off. **This is not the defect finding 2 records.**
  That one was an algebraic identity to `ne_inbreeding` whose only
  distinguishing signal was an unsourced reweighting; this difference is the
  diagonal C&T Eq. 3 explicitly includes. Nor is it a general property of rate
  estimators agreeing on clean pedigrees: on the same MZ sweep `ne_inbreeding`
  separates from `ne_coancestry` by `−1.18%`, `+0.92%` and `+8.68%`. What
  `ne_group_coancestry` adds is the per-cohort series — group coancestry with
  the diagonal, over genome nodes, with a computed `1/(2N)` baseline that
  carries information where `ne_coancestry`'s founder θ̄ of roughly zero
  carries none, and which is the quantity C&T's gene diversity `GD = 1 − f̄` is
  built on. A consumer choosing between the two on the strength of the scalar
  alone is choosing on a difference that is not there.
* **That series is a function of group size, so the scalar assumes a constant
  census, and the record says so.** The `s̄_g/n_g` term above is a
  self-coancestry near 0.5 over the cohort size and does not accumulate at the
  drift rate, so `ln(1 − f̄)` reads a change in census as drift. Six cohorts,
  8 seeds, `Ne_GC` against `Ne_C`:

  | cohort sizes | Ne_GC | Ne_C | ratio |
  |---|---:|---:|---:|
  | constant 40 | 41.39 | 41.36 | 1.00 |
  | decline 40 → 4 | 8.30 | 14.47 | 0.57 |
  | growth 10 → 40 | 31.54 | 24.70 | 1.28 |
  | constant 40, lone last | 3.40 | 40.73 | 0.08 |

  One trailing cohort of a single individual moves the reported Ne from about
  41 to 3.40 while `ne_coancestry` is unmoved. C&T's own Eq. 11 carries the
  same assumption, building `f̄₀ = 1/(2N)` in at `t = 0` and holding `N` fixed
  after, so the formula is faithful and this is an assumption to carry rather
  than a defect to fix. It is carried the way the Ne_LTC asymptote is carried,
  not the way the old `tol` was: `ne` is always reported, and `census_ratio` —
  `max/min` of `n_genomes_per_gen` over exactly the cohorts the fit uses, `1.0`
  when the census is constant — ships beside it as the evidence to distrust it.
  Withholding `ne` above some threshold was rejected twice over, for inventing
  a tolerance no cited paper gives and for recreating the never-reports failure
  this ADR removes from `ne_long_term_contributions`.
* **What the replacement did to the shipped numbers.** The six golden fixtures
  are byte-identical pedigrees across the change, and the regenerated golden
  differs from its predecessor by exactly the removal of the
  `ne_caballero_toro` block and the addition of a `ne_group_coancestry` one;
  the other seven estimator blocks compare equal in every fixture.

  | fixture | old `ne_caballero_toro` | new `ne_group_coancestry` | `census_ratio` |
  |---|---:|---:|---:|
  | `closed_line_5` | 2.265480 | 2.344616 | 1.000000 |
  | `skip_gen` | *none* | 5.154730 | 3.000000 |
  | `small_pedigree` | 2934.679127 | 809.438884 | 1.001004 |
  | `wf_n20_g4` | 15.159835 | 16.678037 | 1.000000 |
  | `wf_n40_g5_birth_years` | 45.309974 | 47.642255 | 1.000000 |
  | `wf_n60_g6` | 67.538291 | 62.152448 | 1.000000 |

  Two of these earn comment. `small_pedigree` moves by a factor of 3.6, which
  is the founder-reachability reweighting of finding 2 leaving: that fixture is
  where founder reach is least saturated, and the early cohorts where the
  reweighting bit hardest are the ones the regression leans on. `skip_gen` goes
  from no estimate at all to 5.15, because the departing statistic's series was
  flat to 1.5e−16, below the slope-noise floor, so it reported `None`; it is
  also the one fixture whose cohorts differ in size, so the new golden ships a
  live example of `census_ratio` firing, at 3.0, on the single number in it
  that should be distrusted.
* Assumptions that a pedigree cannot verify — random mating, asymptotic
  contribution variance, unrelated founders, the choice of reference
  subpopulation — are now stated on the result record that carries the number,
  not only in prose.
* A faithful estimator is not automatically a more accurate one, and this ADR
  does not claim it is. Replicated over 20 Wright-Fisher pedigrees at `N=200`,
  the corrected `ne_individual_delta_f` averages +19.1% high at `t=5`, +10.5%
  at 10, +5.6% at 16 and +3.8% at 24, where the version it replaces sat within
  1.5% throughout. The cause is a founder boundary, not the reduction. Eq. 1
  inverts `F_t = 1 − (1−ΔF)^t`, so Eq. 2 recovers the census size only where
  the pedigree has accumulated `t` generations of drift, and a pedigree whose
  founders are unrelated by construction runs one generation behind: measured
  mean `F` tracks the idealised curve at `g−1` at every generation out to 16.
  That predicts `t/(t−1)`, against which the convexity of `F ↦ ΔF_i` returns
  a fraction of a percent. **The discarded `1/(t−1)` exponent was numerically
  absorbing that lag**, which is why the unsourced formula looked accurate on
  simulated pedigrees and is the sharpest possible illustration of why this
  ADR exists: a formula can match a simulation for a reason that has nothing
  to do with the population it claims to describe. Real founders are where
  record-keeping stopped, not individuals known to be unrelated, so the
  compensation does not transfer. The bias decays as `1/t`, §2.1's standard
  error ships beside the estimate, and none of this is a reason to choose a
  reference subpopulation that lands nearer the census size. Downstream
  consumers gating on ±10% of `N` must widen for this estimator, or deepen
  the pedigrees they gate on, rather than expect the old numbers back.
* **The discarded `1/(t−1)` exponent ships back as `ne_unrelated_founders`,
  named for the assumption it needs and attributed to nobody.** To first order
  this correction *is* that exponent: `ΔF′_i = 1 − (1−F_i)^(1/(t_i−1))`,
  averaged over the reference rows with `t_i > 1`, reported as `1/(2ΔF̄′)`.
  That is precisely why the deleted formula appeared accurate on simulated
  pedigrees, and shipping it back beside the cited estimator is a deliberate
  decision rather than an oversight. Measured against the census size over 20
  Wright-Fisher replicates per cell, `ne` first and the companion second:

  |     N |  t | eq. 2 Ne |    bias | corrected | residual |
  |------:|---:|---------:|--------:|----------:|---------:|
  |   200 |  5 |   238.24 | +19.12% |    190.86 |   −4.57% |
  |   200 |  8 |   226.78 | +13.39% |    198.51 |   −0.75% |
  |   200 | 12 |   214.62 |  +7.31% |    196.77 |   −1.62% |
  |   200 | 16 |   211.14 |  +5.57% |    197.96 |   −1.02% |
  |  1000 |  5 |  1246.59 | +24.66% |    998.40 |   −0.16% |
  |  1000 |  8 |  1157.12 | +15.71% |   1012.73 |   +1.27% |
  |  1000 | 12 |  1114.00 | +11.40% |   1021.24 |   +2.12% |
  |  1000 | 16 |  1065.42 |  +6.54% |    998.87 |   −0.11% |

  The `N=1000, t=5` cell measures +24.66% against the `t/(t−1)` prediction of
  25.00%, the gap being the O(1/N) convexity term. The assumption is true of a
  simulated pedigree and false of a real one, whose founders are merely where
  record-keeping stopped and are generally related: there is no lag to remove
  and the companion introduces a downward bias instead. It is therefore a
  diagnostic for simulated or genuinely founder-complete pedigrees, not a
  second estimator, and it carries no standard error and no count of its own.
  The paper's own remedy for the same effect is methodological rather than
  algebraic — its Table II reports Ne at no pedigree-depth restriction, at
  `t ≥ 4` and at `t ≥ 8`, and its Figs. 3-4 read ΔF_i against equivalent
  generations — which a caller reaches here through `reference=`.
* **The corrected `ne` has a different downstream expectation, and a simpler
  one.** W&T eq. 31 under their p. 51 relation `σ_r² = V(k)/2` is
  `4N/(2 + V(k))`, while a Crow-Kimura-form `Ne_V` is `2N/V(k)`. `V(k)` cancels
  between them, leaving `2/Ne_LTC = 1/N + 1/Ne_V`: the expectation is the
  harmonic mean of the census and variance effective sizes, and no family-size
  variance has to be estimated to state it. The two coincide exactly under
  Wright-Fisher, where `V(k) = 2`, which is why simACE's previous `Ne_V/2`
  looked defensible — it was the right expectation for `n_effective_founders`
  under Wright-Fisher and wrong for `ne` everywhere else. Measured through
  simACE's own simulator at `N = 1000` over 12 replicates, `2/Σc²` lands within
  0.8 standard errors of the harmonic mean under both mating models; the
  committed method is simACE
  `tests/analysis/test_effective_size.py::test_ne_ltc_expectation_matches_simulator_mc`,
  which reads `sum_c_squared` so that it gates the relation under the pinned
  0.8 as well as 0.9.
* `ne_group_coancestry` and `ne_coancestry` use different MZ conventions until
  the `mean_kinship_by_generation` inconsistency is fixed separately (issue
  #25): that function drops the MZ pair but keeps both co-twins' pairs with
  everyone else, so it is neither row-based nor genome-node.

## Rejected

* **Rename `ne_caballero_toro` and keep its numerics.** It would preserve a
  statistic whose only defence is that it already exists, and which is a
  reweighted duplicate of `ne_inbreeding` whose distinguishing signal is a
  transient artifact of incomplete founder reach.
* **Implement C&T Eq. 11 directly** (`Ne ≈ t/(2Δf₀,ₜ)`). Maximum fidelity, but
  it breaks the shape every other rate estimator shares and inherits a
  linearisation the authors themselves flag as "only accurate if `F̄ₖ₋₁` is
  small". The regression uses the exact cumulative form, which is separately
  sourced to Gutiérrez Eq. 1.
* **Literal Eq. 3 over rows** for group coancestry. An MZ pair would contribute
  four identical entries at `(1+F)/2` — one genome counted twice — depressing
  `Ne`. C&T had no MZ twins, so the paper is silent rather than contradicted,
  and ADR 0008 already makes the genome node this package's semantic unit.
* **Reporting only `Ne = 2/Σc²`.** It states an effective size under a
  random-mating assumption the package cannot check. Reporting `N_ef` alongside
  makes the assumption visible instead of silent.
* **Quarantining the unverified estimators** (ship the Gutiérrez fix, mark the
  other two provisional). It would freeze two undefendable contracts into 0.9,
  which is precisely what issue #15 exists to prevent.
