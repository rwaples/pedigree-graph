# Changelog

This file tracks public-API changes per release.  For per-commit
history, see `git log`.  Historical release notes prior to v0.5.0
live on the corresponding GitHub release pages.

## Unreleased

- **Changed: `kinship_matrix`, `approximate_kinship_matrix` and
  `mean_kinship_by_generation` run on the Rust core** (slice 14; ADR 0007
  and 0009 as amended). The three keep their signatures, dtypes, sorted
  read-only CSC arrays, caches and the propagation-pruned support the
  0.7.1 record freezes, and every entry is the `pair_kinship` bit. Bytes
  are identical to 0.9.1 on every parity fixture in three row orders and
  on the study pedigrees (20k to 53k rows, complete and 0.001 matrices,
  summaries) except where 0.9.1 was wrong (below). One depth-major DP in
  the core builds all three, assembles the CSC in graph rows without a
  SciPy permutation copy, and stores rows as owned vectors freed on
  retirement, selected over a port of the 0.9.1 arena by measurement.
  Against the 0.9.1 wheel on one thread: `random_30k` complete matrix
  97 s to 19 s and 13.9 to 4.7 GiB peak; its 0.001 matrix 76 s to 10.5 s
  and 6.9 to 1.5 GiB; its summary 52 s to 5.6 s and 6.3 to 1.3 GiB; the
  536k-row summary 89 s to 16 s and 14.8 to 3.6 GiB. Record:
  `docs/pedigree-graph-0.8-migration/gate/14a/NOTES.md`.
- **Fixed: `mean_kinship_by_generation` no longer reads storage its own
  merge walk has freed.** In 0.9.1's retiring DP a parent row that outgrew
  its slot during a child's merge walk pushed the old slot onto the free
  list, and a later append in the same walk could take that slot back and
  overwrite what the walk was still reading. It needs a row to outgrow its
  first slot (`2 ** (max_depth + 4)` entries, 16 to 4096), which none of the
  repository fixtures do; the 536k-row study pedigree `baseline100K/rep1`
  does, and its deepest generation's mean kinship was 1.4e-5 too low in
  relative terms (`docs/pedigree-graph-0.8-migration/gate/14a/NOTES.md`).
  The native DP stages each walk's relatives before writing, and
  `tests/test_native_kinship_matrix.py` reproduces the 0.9.1 disagreement
  on a small pedigree with tiny slots. `approximate_kinship_matrix` ran the
  same retiring pass to capture its values; a corrupted row there would
  have failed its own completeness assertion rather than returned a wrong
  value, and no study pedigree reached it.
- **Removed (private): the numba kinship DP.** `_kinship_dp`,
  `_kinship_dp_depth`, `_kinship_allocator`, `_kinship_csc` and the
  `_kinship_kernel` facade are gone from the package, as are
  `Topology.translate` and the approximate-support helpers of
  `_kinship_matrix` (`_topological_candidate_index`,
  `_exactify_approximate_support`, `_write_symmetric_values`,
  `_upper_support_chunks`); `_kinship_depth` keeps `_compute_eqg` alone.
  The DP lives on verbatim as the differential oracle
  `tests/oracle/kinship_dp/`. numba remains a dependency for the inbreeding
  walk, the equivalent-generation kernel and the lineage kernels.

- **Fixed (private): `_native.kinship_support_values` rejects every
  malformed support it used to fill with zeros.** A lower entry with no
  upper mirror now raises `kinship_support_asymmetric` (only the upper to
  lower direction was checked), and an `indptr` that ends before
  `len(indices)` raises `value_out_of_range` on its last position. The
  public matrix path always builds a canonical symmetric support and was
  never affected.

## v0.9.1

- **Changed: `pair_kinship` runs on the Rust core** (slice 13; ADR 0005,
  0007 and 0009 as amended). `PedigreeGraph.pair_kinship` and
  `PedigreeView.pair_kinship` keep their three call forms, validation
  codes, read-only float32 results and the ADR 0009 value definition, and
  every value is bit-identical to 0.9.0: a golden lock of the 0.9.0 bits on
  every parity fixture is a permanent test, and the four simACE study
  pedigrees (20k to 536k rows, 16 million pairs) replayed with zero
  differing elements in both endpoint orders. The recurrence now takes
  structural depth as its peel input in graph space, builds one memo per
  call as one small table per lower row, and frees it before returning.
  Against the 0.9.0 wheel on one thread: `random_30k` degree-3 pairs 82.5 s
  to 5.3 s and 762 to 515 MiB peak; the 536k-row study pedigree at degree 3
  31.3 s to 24.8 s and 5.4 to 3.0 GiB; `random_300k` degree 3 completes in
  934 s at 14.1 GiB where 0.9.0 did not finish in an hour. Record:
  `docs/pedigree-graph-0.8-migration/gate/13a/NOTES.md`.
- **Changed: `relationship_kinship_matrix` fills its values in one native
  walk of the CSC support** (`_native.kinship_support_values`) instead of
  streaming pair chunks through a retained memo. On the 536k-row pedigree
  at degree 3 the matrix takes 32.7 s against 49.1 s, at the same peak RSS.
- **Changed: nothing is retained between `pair_kinship` calls.** 0.9.0 kept
  the recurrence memo on the graph under a 1 GiB limit so a repeated query
  on the same graph returned in under a second; each call now walks cold.
  On `random_30k` that repeat is 5.4 s instead of 0.45 s, and a matrix after
  a degree-3 walk 5.8 s instead of 1.2 s; the first call of either is 15
  times faster than before, and no consumer in simACE, fitACE or pedsum
  makes the second call.
- **Removed: `ResourceError("memo_capacity_exceeded")`.** The memo has no
  global capacity any more; a pedigree that exhausts memory raises
  `ResourceError("allocation_failed")` from one of the new `kinship_memo`,
  `kinship_stack` or `kinship_output` families, or completes. The code and
  its `(operation, capacity, maximum)` fields leave `RESOURCE_CODES`.
- **Added (private): `PedigreeValidationError` codes
  `kinship_support_unsorted` and `kinship_support_asymmetric`**, raised by
  the native support walk on a malformed CSC. Unreachable from the public
  matrix path; pinned through the raw binding.
- **Removed (private):** the Numba pairwise kernel and its memo
  (`pairwise_kinship`, `_pairwise_kinship_core`, `_run_kernel`, `_PairMemo`,
  `memoised_kinship`, `PedigreeGraph._pair_memo`, `_release_pair_memo`).
  The readable Python recurrence is now the test oracle
  `tests/oracle/pair_kinship.py`.
- **Added: `pedigree_graph.MAX_DEGREE` and `_native.max_degree_max()`**, the
  deepest degree the relationship APIs accept. Both sides now derive that
  ceiling from the category registry instead of naming a literal, and a test
  pins the Python value equal to the Rust one across the binding (issue #23).
  The value is unchanged at 5.

## v0.9.0

- **Changed: `relationship_pairs` runs on the Rust row-streaming engine**
  (slice 12; ADR 0006, 0007 and 0010 as amended). Graph and view results
  are element for element what the 0.8.4 SciPy matrix engine returned:
  the same 23 blocks, roles, closest-category precedence, canonical-key
  order, coordinate tokens and read-only int32 arrays. The arrays are now
  the core's own allocations handed over without a copy. On `random_300k`
  at degree 5 the call is 0.29x the 0.8.4 wall time on six threads and
  0.19x its peak memory; on a 20M-row pedigree it completes degree 5
  (2.12 billion pairs) in 18.1 GiB, which the matrix engine could not do
  in 30 GiB. Measurements: `benchmarks/bench_pair_emitters.md`.
- **Added: `relationship_pairs(..., execution="speed" | "memory")`** on
  graphs and views. `"speed"` (the default) is the fastest exact assembly
  at about 2.3 times the result in peak memory; `"memory"` holds no copy
  of the result (the result plus engine state) at roughly twice the wall
  time. The blocks are identical either way; any other value is a
  `ValueError`.
- **Changed (private): the allocation test seam is off by default.**
  `_native.fail_next_allocation` arms a process-global one-shot that the
  next reservation of that family consumes, on whichever thread reaches it
  first, so a released wheel refuses it unless the process was started with
  `PEDIGREE_GRAPH_ALLOW_TEST_SEAM=1`. It was never public API.
- **Added: `ResourceError("allocation_failed")`** with fields `operation`,
  `requested_elements` and `dtype`, raised by `relationship_pairs` and
  `relationship_counts` when the engine, a workspace, a row set, a task
  buffer, a result block or the view-sort scratch cannot be allocated,
  instead of aborting the process.
- **Changed: one package-wide Rayon pool.** The native engine builds one
  thread pool per process from the committed `configure_threads` /
  `PEDIGREE_GRAPH_THREADS` budget (ADR 0007). Results do not depend on the
  budget. The pool is owned per process, not per address space: a `fork`
  copies its memory but none of its worker threads, so a forked child
  rebuilds its own pool on first use instead of blocking on a queue nobody
  drains, and the inherited one is leaked rather than joined. Asking for a
  second, different budget raises `RuntimeError`, the same class
  `configure_threads` raises for the same reason.
  `_reset_thread_state()` remains test-only and cannot resize the native
  pool; tests that compare budgets run each in a fresh interpreter.
- **Added: `relationship_pairs` and `relationship_counts` log at INFO.**
  One line when the call starts, naming the degree, category count,
  execution and thread budget, and one when it finishes with the pair
  total and elapsed seconds. The matrix engine logged once per degree; a
  single native call cannot, so an operator watching a long run sees it
  enter and leave.
- **Changed: the relationship engine reuses its per-row buffers.** The
  sorted-set union merges in place, and the lineal, parent-role, cousin,
  removed-cousin and first-arm sets are refilled rather than replaced, so a
  row no longer frees and reallocates the buffers the previous row used.
  The `speed` blocks are also assembled category by category in parallel.
  On `random_300k` at degree 5 this is 0.94 of the previous wall on every
  thread count and execution, with engine memory unchanged; blocks are
  identical. Measurements: `benchmarks/bench_pair_emitters.md`.
- **Fixed: a malformed view map is rejected instead of misread.** A map
  with an entry outside `-1 .. n`, or with two graph rows on one view row,
  raises `ValueError`. Repeated view rows would have given a block equal
  sort keys, and the unstable parallel sort would then have ordered them by
  thread count. Only the native call could supply one; a `PedigreeView`
  always builds a valid map.
- **Added: `relationship_kinship_matrix(..., execution=...)`.** The same
  keyword `relationship_pairs` takes, for the package's heaviest pair
  consumer: it holds the blocks and the support at once. The matrix and its
  cache entry are identical either way.
- **Fixed: a view that selects no rows no longer overflows.** The view-row
  count is taken over selected rows only; a map of all `-1` used to
  sign-extend to `u64::MAX` and wrap, which panicked in a debug build.
  Reachable only through the private native call, since a `PedigreeView`
  of fewer than two rows short-circuits before the engine.
- **Fixed: `distinct_ancestor_counts` checks its topological precondition.**
  The retiring DP needs every parent row before its child row. Violating it
  returned silently low counts and could hand a later row a retired,
  unwritten slot; the kernel now raises `ValueError`. The public path was
  and remains correct, since it reorders when the graph rows are not
  already topological.
- **Removed (private): the SciPy matrix pair extractor** (`_pair_extractor`,
  `_pair_utils`, the graph's lazily cached adjacency powers `_A` to `_A5`,
  `_A2_shared`, `_get_Ak`, the sibling matrices and `_release_pair_matrices`)
  and the `PEDIGREE_GRAPH_DEBUG_EXCLUSIVITY` environment variable. The
  extractor lives on unchanged as the differential test oracle in
  `tests/oracle/relationship_pairs.py`. SciPy remains a dependency for the
  kinship matrices, lineage and effective-size modules.

- **Changed: `distinct_ancestor_counts()` now uses one retiring Numba DP.**
  The old implementation repeatedly multiplied a sparse boolean transitive
  closure and retained every ancestor link. The replacement makes one
  topological pass, merges sorted closed ancestor sets, and reuses a
  power-of-two slot after its row's last direct child. Returned int32 values,
  graph-space order, read-only memoisation, and distinct-path semantics are
  unchanged. On the committed benchmark, the 60-generation stress fixture
  drops from 31.68 s and 532 MiB peak RSS to 0.16 s and 183 MiB; `random_300k`
  drops from 3.42 s and 586 MiB to 0.45 s and 269 MiB. A fresh process pays
  about 42 MiB to load the Numba runtime, so small fixtures have higher total
  peak RSS despite lower operation-specific allocation. The one-method design
  accepts that fixed cost rather than dispatching between implementations.

- **Removed: `estimate_relationship_counts(*, max_degree)`**, replaced by
  `close_relative_counts()` with no selector, per issue #17. It computes only
  MZ, MO, FO, FS, MHS and PHS, exactly under closest-category precedence.
  The result still has all 23 registry keys, with `None` for the other 17.
  Use `relationship_counts(max_degree=5)` for all categories. The scalar
  lineal, cousin and collateral formulas, residual clamps, warning and
  adjacency-power release are deleted. There is no old-name alias.
- **Removed: `RelationshipCountResult.approximate` and `.clamped`** from all
  count results, including exact graph/view counts. `requested` and `exact`
  remain. Consumers must stop reading the removed attributes. The pedsum
  migration is tracked in [pedsum#2](https://github.com/rwaples/pedsum/issues/2).

- **Removed: `RelationshipCountResult.from_pairs`** (issue #16).  Callers
  that already hold a `RelationshipPairs` result can count its requested
  blocks directly; callers that do not need pair arrays should use the
  row-streaming `relationship_counts` API.  simACE migrated its existing-pairs
  consumer in companion issue rwaples/simACE#17.

- **Breaking: `ne_caballero_toro` is replaced by `ne_group_coancestry`**
  (issue #15, ADR 0012).  The departing estimator averaged descendant
  self-coancestry `(1 + F)/2` within each founder genome's reachable set,
  averaged that across founder genomes, and regressed it against a hardcoded
  `0.5` baseline.  Caballero & Toro 2002 (Cons. Genet. 3(3):289-299) contains
  no such statistic: its `s` feeds a diversity partition and its effective size
  (eq. 14) is a contribution-variance formula.  Where every represented founder
  reached every cohort member the statistic was **bitwise** `(1 + F̄)/2`, so it
  reproduced `ne_inbreeding` to 15 significant figures; where founder reach was
  incomplete it differed only by a reweighting no cited paper motivates.

  The replacement is Caballero & Toro 2000 (Genet. Res. 75(3):331-343) eq. 3
  group coancestry, `f̄ = Σ_i Σ_j a_ij / 2N²` over all `N²` ordered pairs with
  self-coancestries and reciprocals included, evaluated per observed cohort
  over the genome-node pedigree of ADR 0008 so that an MZ pair is one genome
  rather than two, and reduced by the same `ln(1 − x)` regression
  `ne_inbreeding` and `ne_coancestry` use.  It does **not** claim to be their
  eq. 11, whose linearisation the authors flag as "only accurate if `F̄ₖ₋₁` is
  small".  The baseline cohort's `f̄` is computed like every other cohort's
  rather than assumed; with unrelated non-inbred founders it lands on the
  `f̄₀ = 1/(2N)` the paper states, exactly, which makes the first `ne_per_gen`
  entry their eq. 11 `Δf₀,₁`.

  The record changes shape with the name.  `mean_group_coancestry_per_gen`
  replaces `mean_self_coancestry_per_gen`; `n_founders_with_descendants_per_gen`
  is gone, because it described the deleted weighting; `n_genomes_per_gen`,
  `n_generations_used` and `census_ratio` are new.  The estimator no longer
  requires closed represented parentage, since it reads no founder
  contributions, so `incomplete_parentage` now gates `ne_long_term_contributions`
  alone.  It streams off the existing kinship DP (the genome-node collapse is a
  row mask and the diagonal needs only `F`), reusing the summary
  `ne_coancestry` already memoises when the pedigree has no MZ twins, so unlike
  `ne_coancestry` it carries no OOM exposure and needs no `skip_` flag.

  Two measured caveats ship with it rather than being discovered downstream.
  Its **scalar is not an independent number**: against `ne_coancestry` it runs
  `−0.64%` at `N=10`, `+0.05%` at `N=20` and under `0.03%` from `N=40` up, and
  sweeping the MZ fraction of every cohort from 0 to 0.9 never separates them by
  more than `0.41%`, because `f̄_g = θ̄_g·(n_g−1)/n_g + s̄_g/n_g` differs from
  `θ̄_g` by an `O(1/n)` term that is near-constant across cohorts and so nearly
  absent from the slope.  What the estimator adds is the per-cohort series, not
  a second opinion on the scalar.  And that series is a function of group size,
  so the **scalar assumes a constant census**: at 6 cohorts and 8 seeds
  `Ne_GC/Ne_C` is 1.00 at a constant 40, 0.57 declining 40 to 4, 1.28 growing 10
  to 40, and 0.08 with a lone trailing individual.  Following the same rule as
  the Ne_LTC asymptote, `ne` is always reported and `census_ratio` — `max/min`
  of `n_genomes_per_gen` over exactly the cohorts the fit uses, `1.0` under a
  constant census — ships beside it as the evidence to distrust it.

  Any persisted 0.8.x value under the old key is not comparable.  On the golden
  fixtures the old and new scalars are 2.27 against 2.34 (`closed_line_5`),
  none against 5.15 (`skip_gen`, where the old series was flat to 1.5e−16 and
  reported nothing), 2934.68 against 809.44 (`small_pedigree`, where founder
  reach is least saturated and the deleted reweighting bit hardest), 15.16
  against 16.68, 45.31 against 47.64, and 67.54 against 62.15.

  The estimator's numba ancestor-set arena (`_ct_ensure_pool_capacity`,
  `_ct_merge_to_pool`, `_ct_accumulators_kernel`) is deleted with it.  Issue #1
  names that arena as a retirement-style DP pattern to reuse, but asks for it
  adapted inline and excludes a shared kernel, so it survives in git history
  rather than as an importer-free module.

- **Fixed, and breaking: `ne_long_term_contributions` reported an effective
  size 4× below the one it cites** (issue #15, ADR 0012).  It computed
  `1 / (2 · Σ_f c_f²)` over the per-cohort mean founder-genome contributions.
  Wray & Thompson 1990 (Genet. Res. 55(1):41-54) eq. 31 is
  `Ne ≈ 2N/(μ_r² + σ_r²)`, and long-term contributions carry `μ_r = 1` by
  construction, so `Σ_i r_i² = N · Σ_f c_f²` and the normalised form is
  `Ne = 2/Σc²`.  Caballero & Toro 2000 (Genet. Res. 75(3):331-343) eq. 19 is
  `N_ef = 1/[(1/N²)Σc²_{i(0,t)}]`, i.e. `N_ef = 1/Σc²` in the same
  normalisation, and the sentence after their eq. 20 states `N_ef = Ne/2`.
  The shipped value was therefore 4× below W&T's `Ne` and 2× below C&T's
  `N_ef`.  It went unnoticed because the estimator almost never reported
  anything to check: `ne` was gated on `asymptote_reached`, itself
  `max |Δc| < tol` at a default `tol = 1e-6`, and the mean contributions of a
  stochastic pedigree fluctuate around 1e-4 from cohort to cohort.  Measured
  over 30 Wright-Fisher replicates at N=200 and 10 generations, the final
  `max |Δc|` ranges 3.6e-4 to 1.2e-3 — `asymptote_reached` is `False` on every
  one of them, so `ne` was `None` on every one of them.  simACE's 0.8 migration
  record corroborates it independently from the other side of the API
  (`docs/pedigree-graph-0.8-migration/README.md:45`: "`ne_long_term_contributions`
  reported no estimate under both versions").  `NeLTCResult` now reports
  **both** quantities, and in that order: `n_effective_founders = 1/Σc²` is
  C&T eq. 19 and is assumption-free — it is what the founder-contribution
  vector measures directly — while `ne = 2 · n_effective_founders` is W&T
  eq. 31 and is derived, holding only under regular random mating (α = 0) and
  long-term contribution variance at its asymptote.  Reporting only `Ne` would
  state an effective size under a mating assumption a pedigree cannot check;
  reporting both puts that assumption on the record that carries the number
  rather than in prose beside it, and leaves a caller who doubts it a quantity
  that still stands.  The asymptote gate is gone with it.  Both estimates are
  now read from the **last observed cohort**, always, and `final_generation`
  names it; `max_delta_final` is the movement into that cohort,
  `max_f |c_last[f] − c_{last−1}[f]|` (`nan` when only one cohort is observed),
  and is the evidence a caller weighs against the asymptotic-variance
  assumption.  `asymptote_reached` stays as that same movement against a fixed
  1e-6 tolerance, descriptive only, gating nothing.  `ne` and
  `n_effective_founders` are `None` only where the graph has no represented
  founder or no observed cohort, which is also the only case where
  `sum_c_squared` stays `0.0`.  Two API changes follow.  The `tol` keyword is
  **removed** from `ne_long_term_contributions`: it no longer moves any
  reported number, only the `asymptote_reached` boolean, so a caller passing
  `tol=1e-3` would reasonably believe they had tuned the estimate and would
  not have — and `max_delta_final` carries strictly more information than the
  boolean, so a caller who wants their own threshold applies it themselves.
  `estimate_effective_sizes` hard-coded `1e-6` anyway, so the batch path never
  exposed the knob.  `n_iterations` is **renamed** `n_cohorts` and holds the
  number of observed cohorts: with the convergence loop deleted, a field named
  after iterations would describe work the estimator no longer does, while the
  count still earns its place by saying how much cohort series stands behind
  `max_delta_final`.  It is `0` in the degenerate case.  The reducer reads one
  cohort vector and one difference instead of scanning every adjacent pair,
  which drops it from O(k · n_genomes) to O(n_genomes).  Measured on
  Wright-Fisher pedigrees with balanced sex and discrete non-overlapping
  generations, where W&T's derivation predicts the census size, over 30
  replicates per cell: at N=200 and 10 generations `2/Σc²` averages 202.30
  (sd 11.90, **+1.15%** of N), where the old formula on the same cohort vectors
  averages 50.58 and reported `None` on all 30.  The residual is O(1/N)
  and shrinks with N — +5.02% at N=60 (8 generations), +3.30% at N=120 and
  +1.15% at N=200 (10 generations each) — and single replicates scatter widely
  around it, from −13.95% to +13.48% at N=200, so the regression test gates the
  replicate mean rather than any one pedigree.  Any persisted 0.8.x value for
  this estimator is not comparable: the old numbers are 4× low where they exist
  at all, and on a realistic pedigree they do not exist.  The golden that held
  the old numbers has since been retired for `tests/data/ne_baseline_0_9`, so
  the parity test gates this estimator rather than excluding it.

- **Fixed, and breaking: `ne_individual_delta_f` now computes the formula it
  cites** (issue #15, ADR 0012).  Gutiérrez et al. 2008 (Genet. Sel. Evol.
  40(4):359-378) eq. 2 is `ΔF_i = 1 − (1 − F_i)^(1/t)` over every individual
  with `t > 0` equivalent complete generations, and their §2.1 reports
  `Ne = 1/(2·ΔF̄)` averaged over a **reference subpopulation**.  The
  implementation used the exponent `1/(t − 1)`, excluded `t ≤ 1`, and
  aggregated by harmonic mean of the per-cohort Ne, which weights every cohort
  equally whatever its size.  All three are corrected against the paper as
  read.  The reference subpopulation defaults to the last observed cohort and
  is overridable with the new keyword-only `reference=`, taking graph rows
  (not a `PedigreeView` — ADR 0006 keeps effective-size operations off views);
  rows are validated like a view selection, under the new
  `reference_row_out_of_range` and `duplicate_reference_row` codes.  A
  duplicate is rejected rather than collapsed, since a repeated row would
  silently reweight ΔF̄.  `NeIndividualDeltaFResult` gains `standard_error`
  (the paper's `σ_Ne = (2/√N)·Ne²·σ_ΔF`, with σ_ΔF at `ddof=1`, this package's
  choice), `n_reference` and `reference_generation`; `ne_per_gen`,
  `mean_eqg_per_gen` and `n_used_per_gen` keep their meanings, but their
  values move because the eligible set widened to `t > 0`.  Measured on closed
  random-mating pedigrees with balanced sex and discrete generations, where
  the paper's derivation predicts the census size: 66.39 against N=60, 132.43
  against N=120 and 200.01 against N=200, versus 61.69, 130.73 and 190.65
  before.  The old values sat near N by cancellation — the inflated exponent
  deflated each Ne by about as much as averaging over the whole genealogy
  inflated it — so the defence of the change is fidelity to eq. 2, not a
  smaller bias.  Replication makes that explicit.  Over 20 Wright-Fisher
  pedigrees at N=200 the corrected estimator averages +19.1% high at `t = 5`
  equivalent complete generations, +10.5% at 10, +5.6% at 16 and +3.8% at 24,
  where the old one sat within 1.5% throughout.  The cause is a founder
  boundary, not the reduction: eq. 1 inverts `F_t = 1 − (1 − ΔF)^t`, so eq. 2
  recovers N only where the pedigree has really accumulated `t` generations of
  drift, and a simulated pedigree whose founders are unrelated by construction
  runs one generation behind (measured mean F tracks the idealised curve at
  `g − 1` at every generation out to 16).  That predicts `t/(t − 1)`, or
  +25.0/+11.1/+6.7/+4.3%, against which the convexity of `F ↦ ΔF_i` returns
  1.8/0.8/0.5/0.3%.  The discarded `1/(t − 1)` exponent was numerically
  absorbing exactly that lag, which is why the old estimator looked accurate
  on simulated pedigrees; it has no such justification on a real one, whose
  founders are where record-keeping stopped rather than genuinely unrelated.
  The bias decays as `1/t`, and §2.1's standard error now ships beside the
  estimate.  The record also gains `ne_unrelated_founders`, a **package-defined
  statistic** that no line of Gutiérrez contains and that therefore ships under
  no attribution: `1/(2·ΔF̄′)` over `ΔF′_i = 1 − (1 − F_i)^(1/(t_i − 1))`,
  averaged over the reference rows with `t_i > 1` and `F_i < 1`, `None` when
  none qualify.  To first order it is algebraically the `1/(t − 1)` exponent
  this same entry deletes as unsourced, which is exactly why the deleted
  formula appeared accurate on simulated pedigrees; shipping it back as a
  named, assumption-gated diagnostic beside the cited estimator is a
  deliberate decision, not an oversight.  Measured against the census size
  over 20 Wright-Fisher replicates per cell, eq. 2 first and the companion
  second: at N=1000, `t = 5`, 1246.59 (+24.66%, against the `t/(t − 1)`
  prediction of 25.00%) versus 998.40 (−0.16%); at N=200, `t = 16`, 211.14
  (+5.57%) versus 197.96 (−1.02%).  ADR 0012 carries the full table.  The
  assumption it needs is true of a simulated pedigree and false of a real one,
  for the reason just given: on a real pedigree there is no lag to remove, and
  the field introduces a downward bias instead.  It is a diagnostic for
  simulated or genuinely founder-complete pedigrees, and `ne` remains the
  estimator.  Its eligibility is stricter than `ne`'s (`t > 1`, not `t > 0`),
  so fewer rows can stand behind it than `n_reference` counts; it reports
  neither a count nor a standard error of its own.  Gutiérrez's own remedy for
  the same effect is methodological rather than algebraic — Table II reports
  Ne at no pedigree-depth restriction, at `t ≥ 4` and at `t ≥ 8`, and
  Figs. 3-4 read ΔF_i against equivalent generations — which a caller reaches
  here through `reference=`.  Any persisted 0.8.x value for this estimator is
  not comparable; the golden that held the old numbers has since been retired
  for `tests/data/ne_baseline_0_9`, so the parity test gates this estimator
  rather than excluding it.

- **Changed: the parent adjacency is built once, lazily, from the edge lists**
  (issue #18).  Construction eagerly built a CSR per parent, `_Am` and `_Af`,
  whose only production reader was their sum in `_A`.  Every graph paid for
  both halves whether or not it ever reached a relationship path.  `_A` now
  assembles itself from one COO over both edge lists on first read, and `_Am`,
  `_Af`, `_build_parent_csr` and `_ensure_parent_csr` are gone, along with the
  `__dict__.pop` calls in `_pair_extractor` and the rebuild in
  `_streaming_counter` that existed to manage them.  The matrix is unchanged
  byte for byte; `check_same_parent` forbids one id in both parent roles, so no
  entry can be written twice.  On `random_300k`, construction drops from 183.6
  to 162.6 MiB peak RSS and reaching `_A` from 183.7 to 172.4 MiB, both with
  disjoint ranges over five interleaved repetitions
  (`benchmarks/bench_parent_adjacency.md`).  No public API changes; ADR 0006
  recorded fitACE reaching into `_Am`/`_Af`, and a sweep of the five family
  repositories found no reader left.

- **Changed: MHS/PHS-only pair queries no longer build the unused `_A2`**
  (issue #24).  The eager trigger now starts at GP, the first code in the
  dependency closure that consumes the 2-hop parent adjacency, so GP and higher
  selections keep their existing build order.  On `random_300k`, an MHS-only
  query drops from 216.84 to 203.29 MiB median peak RSS and from 0.379 to 0.335
  seconds over five interleaved repetitions, with disjoint ranges and identical
  result checksums (`benchmarks/bench_relationship_a2.md`).  No public API or
  relationship result changes.

- **Fixed: `_native.relationship_counts` rejects an out-of-range `max_degree`
  instead of clamping it** (issue #20).  The Rust engine applied
  `max_degree.min(5)`, so the native binding accepted `6`, `9` or `255` and
  returned fifth-degree counts, and `pgr-count --max-degree 9` printed those
  counts under a `"max_degree": 9` label.  The degree is now a checked
  `MaxDegree` newtype built only through `MaxDegree::try_new`, so an
  out-of-range value cannot reach `Engine::new` at all, and `count_pairs` stays
  infallible because its precondition is a type invariant.  The native surface
  now raises the same `PedigreeValidationError` with code
  `max_degree_out_of_range` that the pure-Python selector already raised.
  `PedigreeGraph.relationship_counts` is unchanged, having validated the
  selector all along.

- **Fixed: the uniform-sex warning names the caller on both paths** (issue #19).
  `_warn_if_uniform_sex` used a fixed `stacklevel` tuned for a direct
  `ne_sex_ratio` / `ne_variance_family_size` call, so the same notice raised
  through `estimate_effective_sizes` was attributed to the orchestrator's own
  memo (`_ne_estimate.py`, the `result()` thunk) rather than to the caller's
  line.  It now skips frames inside the package, which is correct at either
  call depth.  `TestWarningAttribution` covers both paths.

- **Changed: the effective-size orchestrator dispatches through a registry**
  (issue #19).  `_compute`'s chain of `if name == ...` branches with
  membership-set guard tests is replaced by `_REGISTRY`, one row per estimator
  naming its metadata guards and its build.  The guards are a function of the
  graph because `ne_hill_overlapping` is the one estimator whose requirements
  depend on it, its birth-year branch reading no generation label.  The
  per-call memo's single `dict[str, Any]` is split so prerequisites and
  completed results no longer share a namespace, every prerequisite accessor
  carries a return type, and the equivalent-complete-generations array joins
  the memo instead of being computed inline.

  The issue's stated prerequisite, that the batch and standalone paths apply
  the same guards in a different order, is **not** the case.
  `ObservedCohorts.for_graph` runs `_require_complete_generation_labels`
  before it densifies, so the standalone path's "cohorts first" already is
  "labels first"; both paths refuse in the order labels, sex, then the
  estimator's own guard.  Measured over a 14-graph by 8-estimator matrix
  comparing refusal code, emitted warnings and result value, all 112 cells
  agree, and `TestPathEquivalence` now holds them there.  No behaviour
  changes.

- **Documented: the depth-versus-label contract on the structural consumers**
  (issue #22).  Structural results derive from structural depth in the parent
  DAG, and a supplied generation label never enters them; cohort-indexed
  results group by generation label, falling back to structural depth only
  when the whole pedigree is unlabelled.  The rule held in the code and was
  covered by `test_generation_labels_do_not_drive_structure`, but an
  `inspect.getdoc` scan of the public surface found it stated nowhere: none of
  `kinship_matrix`, `approximate_kinship_matrix`, `relationship_kinship_matrix`,
  `pair_kinship`, `inbreeding`, `relationship_pairs`, `relationship_counts`,
  `distinct_ancestor_counts` or `descendant_path_counts` mentioned depth,
  labels or cohorts.  All nine now say it, as do the three view methods, and
  the rule is stated once in `CONTEXT.md`'s `## Relationships` section.
  `test_generation_labels_stay_off_the_structural_path` pins the six modules
  that read the `generation_labels` property — the property itself plus five
  cohort-side effective-size modules — so a structural module reading labels
  fails the suite rather than waiting on a reviewer.  No behaviour changes.

- **Fixed: `mean_kinship_by_generation` counts MZ co-twins as one genome**
  (issue #25).  It used a third convention that was neither row-based nor
  genome-node: `_finalize_summary` dropped the MZ pair from both the numerator
  and the denominator, but nothing removed the duplicate co-twin row, so that
  genome's relationships with the rest of the cohort were counted twice.  On
  the issue's fixture a cohort of `{4, 5, 6}` with `4`/`5` co-twins reported 2
  pairs where row-based is 3 and genome-node is 1.  It is now genome-node
  (ADR 0008), matching `kinship_matrix`, which already returns
  `K[4,5] == K[4,4] == 0.5`, and matching the `ne_group_coancestry` of issue
  #15.  Both estimators now share one convention and one memoised summary, so
  a pedigree with MZ twins pays one DP pass rather than two.

  The direction was measured before the convention was chosen, because the
  issue asked for it and the answer is not what a bias correction looks like:
  over 30 seeds per twin fraction the change in `ne_coancestry` has **no
  consistent sign**, a median of −0.37% to −1.17% with individual replicates
  from −6.0% to +6.7%, positive in 8/30, 8/30 and 14/30 of replicates at 10%,
  20% and 40% twinning.  Double-counting one genome over-weights it, and
  whether that raises or lowers mean θ depends on whether that genome happens
  to be more or less related than its cohort average.  `small_pedigree` is the
  golden's only twin-bearing fixture and moves `ne_coancestry` from
  811.4589907 to 809.9403738 (−0.187%); the other five are byte-identical.

  `unlabelled_individual_count` keeps its meaning.  The collapse is a row mask
  and a masked row lands in the same sentinel bucket an unlabelled row does,
  which would have tallied collapsed co-twins as unlabelled; the count still
  reports only rows whose *supplied* label is unknown.

  Making the summary genome-node also required the collapse itself to become
  row-order invariant, which it was not.  `_genome_node_labels` chose the
  representative by lower row index, following `_genome_of`.  That is harmless
  for kinship and inbreeding, where both co-twins report identical values
  whichever is canonical, but here it decides *which cohort keeps the genome*,
  and co-twins may carry different generation labels.  Reversing the row order
  of a pedigree whose co-twins straddle two cohorts moved the unreleased
  `ne_group_coancestry` from `n_genomes_per_gen` `[6, 3, 1]` to `[6, 1, 3]` and
  its `ne` from 0.979 to `None`.

  The representative is chosen on the data, not on an identifier: the co-twin
  carrying a label beats one that does not, and between two labelled co-twins
  the earlier label wins, so a genome enters at the earliest cohort claimed for
  it.  Only when both rows agree does the row index settle it, and that is
  unobservable — co-twins are one genome, so they carry identical kinship to
  every other row and identical `F`, and dropping either leaves the same pairs
  in the same cohort.  Keying the choice on the `ids` instead was tried and
  rejected: an id is as arbitrary as a row, and two pedigrees identical in
  structure, sex, labels and row order but differing in which co-twin held the
  smaller id gave `ne` 1.233 against `None`.  `tests/test_row_order.py` gains
  regression tests over five row permutations and three id renumberings; four
  of the five and two of the three fail under the respective old rules.

  The frozen 0.7.1 parity baseline records the old convention rather than
  independently attesting it — rebuilding that convention from the dense
  kinship matrix reproduces the stored `per_gen_mean_kinship` exactly — so it
  joins `inbreeding` (ADR 0008) and the `deep_inbred_60g` pair kinship
  (ADR 0009) as a documented divergence rather than being regenerated.  The
  thirteen small fixtures with no MZ twin still compare in full.

## v0.8.4

- **Fixed: the transient adjacency matrices are released on the failure path.**
  `relationship_pairs` and `estimate_relationship_counts` released `_A` through
  `_A5` only on success, so an exception left them resident for the graph's
  lifetime.  `estimate_relationship_counts` reached that path through its own
  clamping `RuntimeWarning` under an `error` filter.  Both releases now run in a
  `finally`.

- **Fixed: `estimate_effective_sizes` no longer hides unrelated warnings.**  The
  Hill fallback suppressed every `RuntimeWarning` raised while computing the
  variance estimator.  It now hides only the duplicate uniform-sex notice, which
  the caller already received under the Hill estimator's own name.

- **Fixed: the one-of-two selector error no longer names `relationship_pairs()`.**
  `relationship_counts` and `relationship_kinship_matrix` share the selector, so
  a caller of either was told a function it had not called was at fault.

- **Changed: relationship `max_degree=` selectors require an integer.**  Values
  implementing the integer index protocol, including NumPy integer scalars,
  remain valid.  Floats, strings, and booleans now raise `TypeError` instead of
  being silently coerced.  Integer values outside `[0, 5]` still raise
  `PedigreeValidationError` with code `max_degree_out_of_range`.

- **Changed: relationship result mappings enforce immutability at construction.**
  `RelationshipPairs` and `RelationshipCountResult` now copy their input mapping
  into a read-only proxy.  A mapping missing a registry code or using the wrong
  order raises `ValueError`; these checks no longer disappear under `python -O`.

- **Changed: `NeHillResult` enforces its three producer states.**  Sentinel,
  birth-year-empty, and birth-year-populated results are distinguished by
  `collapses_to_ne_v` and `n_eligible_cohorts`.  Direct construction of a record
  whose diagnostics contradict its state now raises `ValueError`.  Estimator
  output is unchanged.

- **Changed: `relationship_kinship_matrix` caches by the codes selected, not by
  the selector written.**  `max_degree=2` and the explicit list of the codes it
  names are one selection, so they now share one cache entry and return the same
  matrix object; previously each selector shape got its own entry and the second
  call recomputed a matrix the graph already held.  Cache entries are keyed by
  the selected codes in registry order, so code order and duplicates in
  `categories=` no longer matter.  Counts and pairs are unchanged.

  Internally the one-of-two `max_degree=` / `categories=` selector is now parsed
  once at the public boundary into a frozen `RelationshipSelection`
  (`_selection.py`) that the pair, count, and matrix engines share, replacing a
  private helper of `_pair_extractor` that the other two modules imported.  A
  one-shot `categories` iterable — a generator, say — is consumed exactly once
  and is now safe at every endpoint.  The public signatures are unchanged and
  the type is private.

- **Changed: the relationship-counting invariants live in the Rust core, not
  the PyO3 binding.**  `_native.relationship_counts` now takes the graph's own
  validated `BuiltPedigree` in place of five loose column arrays, and returns a
  `dict` keyed by registry code instead of one positional int64 array, so a
  reordering of the Rust categories can no longer permute the counts unnoticed.
  The column-length and row-range checks the binding duplicated moved into
  `Pedigree::try_new` / `PedigreeColumns::try_borrow`, the only ways to build
  the core's borrowed engine input now that its slices are private; every Rust
  caller gets them, and columns are checked twice instead of three times.  All
  of this is behind the private `_native` boundary: `PedigreeGraph`,
  `PedigreeView`, and `RelationshipCountResult` are unchanged.

- **Removed: the experimental Python BFS relationship counter** (issue #7).
  `pedigree_graph.experimental.count_pairs_bfs`, the `_bfs_engine` and
  `_bfs_kernel` modules behind it, the `experimental` module that exposed it,
  and the plan fields that described its divergence
  (`REL_PLAN[...].bfs_diverges_under_inbreeding`, `bfs_divergent_codes()`) are
  gone.  The Rust row-streaming engine of 0.8.3 is the one
  relationship-counting implementation; `relationship_counts`,
  `relationship_pairs`, and `estimate_relationship_counts` are unchanged.
  The counter warned `FutureWarning` on every call and was never re-exported
  at the package root, so no deprecation cycle was owed.  It has no
  replacement; use `relationship_counts` for closest-category counts.

## v0.8.3

- **Changed: `relationship_counts` runs in the Rust row-streaming engine
  (ADR 0010, as amended).**  `PedigreeGraph.relationship_counts` and
  `PedigreeView.relationship_counts` no longer build the pair lists of
  `relationship_pairs` and take their lengths.  The engine classifies every
  pair one row at a time, folds closest-category precedence in the row, and
  counts, so peak memory is O(N) whatever the pair density and the call fits
  pedigrees where `relationship_pairs` would not.  The counts are unchanged:
  bit-identical to the block lengths of `relationship_pairs` on every parity
  fixture, every selector, every row order, and every view
  (`tests/test_native_relationship_counts.py`, a live differential against
  the matrix engine, plus `crates/core/tests/parity.rs` against
  `relationship_counts(max_degree=5)` on the 26 dumped fixtures).  A view
  crosses the boundary as a row mask; classification still runs through the
  full graph.  The package thread budget sizes a per-call Rayon pool; the
  integer counts are the same under any budget.
  `RelationshipCountResult.from_pairs` stays, for counting a
  `RelationshipPairs` a caller already holds (simACE's stats runner does);
  `estimate_relationship_counts` is unchanged.

- **Changed: the Rust engine's own semantics are the published ones.**
  `pgr-count` and `crates/core::relationships::count_pairs` return
  closest-category counts; the pre-fold 0.7.1 counts the engine reproduced
  before are no longer a mode.  `tests/parity/dump_relationship_counts.py`
  (0.8 API) writes the fixture oracles from a graph rebuilt from each TSV.

## v0.8.2

- **Changed: pedigree construction runs in the Rust core.**  `from_frame` and
  `from_arrays` still coerce host input in Python (`_input.py`: presence,
  shape, length, and the lossless int64 form of numpy dtypes, pandas nullable
  columns, and host nulls), then hand int64 columns to
  `pedigree_graph._native.build_pedigree`, which applies every pedigree rule of
  ADR 0006 in the 0.8.1 order (per-field range, sex encoding, `duplicate_id`,
  `same_parent_id`, id→row resolution, the topological check and `cycle`
  witness, the four MZ codes, wholly-unknown optional columns collapsing to
  `None`, and `birth_year_topology`) and returns owned numpy columns.  The
  Python `PedigreeInput`, `parse_pedigree_input`, `parse_pedigree_arrays`,
  `validate_id_field`, `IdIndex`, and `PedigreeGraph._validate_birth_year_topology`
  are deleted, not kept as fallbacks; the 0.8.1 rules live on as the readable
  oracle in `tests/oracle/construction.py`, and `tests/test_native_construction.py`
  compares the native builder against it under Hypothesis on structured
  pedigrees with planted defects and on chaos input where defects coexist, down
  to the error message.  Every structured error keeps its code, fields, and
  prose; an unknown `sex_encoding` is still a plain `ValueError`.  Nothing in
  the public API changed.

- **Changed: the private topology kernels reject an out-of-range parent row
  with `ValueError`** instead of indexing past the pedigree.

## v0.8.1

- **Changed: the package is a maturin-built mixed Python/Rust distribution.**
  `pedigree_graph._native` is a PyO3 extension module (`abi3-py313`, so one
  wheel per platform serves every CPython >= 3.13) over the host-neutral
  `crates/core`.  The version is `[workspace.package].version` in the root
  `Cargo.toml`; setuptools-scm is gone.  Binary wheels are published for
  manylinux x86-64 and AArch64, macOS x86-64 and Apple Silicon, and Windows
  x86-64; the sdist needs a Rust toolchain (`rust >= 1.85`).  Nothing in the
  public API changed.

- **Changed: structural depth, the topological-order check, the cycle
  witness, and the depth-major permutation are computed by the Rust core.**
  The numba `_compute_depth` / `_check_topological` kernels and the NumPy
  Kahn peel are deleted, not kept as fallbacks (ADR 0007); readable oracles
  live in `tests/oracle/topology.py` and `tests/test_native_topology.py`
  compares the native kernels against them under Hypothesis, including the
  exact `cycle` witness tuple.  `PedigreeValidationError("cycle")` is now
  raised across the host boundary from a Rust `Error` enum with the same
  code, fields, and prose.  Private `structural_depth(mother_rows,
  father_rows)` dropped its row-count argument and
  `_compute_eqg(m, f, depth, n)` takes the graph's depth instead of
  recomputing it.  At 300k rows the depth kernel is 2.3x faster and the
  permutation 5x faster than the warmed numba path.

## v0.8.0

- **Removed: the 0.7.1 compatibility surface** (ADR 0006, slice 7).  Every
  `0.8.0-DELETE` adapter described in the entries below is gone, the package
  root is frozen to `PedigreeGraph`, `PedigreeView`, `RELATIONSHIPS`,
  `RelationshipCategory`, `RelationshipPairs`, `RelationshipPairBlock`,
  `RelationshipCountResult`, the three errors, and `configure_threads`, and
  `tests/test_architecture_guardrails.py` fails the suite if an old name or
  a delete marker reappears.  Public non-root modules are
  `pedigree_graph.relationships`, `pedigree_graph.summaries`,
  `pedigree_graph.effective_size`, and `pedigree_graph.typing`.

  | 0.7.1 | 0.8.0 |
  |---|---|
  | `PedigreeGraph(data)`, `PedigreeGraph.from_dataframe(df)` | `PedigreeGraph.from_frame(frame_or_dict)`; no sex default, no generation fallback |
  | `from_arrays(ids, mothers, fathers, ...)` and the `mothers=` / `fathers=` / `twins=` keywords | `from_arrays(ids=, mother_ids=, father_ids=, twin_ids=, sex=, generation=, birth_year=, sex_encoding=)`, keyword-only |
  | `PedigreeGraph.from_subsample(full, sub)` | `PedigreeGraph.from_frame(full).view(ids=...)`; pairs, counts, and kinship in view rows |
  | `pg.n`, `pg.mother` / `father` / `twin`, `pg.generation` | `n_individuals`, `mother_rows` / `father_rows` / `twin_rows`, `generation_labels` (or `depth`) |
  | `extract_pairs(max_degree=)` -> `{code: (lo, hi)}` | `relationship_pairs(max_degree=)` -> `RelationshipPairs` of role-oriented blocks, one closest category per pair |
  | `count_pairs(max_degree=)` | `relationship_counts(max_degree=)` -> `RelationshipCountResult` |
  | `count_pairs_streaming(max_degree=)` (unfolded raw counts) | `estimate_relationship_counts(max_degree=)` (fold-aware; MHS / PHS minus the parent-offspring overlap) |
  | `compute_pair_kinship(pairs)` | `pair_kinship(pairs)` or `pair_kinship(first_rows, second_rows)` |
  | `compute_inbreeding()` | `inbreeding()` |
  | `compute_n_ancestors()`, `compute_n_descendants()` (int32) | `distinct_ancestor_counts()`, `descendant_path_counts()` (int64, never overflows) |
  | `per_gen_mean_kinship()` (dense `max(label)+1` array) | `mean_kinship_by_generation()` -> `GenerationKinshipSummary` over observed labels |
  | `kinship_matrix(min_kinship=0)`, `kinship_matrix(min_kinship=t)`, `kinship_matrix(max_degree=d)` | `kinship_matrix()`, `approximate_kinship_matrix(min_propagated_kinship=t)`, `relationship_kinship_matrix(max_degree=d)` |
  | root `compute_all_ne`, `ne_*`, `Ne*Result`, `CohortWindow`, `eligible_cohort_range`, `GenerationInterval` | `pedigree_graph.effective_size.estimate_effective_sizes` and the same names in `pedigree_graph.effective_size` |
  | root `FrameLike` | `pedigree_graph.typing.FrameLike` |
  | `REL_REGISTRY`, `PAIR_KINSHIP`, `RelType` | `RELATIONSHIPS[code]` (`nominal_kinship`, `up`, `down`, `ancestor_count`; collateral categories are stored `up >= down`) |
  | `pedigree_graph._registry.streaming_exact_codes()` | `estimate_exact_codes()` (ADR 0011) |

  Three facts to carry across.  Pair and matrix kinship values are the
  pinned float32 recurrence of ADR 0009: exports that widen them to
  float64 carry float32-origin values, and consumers comparing against a
  float64 recurrence must allow its envelope.  Generation summaries and
  effective-size records are indexed by *observed* generation label
  (`generations` on each record), never by a dense `0 .. max(label)`
  range; a label nobody carries has no row.  `approximate_kinship_matrix`
  has propagation-pruned *candidate* support, not a final-value cutoff:
  a retained coefficient is exact, an absent one may be nonzero, so it is
  not a substitute for `relationship_kinship_matrix` or the complete matrix.

- **Fixed after review of slice 6c.**  A flat mean series (a non-inbred
  pedigree's coancestry or self-coancestry) no longer regresses to an Ne of
  order 1e15 from least-squares noise: the scalar regression reports no
  estimate unless the slope is below `-1e-12`.  `NeHillResult.vk_scaled`
  records the requested `vk_scale` on the collapsed and empty-graph branches
  too.  `GenerationKinshipSummary` compares field-wise (NaN equal to NaN)
  and is unhashable, like the effective-size records, instead of raising on
  `==`.  The `compute_all_ne` adapter's skipped-coancestry sentinel is
  length zero on an empty graph, aligned with every other array, and the
  adapter no longer refuses all eight estimators when one row disables the
  founder-based or sex-dependent ones.

- **Added: `estimate_effective_sizes()`** (ADR 0006, ADR 0007, slice 6c-3).
  `pedigree_graph.effective_size.estimate_effective_sizes(pg, estimators=ALL_EFFECTIVE_SIZE_ESTIMATORS, *, hill_vk_scale=False)`
  runs the selected estimators over one per-call memo of lazily built
  prerequisites (observed cohorts, F, the generation kinship summary, the
  represented founders, the generation and birth-year family tables, the
  founder means, the Caballero-Toro accumulators, the generation interval,
  the cohort window) and returns an `EffectiveSizeResults`: a deeply
  immutable, tuple-backed `Mapping` with all eight keys in canonical order.
  An unselected estimator maps to `UnavailableEffectiveSize(reason="not_requested")`;
  a selected estimator that refused the pedigree with
  `MissingMetadataError` maps to `reason="missing_metadata"` with that
  error's code and fields, each estimator naming itself; any other
  exception propagates.  Every prerequisite is built at most once and
  only when a selected estimator needs it, so Hill's absent-birth-year
  collapse reuses a selected Ne_V result or computes one privately while
  the public key stays `not_requested`.  The selector is any finite
  iterable of names, materialized and validated before any work (`None`
  or a bare string is a `TypeError`, an unknown name a `ValueError`);
  `hill_vk_scale` must be a `bool`.  Direct and orchestrated calls share
  the evaluators, so their records compare equal; the final records define
  NaN-aware equality.  `to_dict()` serializes every nested result.

- **Changed: no worker pool on the final effective-size path** (ADR 0007
  amended).  `estimate_effective_sizes` has no `n_threads`: the estimators
  and prerequisites run serially, the package thread budget is committed
  once after selection is validated, and kernels honor it.  The old pool
  dispatched formulas only after eagerly building the expensive
  prerequisites, and running the kinship, founder, and Caballero-Toro
  prerequisites concurrently would multiply peak memory.
  `benchmarks/bench_effective_size.py` gates the serial path against the
  pooled `compute_all_ne` adapter, which keeps its pool until slice 7.

- **Added: the effective-size metadata matrix** (ADR 0006, slice 6c-2).
  Every estimator in `pedigree_graph.effective_size` validates the metadata
  it needs, in a fixed order, before any work, and raises
  `MissingMetadataError` naming itself in `operation`; the matrix is on the
  module docstring.  New raise sites: `missing_sex` (`status` `"absent"`
  when the graph carries no sex, `"partial"` with the `-1` count) from
  `ne_variance_family_size`, `ne_sex_ratio`, and both Hill branches, since
  dropping unknown rows would change offspring numerators and cohort
  denominators; uniform but known sex stays valid and warns.
  `incomplete_parentage` (new code: `operation`, `affected_count`,
  `first_row`, `first_id`, `represented_parent_role`,
  `unrepresented_parent_role`, `unrepresented_parent_status` of `"missing"`
  or `"external"`) from `ne_long_term_contributions` and
  `ne_caballero_toro` only, because a child with one represented parent
  would keep half its founder ancestry; every other estimator runs.
  `PedigreeGraph.generation_interval` returns `None` only when birth years
  are absent and raises `insufficient_parent_age_data` with the roles that
  have no known-age edge, so Hill's absent-birth-year collapse to Ne_V is
  unchanged while inadequate parent ages surface instead of collapsing.
  `eligible_cohort_range` raises `missing_birth_year` (`status="absent"`)
  and `insufficient_parent_age_data` with both roles instead of plain
  `ValueError`; one role with known ages is enough for its percentile.
  Hill's birth-year branch ignores generation labels.  Empty graphs bypass
  every guard.  The `compute_all_ne` adapter still rejects partly known
  generation labels up front; every other refusal disables only the
  estimator that raised it, which reports `ne=None` in its 0.7.1 record
  (0.7.1 had no metadata refusals, so one such row never blocked the other
  seven estimators there either).

- **Added: `pedigree_graph.effective_size`** (ADR 0006, slice 6c-1).  The
  final effective-size surface: the eight `ne_*` estimators with frozen
  signatures (`ne_long_term_contributions(pg, *, tol=1e-6)` and
  `ne_hill_overlapping(pg, *, vk_scale=False)` take no other arguments; none
  accepts an injected prerequisite) and their result records.  Estimators
  group by the **observed** generation labels: every cohort array carries its
  labels (`generations`, `parent_generations` for the family-size variance,
  `cohort_years` for Hill), is sized by the number of distinct labels rather
  than by the largest value, and the inbreeding, coancestry, and
  Caballero-Toro records name each adjacent transition (`transition_from`,
  `transition_to`).  A transition across a label gap `h` reports the
  per-generation rate `1 - ((1 - x_b) / (1 - x_a)) ** (1 / h)` of Gutiérrez
  et al. 2008; at `h = 1` it is bit-identical to the one-step arithmetic
  (`tests/test_ne_h1_parity.py` pins this against the slice-6b outputs on six
  fixtures).  Scalar regressions use the label offset, so rebasing every
  label changes nothing.  Every result array is an owned read-only copy and
  the record checks its lengths against its labels; Hill's `age_table` is an
  immutable mapping.  An empty graph returns every record with `ne=None` and
  zero-length arrays.  `NeLTCResult.n_iterations` is now the number of
  adjacent-cohort comparisons and `final_generation` names the cohort whose
  vector produced `sum_c_squared`.

- **Changed: founders are represented founders** (slice 6c-1).  A founder is
  a row with no represented mother or father, whatever its label and whether
  its parents are missing or external, where 6b took `generation == 0`.
  Long-term-contribution and Caballero-Toro columns represent **founder
  genomes** (ADR 0008): parentless MZ co-twins share one column, so their
  descendants inherit one lineage, and a founder row is never its own
  descendant.  On a pedigree with parentless MZ pairs this lowers
  `n_founders_with_descendants_per_gen` by the number of pairs and moves the
  long-term-contribution sum of squares in the last bit; on every other
  fixture the outputs are unchanged.  The family-size variance now keeps a
  usable entry for a maximum-labelled parent cohort with represented
  offspring.  Contribution propagation follows structural depth, never the
  labels, so labels that merge depths or place a parent and child in one
  group change only the grouping.

- **0.8.0-DELETE adapters.** The package-root `ne_*`, the root `Ne*Result`
  names, and `compute_all_ne()` keep their 0.7.1 signatures, dense
  `0 .. max(label)` layouts (gaps filled `0.0` for mean F, `NaN` for every
  other statistic, `0` for counts; rates placed at their `transition_to`
  label; variance arrays of length `max(label)`, which drops the new
  maximum-label entry), and the 0.7.1 `n_iterations` meaning.  They call the
  same evaluators as the final surface, so a corrected sparse-gap rate is
  the value they return too.

- **Added: `distinct_ancestor_counts()`, `descendant_path_counts()`, and
  `connected_component_ids()`** (ADR 0006, slice 6b).  The names carry the
  semantics: distinct ancestors count a looped ancestor once (read-only
  int32); descendant paths count a looped descendant once per path
  (read-only int64, so no int32 overflow error on this surface); component
  ids are input-row aligned int64 values, each the smallest original ID in
  the row's *represented* parent-edge component.  External or missing
  parents add no edge, so two rows naming the same external parent stay
  apart, and MZ co-twins join only through their parents.  All three are
  computed once per graph.  `compute_n_ancestors()` and
  `compute_n_descendants()` are now `# 0.8.0-DELETE` adapters returning the
  same values as the 0.7.1 writeable int32 arrays, the descendant adapter
  keeping its `arithmetic_overflow` error.  The component values equal
  fitACE's current `founder_family_ids` construction on every fixture, the
  target for that consumer's migration.

- **Added: `PedigreeGraph.mean_kinship_by_generation()`** (ADR 0006, slice 6a).
  Returns a frozen `pedigree_graph.summaries.GenerationKinshipSummary` with
  read-only `generations` (int32, ascending, observed labels only),
  `mean_kinship` (float64, NaN where no pair is averaged), `pair_counts`
  (int64), and `unlabelled_individual_count`.  Wholly absent generation labels
  fall back to structural depth; partial labels exclude the `-1` rows and
  report their count instead of raising; sparse or rebased labels return only
  the labels some row carries.  An MZ twin pair leaves a group's sum and
  denominator only when both co-twins are in that group; a twin whose partner
  is unlabelled or elsewhere is an ordinary member.  The kinship is streamed
  from the retiring DP, or walked from the complete matrix when that is already
  cached, and the accumulator is sized by the number of distinct labels rather
  than by the largest label value.  Computed once per graph.
  `per_gen_mean_kinship()` is now a `# 0.8.0-DELETE` adapter over the summary:
  same `max(label) + 1` array, same partial-label rejection, same
  `min_kinship` cache, and bit-identical values on the v0.7.1 parity corpus.

- **Breaking: cached kinship-matrix arrays are read-only.** Every matrix
  family marks `data`, `indices`, and `indptr` non-writeable so a caller
  cannot corrupt a cached result in place. `scipy.sparse` constructors do not
  copy by default, so a derived matrix shares those buffers and in-place
  SciPy operations on it now raise: `setdiag` gives
  `ValueError: assignment destination is read-only`, and `eliminate_zeros`
  gives `ValueError: WRITEBACKIFCOPY base is read-only`. Callers that mutate
  a matrix derived from `kinship_matrix()` must copy first, for example
  `sp.csc_matrix(K, copy=True)` or `K.copy()`. This matters across repos:
  `fitACE` mutates a derived GRM in `fitace/kinship/grm_io.py`, on a branch
  reachable only for one-triangle input, which pedigree-graph never returns.

- **Added: `PedigreeGraph.inbreeding()`** (ADR 0006, ADR 0008, slice 5c).  The
  canonical name for the per-individual inbreeding coefficient *F*.  It returns
  a read-only float64 array of length `n_individuals`, computed once and
  memoised, so every later call hands back the same frozen object rather than an
  equal one.  The values are those `compute_inbreeding()` already returned, from
  the genome-node Meuwissen-Luo walk recorded below.  `F_i = 2 * phi(i, i) - 1`
  against `pair_kinship` self pairs and the `kinship_matrix()` diagonal is a
  tested invariant of this surface: exact on the shallow dyadic ADR 0008
  fixtures, and within `2**-22` on the 60-generation closed herd (measured
  1.21e-07 against a bound of 2.38e-07).  Full-graph only, because ADR 0006
  keeps inbreeding off views.  The call commits the package thread budget like
  every 0.8 operation.  Wall time and peak RSS at production scale are in
  `benchmarks/bench_inbreeding.md`, the baseline the ADR 0007 Rust port ports
  against.
- **Changed: `compute_inbreeding()` is a compatibility adapter**
  (0.8.0-DELETE).  It returns exactly the object `inbreeding()` returns, so the
  array is read-only now, as every 0.8 result is, and a consumer that mutated it
  in place must copy first.  Like `compute_pair_kinship`, it preserves 0.7.1
  thread behaviour and leaves the package thread budget uncommitted, as does the
  `compute_all_ne` / `ne_*` surface, which reads F through the same
  non-committing path and keeps its own `n_threads` argument until slice 6c.

- **Added: three explicit kinship-matrix families** (ADR 0006, ADR 0009,
  slice 5b). `kinship_matrix()` is complete;
  `relationship_kinship_matrix(max_degree=...)` or `(categories=...)` contains
  exactly the selected closest-category pairs plus the diagonal; and
  `approximate_kinship_matrix(min_propagated_kinship=0.001)` preserves the
  0.7.1 propagation-pruned candidate support plus the diagonal. The approximate
  threshold applies to intermediate propagation values, is not a final-value
  cutoff, and can admit or omit pairs relative to thresholding
  pedigree-expected coefficients. Every retained value in every family is now
  the pinned float32 recurrence and is bit-identical to `pair_kinship`; the
  approximate family discards the old propagated values after selecting their
  support and captures exact candidate coefficients during one complete
  retiring-DP pass; sparse relationship support uses deterministic bounded pair
  chunks. All matrices
  are cached by operation and selector and return CSC with float32 data, int32
  indices/indptr, sorted rows, and read-only arrays. A zero approximate
  threshold delegates to the complete matrix; non-finite or out-of-range
  thresholds are `ValueError`. The 0.7.1 overloaded
  `kinship_matrix(min_kinship=..., max_degree=...)` remains temporarily, routing
  positive resolved thresholds to the approximate-support family rather than
  describing propagation pruning as exact relationship support. Profiling and
  the runtime/memory decision are recorded in `benchmarks/matrix_exactification.md`.

- **Added: `PedigreeGraph.pair_kinship` and `PedigreeView.pair_kinship`**
  (ADR 0006, ADR 0009, slice 5a).  Three call forms:
  `pair_kinship(first_rows, second_rows)` for any pairs, self pairs included;
  `pair_kinship(block)` for one `RelationshipPairBlock`; and
  `pair_kinship(pairs)` for a whole `RelationshipPairs`, which runs one
  recurrence with one shared memo and returns an immutable mapping over all
  23 codes.  Values are read-only float32, positionally aligned to the input.
  Each value is the pinned float32 recurrence: peel the endpoint with the
  greater structural depth, ties by the greater row, every half-sum rounded
  once to float32.  Within one receiver the value is bit-identical to the
  `kinship_matrix` entry for the same pair, to the reversed endpoint order,
  and to itself before and after a matrix is cached, because the call never
  reads a cached matrix (issue #6).  A returned `0` is an exact `0`.  Two
  graphs built from one pedigree in different row orders agree within
  `2 * (depth_a + depth_b + 1) * 2**-25` on deep inbred pairs (measured: at
  most 2 ulp on the 60-generation closed herd, none on any shallow fixture).
  Widen to float64 before comparing against a non-dyadic cutoff.  A block or
  collection from another receiver fails with `coordinate_space_mismatch`;
  row arguments fail with `invalid_shape`, `invalid_integer_value`,
  `pair_row_out_of_range`, or `pair_length_mismatch`; a memo past its cap
  is `ResourceError("memo_capacity_exceeded")`.  The call commits the package
  thread budget like every 0.8 operation and runs on one thread.
- **Changed: `compute_pair_kinship` returns float32** (0.8.0-DELETE adapter).
  The dict form and the caller-space rows of a `from_subsample` graph are
  kept, but the values are those of `pair_kinship`, so on pedigrees deep
  enough for a kinship to need more than 24 significant bits (about 24
  generations of sustained inbreeding loops) they differ from the 0.7.1
  float64 recurrence within the envelope above.  The arrays are also read-only
  now, as every 0.8 result is, so a consumer that mutated them in place must
  copy first.  The cached-matrix sampling branch is gone; results no longer
  depend on call history.
- **Changed: the pairwise kernel is float32 and runs in the stable
  depth-major order**, where peeling the greater row is the ADR 0009 rule.
  On the `random_30k` fixture (interleaved fresh processes, medians of three)
  the degree-3 batch of 566,720 pairs and the 30,300 self pairs run at the
  0.7.1 kernel's wall time with 25 percent less kernel-attributable RSS
  (560 MB against 747 MB, 289 MB against 384 MB).

- **Fixed: the scalar counter no longer allocates a table the size of the
  largest parent id.**  `_per_sex_anchor_sums` binned per-parent sums by
  original parent id, so a pedigree with ten-digit ids allocated a dense
  table of that size per sex side and per degree (480 MB peak on a 20-row
  pedigree with ids near 10^7).  Parents are now grouped by dense index.
  Counts are unchanged; on the parity fixtures, whose ids start at 10^7,
  `estimate_relationship_counts(max_degree=5)` drops from 0.45 s to 0.04 s
  on 30k rows and from 0.88 s to 0.46 s on 300k rows, with peak RSS at 0.28x
  and 0.42x of before (`benchmarks/bench_estimate_counts.py`, interleaved,
  medians of ten).  This predates 0.8 (`count_pairs_streaming` had it too).

- **Added: `PedigreeGraph.estimate_relationship_counts(max_degree=...)`**
  (ADR 0006, ADR 0011, slice 4c).  The memory-bounded scalar estimate that
  `count_pairs_streaming` computed, returned as a `RelationshipCountResult`:
  `None` above the cutoff, and per code whether the value is `exact`,
  `approximate`, or `clamped` (an inclusion-exclusion residual that
  underflowed and was floored at `0`; that `0` is not a true absence).
  MZ, MO, FO, FS, MHS, and PHS are exact and equal `relationship_counts`
  (the half-sib pairs a parent-offspring category claims under the
  precedence fold are subtracted).  Every other code is approximate: GP,
  GGP, GGGP, and G3GP are raw ancestor-path counts that over-count a pair
  also related at a shorter depth, as a half-sib, or as a closer collateral,
  and the cousin / collateral formulas assume a full complement of known
  ancestors.  The exact set is `REL_PLAN.estimate_exact` /
  `estimate_exact_codes()`; why it excludes the lineal codes is ADR 0011
  (`docs/adr/0011-scalar-estimate-exact-set-excludes-lineal-codes.md`).
  The result for each `max_degree` is computed once per graph and the same
  frozen object is returned afterwards.  The computation, and only the
  computation, emits one `RuntimeWarning` naming the clamped codes in
  registry order when there are any, before the result is cached; a cached
  retrieval is silent, and a different cutoff computes and warns on its
  own.  The call commits the package thread budget like every 0.8
  operation; the counter itself is single-threaded and its integer results
  do not depend on the budget.  Full-graph only.  `count_pairs_streaming`
  is now an adapter over the same computation (the unfolded raw counts, `0`
  above the cutoff, the 0.7.1 dict and scope rules kept, no thread-budget
  commit, no longer written to the matrix count cache), so it can emit that
  `RuntimeWarning` where 0.7.1 wrote a `logging` warning, and it shares the
  per-cutoff cache with the new method.

- **Added: `PedigreeView.relationship_pairs`, `relationship_counts` on both
  receivers, and `RelationshipCountResult`** (ADR 0006, slice 4b).
  `view.relationship_pairs(max_degree=...)` / `(categories=...)` classifies
  through the full graph, so a relationship whose connecting ancestors are
  unselected is still found, then reports only the pairs with both endpoints
  in the view, as view rows (`0 <= row < len(view)`).  Asymmetric blocks keep
  their role orientation; symmetric blocks store `first < second` in view
  rows; every block is sorted by the canonical unordered view-row key and
  carries the view's own coordinate token.  A view of fewer than two rows
  returns all-empty blocks after selector validation.
  `graph.relationship_counts(...)` and `view.relationship_counts(...)` take
  the same selectors and return `RelationshipCountResult`, a frozen mapping
  over all 23 codes to the exact block length (`None` where unrequested),
  with `requested` / `exact` / `approximate` / `clamped` code sets (in this
  release `exact == requested` and the other two are empty).  Exported from
  `pedigree_graph.relationships` and the package root.  The 0.7.1
  `from_subsample` now builds the full graph first and resolves the
  subsample ids through `view(ids=...)`, so a full pedigree that is both
  invalid and missing subsample ids reports the pedigree fault before
  `unknown_view_id`; its `extract_pairs` output is unchanged.

- **Added: `PedigreeGraph.relationship_pairs`, `RelationshipPairs`, and
  `RelationshipPairBlock`** (ADR 0006, slice 4a).  `relationship_pairs(
  max_degree=...)` or `relationship_pairs(categories=...)` (exactly one
  selector) returns an immutable mapping over all 23 registry codes.  Each
  block owns read-only int32 graph rows: for an asymmetric category
  `first_rows` carries `first_role` (offspring, descendant, niece_nephew,
  junior_cousin) and `second_rows` the counterpart; a symmetric category
  stores `first < second`.  Every unordered pair appears in exactly one
  category (lowest degree, then registry order), a pair valid in both
  orientations of an asymmetric category appears once with the lower graph
  row first, and blocks are sorted by the canonical unordered row key, so the
  output is bit-identical across thread counts.  Selection is an output
  filter: the closer categories a selected one depends on are always
  resolved.  Unselected blocks are empty with `requested=False`.  The engine
  honours `configure_threads`.  Both types are exported from
  `pedigree_graph.relationships` and the package root.  `extract_pairs` and
  `sibling_pairs` keep their 0.7.1 orientation and membership until 0.8.0
  removes them, but within a code they may now emit pairs in a different
  order.

- **Added: a Rust row-streaming relationship engine with exact,
  memory-bounded pair counts** (ADR 0010; issues #11 and #9).  `crates/core`
  (`pedigree-graph-core`) classifies every relationship pair up to degree 5
  one individual at a time, so peak memory is linear in the pedigree size:
  the 20M-row simACE pedigree counts all 23 categories exactly in 83 s on 12
  threads within 2.9 GiB, where the matrix engine needed an estimated
  150 GiB.  Counts are bit-identical to `count_pairs(max_degree=5)` on every
  parity fixture and bit-identical across thread counts.  Path multiplicity
  is saturated at two, which is provably exact for the engine's predicates
  and removes the overflow question of issue #9.  Not yet reachable from
  Python: `pgr-count` is a benchmark and parity CLI over the array dump from
  `tests/parity/dump_relationship_inputs.py`; the binding lands with the
  native scaffold.  The pixi manifest now provides the Rust toolchain.

- **Added: canonical construction** (ADR 0006).  `PedigreeGraph.from_frame(frame,
  *, sex_encoding="simace")` takes a dict of columns or any FrameLike table, and
  `PedigreeGraph.from_arrays(*, ids=, mother_ids=, father_ids=, twin_ids=None,
  sex=None, generation=None, birth_year=None, sex_encoding="simace")` takes the
  columns separately.  Neither applies a default: an omitted or wholly unknown
  `sex`, `generation`, or `birth_year` reads as absent rather than as a
  fabricated column.  The 0.7.1 entry points — `PedigreeGraph(data)`,
  `from_dataframe`, `from_subsample`, and the positional
  `from_arrays(ids, mothers, fathers, ...)` — keep their names, their
  depth-derived `generation` fallback, and their all-female `sex` default until
  0.8.0 removes them.  `from_arrays` serves both call forms: the canonical one
  is keyword-only and the 0.7.1 one is selected by any positional argument or by
  `mothers=`/`fathers=`/`twins=`.  Mixing the two, or naming neither, is a
  `TypeError`, as is passing `sex_encoding=` to the 0.7.1 form.

- **Added: read-only properties** (ADR 0006).  A graph now exposes `ids`,
  `mother_ids` / `father_ids` / `twin_ids` (int64, `-1` missing), `mother_rows` /
  `father_rows` / `twin_rows` (int32, `-1` absent or external), `sex` (int8 or
  `None`), `depth` (int32, always present), `generation_labels` (int32 or
  `None`), `birth_year` (int32 or `None`), `n_individuals`, and `len(pg)`.  Each
  array property hands back the graph's own storage — the same object on every
  access, read-only, so writing into what you read raises instead of silently
  changing the graph.  `depth` is structural and is computed on first access,
  never at construction.  The 0.7.1 names `mother`, `father`, `twin`,
  `generation`, and `n` still work and still mean rows, depth-fallback labels,
  and the row count; 0.8.0 removes them.

- **Added: canonical relationship registry** (ADR 0006, slice 2).  The root
  now exports `RELATIONSHIPS`, an immutable ordered mapping of all 23
  relationship codes to frozen `RelationshipCategory` records, also importable
  from `pedigree_graph.relationships`.  Each record carries `code`, `label`,
  `degree`, `nominal_kinship`, `up`, `down`, `ancestor_count`, `first_role`,
  and `second_role` (roles drawn from the closed `RelationshipRole` literal set
  exported by `pedigree_graph.relationships`), replacing the two parallel `REL_REGISTRY` / `PAIR_KINSHIP`
  lookups with one.  Iteration order is the documented same-degree precedence
  for closest-category classification.  `first` is the pair member with at
  least as many meioses to the shared ancestor(s), so `up` counts meioses from
  `first` up to the ancestor(s), `down` counts them from the ancestor(s) down
  to `second`, and `up >= down` holds for every category.  This flips the
  stored collateral orientation: `Av` is now `up=2, down=1`, where 0.7.1's
  `RelType` stored `up=1, down=2`.  Asymmetric categories name their two
  positions (`offspring`/`mother`, `offspring`/`father`,
  `descendant`/`ancestor`, `niece_nephew`/`aunt_uncle`,
  `junior_cousin`/`senior_cousin`, where junior means generationally further
  from the shared ancestors, not younger by birth year); the seven symmetric
  categories carry `None` for both and report `symmetric` as `True`.
  `REL_REGISTRY`, `PAIR_KINSHIP`, and `RelType` keep their 0.7.1 names, values,
  and orientation, but are now detached snapshots built once from
  `RELATIONSHIPS`: mutating them no longer changes the registry or any engine
  output.  0.8.0 removes all three.

- **Added: pedigree views** (ADR 0006, slice 3).  `graph.view(ids=[...])` and
  `graph.view(rows=[...])` return a `PedigreeView` over exactly that selection,
  in exactly the order given; naming both keywords, or neither, is a
  `TypeError`.  An empty selection of any dtype is a valid empty view.  A view
  exposes read-only `ids` (int64) and `graph_rows` (int32), each its own
  contiguous storage handed back unchanged on every access, plus
  `n_individuals` and `len(view)`; mutating the selection array afterwards
  cannot change the view.  A bad selection raises one of four structured
  errors: `duplicate_view_id`, `unknown_view_id` (which is also how a negative
  id reads), `duplicate_view_row`, and `view_row_out_of_range` (there is no
  negative indexing).  Each selector checks single entries before pairs:
  membership or range first, duplicates last, so a value too large for
  int64 reads as unknown or out of range rather than as a fifth code.  The
  graph memoises a sorted-id index on the first `ids=` view, so later id
  views cost the selection's size, not a sort of the whole pedigree.
  Shape and lossless-integer failures report
  `invalid_shape` and `invalid_integer_value` naming the `ids` or `rows`
  argument.  Each graph and each separately built view owns a distinct opaque
  coordinate token, so equivalent selections from two `view(...)` calls are not
  interchangeable receivers.  Relationship methods on views arrive in a later
  slice, and `from_subsample` is unchanged for now.

- **Changed: MZ pairs are validated at construction.**  Every constructor now
  rejects a represented MZ reference that is self-directed
  (`mz_self_reference`), not reciprocated (`mz_nonreciprocal`, which is also how
  a third row pointing into a pair is reported), names different parents
  (`mz_parent_mismatch`, with the offending roles in `parent_roles`), or pairs
  two individuals of different known sex (`mz_sex_mismatch`).  Parents are
  compared by id, so co-twins sharing one unrepresented parent agree.  A co-twin
  outside the represented rows forms no pair and is not checked.
  `compute_inbreeding()` no longer performs this check; it raised
  `mz_nonreciprocal` and `mz_parent_mismatch` lazily in 0.7.1, and a pedigree
  that used to construct and fail later now fails at construction.

- **Changed: partly known generation labels are rejected.**  A supplied
  `generation` column containing `-1` now raises
  `MissingMetadataError("missing_generation_labels", status="partial")`, with
  `missing_count`, from `per_gen_mean_kinship()`, every generation-indexed Ne
  estimator, and `compute_all_ne`.  Previously the `-1` rows wrapped into the
  last cohort bucket of the kinship DP theta sums and the Caballero-Toro
  founder sweep, silently biasing `ne_caballero_toro` and `ne_inbreeding`,
  while `ne_coancestry` and `per_gen_mean_kinship()` failed with an
  unstructured `ValueError` from `np.bincount`.  A wholly absent column is
  unchanged: the 0.7.1 estimators still fall back to structural depth.

- **Added: `configure_threads` is exported from the package root.**  One
  package-wide budget resolving `configure_threads(n)` >
  `PEDIGREE_GRAPH_THREADS` > `1`, committed the first time it is read.
  Repeating the committed value is accepted; changing it afterwards raises
  `RuntimeError`.  There is no per-call thread argument (ADR 0007).

- **Changed: any acyclic input row order is accepted.**  A parent no longer
  has to precede its child; construction rejects only genuine cycles.  Public
  outputs stay aligned to the input rows.  Internally the graph derives one
  private stable depth-major order (`_topology.build_topology`) and runs the
  kernels that need parents first — the Meuwissen-Luo inbreeding walk, the
  descendant path-count sweep, the pairwise kinship recurrence, the kinship
  DP, and the Caballero-Toro founder sweep — in that order, mapping their
  results back to graph rows.  Ties inside a depth keep input row order and no
  original id ever enters the ordering.

- **Changed: a supplied `generation` label no longer affects any relationship
  or kinship output.**  Structural depth drives the kinship DP and every other
  order-dependent kernel; labels are metadata.  `per_gen_mean_kinship()` still
  groups its cohorts by the supplied label, and so do the label-indexed
  effective-size results.  Callers who passed labels that disagreed with
  structural depth will see kinship and relationship results change to the
  structurally correct values.

  Two floating consequences of accepting arbitrary order, both governed by
  ADR 0009: two graphs built from the same pedigree in different row orders
  agree exactly on every integer, category and pair result, and agree on
  kinship within the recurrence envelope
  `abs(a - b) <= 2 * (depth_a + depth_b + 1) * 2**-25`.  The propagated
  `kinship_matrix(min_kinship > 0)` support is approximate by construction
  (ADR 0005) and is the one output whose *support* a permutation can move,
  by a fraction of a percent on the test corpus.

- **Added: structured errors** (ADR 0006).  `PedigreeValidationError` and
  `MissingMetadataError` (both `ValueError`) and `ResourceError` (a
  `RuntimeError`) are exported from the package root.  Each carries a stable
  `.code` string and an immutable `.fields` mapping naming the offending
  field, row, value, or limit; messages are prose and are not a contract.
  Construction failures, `max_degree` range failures, MZ-invariant failures,
  and the `compute_n_descendants()` int32 overflow (previously an
  `OverflowError`) now raise these instead of bare `ValueError` /
  `OverflowError`.  Tests should assert `.code`, not the message text.

- **Changed: only `id`, `mother`, and `father` are required.**  `twin`,
  `sex`, `generation`, and `birth_year` are optional in every dict and frame
  input; a frame column that is wholly missing (all `-1` or all host nulls)
  now reads exactly like an omitted one.  The 0.7.1 attributes keep their
  0.7.1 defaults for now — `pg.sex` is all-female, `pg.generation` falls back
  to structural depth — but `pg.birth_year` is `None` for a wholly unknown
  birth-year column, where 0.7.1 returned an all-`-1` array.

- **Changed: numeric input is coerced losslessly and range-checked.**
  Integer, integral-float, and object columns (pandas nullable, mixed
  lists) are accepted; polars nulls and pandas `pd.NA` become the `-1`
  missing sentinel everywhere except `id`, which has no missing value.
  Non-integral floats, infinities, `bool` columns, strings, `uint64` values
  above the int64 maximum, and out-of-range values are rejected with
  `invalid_integer_value` or `value_out_of_range` naming the position.
  `sex_encoding="plink"` maps `1 -> 1` male, `2 -> 0` female, `0 -> -1`
  unknown; the default `"simace"` encoding stores `0` / `1` / `-1` as given.

- **Added: cyclic parent references are rejected** with a `cycle` error
  carrying one deterministic witness — the tuple of ids around the cycle,
  the same for a given graph whatever order its rows arrive in.  A child
  naming one id in both parent roles is `same_parent_id`, external ids
  included.

- **Changed: construction owns its arrays.**  Every column is copied into
  contiguous, read-only storage, so mutating the caller's arrays after
  construction cannot change the graph.

- **Changed: `compute_inbreeding()` is MZ-aware** (#8, ADR 0008).  The
  Meuwissen–Luo walk now runs over the genome-node pedigree, in which MZ
  co-twins share one node, so `F` equals `2 * phi(i, i) - 1` from
  `compute_pair_kinship()` and from the `kinship_matrix()` diagonal on every
  pedigree.  Previously co-twins were walked as two individuals: twins with
  parents counted as full sibs and founder co-twins as unrelated founders,
  so an inbreeding path through both members of an MZ pair was
  under-weighted or missed.

  **Numeric change for existing callers** on pedigrees containing MZ twins:
  `F` never decreases; on a 1M-row simACE pedigree 0.12% of individuals
  change, by at most 1/128, and mean `F` moves under 1%.  Everything
  derived from `compute_inbreeding()` (`ne_inbreeding`,
  `ne_individual_delta_f`, the Caballero–Toro estimators, pedsum's
  inbreeding section) shifts accordingly.  Wall time is unchanged; peak
  memory rises by about 2% on pedigrees that contain twins.

  `compute_inbreeding()` now raises `PedigreeValidationError`
  (`mz_nonreciprocal` or `mz_parent_mismatch`, both `ValueError`s) when a
  represented MZ reference is not reciprocal or the co-twins do not share
  both parent rows.  An absent co-twin (`twin == -1`) is not an MZ pair.

## v0.7.1

- **Fixed: founder MZ co-twins were dropped by the kinship DP** (#5).  The MZ
  twin pass ran only inside the depth ≥ 1 loop, so a twin pair sitting at
  depth 0 — both co-twins founders — never had its off-diagonal written.
  `kinship_matrix()` returned `0.0` for the pair on the capped and uncapped
  paths alike, disagreeing with `compute_pair_kinship()`, which was correct.

  The missing edge was not the whole cost: because it was absent at depth 0,
  every merge walk below the pair propagated the zero, so **descendants of
  founder co-twins were unrelated to each other** in the returned matrix
  (children of co-twins: `0.0` against an exact φ = 0.125).  Twins with
  parents were never affected.

  The pass is now a shared `_mz_twin_pass()` run once per depth, depth 0
  included, before that depth's retirement.

  **Numeric change for existing callers** on pedigrees containing founder
  co-twins: `kinship_matrix()` gains those entries and everything derived from
  them, and `per_gen_mean_kinship()` rises accordingly for the generations
  below such a pair.  Pedigrees with no founder co-twin pair are unaffected.

## v0.7.0

- **Structural frame protocol (`FrameLike`), exported.**  Every constructor
  (`PedigreeGraph(...)`, `from_dataframe`, `from_subsample`) now accepts any
  column-addressable table exposing `.columns`, string `__getitem__`, and
  column `.to_numpy()` — pandas *and* polars DataFrames both qualify — while
  the package continues to import neither frame library at runtime.
  `dict[str, np.ndarray]` input remains accepted everywhere.  Columns are
  extracted via `.to_numpy()` (previously pandas-only `.values`); NA-free
  pandas nullable-integer columns are now accepted.  A frame missing a
  required column now reports the uniform `ValueError` instead of a raw
  `KeyError`.  `from_dataframe` is kept as a compatibility name.
  Coercion lives in the new focused `pedigree_graph/_frames.py` module.
- Test fixtures serve polars frames (the family's primary library);
  focused pandas compatibility coverage — including nullable integers —
  lives in `tests/test_frame_inputs.py`.  `polars` joins pandas in the
  `test` extra only; runtime dependencies are unchanged.

## v0.6.0

- **First release published to PyPI.**  `pip install pedigree-graph` now
  works, retiring the `git+https://...@vX.Y.Z` install form that consumers
  had to carry because the project was unavailable on the index.
  Distributions are built and uploaded by a tag-triggered GitHub Actions
  workflow using PyPI trusted publishing (OIDC), so no API token is
  stored in the repository or in CI secrets.  Downstream packages pinning
  a git URL can move to a version range such as
  `pedigree-graph>=0.6,<0.7`; the bound is worth keeping tight because
  `PAIR_KINSHIP` and `extract_pairs` are consumed directly by simace,
  fitace, and pedsum.
- **Packaging metadata completed for the index listing.**  `readme`,
  `license` (SPDX `MIT`), `license-files`, `authors`, `classifiers`, and
  `[project.urls]` are now declared, so the PyPI page renders the README
  and links back to the repository and this changelog.  The build backend
  floor moved from `setuptools>=64` to `setuptools>=77`, which is where
  PEP 639 SPDX license support lands.
- **`py.typed` marker added.**  The package now advertises inline type
  information under PEP 561, so type checkers read its annotations when
  it is installed as a wheel rather than as an editable source checkout.
  Runtime code is unchanged.

## v0.5.4

- **Self-kinship diagonal fixed for inbred individuals when rows are not
  in generation-monotonic order.**  The matrix-DP kinship kernel assumed
  every relative discovered during a row's merge walk had a smaller row
  index, but `from_arrays` only requires topological order (parents
  before children).  When a relative at an earlier generation had a
  higher row index, the diagonal append broke the row's sorted order and
  the binary search reading `phi(mother, father)` silently returned 0 —
  so the self-kinship diagonal read `(1+0)/2` instead of `(1+F)/2`, and
  the GRM diagonal consumed downstream was wrong for inbred individuals.
  Off-diagonals and the pairwise `compute_pair_kinship` path were
  unaffected, as was any pedigree loaded in generation order (the common
  case, which now hits a zero-overhead fast path).

- **Pair-set subtraction and id validation no longer go through
  `np.unique`/`np.isin`.**  Three internal hot spots were rewritten with
  sort/searchsorted or diff-based equivalents, verified output-identical
  on real data and under differential fuzzing (5,000 randomized trials
  per site, including empties, duplicates, and ids near the int64
  pair-key bound):
  - `PedigreeGraph._subtract_pairs` now sorts the remove keys and
    binary-searches candidates instead of `np.isin` (~6.9x on a 300k-N
    pedigree; halves `sibling_pairs()` wall time).  Same pattern —
    and rationale — as `extract_from_sparse` already used.
  - The duplicate-id check in `_validate_id_column` counts equal
    adjacent elements after a sort instead of `len(np.unique)` (~37x).
  - `pairs_from_groups` detects group boundaries by diffing its
    already-sorted key array instead of re-running `np.unique` on it.

  Returned pairs, counts, orderings, and error messages are unchanged —
  this change is performance-only.

## v0.5.3

- **`count_pairs_streaming` warns when a cousin/collateral residual
  underflows.**  The scalar engine derives `H1C`, `1C1R`, `1C2R`, and
  `H1C1R` by inclusion–exclusion — subtracting closer-relationship
  contributions with fixed coefficients that are exact only on non-inbred,
  single-mating pedigrees.  On inbred or structurally complex real
  pedigrees those corrections can over-count, driving the raw residual
  negative; it was then silently clamped to `0`, indistinguishable from a
  true absence (e.g. millions of `1C` but `H1C == 0`).  The clamp now logs a
  `WARNING` naming the code and the underflow magnitude and points to the
  matrix engine (`extract_pairs`) for an exact count.  Returned counts are
  unchanged — only the diagnostic is new.

## v0.5.2

- **`count_pairs_streaming()` releases its transient matrices on exit.**
  The scalar streaming counter builds the adjacency powers `_A`…`_A5` and
  now drops them via `_release_pair_matrices()` before returning, exactly
  as `extract_pairs()` already did.  Previously they stayed resident for
  the graph's lifetime, inflating peak memory of any later inbreeding / Ne
  / lineage work on the same graph (~400–520 MiB on a 1M-row pedigree).
  The counts remain cached and the matrices rebuild lazily via
  `_ensure_parent_csr()` if pair work runs again, so callers that reached
  into the private `_release_pair_matrices()` after a streaming call (e.g.
  pedsum `summarize`) can drop that workaround.  Fixes #4.

## v0.5.0

- **Registry-aligned `max_degree` semantics.**  `extract_pairs`,
  `count_pairs`, and `count_pairs_streaming` now include exactly the
  relationship categories whose `REL_REGISTRY[code].degree` is less
  than or equal to the cutoff.  `max_degree=0` is MZ-only,
  `max_degree=2` stops before 1st cousins, and `max_degree=3`
  includes 1st cousins plus the other degree-3 categories.  The public
  defaults changed from `2` to `3` to preserve the old default behavior
  of including 1st cousins.

- **`PedigreeGraph.compute_n_ancestors()`** — new cached method.
  Returns the per-individual count of *distinct* strict ancestors
  (`int32`, length `n`).  Backed by a sparse boolean transitive
  closure of the parent graph; memory scales with the total closure
  size.  Suitable for pedigrees up to ~1M rows on commodity hardware;
  deeper / wider pedigrees may need a future retirement-style DP
  variant.

- **`PedigreeGraph.compute_n_descendants()`** — new cached method.
  Returns the per-individual descendant *path count* (`int32`, length
  `n`).  In non-inbred pedigrees this equals the unique-descendant
  count; in inbred pedigrees it over-counts a descendant reachable via
  multiple ancestor paths.  Matches the convention used historically
  by `pedsum` (`compute_descendants`) and by the matrix engine's GP /
  Av / 1C pair counts.  Raises `OverflowError` if any per-individual
  path count exceeds `int32` max (the kernel accumulates in `int64`
  and the cast happens after a bounds check, so deeply inbred
  pedigrees cannot silently wrap).

- **`PedigreeGraph.from_arrays(...)`** — accepts a new optional `sex`
  kwarg (`np.ndarray | None`).  When omitted, behaviour is unchanged
  (sex defaults to zeros).  Existing callers do not need updates.

- **Defensive warning for the `sex`-default foot-gun.**
  ``ne_sex_ratio`` and ``ne_variance_family_size`` now emit a
  ``RuntimeWarning`` when ``pg.sex`` is uniformly 0 or 1 — almost
  always a sign that the caller forgot to pass ``sex=`` to
  ``from_arrays`` and is consuming silently-degenerate (single-sex)
  Ne results.  The estimator return values are unchanged (``ne=None``);
  the warning is the new diagnostic.  Kinship-only callers
  (relationship-pair extraction, GRMs, PA-FGRS) are not affected
  because they don't invoke the sex-aware estimators.

- New private kernel module `pedigree_graph/_lineage_kernel.py` houses
  the descendant (numba-JIT) and ancestor (scipy sparse) primitives.

- **`PedigreeGraph.count_pairs_streaming(max_degree=2, scope="full")`**
  — new method.  Memory-bounded relationship pair counts via pure
  scalar arithmetic; no pair-key arrays are ever materialized.  Peak
  memory is O(N) regardless of pedigree density.  Returns all 23
  codes from `REL_REGISTRY`.  Bit-identical to `count_pairs` for
  the 10 simple codes (`MZ`, `MO`, `FO`, `FS`, `MHS`, `PHS`, `GP`,
  `GGP`, `GGGP`, `G3GP`); approximate (~1% on deep low-inbreeding
  pedigrees) for the 13 cousin / collateral codes (`Av`, `1C`,
  `H1C`, `HAv`, `GAv`, `GGAv`, `G3Av`, `HGAv`, `HGGAv`, `1C1R`,
  `H1C1R`, `1C2R`, `2C`).  The scalar path is **full-graph only**:
  `scope='subsample'` raises `NotImplementedError` on graphs built
  via `from_subsample` (use `count_pairs` for subsample-restricted
  counts).  See `LIMITATIONS.md` for the full precision contract.
  Benchmark: 5 seconds on a 783K-row stallion-heavy livestock
  pedigree where both matrix and BFS engines OOM at 30 GB.

- **`max_degree` validation** — `extract_pairs`, `count_pairs`, and
  `count_pairs_streaming` now reject `max_degree` outside `[0, 5]`
  with `ValueError`.  Degree 0 is accepted (cheap codes MZ / MO /
  FO / FS are computed regardless; the cap controls the expensive
  matrix products at degree 2 and above).
