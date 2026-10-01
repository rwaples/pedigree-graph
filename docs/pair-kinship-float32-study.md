# pair_kinship float32 decision study

Date: 2026-09-04. The evidence behind [ADR 0009](adr/0009-kinship-is-a-pinned-float32-recurrence.md),
kept as a frozen record; written as a simACE planning draft and moved here on
2026-10-01 so the ADR's citation resolves. Investigation output only; no
production code was changed. The instruments (`f32kernel.py`, `corpus.py`,
`study.py`, `characterize.py`, `prep.py`, `run_one.py`, `bench.sh`,
`summarize.py`, `consumers/`) were throwaway scratch files and were not kept,
so the figures below cannot be re-derived from this repository. Paths such as
`results/...` and `scratchpad/f32/...` refer to the simACE working tree of
that session.

## Environment

| Item | Value |
|---|---|
| pedigree-graph | `4fd07ba` plus the concurrent inbreeding session's uncommitted ADR 0008 edits (a comment-only change in `_kinship_pairwise.py`) |
| fitACE | `2289cd6` |
| pedsum | `f2fb861` |
| Python / numpy / numba | 3.13.15 / 2.5.2 / 0.67.0, via `pixi run --manifest-path external/pedigree-graph/pixi.toml` |
| Machine | 12 cores, 30 GB RAM |

## Candidates

- `direct64`: current float64 recurrence (`pairwise_kinship`).
- `rounded32`: `direct64.astype(np.float32)`. Reference for an output-only float32 API.
- `internal32`: experimental twin of the numba kernel with a float32 memo and float32 arithmetic at every combine (`f32kernel.py`). A pure-Python float32 recurrence was used as its bit-oracle; the two agreed on every case where the oracle ran.
- `matrix32`: values sampled from the complete float32 `kinship_matrix(0.0)`.

## Correctness corpus

Commands:

```
pixi run --manifest-path external/pedigree-graph/pixi.toml python study.py
pixi run --manifest-path external/pedigree-graph/pixi.toml python characterize.py
```

All 23 registry motifs, founders, one-known-parent, sibling mating, selfing chain (30), backcross to one founder (40), double first cousins, MZ founders and descendants, deep MZ descendants, the seven ADR 0008 MZ fixtures, a 60-generation two-line multipath lineage, a 50-generation closed herd, and 120 random pedigrees (40 shallow, 40 deep inbred with a 12-row mating window, 40 MZ-heavy). Every unordered pair including self-pairs was requested, forward and reversed.

| case | n | pairs | internal32 != rounded32 | max ulp | rounded32 != direct64 | max abs | matrix32 != rounded32 | max ulp | public after-matrix != fresh |
|---|---|---|---|---|---|---|---|---|---|
| selfing_chain_30 | 31 | 496 | 0 | 0 | 28 | 2.98e-08 | 0 | 0 | 28 |
| backcross_to_founder_40 | 43 | 946 | 0 | 0 | 341 | 2.98e-08 | 0 | 0 | 341 |
| deep_lineage_60_multipath | 124 | 7750 | 633 | 1 | 2925 | 2.98e-08 | 633 | 1 | 2925 |
| closed_herd_deep | 304 | 46360 | 18197 | 2 | 39685 | 2.98e-08 | 18197 | 2 | 39709 |
| rand_deep_inbred | 118 | 96925 | 4675 | 2 | 15474 | 2.98e-08 | 4671 | 2 | 15562 |

All other cases were bit-identical across every candidate and comparison, zero differences in every column (38 rows: registry:MZ, registry:MO, registry:FO, registry:FS, registry:MHS, registry:PHS, registry:GP, registry:Av, registry:GGP, registry:HAv, registry:GAv, registry:1C, registry:GGGP, registry:HGAv, registry:GGAv, registry:H1C, registry:1C1R, registry:G3GP, registry:HGGAv, registry:G3Av, registry:H1C1R, registry:1C2R, registry:2C, founders_only, one_known_parent, sib_mating_5gen, double_first_cousins, mz_founders_descendants, mz_descendants_deep, adr8:mz_ancestry_no_loop, adr8:mz_only_link, adr8:mz_plus_full_sib_loop, adr8:double_mz_grandparents, adr8:founder_mz_twins_only_link, adr8:inbred_twins, adr8:twins_mate_each_other, rand_shallow, rand_mz_heavy). Random rows aggregate 40 pedigrees each (max n shown, pair counts summed).

Observations:

- Forward versus reversed endpoint order: zero differences for every candidate on every case.
- Zero flips (`rounded32 == 0` differing from `direct64 == 0`): none.
- `rounded32` error: max abs 2.98e-8 (2^-25), max rel 6.0e-8. Every non-representable value has more than 24 significant bits in its dyadic expansion, and these occur only in the deep inbred constructions (selfing chain, 40-generation backcross, deep lineage, closed herd, random deep inbred). Nothing in the registry motifs, MZ fixtures, or shallow random pedigrees is affected.
- `internal32 != rounded32` on the deep cases: up to 2 float32 ulps, with an absolute error up to 4.9 times the correctly rounded error (1.46e-7 versus 2.98e-8 on the closed herd). Internal float32 is therefore not correctly rounded; per-step rounding accumulates.
- `matrix32 != rounded32` on the same deep cases (1 to 2 ulps). `matrix32` mostly equals `internal32` but not always (248 differences on the random deep inbred set), because the DP computes `(mv + fv) / 2.0` in float64 before narrowing while the experimental kernel rounds each operand.
- Issue #6 call history: on the deep cases the public `compute_pair_kinship` returns different float64 values before and after `kinship_matrix(0.0)` has been built (float32-rounded values widened to float64). Casting the after-matrix values to float32 does not recover `rounded32` either. On every shallow case and every simACE pedigree the two paths agree bit-for-bit.

## simACE pedigrees at scale

Pedigrees: `results/dev/dev_mean_n10k/rep1` (20,400 rows), `results/base/baseline10K/rep1` (53,466), `results/dev/dev_laplace_am_strong_50k/rep1` (102,000), `results/base/baseline100K/rep1` (536,036). All have 5 generations. Pair batches: all pairs extracted at `max_degree=3`, all self-pairs, 10,000 random pairs, degree-3 pairs restricted to the last generation, and degree-3 plus self (the fitACE matrix builder's call shape). The 20k pedigree was also run at `max_degree=5` (pedsum's shape, 1,595,199 pairs, 23 codes).

Parity (identical result in every batch at every size):

| Metric | Value |
|---|---|
| `rounded32 != direct64` | 0 |
| `internal32 != rounded32` | 0 |
| forward != reversed | 0 |
| zero flips | 0 |
| distinct kinship values per batch | 9 to 112 |
| smallest nonzero value | 1/64 at degree 5 |

Complete matrix at 20k (13.2M nnz) and 53k (39.4M nnz): `matrix32 == rounded32` bit-for-bit, and the public call after matrix construction equals the fresh call. No matrix was built at 102k or 536k.

Memo size (`_pairwise_kinship_with_stats`) for the degree-3 batch: 3.6M entries at 20k, 12.7M at 53k, 23.3M at 102k, 145M at 536k (capacity 2^28 slots). The 5.36M-row `baseline1M` pedigree was not attempted: linear extrapolation puts the memo near 1.5G entries, above the 2^31 slot cap.

## Production-scale benchmark

Each cell is a fresh process (`run_one.py`) that warms Numba on a 4-row pedigree, loads the pre-extracted pair arrays from `prep_<tag>.npz`, and times one kernel call. Baseline (`d64`, the production kernel) and candidate (`i32`) runs were interleaved per rep. Peak RSS is `ru_maxrss` after the call; the kernel-attributable column subtracts `ru_maxrss` measured just before the call. The 20k and the first 53k pass ran while another job was active; the 53k, 102k and 536k figures below are from a quiet rerun. Commands:

```
./bench.sh p20k 5 ext3 self sparse10k lastgen_ext3 ext3_plus_self
./bench.sh p53k 5 ext3 self sparse10k lastgen_ext3
./bench.sh p102k 5 ext3 self sparse10k lastgen_ext3
./bench.sh p536k 3 ext3 self sparse10k lastgen_ext3
python summarize.py bench_p*.jsonl
```

| pedigree | batch | reps | wall d64 s (spread) | wall i32 s (spread) | wall Δ | peak RSS d64 MB | peak RSS i32 MB | RSS Δ | kernel-attributable RSS d64 → i32 MB | output d64 → i32 MB | cast ms |
|---|---|---|---|---|---|---|---|---|---|---|---|
| p20k | ext3 | 5 | 2.00 (0.10) | 1.99 (0.13) | -0.3% | 361 | 313 | -13.3% | 192 → 144 | 3.09 → 1.55 | 0.5 |
| p20k | self | 5 | 0.25 (0.03) | 0.24 (0.04) | -5.1% | 211 | 200 | -5.4% | 48 → 37 | 0.16 → 0.08 | 0.0 |
| p20k | sparse10k | 5 | 0.88 (0.05) | 0.84 (0.12) | -4.6% | 260 | 236 | -9.2% | 98 → 74 | 0.08 → 0.04 | 0.0 |
| p20k | lastgen_ext3 | 5 | 1.04 (0.14) | 0.99 (0.09) | -5.5% | 260 | 236 | -9.1% | 97 → 74 | 0.12 → 0.06 | 0.0 |
| p53k | ext3 | 5 | 5.68 (1.04) | 5.63 (1.01) | -0.8% | 948 | 756 | -20.2% | 769 → 577 | 8.44 → 4.22 | 1.4 |
| p53k | self | 5 | 0.73 (0.09) | 0.64 (0.13) | -11.9% | 357 | 309 | -13.4% | 193 → 145 | 0.43 → 0.21 | 0.0 |
| p53k | sparse10k | 5 | 0.45 (0.11) | 0.43 (0.08) | -4.2% | 261 | 237 | -9.1% | 98 → 74 | 0.08 → 0.04 | 0.0 |
| p53k | lastgen_ext3 | 5 | 2.24 (0.17) | 2.18 (0.20) | -2.8% | 549 | 453 | -17.5% | 385 → 289 | 0.37 → 0.19 | 0.1 |
| p102k | ext3 | 5 | 17.01 (0.29) | 16.71 (0.11) | -1.8% | 963 | 771 | -19.9% | 770 → 578 | 15.44 → 7.72 | 2.3 |
| p102k | self | 5 | 1.02 (0.02) | 0.98 (0.05) | -3.7% | 359 | 311 | -13.4% | 194 → 146 | 0.82 → 0.41 | 0.1 |
| p102k | sparse10k | 5 | 0.36 (0.03) | 0.35 (0.01) | -4.3% | 263 | 239 | -9.1% | 99 → 75 | 0.08 → 0.04 | 0.0 |
| p102k | lastgen_ext3 | 5 | 4.41 (0.08) | 4.24 (0.12) | -3.9% | 935 | 743 | -20.5% | 770 → 578 | 0.64 → 0.32 | 0.1 |
| p536k | ext3 | 3 | 33.24 (0.53) | 31.30 (0.84) | -5.8% | 6484 | 4948 | -23.7% | 6156 → 4620 | 83.42 → 41.71 | 7.8 |
| p536k | self | 3 | 6.58 (0.23) | 6.24 (0.38) | -5.2% | 1725 | 1341 | -22.3% | 1548 → 1164 | 4.29 → 2.14 | 0.9 |
| p536k | sparse10k | 3 | 0.45 (0.03) | 0.43 (0.03) | -3.3% | 278 | 254 | -8.6% | 109 → 85 | 0.08 → 0.04 | 0.0 |
| p536k | lastgen_ext3 | 3 | 16.58 (0.26) | 15.75 (1.38) | -5.0% | 3260 | 2492 | -23.6% | 3084 → 2316 | 3.77 → 1.88 | 0.7 |

Reading the table:

- The memo dominates memory, not the output. At 536k degree 3 the memo capacity is 2^28 slots; the float64 peak of 6.2 GB above baseline is the 4.3 GB table plus the 2.1 GB predecessor held during the last rehash. The output array is 83 MB, 1.4 percent of that peak.
- Output-only float32 saves the output's second half (42 MB at 536k) and costs the cast (7.8 ms at 536k, under 3 ms everywhere else). Neither moves the totals.
- Internal float32 shrinks every memo slot from 16 to 12 bytes and delivers a consistent 20 to 24 percent lower peak RSS on the large batches (9 to 13 percent on the small ones, where the process baseline dominates). Wall time is 0 to 6 percent lower; at 20k to 102k the difference is inside the run-to-run spread, at 536k it is just outside it (spread 0.5 to 0.8 s on a 2 s gap).


## Consumer impact

Delegated study; full report with per-path tables, file:line citations, and commands in `scratchpad/f32/consumers/REPORT.md`. It patched `PedigreeGraph.compute_pair_kinship` to return `astype(np.float32)` and ran each consumer twice inside its own Pixi environment (fitACE manifest, pedsum manifest). Both environments hold the released pedigree-graph 0.7.1, not the dirty checkout; the kernel recurrence is the same, so this only matters if the 0.8.0 kernel changes value widths.

| Consumer path | Outcome under float32 |
|---|---|
| `fitace.exports.tables.export_pairwise_relatedness`, 9 thresholds (dyadic and non-dyadic) x 2 fixtures | TSV byte-identical; 0 rows gained or lost; kinship column stays Float64 |
| `fitace.kinship.kinship.build_kinship_matrix`, degree 2 and 3, 3 fixtures | shape, nnz, indices, indptr identical; data bit-identical (uint32 view); inbred count 301 both |
| PA-FGRS scoring via `build_kinship_matrix`, 20k pedigree, degree 2 and 3 | est and var max abs diff 0.0; package suite 119 passed under both |
| TetraHer adapter `.rel` emit, degree 2 and 3 | r_A bit-identical, serialized file sha256 identical, schema Float64 both |
| pedsum `build_relative_pairs` `kinship_exact` | bit-identical, schema Float64 both, 315 tests passed under both |
| Canonical inbreeding F from float32 phi(i,i) | 0 flag flips on every fixture; max abs F diff 5.96e-8 only on synthetic 30 and 60 generation full-sib chains |

Test suites under the patch: fitACE kinship and exports 105 passed, PA-FGRS 119 passed, pedsum 315 passed.

One hazard was identified and bounded. A float32 result changes the semantics of `kin >= min_kinship` in consumer code: under NEP 50 the comparison runs in float32, so a non-dyadic threshold within one float32 ulp above a representable phi (for example 0.062500001 against 0.0625) rounds down onto the value and admits a pair that float64 excludes. Measured at the numpy level; not reachable through `export_pairwise_relatedness` today because its degree gate (`tables.py:143-150`) already restricts extraction to codes whose nominal kinship clears the threshold, so the row filter at `tables.py:183` never drops anything (verified over 12 thresholds x 2 fixtures). It becomes reachable for any caller that thresholds `pair_kinship` output directly without a matching degree gate. A one-line `kin.astype(np.float64)` at the comparison, or the API contract stating that thresholds should be compared in float64, closes it.


## Findings against the decision rules

1. Threshold decisions: none changed on any corpus (consumer study, plus zero flips against zero in the kernel study).
2. Consumer outputs: bit-identical everywhere; all suites pass under the patch.
3. Float32 error characterized: absolute error at most 2^-25 (2.98e-8), relative at most 6.0e-8, and only for values with more than 24 significant dyadic bits. On the four simACE pedigrees (20k to 536k rows, 5 generations) and pedsum's degree-5 shape, no such value exists; the largest batch has 112 distinct values. Producing one requires on the order of 24 or more generations of sustained inbreeding loops (first inexact value at full-sib chain depth 24). That is outside the supported simACE use case and, for real human pedigrees, outside recorded depth.
4. Public contracts needing more precision: none found. TetraHer, pedsum and the fitACE pair table publish Float64 columns, but their values are the widened float32 and stay identical on every fixture. Whether those schemas should switch to Float32 is a consumer-side choice; nothing forces it.
5. Cached-matrix parity: `matrix32 == rounded32` on all simACE pedigrees and shallow cases, but the DP rounds per generation and diverges by 1 to 2 ulps on deep inbred pedigrees, where `internal32` also diverges (and the two do not even agree with each other).

## Locked decisions (grill-with-docs, 2026-09-04)

The recommendations below were the study's pre-interview position. The interview
locked a different internal design, recorded in pedigree-graph ADR 0009:

1. Public `pair_kinship` returns read-only float32.
2. Internal: float32 memo and arithmetic, with a pinned peel rule (peel the
   deeper endpoint, ties by larger row). Follow-up measurement
   (`scratchpad/f32/parity_depth.py`): with that rule the pairwise kernel is
   bit-identical to the matrix on every deep corpus and on row-permuted
   pedigrees, 0 differences over 490k pairs, where row-order peeling gave 5,807.
3. `pair_kinship` is recurrence-only; issue #6 closes; parity with the matrix is
   a property test.
4. `inbreeding()` stays the float64 Meuwissen-Luo walk (1.0 s and 19 MB at 536k
   versus 6.2 s and 1.2 GB via self pairs); the identity is tested with tolerance
   `2^-22`.
5. Threshold rule is an ADR and docstring contract line; fitACE widens at
   `tables.py:183` when it migrates.
6. Recorded as ADR 0009; ADR 0006 kinship bullets amended; ADR 0005 marked
   superseded in part; ADR 0007 gains the memo-layout requirement.
7. The no-copy memo grow (about 30 percent of peak RSS, dtype-independent) is a
   Rust-core requirement, not a 0.8.0 Python change.

Draft closing comment for issue #6:

> Resolved by ADR 0009. `pair_kinship` is recurrence-only and never reads a
> cached matrix, so a query no longer depends on call history. Both kernels now
> implement one pinned float32 recurrence (peel the deeper endpoint, ties by
> row); measured bit-identical on 23 registry motifs, the ADR 0008 MZ fixtures,
> 50 to 60 generation inbred lineages, 100 random deep-inbred and row-permuted
> pedigrees, and simACE pedigrees to 536k rows. Parity is a property test in
> slice 5. The fitACE second-graph workaround can go once 0.8.0 ships.

## Recommendations (pre-interview)

**Public dtype: float32, output-only.** Contract wording for `pair_kinship`: "returns the exact kinship (a dyadic rational computed in float64) rounded once to the nearest float32; forward and reversed endpoint order give identical values; a value is 0 if and only if the exact kinship is 0." Add: "callers applying a non-dyadic threshold should compare in float64 (`values.astype(np.float64) >= t`) or use a dyadic threshold". The measured benefit is small (half the output array, no wall-time change), so the case rests on consistency with the float32 matrix and every consumer's own storage, not on performance. Nothing in the study gives a concrete correctness, scientific, consumer or performance reason to keep float64 at the boundary.

**Internal dtype: float64, unchanged.** Internal float32 fails the strict criterion: it is not correctly rounded (up to 2 ulps, absolute error up to 4.9 times the rounded32 error, closed herd 1.46e-7 versus 2.98e-8). Its benefit is real but modest, 20 to 24 percent lower peak RSS at scale and wall time within or just outside noise. The saving comes from a 12-byte instead of 16-byte memo slot, which the Rust core redesign can obtain without precision loss through memo layout (for example a float64 value with a compacted key) if the memory matters. Keep the recurrence and memo float64 and cast once at the boundary.

**Canonical inbreeding: keep float64 internally.** `F = 2 phi(i,i) - 1` from float32 phi loses up to 6e-8 near F = 1; the public `inbreeding()` should derive from the float64 self-kinship before the cast. No flag flips were observed either way.

**Cached-matrix reuse (issue #6): remove the branch.** The matrix is a per-generation-rounded float32 artifact; it cannot reproduce the "rounded once" contract on deep inbred pedigrees, and making it exact would require a float64 matrix (double the dominant memory cost of the package). The recurrence is bounded and measured (33 s and 6.2 GB for 10.4M pairs at 536k rows) so there is no need for the shortcut. `pair_kinship` should always run the recurrence; document that `kinship_matrix` values may differ from `pair_kinship` by at most 2 float32 ulps and only for values with more than 24 significant bits. That resolves issue #6 with an explicit call-history contract: none, because the result no longer depends on prior calls.

## Proposed ADR 0006 amendment

Replace the `pair_kinship` float64 clause with:

> `pair_kinship(...)` returns read-only float32 arrays (or a mapping of them for `RelationshipPairs`). Each value is the exact dyadic kinship, computed by the float64 recurrence, rounded once to float32. The result is independent of endpoint order and of any prior `kinship_matrix` call. Zero is exact. Consumers that threshold these values against a non-dyadic cutoff should widen to float64 first. The internal recurrence, memo, and the inbreeding derivation stay float64 (study: this document).

Not applied here: ADR 0006 currently carries the concurrent inbreeding session's uncommitted edits, and the handoff routes the lock through a `grill-with-docs` session. The slice plan's `pair_kinship` entries should change from float64 to float32 in the same step.

## Unverified

- The 5.36M-row `baseline1M` pedigree was not run (memo extrapolates above the 2^31 slot cap). Parity at that scale is inferred from the value-width argument, not measured.
- Consumer paths ran against installed pedigree-graph 0.7.1, not the dirty checkout.
- PA-FGRS `score_probands(kmat=None)` builds `kinship_matrix()` directly and never calls `compute_pair_kinship`; it was not exercised and is unaffected by this change.
- fitACE_epimight was not exercised.
- No real (non-simulated) pedigree was tested; the depth argument for human pedigrees is reasoning, not measurement.

