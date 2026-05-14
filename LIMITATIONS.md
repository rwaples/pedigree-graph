# Limitations

Notes on known scaling and correctness limitations of the
relationship-pair engines, captured 2026-05 after a count-only
streaming experiment that didn't ship.  Read before re-attempting
either approach.

## Pair counting is O(answer size)

Both engines (`matrix` and `bfs`) materialise every pair as an
`(idx1, idx2)` array before counting it:

- `extract_pairs()` returns `dict[code, (np.ndarray, np.ndarray)]`.
- `count_pairs()` is a thin facade — it runs `extract_pairs()` and
  returns `{code: len(idx_array)}`.

Memory is therefore proportional to the total relationship-pair
count, NOT to the pedigree size.

### Worst case: prolific-stallion livestock pedigrees

Stallion-driven half-sib density is the engine's hard wall.  On a
real horse breed pedigree (783K individuals, one all-time-great
stallion siring 2,500 horses, top sire's grand-offspring set ~50K
individuals):

- Paternal half-sib pair count = ~156 M
- `_half_sib_matrix` materialisation = ~312 M nonzeros (~3 GB) just
  for the symmetric MHS+PHS matrix
- Per-grandparent grandchild bucket for `_cousin_pairs` enumeration
  reaches `C(50K, 2) ≈ 1.25 B` candidate pairs through one stallion
  grandparent (~10 GB of int64 keys)
- `_A2 @ _A3.T` for 1C1R / H1C1R has row nnz ~2,500 (one stallion's
  great-grandchildren spread); chunk sizes blow up at default
  `chunk_rows`

Both engines OOM on this pedigree well before producing a count,
even on 30 GB hosts.  The matrix engine OOMs in `A_f @ A_f.T` (PHS
sparse product); the BFS engine OOMs in the numba cousin
enumeration kernel.

### Operational workaround

`pedsum --no-pairs` skips the pair-counting stage entirely.  The
horse pedigree completes in ~30s with 1 GB peak RSS this way,
producing every other section (size structure, family, mating,
lineage, founder contribution, inbreeding, effective size) but
returning stub values for the 23 relationship counts (`pairs: {}`
and `relationship_summary.computed: false`).

This is the right escape hatch for pair-dense pedigrees today.

## A "count-only streaming" engine was attempted and abandoned

The natural question is "can we count pairs without materialising
them?"  We tried:

### Attempt 1 — enumeration with sub-chunking (matched matrix semantics)

Mirror the matrix engine's per-grandparent / per-mating-pair
enumeration in chunks of fixed size, accumulating canonical
`(lo, hi)` int64 keys into a sorted global buffer with `np.unique`
merges.  Maintain the inclusion-exclusion subtractions exactly as
`_extract_from_sparse` does.

**Failed at scale.**  The unique answer itself is enormous on the
horse pedigree — 1C count is hundreds of millions of pairs, H1C
similarly.  Even with bounded per-chunk emission, the global key
buffer would have to hold gigabytes.  And the inclusion-exclusion
subtractions need full closer-pair-key sets, so MHS / PHS pair
arrays can't be avoided (the 156 M-pair blowup we were trying to
escape).

### Attempt 2 — pure scalar per-anchor arithmetic (BFS-distinct cousin semantics)

Compute counts via per-anchor `C(k, 2)` sums and `_A^k.nnz` reads —
no pair-key arrays anywhere.  Memory truly O(N).

**Shipped as `count_pairs_streaming`** (2026-05-14).  Completes on
the 783K-row horse pedigree in ~5 seconds, peak RSS 730 MB.  Returns
all 23 codes with the following precision contract:

- **Exact on lineal + sibling + MZ codes** (10 of 23):
  `MO`, `FO`, `GP`, `GGP`, `GGGP`, `G3GP`, `MZ`, `FS`, `MHS`, `PHS`.
  These match `count_pairs` bit-identically on every input.
- **Approximate on cousin / collateral codes** (13 of 23):
  `Av`, `1C`, `H1C`, `HAv`, `GAv`, `GGAv`, `G3Av`, `HGAv`, `HGGAv`,
  `1C1R`, `H1C1R`, `1C2R`, `2C`.  Scalar formulas assume each
  individual has the full complement of known grandparents at the
  relevant depth; constants like `4*FS` in the `H1C` correction
  over-subtract on shallow pedigrees and `H1C` may clamp to `0` on
  depth ≤ 3 fixtures.  On the synthetic `small_pedigree` (3000
  rows, depth 3, ~0.5% sib-mating, 10 twin pairs): `Av` off by 3,
  `HAv` off by 11, `1C` off by 30, `H1C` clamped to 0.  On deep
  livestock pedigrees (depth ≥ 5, low inbreeding) the formulas
  are accurate to better than 1%.

**Worked exactly for the simple codes** on synthetic test fixtures.
**Failed bit-identity on 4 codes** (`1C`, `H1C`, `Av`, `HAv`) due
to inclusion-exclusion edge cases:

1. **Twins**: excluded from `FS` / `MHS` / `PHS` group-by per
   `sibling_pairs()` (`_core.py:564`), but their children ARE valid
   grandchildren for cousin counting.  Scalar `pair_sum_d1`
   aggregates only via non-twin intermediates, under-counting
   cousins through twin parents.

2. **Sib-mating inbreeding**: rare in real pedigrees (small_pedigree
   has 2 such offspring among 3,000), but creates pairs that share
   both grandparents via the same intermediate path.  Inclusion-
   exclusion in scalar form needs an extra term per inbreeding
   pattern.

3. **Per-mating-pair vs per-single-grandparent multiplicities**:
   the matrix engine's `_cousin_pairs` enumerates per single
   grandparent and uses `count >= 2` thresholding.  Scalar
   `Σ C(grandchildren_via_pair, 2)` over mating-pair grandparents
   gives a related-but-different count that requires careful
   subtraction of FS / MHS / PHS contributions to align.  The
   subtractions cascade — fixing `1C` requires re-deriving `H1C`,
   `Av`, `HAv` simultaneously.

The pure-scalar approach is correct on **non-inbred pedigrees with
no twins**, but real data — even synthetic test fixtures — has
both.  Getting bit-identity required adding edge-case-specific
correction terms that are about as much code as the enumeration
approach.

**Outcome**: shipped pure-scalar as `count_pairs_streaming` with a
documented precision contract (see method docstring).  Exact for the
10 simple codes; approximate within ~1% for the 13 cousin / collateral
codes on deep low-inbreeding pedigrees.  208 tests pass; 6 new tests
pin the precision contract on the `small_pedigree` fixture.

## Cousin-code matrix/BFS divergence on inbred input

Independent of scaling, the `matrix` and `bfs` engines give
**different counts** on inbred pedigrees for the four
cousin-multiplicity codes: `1C1R`, `H1C1R`, `1C2R`, `2C`.

- **Matrix engine** uses path multiplicity: `M.data >= 2` (full) or
  `M.data == 1` (half) thresholds on `_A2 @ _A3.T` / `_A2 @ _A4.T`
  / `_A3 @ _A3.T`.  A pair sharing an ancestor via two distinct
  paths is counted twice in the matrix entry; the threshold
  classifies based on this multiplicity.
- **BFS engine** uses distinct-shared-ancestor semantics: a pair
  sharing N distinct ancestors at the relevant depth is counted
  once regardless of paths.

The divergence is documented in
`tests/test_experimental.py:171` (the `inbred_with_cousins_pedigree`
fixture) and asserted in `test_inbred_with_cousins_cousin_codes_diverge`.

`extract_pairs(scope="full")` returns matrix-engine values by
default.  Callers needing BFS-distinct semantics on inbred input
must use `pedigree_graph.experimental.count_pairs_bfs` and accept
the matrix-vs-BFS difference for those four codes.

This divergence was a hard constraint on the count-only experiment:
matching matrix semantics required enumeration with multiplicity
thresholding (no scalar shortcut), and matching BFS semantics
required different inclusion-exclusion than matrix's
`_extract_from_sparse` provides.

## Half-founders and missing parents

Both engines accept half-founders (one parent known, one missing).
The sibling group-by (`_core.py:573-575`) filters to known parents
only:

- `FS` requires BOTH parents known on both individuals.
- `MHS` only considers individuals with mother known.
- `PHS` only considers individuals with father known.

This matches the standard convention but can surprise callers who
expect half-founders to contribute to half-sib counts on the
"missing" side.  They don't.

## Subsample-restricted counts are O(full pair count)

`PedigreeGraph.from_subsample(...)` builds a graph that returns
subsample-filtered pair arrays from `extract_pairs`, but the
underlying enumeration runs over the FULL pedigree first
(`_core.py:1174` saves raw counts, then line 1177 applies the
mask).  Memory is bounded by full-pedigree pair counts, not the
subsample.

For a 10% subsample of a stallion-heavy pedigree, this is still
OOM-prone because the full-pedigree intermediate doesn't shrink.

## What would unlock the horse-pedigree scale

In rough order of effort:

1. **Sub-chunked per-bucket enumeration with continuous global
   dedup.**  Avoid the `C(k, 2)` single-allocation in
   `_pairs_from_groups` when a bucket exceeds threshold.  Peak
   memory still O(answer), but no longer multiplied by transient
   bucket-level intermediates.  Estimated horse run: 5-10 GB peak,
   minutes wall.

2. **Replace `_half_sib_matrix` with on-the-fly streaming per
   parent**: the `_collateral_pairs(hsm, ...)` form is `A^(down-1)
   @ hsm`, which can be computed as a sum over per-parent
   half-sibship contributions without materialising the symmetric
   half-sib matrix.  Removes the 312 M-entry blowup.  Estimated
   savings: 3 GB.

3. **Pure scalar formulas for non-inbred twin-free pedigrees**, with
   a documented "inbred / twins not supported" contract and an
   explicit fallback to the matrix engine when twins or sib-mating
   are detected.  Smallest code path; widest semantic compromise.

4. **Pair-iterator API** (`extract_pairs_iter()`) that yields
   pair arrays per code WITHOUT caching them.  Callers that only
   need counts call `len()` on the yielded array and drop it.
   Doesn't reduce peak memory below the largest per-code array
   (`PHS` is still ~1.2 GB), but releases earlier and avoids
   holding all 23 codes in memory simultaneously.

None of these were implemented.  The operational answer remains
`pedsum --no-pairs`.

## What this file does NOT cover

- Lineal-code counting limitations (none significant — `_A^k.nnz`
  is O(N · depth) and tractable to N=10M+).
- F (inbreeding coefficient) scaling — covered by
  `pedigree_graph._kinship_kernel` and its own row-retirement
  optimisation work.
- Effective size estimator scaling — covered by
  `pedigree_graph._effective_size` and the `skip_ne_coancestry`
  knob.
- BFS engine internal limitations — see
  `external/pedsum/STATUS.md` for the historical follow-ups against
  this package.

## Last updated

2026-05-14 — after `count_pairs_streaming` rollback.
