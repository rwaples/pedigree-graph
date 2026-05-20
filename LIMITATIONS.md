# Limitations

Current scaling and correctness limitations of the relationship-pair
engines.  Read before reaching for `extract_pairs` / `count_pairs` on
pair-dense pedigrees.

## Pair counting is O(answer size) for the matrix and BFS engines

The matrix and BFS engines both materialise every pair as an
``(idx1, idx2)`` array before counting it:

- ``extract_pairs()`` returns ``dict[code, (np.ndarray, np.ndarray)]``.
- ``count_pairs()`` is a thin facade — it runs ``extract_pairs()`` and
  returns ``{code: len(idx_array)}``.

Memory is therefore proportional to the total relationship-pair count,
**not** the pedigree size.

### Worst case: prolific-stallion livestock pedigrees

Stallion-driven half-sib density is the engine's hard wall.  On a real
horse-breed pedigree (783K individuals, one all-time-great stallion
siring 2,500 horses, top-sire grand-offspring set ~50K):

- Paternal half-sib pair count = ~156 M
- ``_half_sib_matrix`` materialisation = ~312 M nonzeros (~3 GB) just
  for the symmetric MHS+PHS matrix
- Per-grandparent grandchild bucket for ``_cousin_pairs`` enumeration
  reaches ``C(50K, 2) ≈ 1.25 B`` candidate pairs through one stallion
  grandparent (~10 GB of int64 keys)
- ``_A2 @ _A3.T`` for 1C1R / H1C1R has row nnz ~2,500 (one stallion's
  great-grandchildren spread); chunk sizes blow up at default
  ``chunk_rows``

Both engines OOM on this pedigree well before producing a count, even
on 30 GB hosts.  Matrix OOMs in ``A_f @ A_f.T`` (PHS sparse product);
BFS OOMs in the numba cousin enumeration kernel.

### Operational workarounds

1. ``pedsum --no-pairs`` skips the pair-counting stage entirely.  The
   horse pedigree completes in ~30s with 1 GB peak RSS this way,
   producing every other section (size structure, family, mating,
   lineage, founder contribution, inbreeding, effective size) but
   returning stub values for the 23 relationship counts (``pairs: {}``
   and ``relationship_summary.computed: false``).

2. ``PedigreeGraph.count_pairs_streaming()`` — pure-scalar per-anchor
   arithmetic, O(N) memory, exact on 10 simple codes, approximate
   within ~1% on 13 cousin / collateral codes for deep low-inbreeding
   pedigrees.  See the precision contract below.

## ``count_pairs_streaming`` precision contract

The scalar path is **full-graph only** — ``scope='subsample'`` raises
``NotImplementedError`` on graphs constructed via ``from_subsample``.
Use ``count_pairs`` for subsample-restricted counts.

- **Exact** on 10 codes (bit-identical to ``count_pairs`` on every
  input):
  ``MZ``, ``MO``, ``FO``, ``FS``, ``MHS``, ``PHS``,
  ``GP``, ``GGP``, ``GGGP``, ``G3GP``.

- **Approximate** on 13 cousin / collateral codes:
  ``Av``, ``1C``, ``H1C``, ``HAv``, ``GAv``, ``GGAv``, ``G3Av``,
  ``HGAv``, ``HGGAv``, ``1C1R``, ``H1C1R``, ``1C2R``, ``2C``.

  Scalar formulas assume each individual has the full complement of
  known grandparents at the relevant depth, so constants like
  ``4*FS`` in the ``H1C`` correction over-subtract on shallow
  pedigrees; ``H1C`` may clamp to ``0`` on depth ≤ 3.  Twin parents
  and sib-mating offspring also push the formulas off bit-identity
  because the inclusion-exclusion terms assume neither pattern.

  On the synthetic ``small_pedigree`` fixture (3000 rows, depth 3,
  ~0.5% sib-mating, 10 twin pairs): ``Av`` off by 3, ``HAv`` off by
  11, ``1C`` off by 30, ``H1C`` clamped to 0.  On deep livestock
  pedigrees (depth ≥ 5, low inbreeding) the formulas are accurate to
  better than 1%.

The horse-pedigree benchmark (N=783K, mean F=0.007) completes in
~5 seconds with peak RSS ~730 MB.

## Cousin-code matrix/BFS divergence on inbred input

Independent of scaling, the ``matrix`` and ``bfs`` engines give
**different counts** on inbred pedigrees for the four
cousin-multiplicity codes: ``1C1R``, ``H1C1R``, ``1C2R``, ``2C``.

- **Matrix engine** uses path multiplicity: ``M.data >= 2`` (full) or
  ``M.data == 1`` (half) thresholds on ``_A2 @ _A3.T`` / ``_A2 @ _A4.T``
  / ``_A3 @ _A3.T``.  A pair sharing an ancestor via two distinct
  paths is counted twice in the matrix entry; the threshold
  classifies based on this multiplicity.
- **BFS engine** uses distinct-shared-ancestor semantics: a pair
  sharing N distinct ancestors at the relevant depth is counted once
  regardless of paths.

The divergence is documented in
``tests/test_experimental.py:171`` (the ``inbred_with_cousins_pedigree``
fixture) and asserted in ``test_inbred_with_cousins_cousin_codes_diverge``.

``extract_pairs(scope="full")`` returns matrix-engine values by
default.  Callers needing BFS-distinct semantics on inbred input must
use ``pedigree_graph.experimental.count_pairs_bfs`` and accept the
matrix-vs-BFS difference for those four codes.

## ``int8`` overflow risk in BFS ``P_k`` boolean matmul

The BFS engine (`pedigree_graph.experimental.count_pairs_bfs`) uses
``np.int8`` for ``P_k.data`` during the boolean matmul stages.
Theoretically vulnerable to silent path-count overflow under extreme
consanguinity: more than 127 distinct paths to a single ``(i, X)`` pair
before the ``M.data[:] = 1`` clamp will wrap.

Empirically not seen on any tested pedigree. Switch to ``int32`` if it
ever bites — the change is a one-line dtype swap at the matmul site;
memory cost is 4× on the intermediate matrices.

## ``compute_n_ancestors`` memory scales with ``sum_i n_ancestors[i]``

``PedigreeGraph.compute_n_ancestors`` is a sparse boolean transitive
closure of the parent graph (``_lineage_kernel._compute_n_ancestors``).
Memory scales with ``sum_i n_ancestors[i]``, so very deep / very wide
pedigrees can hit RAM limits:

- N=100K, G=10, random mating → 2.2 s, peak RSS ~0.5 GB.
- N=10M with saturated ancestry → extrapolates beyond commodity hardware.

A retirement-style DP (analogous to the F kernel's row-retirement
optimisation in ``_kinship_kernel``) would bound peak memory to the
live frontier rather than the cumulative ancestor set. Deferred until
a user hits the wall.

## Half-founders and missing parents

Both engines accept half-founders (one parent known, one missing).
The sibling group-by filters to known parents only:

- ``FS`` requires BOTH parents known on both individuals.
- ``MHS`` only considers individuals with mother known.
- ``PHS`` only considers individuals with father known.

This matches the standard convention but can surprise callers who
expect half-founders to contribute to half-sib counts on the
"missing" side.  They don't.

## Subsample-restricted counts are O(full pair count)

``PedigreeGraph.from_subsample(...)`` builds a graph that returns
subsample-filtered pair arrays from ``extract_pairs``, but the
underlying enumeration runs over the FULL pedigree first (raw counts
saved, then sample mask applied).  Memory is bounded by full-pedigree
pair counts, not the subsample.

For a 10% subsample of a stallion-heavy pedigree, this is still
OOM-prone because the full-pedigree intermediate doesn't shrink.

## What this file does NOT cover

- Lineal-code counting limitations (none significant — ``_A^k.nnz`` is
  O(N · depth) and tractable to N=10M+).
- F (inbreeding coefficient) scaling — covered by
  ``pedigree_graph._kinship_kernel`` and its own row-retirement
  optimisation work.
- Effective size estimator scaling — covered by
  ``pedigree_graph._effective_size`` and the ``skip_ne_coancestry``
  knob.
- BFS engine internal limitations — see the ``int8`` overflow section
  above for the path-count overflow case, and GitHub issues
  [#2 (numba kernel parallelisation)](https://github.com/rwaples/pedigree-graph/issues/2)
  and [#3 (10M+ scaling test)](https://github.com/rwaples/pedigree-graph/issues/3)
  for open performance / scalability questions.

## Last updated

2026-05-20 — ``int8`` overflow risk and ``compute_n_ancestors``
scalability sections added; BFS internal follow-ups re-homed from
retired ``external/pedsum/STATUS.md`` to GitHub issues #2 and #3.

2026-05-19 — ``count_pairs_streaming`` precision contract
reconciled; ``Av`` documented as approximate; stale
"count-only-experiment-didn't-ship" narrative removed.
