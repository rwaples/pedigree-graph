# Limitations

Current scaling and correctness limitations of the relationship-pair
engines. Read before choosing between `relationship_pairs`,
`relationship_counts` and `close_relative_counts` on pair-dense pedigrees.

## Pair *lists* are O(answer size); pair *counts* are O(N)

Since 0.8.3 ``PedigreeGraph.relationship_counts`` and
``PedigreeView.relationship_counts`` run in the Rust row-streaming engine
(ADR 0010): every pair is classified in its own row and counted, no pair
list exists, and peak memory is linear in the pedigree size (2.9 GiB for
2.1 billion pairs on a 20M-row pedigree).  The rest of this section is
about ``relationship_pairs``, which materialises every pair as an
``(idx1, idx2)`` array.  Its memory is proportional to the total
relationship-pair count, **not** the pedigree size.

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

``relationship_pairs`` OOMs on this pedigree well before producing a
list, even on 30 GB hosts, in ``A_f @ A_f.T`` (the PHS sparse product).

### Operational workarounds

1. ``pedsum --no-pairs`` skips the pair-counting stage entirely.  The
   horse pedigree completes in ~30s with 1 GB peak RSS this way,
   producing every other section (size structure, family, mating,
   lineage, founder contribution, inbreeding, effective size) but
   returning stub values for the 23 relationship counts (``pairs: {}``
   and ``relationship_summary.computed: false``).

2. ``PedigreeGraph.relationship_counts(max_degree=5)`` counts all 23
   categories exactly without materialising pair lists. Use this when
   counts, rather than pair coordinates, are the goal.

3. ``PedigreeGraph.close_relative_counts()`` uses scalar sibling-group
   arithmetic and parent-edge counts for six exact categories only.
   See the coverage contract below.

## ``close_relative_counts`` precision and coverage

The scalar path is full-graph only and accepts no selector. It computes
``MZ``, ``MO``, ``FO``, ``FS``, ``MHS`` and ``PHS`` exactly, including on
inbred pedigrees. Half-sib pairs also related as parent-offspring are
subtracted to match the package's closest-category precedence.

The result retains all 23 registry keys. The six computed codes are in
both ``requested`` and ``exact``; all other values are ``None``, not zero.
There are no approximate counts, clamps or warning, and the shared
``RelationshipCountResult`` no longer has ``approximate`` or ``clamped``
fields. The old ``estimate_relationship_counts`` method is removed.

"Close" is not a degree-2 cutoff: grandparents and avuncular pairs are
not included. Use ``relationship_counts`` for those categories and for
counts restricted to a ``PedigreeView``. Both counting methods use O(N)
memory; the scalar method needs no adjacency powers or pair arrays.
See the amendment to ADR 0011 for the API decision.

## ``distinct_ancestor_counts`` memory follows its live parent frontier

``PedigreeGraph.distinct_ancestor_counts`` keeps a sorted closed ancestor set
for each row that still has an unprocessed direct child. It reuses that row's
power-of-two storage slot after the last child. This removes the old complete
sparse closure, but it does not guarantee memory proportional to one generation:
a parent with a late last child remains live, and a pathological pedigree can
keep much of its historical ancestry in the frontier.

The committed A/B benchmark in ``benchmarks/bench_distinct_ancestors.py``
measured the retiring DP against the removed closure:

- `random_30k`: 0.257 s and 173 MiB became 0.043 s and 178 MiB;
- `random_300k`: 3.42 s and 586 MiB became 0.45 s and 269 MiB;
- a closed 60-generation pedigree: 31.68 s and 532 MiB became 0.16 s and 183 MiB.

The DP itself allocated less memory on every fixture. In a fresh process,
loading the Numba runtime adds about 42 MiB before the timed call, so total peak
RSS is higher on small inputs even though their timed RSS growth is lower. The
package uses one implementation rather than dispatching at a measured size or
depth threshold. Full results, allocator telemetry, and the environment are in
``benchmarks/bench_distinct_ancestors.md``.

## Half-founders and missing parents

Both engines accept half-founders (one parent known, one missing).
The sibling group-by filters to known parents only:

- ``FS`` requires BOTH parents known on both individuals.
- ``MHS`` only considers individuals with mother known.
- ``PHS`` only considers individuals with father known.

This matches the standard convention but can surprise callers who
expect half-founders to contribute to half-sib counts on the
"missing" side.  They don't.

## View pair lists still enumerate the full graph

``PedigreeView.relationship_pairs`` extracts full-graph pairs before
projecting them onto the view. A small view therefore does not protect
pair-list extraction from full-pedigree memory costs.

Use ``PedigreeView.relationship_counts`` when only counts are needed.
It passes a row mask to the Rust engine and builds no pair list.

## What this file does NOT cover

- Historical scalar lineal-count performance. Those formulas are removed;
  current lineal counts use ``relationship_counts``.
- F (inbreeding coefficient) scaling — covered by
  ``pedigree_graph._inbreeding_kernel``.
- Effective size estimator scaling — covered by
  ``pedigree_graph._effective_size`` and the ``skip_ne_coancestry``
  knob.

## Last updated

Issue #17: replace the scalar estimator with six exact close-relative
counts, remove approximate/clamped metadata, and update view-count guidance.

2026-09-16 — replace the ``distinct_ancestor_counts`` sparse closure with the
retiring sorted-set DP; record the A/B benchmark and the fixed Numba runtime
cost.

2026-09-10 — the experimental Python relationship counter is gone
(issue #7), and with it the cousin-code divergence and ``int8``
overflow sections; the Rust engine of 0.8.3 is the one
relationship-counting implementation.

2026-05-19 — ``count_pairs_streaming`` precision contract
reconciled; ``Av`` documented as approximate; stale
"count-only-experiment-didn't-ship" narrative removed.
