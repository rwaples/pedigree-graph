# Limitations

Current scaling and correctness limitations of the relationship-pair
engines. Read before choosing between `relationship_pairs`,
`relationship_counts` and `close_relative_counts` on pair-dense pedigrees.

## Pair *lists* are O(answer size); counts and burden are O(N)

``PedigreeGraph.relationship_counts`` and ``PedigreeView.relationship_counts``
run in the Rust row-streaming engine (ADR 0010): every pair is classified in
its own row and counted, no pair list exists, and peak memory is linear in
the pedigree size (2.9 GiB for 2.1 billion pairs on a 20M-row pedigree).
``PedigreeGraph.relationship_burden`` streams the same engine into
per-person counts of relatives at degrees 1 to 5, also without a pair list.

``relationship_pairs`` is different: it returns every pair, as two int32
row arrays per category, so the result alone costs 8 bytes per pair and its
memory is proportional to the total relationship-pair count, **not** the
pedigree size.  Assembly adds to that: ``execution="speed"`` (the default)
peaks at about 2.3 times the result, and ``execution="memory"`` at the result
plus engine state, for roughly twice the wall time.  The blocks are identical
either way.

### Worst case: prolific-stallion livestock pedigrees

Stallion-driven half-sib density is where pair lists stop being practical.
On a real horse-breed pedigree (783K individuals, one all-time-great stallion
siring 2,500 horses, top-sire grand-offspring set ~50K) the paternal
half-sib pairs alone number ~156 M, about 1.2 GB as a pair block, and a
degree-5 list is far larger again, dominated by cousins through the one
stallion grandparent.

### What to use instead

1. ``relationship_counts(max_degree=5)`` counts all 23 categories exactly
   without materialising pairs.  Use it when counts, not pair coordinates,
   are the goal.
2. ``relationship_burden()`` gives each person's count of relatives by
   degree, again without materialising pairs.
3. ``close_relative_counts()`` uses scalar sibling-group arithmetic and
   parent-edge counts for six exact categories only.  See the coverage
   contract below.
4. When a list is needed, ``execution="memory"`` lowers the peak to the
   result plus engine state.

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
See ADR 0011 for the API decision.

## ``distinct_ancestor_counts`` memory follows its live parent frontier

``PedigreeGraph.distinct_ancestor_counts`` keeps a sorted closed ancestor set
for each row that still has an unprocessed direct child, sized exactly, and
frees it after the last child. This removes the old complete sparse closure,
but it does not guarantee memory proportional to one generation:
a parent with a late last child remains live, and a pathological pedigree can
keep much of its historical ancestry in the frontier.

The sweep runs in the Rust core since 0.9.4, with one exactly sized set per
live row. Against the 0.9.3 Numba pool it measured 0.31x to 0.62x the wall
and 0.40x to 0.69x the peak RSS on every benchmarked pedigree, from 1k rows to
the 536k-row ``baseline100K``, including closed 60-generation and
2,000-wide pedigrees. The Numba runtime's fixed RSS is gone with it. Record:
``docs/pedigree-graph-0.8-migration/gate/15a/NOTES.md``.

## Half-founders and missing parents

Both engines accept half-founders (one parent known, one missing).
The sibling group-by filters to known parents only:

- ``FS`` requires BOTH parents known on both individuals.
- ``MHS`` only considers individuals with mother known.
- ``PHS`` only considers individuals with father known.

This matches the standard convention but can surprise callers who
expect half-founders to contribute to half-sib counts on the
"missing" side.  They don't.

## View pair lists

``PedigreeView.relationship_pairs`` and ``PedigreeView.relationship_counts``
classify every pair through the view's ancestry, so unselected relatives
still connect selected ones.  For a sparse view (at most a tenth of a graph
of at least 20,000 rows) both first restrict the engine to the ancestry the
view needs (``_should_compact_view`` in ``pedigree_graph/_relationship_pairs.py``);
otherwise the engine runs over the full graph and keeps the pairs with both
rows selected.  Either way a view pair list costs its own answer size, and
view counts build no pair list.

## What this file does NOT cover

- Historical scalar lineal-count performance. Those formulas are removed;
  current lineal counts use ``relationship_counts``.
- F (inbreeding coefficient) scaling: the Meuwissen-Luo walk runs in the
  Rust core (``crates/core/src/kinship/inbreeding.rs``).
- Effective size estimator scaling: see ``pedigree_graph.effective_size``;
  ``ne_coancestry`` runs the kinship DP and is the expensive one.

## Last updated

2026-09-24 — pair-list section rewritten for the Rust engine (the SciPy
matrix internals it described are gone), ``relationship_burden`` and the
``execution`` modes added, pedsum's retired ``--no-pairs`` workaround dropped,
and the compact view path described.

2026-09-23 — ``distinct_ancestor_counts`` runs on the Rust core (0.9.4);
the Numba runtime cost is gone.

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
