# ADR 0019: relationship queries are total in the core

**Status:** accepted
**Date:** 2026-11-03
**Context:** issue #52 (both hosts repeated the wiring around every
relationship query). Amends the glossary's **Receiver** entry, ADR 0013's
degenerate-call paragraph, ADR 0014's "Views and the compact path" and the
entry-point list of ADR 0017.

## Context

Every relationship query (pairs, counts, moments, relatives per person)
went through the same steps in each host: resolve progress, commit the
thread budget, derive `max_degree` from the requested categories, return an
empty result for an empty selection or a receiver of fewer than two rows,
choose whether a view runs on its compact pedigree, then call the core.

- The empty-call shortcut was written three ways in Python and once more in
  R (`walk_of`). Python's `relationship_counts` skipped the thread-budget
  commit on an empty selection, unlike every other call and unlike R.
- Both hosts built the same degenerate moments table by hand.
- `_should_compact_view`, a measured threshold, lived only in Python. Three
  sibling modules imported it, and an R view binding would have had to copy
  it.
- The core's entry points took both `max_degree` and `requested`, so a
  category could be requested above the walk and come back empty. Counts
  took a `selected` mask or a second entry point
  (`count_view_pairs_compact`) instead of a `Receiver`.

## Decision

`relationships::query` is the module hosts call. Its four entry points,
`pairs`, `counts`, `moments` and `relatives`, take a `CategorySet` and a
`query::Receiver`, and own the wiring:

1. **Depth.** The engine walks to the requested set's top degree. No entry
   point takes a `MaxDegree`. `MaxDegree` stays public for `Engine`,
   `reduce_pairs` and `reduce_pairs_in` (the Reducer seam, ADR 0013) and the
   CLIs.
2. **Calls that walk nothing.** An empty set, or a receiver of fewer than
   two rows, returns `PairBlocks::empty`, zero `Counts`,
   `Moments::degenerate` (planned like any call, so the budget refuses it
   the same way; width 1, zero table, no lanes) or `RelativesPerPerson::zeros`.
   The entry points still run their argument checks first, so a Rust caller
   gets the same errors either way.
3. **`query::walks(requested, rows)`** says, in O(1), whether a call walks.
   A host calls it before borrowing or copying its columns, and on `false`
   returns the same constructor result. The Python binding does this before
   its numpy borrows. R does it before copying a graph's rows into the job.
   The predicate exists because the borrow, the core's column check and the
   view map are O(n): on a 300k-row graph they would add 0.1–0.5 ms to calls
   that take 5–40 µs today (issue #52's plan, §2). So this skip skips the
   input checks the core runs; both hosts check those columns themselves
   before committing the budget, so no public error changes.
4. **A view crosses as its `graph_rows`**, the graph row of each view row.
   Its row count is then O(1). The core builds the graph-to-view map only
   when it walks, and counts on the full path build a row mask instead.
   `graph_rows` errors are `value_out_of_range` on `graph_rows` and
   `InvalidViewMap` (now naming its `field`) for a row selected twice.
5. **Compaction is the core's choice.** `query::Compaction::Auto` applies
   the rule Python applied: a view of at most a tenth of a graph of at least
   20 000 rows runs on its compact pedigree (measured on `random_30k`; see
   `query::compacts`). `Full` and `Compact` force a path, for tests and
   benchmarks. The result is in view rows on every path; progress totals
   follow the pedigree the engine walks, as ADR 0017 already said.

Progress polling and cancellation (ADR 0017), the thread-budget commit and
the result wrapping stay in the hosts. Python commits the budget after
validating its arguments and before the native call, so rejected arguments
leave it uncommitted (#51), and every call, an empty one included, commits
it.

The former entry points (`pair_blocks`, `relationship_moments`,
`relatives_per_person`, `count_pairs`, `count_view_pairs_compact` and the
top-level `relationships::Receiver`) stay unchanged until pg-phenotype has
moved to `query`, then go.

## Consequences

- The empty-call rule, the compaction threshold and the degenerate results
  each have one home and are tested once, in the core.
- An R view binding needs no heuristic: it passes `graph_rows` and `Auto`.
- Calls that walk nothing cross into native once more than before in
  Python. The perf gate held those cells to an absolute bound (+5 µs per
  call) rather than the 5% ratio. Pairs and counts meet it; relatives costs
  +8 to +12 µs per call and moments +5 to +10 µs (measured against
  0.14.0-13, random_30k and random_300k). That is the native call's fixed
  cost and the result wrapping, and it was accepted for calls that do no
  work rather than keep a second copy of the rule in Python.
- A zero-row view's moments report the zero cells their axes span. Before,
  Python's degenerate path reported a plan of one label cell for the peak.

## Rejected

- **Hosts always call the core.** No host branch at all, but every empty
  call pays the O(n) borrow, check and map (§2 of the plan): 10–80x on
  those cells.
- **A Python copy of the predicate** (B-py). It keeps empty calls at
  today's cost, but the rule has two owners held equal by a test.
- **Entry points that take unchecked columns and project lazily.** No
  binding branch in Python, but it undoes #53's checked projection and
  would recheck a Rust host's owned, validated graph (#57) on every call.
