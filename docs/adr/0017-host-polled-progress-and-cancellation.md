# ADR 0017: the host polls the row-streaming engine for progress and cancels it

**Status:** accepted
**Date:** 2026-10-02
**Context:** issue #37 ("relationship_counts gives no progress feedback on
long runs"). Plan: simACE `plans/pedigree-graph-37-progress.md`
(session-local draft). Amends ADR 0006 (a `progress=` keyword and the
`RelationshipProgress` root export). Keeps ADR 0007's host-neutral core.

## Context

A degree-5 count on a large, deep pedigree (pedsum's horse pedigree) runs
for many minutes. Between its start and `total:` log lines it printed
nothing, so an operator could not tell a slow call from a hung one. Ctrl-C
did nothing either: the GIL is released for the whole native call, and
Python only handles a signal when it runs bytecode.

Two constraints shape the fix. The core never calls its host (ADR 0007),
so the engine can't log or raise a Python exception itself. And a signal
reaches only the main thread, which is blocked in the native call.

## Decision

**The engine publishes atomics; the calling thread polls them.**

* **Public keyword (D1, D3, D7).** `progress=` on `relationship_counts`,
  `relationship_pairs`, `relationship_moments`, `relatives_per_person`
  (graph and view) and `relationship_burden` (graph only). `None`, the
  default, logs a line at INFO through the method's module logger.
  `False` emits no progress. A callable receives a `RelationshipProgress`
  (`phase`, `rows_done`, `rows_total`, `elapsed`) about once a second, and
  nothing is logged. Anything else, `True` included, is a `TypeError`.
* **What `False` silences (D2, D10).** Progress lines only. The existing
  start and `total:` lines stay, and consumers mute those with the logger
  level as before. `relationship_burden` never had start or total lines and
  gains none: it logs progress lines only, so a short burden call still
  logs nothing.
* **Cadence (D5).** The default logger writes its first line at 30 s and
  then one every 30 s, in one form per phase:
  `relationship_pairs: preparing after 45s`,
  `relationship_counts: 195,758/783,029 rows (25%) after 3m12s`,
  `relationship_pairs: all 783,029 rows walked, assembling after 9m40s`.
  The percentage is truncated, so it never reads 100% before the walk ends.
  There is no ETA.
* **Cancellation (D4).** Ctrl-C, or an exception the callback raises,
  cancels the call, and that exception is raised in place of the result.
* **`relationship_kinship_matrix` (D8)** keeps the default logging of its
  internal `relationship_pairs` call, under that module's logger. It gets
  no keyword of its own.

### Core: `relationships::Progress`

`crates/core/src/relationships/progress.rs` holds four atomics: the phase,
rows done, rows total and a cancel flag. Every entry point takes a trailing
`&Progress`: `count_pairs`, `count_view_pairs_compact`,
`relationship_burden`, `reduce_pairs` (and so `relationship_moments` and
`relatives_per_person`), `pair_blocks` and `pair_blocks_compact`. The R host
passes `&Progress::default()` and gains no keyword.

**Phases (D9).** A call starts in *preparing*: view compaction and engine
setup, before the total is known. `walk(n)` publishes the number of row
visits and moves to *walking*. `n` is the pedigree the engine actually
walks: the compact pedigree's rows for a compact view. `finish()` moves to
*finishing* once the walk returns. What runs in each phase:

| Path | Walking | Finishing |
|---|---|---|
| `count_pairs` | rows; per-task counts merged inside the walk | nothing material |
| `relationship_burden` | rows; atomic counters | moving the counters out |
| `reduce_pairs` (moments, relatives) | rows | lane merges |
| `pair_blocks`, `"speed"` | rows | per-category copy (categories in parallel); view sort |
| `pair_blocks`, `"memory"` | pass 1; per-category allocate and zero-fill; pass 2 | view sort only |

A **row visit** is one row walked once. The `"memory"` execution walks
every row twice, so its total is `2n`. Between its passes `done == n` and
the phase stays *walking*: the line reads 50%, which is right for row
visits, and the allocation between the passes is short next to either pass.

**Memory ordering.** `walk` stores the total `Relaxed` and then the phase
`Release`; `finish` stores the phase `Release`; rows advance with a relaxed
`fetch_add` once per 2048-row task. The observer loads the phase `Acquire`
before the counters, so it never sees *walking* with a total of 0 or
*finishing* with a stale count. Every advance happens before `finish`,
because rayon's join completes every task before the walk returns to the
thread that calls it. The snapshot is normalized per phase: *preparing*
carries no counts, *walking* carries `done.min(total)`, *finishing* only
the total. The cancel flag is `Relaxed` both ways; it carries no data.

**Checkpoints.** The engine checks the cancel flag every 64 rows
(`CHECK_EVERY`) in all four row loops, and at these named checkpoints:

* `Compacted`, after the compact pedigree of a view is built;
* `Walk` and `Finish`, at the phase changes;
* `BetweenPasses` after pass 1 of `"memory"`, and `CategoryAlloc` before each
  category's allocation and zero fill;
* `CategoryCopy` at the start of each parallel copy task of `"speed"`, and
  `CategorySort` before each category's view sort;
* `LaneMerge` before each lane merge of `reduce_pairs`.

The steps that cannot be interrupted are one compaction, one engine setup,
one category's allocate and zero fill, one category's copy, one parallel
sort, and one lane merge. After the host cancels, the call returns once the
next tick has seen the signal or the callback error (at most one tick,
1 s), and every worker has finished its current 64 rows (in the walk, the
slowest such block across workers) or its current uninterruptible step.
Nothing promises "one tick plus one row".

The check was once per row at first. The overhead benchmark (capped clock,
five interleaved runs) put `relationship_counts` 2–5% behind 0.12.0 in every
cell, up to +4.6% on 5,000 rows at six threads, with `progress=False`
costing the same as the default: the per-row check, not the watcher. A
check every 64 rows keeps the cancel bound short next to a one-second tick.
Task ranges are 2,048 rows, a multiple of 64 (a compile-time assert), so
every task still checks on its first row.

`Error::Cancelled` reports a cancel. Its class is `Usage`, so its code is
empty and the ADR 0006 code tables don't change. The Python host never
surfaces it, because it raises the exception that caused the cancel. If
`Cancelled` ever reaches `to_pyerr` without one, that is a bug, and it shows
up as the `Usage` `ValueError`.

### Binding: `run_watched` (D6, D11)

`crates/python/src/lib.rs` routes the six native entry points through one
helper. It releases the GIL, then runs the engine as one job in the existing
package pool with `ThreadPool::in_place_scope`, while the calling thread
polls. No OS thread is spawned per call. The poll loop parks for one tick
(`TICK_S`, 1 s, passed from `_progress.py`). On each tick it attaches to
Python, calls `check_signals()`, and passes the callback
`(phase, rows_done, rows_total)` from the snapshot. On the first error it
cancels the `Progress`, keeps that exception, stops calling the callback,
and keeps polling until the job ends.

A drop guard on the job sets `finished` and unparks the caller, both when
the job returns and while it unwinds. So a short call returns as soon as its
job ends, not on the next tick, and a panicking job wakes the caller at
once. `in_place_scope` then re-raises the panic, and pyo3 turns it into
`PanicException` at the function boundary as before. The test hook
`_panic_in_watched_worker_for_test` pins both wake-ups against a 30 s tick.

## Rejected

* **B: a core callback from worker threads.** It is silent while a long
  task chunk runs, and it can't deliver Ctrl-C, because signals reach only
  the main thread. It also has the core call its host, against ADR 0007.
* **C: the `log` crate with a `pyo3-log` bridge.** It adds dependencies and
  has the same thread problem as B.
* **A watcher OS thread per call**, the first form of the host-polled
  design. Review found its fixed per-call cost too high for workloads of
  many small queries. Running the engine as a job in the existing pool while
  the caller polls (D11) costs one job spawn and no new thread.
  `std::thread::scope` remains the fallback if the pool form shows a
  problem.

## Consequences

* A long call shows that it is moving and can be stopped; a short call adds
  no output. Results are unchanged. The counting code never reads the
  counters, and `tests/test_progress.py` checks counts, pairs, moments,
  relatives and burden for bit identity under `None`, `False` and a
  callable at budgets 1 and 4.
* The cost is one relaxed load every 64 rows and one relaxed add per
  2,048-row task.
* Every core entry point named above takes a `&Progress`, so a new
  relationship kernel threads one through, reports its phases, and adds its
  checkpoints to the per-path test table in `progress.rs`. That table pins
  where every check is.
* Kinship kernels (`pair_kinship`, the matrix DP, inbreeding) have no
  progress yet.
