# ADR 0007: Host-neutral Rust core, host ownership, threading, build, and release

**Status:** accepted
**Date:** 2026-09-04
**Context:** the pedigree-graph Rust core redesign
(simACE `docs/plans/pedigree-graph-rust-core.md`); builds on ADR 0006

## Context

pedigree-graph's production kernels are Python, SciPy, and Numba. They work at
simACE scale but carry structural costs: Numba compile and cache warm-up,
`Vec`-of-sparse-matrix intermediates that are hard to bound, a separate Python
thread pool for effective-size orchestration
(`_effective_size.py:132-206`), and no path to an R host.

A Rust pair-engine spike (branch `rust-spike`, commit `659aa0c`) matches the
current Python pair sets on fixtures, inbred random pedigrees, and simulated
pedigrees through 300,000 rows. It is evidence that the relationship engine
ports cleanly. It does not cover the redesigned API, graph views, semantic
orientation, arbitrary input order, bindings, kinship, or R, and it carries
unchecked `i32` CSR multiplicity arithmetic (issue #9).

ADR 0006 fixes the public semantics in pure Python first. This ADR fixes how
the native implementation is structured, how it hands data to hosts, and how
it is built and released.

## Decision

### One host-neutral core

```text
Cargo.toml
crates/core/     pedigree-graph-core: no Python/R imports; publish = false
crates/python/   PyO3 module pedigree_graph._native
pedigree_graph/  typed Python facade and host representations
r/               extendr package, added in 0.9.0
```

The core type is `pedigree_graph_core::PedigreeGraph`. Core state is original
IDs with typed missing/external/internal parent references, input-row ↔
private-topological-row maps, sex / optional generation labels / optional
birth years, derived structural depth, a shared execution context, and
bounded one-shot memoisation. `PedigreeView` holds an `Arc<PedigreeGraph>`,
ordered view-row ↔ graph-row maps, and its own coordinate-space token.

Host `-1` sentinels never enter the domain model. Internally the core uses
typed `IndividualId`, `RowId`, and explicit reference states in
structure-of-arrays storage.

The crate stays unpublished until 1.0: Python uses a workspace path, R vendors
the source.

### Relationship engine invariants

* `Ak(0)` is an `Identity` variant, never an accidental parent hop.
* Pair subtraction uses one canonical 64-bit key from two 32-bit rows; public
  semantic orientation is restored only after key operations.
* Full/half classification preserves path multiplicity through the `>= 2`
  decision. Multiplicity arithmetic is checked; production code may not
  silently wrap (issue #9 gates promotion of the spike).
* `CousinSplit { full, half }` replaces degree-gated side-channel caches.
* `[PairBlock; 23]` indexed by the relationship enum makes registry coverage
  structural.
* Exclusion lists stay explicit per category; a global uniqueness/precedence
  check catches a missing exclusion.
* Query-local intermediates are freed at call completion.

### Kinship DP storage

No row-storage representation is preselected. `Vec<Vec<_>>` is the first
simplicity prototype and must pass the wall/RSS gate against the warmed
current implementation; if it fails, a slab/arena design with retirement is
ported instead. A Rust port alone is not assumed to improve the
output-dominated DP.

### Pairwise kinship memo layout

The pairwise kernel implements the pinned float32 recurrence of ADR 0009
(float32 values, peel the deeper endpoint, ties by row) and must match the
matrix bit-for-bit. Its memo, not its output, dominates memory: at 536k rows
and degree 3 the Python kernel holds 145M entries in 2^28 slots, and the
rehash keeps the predecessor table alive, so peak RSS is about 1.5 times the
final table (6.2 GB float64, 4.6 GB float32). A layout that grows without a
copy, or sizes from a bound, is a requirement here; it is worth about 30
percent of peak independent of dtype and was deliberately left out of the
0.8.0 Python kernel.

*Amended 2026-09-23 (slice 13, 0.9.1).* The Rust core's memo is one small
open-addressing table per lower row, `(hi: u32, value: f32)` slots of 8
bytes, each row doubling on its own, so a rehash copies one row and nothing
else. It was selected over a Rust port of the 0.9.0 flat table by
measurement (`docs/pedigree-graph-0.8-migration/gate/13a/NOTES.md`): on the
536k-row degree-3 batch peak RSS fell from 5.4 GiB (wheel) to 3.0 GiB, on
`random_30k` the degree-3 walk went from 82.5 s to 5.3 s, and on
`random_300k` the walk completes in 934 s at 14.1 GiB where the wheel did not
finish in an hour. The memo lives for one call and is freed before the call
returns; nothing is retained on the graph, which closes the migration plan's
"pairwise kinship" step. The flat port measured within 12 to 25 percent of
the wheel, so the layout, not the language, is the saving.

### Memoisation and ownership

The public graph is immutable; safe interior memoisation is allowed.

* `OnceLock`-style caches for modest single-valued host-neutral results
  (inbreeding, lineage vectors, generation summaries).
* No unbounded caches for arbitrary pairwise requests or arbitrary
  category/view query combinations.
* Large relationship and CSC buffers require an ownership benchmark before a
  strategy is chosen: Rust-owned caching plus host conversion versus a single
  ownership transfer with only the read-only host representation cached.
* No compatibility cache field names remain public or test-observable.
* Relationship result objects never retain the graph.

### Threads and determinism

One package-wide Rayon pool, configured before first parallel work:
explicit `configure_threads(n)` > `PEDIGREE_GRAPH_THREADS` > default 1.
Reconfiguration after initialisation is an error unless it repeats the
existing value. There are no per-call thread counts; fitACE's thread-cap
helper sets `PEDIGREE_GRAPH_THREADS`. `estimate_effective_sizes` prepares
prerequisites on this pool and applies the Python formulas serially; the
separate Python worker pool is removed.

Before the Rust core exists, the pure-Python 0.8.0 package already exposes
`configure_threads` with the same precedence and reconfiguration rule. In
0.8.0 it is a package-level budget read by Python-level worker pools on
new-API paths, currently the pair extractor. Effective-size orchestration and
its lazy prerequisites run serially: the old pool dispatched formulas only
after eagerly building the expensive prerequisites, while concurrent kinship,
founder, and Caballero-Toro prerequisites would multiply peak memory. The
entry point still resolves `thread_budget()` once so package configuration is
committed consistently, and kernels it calls may honor that budget. It never
calls `numba.set_num_threads`: that global is process-wide, consumers such as
simACE pin it themselves, and no production kernel in the package runs
`parallel=True` (the one that does is the experimental BFS engine, which issue
#7 removes). Adapters marked for deletion keep 0.7.1 execution behaviour and
their own thread arguments until slice 7. When the Rayon pool lands, the same
function configures it and may prepare independent effective-size prerequisites
without changing the serial Python formulas.

Acceptance criterion, recommended pending final sign-off: integer outputs are
bit-identical across thread counts; floating reductions use fixed partitions
and ordered combination where practical. Any tolerance is declared per kernel
and justified by benchmarked cost.

### Safety and allocation

`pedigree-graph-core` is `#![forbid(unsafe_code)]`. CI asserts core types are
`Send + Sync`, rejects PyO3/extendr dependencies in core, runs Clippy with
warnings as errors, and forbids user-reachable panics. Potentially large
buffers use fallible reservation and propagate structured `ResourceError`s
through Rayon and both host bindings; subprocess tests verify adversarial
allocation failures raise rather than abort. No default memory budget is
invented before profiling.

### Structured errors

Core errors are enums carrying IDs, rows, capacities, and operation context.
Python maps them to the exception classes in ADR 0006; R uses classed
conditions.

### Python packaging

Maturin is the preferred backend, conditional on a scaffold that proves mixed
Python/Rust packaging as `pedigree_graph._native`, `abi3-py313` wheels,
editable install through the pedigree-graph pixi manifest, wheel-install
tests independent of the source tree, clean sdist → wheel builds, and shipped
type stubs plus `py.typed`. A failed gate is documented before falling back
to setuptools-rust. There is no production Python fallback once an operation
has migrated.

Initial wheels: CPython 3.13+ via ABI3 on manylinux x86-64/AArch64, macOS
x86-64/Apple Silicon, Windows x86-64, plus an sdist requiring Rust elsewhere.
No initial PyPy, musllinux, Windows ARM, or 32-bit guarantee.

### Versioning

0.8.0 may be the final setuptools-scm release. Once the Cargo workspace lands,
`[workspace.package].version` is authoritative; Maturin reads the Python
version from Cargo; the release tool updates Cargo and `r/DESCRIPTION`
together; CI asserts wheel, sdist, Cargo metadata, DESCRIPTION, and Git tag
agree. pedigree-graph keeps independent SemVer outside the simACE/fitACE
CalVer family.

### R 0.9.0

A deliberately small extendr surface: `pedigree_graph(df)`,
`relationship_pairs(pg, max_degree=)`, `pair_kinship(pg, first, second)`,
`kinship_matrix(pg)` (float32 values promoted into `Matrix::dgCMatrix`), and
`inbreeding(pg)`. `relationship_pairs` returns a named list of all 23
categories, each a data frame with integer 1-based `first`/`second`, role
attributes, requested status, and deterministic order. Views, count
estimation, lineage, connectivity, and effective size are deferred.

The source tarball compiles offline: the core is staged under `r/src/rust`,
`cargo vendor` captures the full locked dependency graph, and network-disabled
CI proves `cargo --offline` plus `R CMD check` against the final tarball. The
layout is CRAN-compatible from the first release; submission is deferred.

### Migration sequence

Each published slice is green, deletes the production implementation it
replaces, and passes the cross-repository release gate. In order:

1. 0.8.0 — pure-Python API redesign (ADR 0006), frozen baseline.
2. Native construction, structured errors, depth; PyO3 module and facade;
   Maturin and Cargo-authoritative versioning once scaffold gates pass.
3. Remove the BFS engine (issue #7) before touching relationships.
4. Streaming relationship-count estimator.
5. Relationship-pair engine (after issue #9), compared against the oracle and
   the 0.8.0 baseline at fixture, inbred, 30k, and 300k scales.
6. Pairwise kinship (issue #6 resolved by ADR 0009: recurrence-only, pinned
   float32 recurrence); Python/Numba production kernel deleted, independent
   oracle retained.
7. Complete and relationship-limited CSC matrices; DP storage chosen by
   benchmark.
8. Inbreeding, generation summary, lineage, effective-size prerequisites;
   Numba removed when nothing uses it.
9. 0.9.0 — R package. 10. 1.0.0 — stabilisation, guardrails, decide whether
   to publish the crate.

### Gates

* Correctness: the twelve gates listed in the plan (all 23 categories in
  every result, roles/exclusivity/ordering, graph/view conversion under
  reorder and empty views, arbitrary input order, multiplicity without
  overflow, `Ak(0)` identity, pairwise kinship edge cases, `F = 2·phi − 1`
  within `2^-22`, matrix entries bit-identical to pairwise values (ADR 0009),
  pinned CSC dtypes,
  partial-metadata rules, structured errors at both host boundaries).
* Differential: a readable independent Python oracle, property tests against
  it, large differential tests against the released 0.8.0 baseline,
  Rust-native invariant tests. Replaced Python is never kept as a fallback.
* Performance: the 5% median wall/RSS blocker applies only to
  behaviour-equivalent comparisons, warmed Python/Numba versus release-mode
  Rust, same thread budget, fresh interleaved processes, medians with
  uncertainty; block only on a confident >5% regression. Any exception needs
  maintainer sign-off and documentation.
* Cross-repository release gate before 0.8.0 and every migration patch:
  pedigree-graph (pytest, Ruff, type check, then Cargo test/rustfmt/Clippy),
  simACE test groups plus workflow smoke, fitACE core and every consuming
  method package, fitACE_epimight integration, pedsum tests plus CLI smoke,
  and from 0.9.0 testthat plus offline `R CMD check`.

## Consequences

* Consumers gain a compile-free import (no Numba warm-up) and one thread
  configuration knob; fitACE must set `PEDIGREE_GRAPH_THREADS` where it
  previously capped threads elsewhere.
* Contributors need a Rust toolchain for source installs; binary wheels cover
  the listed platforms.
* No migrated operation has a Python fallback, so a native build failure is a
  hard failure rather than a silent slow path.
* The release tool and CI grow version-agreement checks across four
  artifacts; the package leaves setuptools-scm.
* R users get a small, correct surface in 0.9.0 rather than a parity port.

## Alternatives considered

* **Keep Python/Numba and optimise in place** — rejected. It cannot serve an
  R host, and the memory shape of the intermediates is the bottleneck, not
  kernel speed.
* **Port the BFS engine too** — rejected (issue #7). It is experimental and
  would be adapted to Rust-owned adjacency for no production benefit.
* **Per-call thread arguments** — rejected. Nested pools and mismatched
  budgets between Python orchestration and Rayon are how the current double
  pool arose.
* **Publish the crate from the start** — deferred to 1.0. Publishing freezes
  a core API that ADR 0006's Python surface is still exercising.
* **Pre-accept `Vec<Vec<_>>` DP rows for simplicity** — rejected; it is a
  prototype that must pass the same gate as everything else.
* **Ship a pure-Python fallback wheel** — rejected. Two production
  implementations per operation is the state this migration exists to end.

## Amended 2026-09-17 (slice 12, pair emission ownership and the R milestone)

**Ownership strategy for relationship pair blocks.** The benchmark this ADR
required before choosing an ownership strategy for large relationship
buffers has been run (`benchmarks/bench_pair_emitters.md`: screening on
`random_30k` and `random_300k`, qualification against the PyPI 0.8.4 wheel
on graph and reordered-view receivers at degrees 3 and 5 with five
interleaved fresh processes, a 2M run, and the 20M capability runs). Three
assemblies of the row-streaming engine's per-row output were measured, all
producing identical blocks:

* `buffered` (collect every task's chunks, then copy per category) is the
  fastest at six threads on every graph cell of 300k rows and above and
  peaks at about 2.3 times the raw `int32` payload. It backs
  `execution="speed"`.
* `two_pass` (count, size every block exactly, fill disjoint task slices in
  place) holds no copy of the result and peaks at the payload plus engine
  state (1.11 times the payload at 20M rows, degree 5), at 1.6 to 1.9 times
  the wall of `buffered`. It backs `execution="memory"`, and it is what
  serves the 20M degree-5 query (2.12 billion pairs, 18.1 GiB, 291 s on six
  threads) within the 30 GiB target machine.
* `bounded_wave` (append wave by wave) won neither wall nor peak and is not
  kept.

Either way the core hands the host two owned `Vec<u32>` per block in one
transfer; the host marks them read-only without a copy and caches nothing.
No Rust-side cache of pair results exists, in keeping with the "no unbounded
caches" rule above.

**Threads.** Slice 11 built a Rayon pool per relationship call as a
temporary measure. Slice 12 restores the model this ADR states: one
package-owned pool, sized from the committed Python `thread_budget()`, used
by counts and pairs alike, with repeated initialisation at the same value
allowed and a conflicting value an error. Integer outputs stay bit-identical
across thread counts.

**Allocation safety.** The shared relationship engine becomes fallible end
to end: CSR construction and transpose, workspaces, set growth, task chunks
and count tables, final blocks, and view-sort scratch all use fallible
reservation and surface as one structured core error,
`allocation_failed` (fields `operation`, `requested_elements`, `dtype`,
class Resource), which Python maps to `ResourceError`. Subprocess tests
force each allocation family to fail through a lowered private limit.

**Versioning and the R milestone.** The release carrying this work is
0.9.0, not a 0.8 patch, because the unreleased branch already removed
`estimate_relationship_counts` (CHANGELOG). Step 9 of the migration
sequence above therefore moves: the R package is the 0.10.0 milestone, and
the "R 0.9.0" section and the `r/` line of the layout should be read as
0.10.0. Nothing about the R surface itself changes.

## Amended 2026-09-23 (slice 14, the kinship matrix DP)

**DP row storage.** The bake-off this ADR required ran on the 0.9.1 wheel
and two source layouts in one interleaved sweep of nine cells
(`docs/pedigree-graph-0.8-migration/gate/14a/NOTES.md`). One owned vector
pair per row, freed on retirement, is 0.11x to 0.28x the wheel's median
wall and 0.21x to 0.80x its peak RSS on every cell, and beats the ported
0.9.1 slab allocator on every cell on both metrics; on the 536k-row summary
it peaks at 3.6 GiB against the wheel's 14.8 GiB and the arena's 8.6 GiB.
The "first simplicity prototype" above therefore ships, and the arena was
deleted with the record. The DP runs in stable depth-major order inside the
core and assembles the CSC in graph rows without a sort, so the SciPy
permutation copy is gone; the three products (`kinship_matrix`,
`approximate_kinship_matrix`, the generation summary) are one kernel with
three sinks, and step 7 of the migration sequence is done with the
generation summary taken from step 8.

**Ownership.** The three CSC arrays and the sum vector are handed over once
and cached read-only on the graph, as the slice 12 pair blocks are; nothing
native is retained. The ownership benchmark this ADR asked for on large CSC
buffers is the slice 12 measurement adopted as is: same shape, one transfer,
no native cache.

**A 0.9.1 defect the differential found.** The numba DP's retiring path
could recycle, within one merge walk, the slot that walk was still reading;
`mean_kinship_by_generation` was wrong on the 536k study pedigree by 1.4e-5
relative in its deepest bucket. The record shows the mechanism and the
native DP is held to the relocation-free walk.

## Amended 2026-09-23 (slice 15, the last Numba kernels)

**Numba is out of the wheel.** The Meuwissen-Luo inbreeding walk, the
distinct-ancestor and descendant-path sweeps, Maignel's equivalent complete
generations and the per-cohort founder-contribution means moved to the core
(`kinship/inbreeding.rs`, `lineage.rs`, `kinship/generations.rs`), and `numba`
left the runtime dependencies; the three Numba modules are kept verbatim as
test oracles under the `test` extra. Every kernel the package runs is now
Rust, SciPy or NumPy.

**Storage and caches.** The distinct-ancestor sets are one exactly sized
slice per row with children, dropped after its last child, the layout slice
14 selected for the DP rows. It measured 0.40x to 0.69x the 0.9.3 pool's
peak RSS on all twelve benchmarked pedigrees, so no arena was ported. As in
slices 12 to 14, results are handed over once and memoised on the graph;
nothing native is retained.

**Order.** The sweeps that need only parents before children walk graph rows
when construction found them parents-first, and only then sort by depth,
which they then take and check; `kinship/depth_order.rs` holds the sort the
kinship DP shares. Construction's validated columns and its parents-first
flag are trusted at these bindings rather than re-checked, because a second
pass was the difference on sweeps of a few milliseconds.

**Build.** The release profile uses one codegen unit: under sixteen, adding
this slice's modules moved the unchanged generation-summary DP's inlining and
cost it 12%. Record: `docs/pedigree-graph-0.8-migration/gate/15a/NOTES.md`.
