# ADR 0007: Host-neutral Rust core, host ownership, threading, build, and release

**Status:** accepted
**Date:** 2026-09-04
**Context:** the pedigree-graph Rust core redesign
(simACE `docs/plans/pedigree-graph-rust-core.md`); builds on ADR 0006

Revised 2026-09-24 to match 0.10.0; earlier wording in git history.

## Context

In 0.8.0 pedigree-graph's production kernels were Python, SciPy, and Numba.
They worked at simACE scale but carried structural costs: Numba compile and
cache warm-up, `Vec`-of-sparse-matrix intermediates that were hard to bound,
a separate Python thread pool for effective-size orchestration
(`_effective_size.py:132-206`), and no path to an R host.

A Rust pair-engine spike (branch `rust-spike`, commit `659aa0c`) matched the
Python pair sets on fixtures, inbred random pedigrees, and simulated
pedigrees through 300,000 rows. It was evidence that the relationship engine
ports cleanly. It did not cover the redesigned API, graph views, semantic
orientation, arbitrary input order, bindings, kinship, or R, and it carried
unchecked `i32` CSR multiplicity arithmetic (issue #9).

ADR 0006 fixes the public semantics in pure Python first. This ADR fixes how
the native implementation is structured, how it hands data to hosts, and how
it is built and released.

## Decision

### One host-neutral core

```text
Cargo.toml       workspace; [workspace.package].version is the package version
crates/core/     pedigree-graph-core: no Python/R imports; publish = false
crates/python/   PyO3 module pedigree_graph._native
pedigree_graph/  typed Python facade and host representations
r/               extendr R package `pedigreegraph`; binding crate in r/src/rust
```

The core graph is `pedigree_graph_core::graph::PedigreeGraph`: validated
columns of ids, parent and twin ids (`-1` when missing), parent and twin rows
(`-1` when missing or external), optional sex, generation labels, and birth
years, and a flag for whether every parent row precedes its child row.
Views are host objects. The host keeps the view's rows and its
coordinate-space token and passes a view-row map to the core, which relabels
and sorts pairs into view order.

The crate stays unpublished until 1.0. Python depends on it by workspace
path; the R source tarball copies it in and vendors its dependencies.

### Relationship engine

The engine classifies pairs one row at a time (ADR 0010). It builds no
adjacency powers and no degree-gated caches, so the `Ak(0)` parent-hop and
side-channel cache bugs of the matrix engine have no counterpart.

* Path multiplicity saturates at two (`relationships/multiplicity.rs`). The
  full/half decision needs only zero, one, or at least two, so the
  arithmetic cannot overflow (issue #9).
* Results are `PairBlocks`, one block for each of the 23 registry
  categories, so registry coverage is structural.
* Exclusions stay explicit per category (`EXCLUSIONS` in
  `relationships/engine.rs`). After a row's sets are final, each category
  loses the members an earlier registry category claims, so a pair lands in
  its closest category only.
* Blocks sort by the canonical unordered key `min * n + max`.
* Nothing native is retained after a call.

### Kinship DP storage

The DP keeps one owned vector pair per row and frees it when the row
retires (`kinship/rows.rs`). This was the simplicity prototype, and it had to
pass the wall/RSS gate before shipping. In a bake-off against the 0.9.1
wheel and a Rust port of its slab allocator
(`docs/pedigree-graph-0.8-migration/gate/14a/NOTES.md`), it ran at 0.11x to
0.28x the wheel's median wall and 0.21x to 0.80x its peak RSS on every cell,
and beat the slab port on every cell on both. On the 536k-row summary it
peaked at 3.6 GiB against the wheel's 14.8 GiB and the slab port's 8.6 GiB.
The slab port was deleted.

The DP runs in stable depth-major order and assembles the CSC in graph rows
without a sort. `kinship_matrix`, `approximate_kinship_matrix`, and the
generation summary are one kernel with three sinks. The differential against
the 0.9.1 Numba DP found that its retiring path could recycle a slot the same
merge walk was still reading; the native DP does not relocate rows during a
walk.

### Pairwise kinship memo layout

The pairwise kernel implements the pinned float32 recurrence of ADR 0009
(float32 values, peel the deeper endpoint, ties by row) and matches the
matrix bit for bit. Its memo, not its output, dominates memory. In the 0.8.0
Python kernel, at 536k rows and degree 3, the memo held 145M entries in 2^28
slots, and the rehash kept the predecessor table alive, so peak RSS was
about 1.5 times the final table.

The core's memo is one small open-addressing table per lower row, with
`(hi: u32, value: f32)` slots of 8 bytes (`kinship/memo.rs`). Each row grows
on its own, so a rehash copies one row. It was chosen over a Rust port of the
0.9.0 flat table by measurement
(`docs/pedigree-graph-0.8-migration/gate/13a/NOTES.md`): on the 536k-row
degree-3 batch, peak RSS fell from 5.4 GiB (wheel) to 3.0 GiB; on
`random_30k` the degree-3 walk went from 82.5 s to 5.3 s; on `random_300k`
it completed in 934 s at 14.1 GiB where the wheel did not finish in an hour.
The flat port measured within 12 to 25 percent of the wheel, so the saving
came from the layout, not the language. The memo lives for one call.

### Memoisation and ownership

The public graph is immutable. The core retains nothing between calls.
Results cross to the host in one transfer and the host memoises them on its
graph object, read-only.

* Relationship pair blocks: the core hands over two owned `Vec<i32>` per
  block. Python marks them read-only without a copy and caches nothing. The
  ownership benchmark (`benchmarks/bench_pair_emitters.md`) measured three
  assemblies of the engine's per-row output, all producing identical blocks.
  `buffered` (collect every task's chunks, then copy per category) was the
  fastest at six threads on every graph cell of 300k rows and above and
  peaked at about 2.3 times the raw `int32` payload; it backs
  `execution="speed"`. `two_pass` (count, size every block exactly, fill
  disjoint slices in place) peaked at the payload plus engine state, at 1.6
  to 1.9 times the wall of `buffered`; it backs `execution="memory"` and
  serves the 20M-row degree-5 query (2.12 billion pairs, 18.1 GiB, 291 s on
  six threads). `bounded_wave` won neither wall nor peak and was not kept.
* Kinship CSC arrays, the generation sums, inbreeding, and lineage vectors
  follow the same shape: one transfer, cached read-only on the host graph,
  no native cache.
* No compatibility cache field names remain public or test-observable.
* Relationship result objects never retain the graph.

### Threads and determinism

One package-owned Rayon pool per process (`crates/core/src/pool.rs`). The
budget resolves as `configure_threads(n)` > `PEDIGREE_GRAPH_THREADS` >
default 1, and is committed on first use (`pedigree_graph/_threads.py`; R
follows the same rules). Reconfiguring to a different value after that is an
error; repeating the committed value is allowed. There are no per-call thread
counts. A forked child rebuilds its own pool on first use.

`estimate_effective_sizes` has no worker pool. It commits the budget once
and applies the Python formulas serially; the kernels its prerequisites call
use the budget. Running the kinship and founder prerequisites concurrently
would multiply peak memory.

Every output, integer and float, is bit-identical across thread budgets.
There is no per-kernel tolerance. `tests/test_architecture_guardrails.py`
enforces this: it maps every core module that uses Rayon or the pool
(`PARALLEL_MODULES`: `relationships/pairs.rs`, `relationships/mod.rs`,
`relationships/burden.rs`; `POOL_INFRASTRUCTURE`: `pool.rs`) to a test that
compares budget 1 with budget 4 for bit equality. An unmapped parallel module
or a missing mapped test fails. The kinship, inbreeding, and lineage kernels
are serial.

### Safety and allocation

`pedigree-graph-core` is `#![forbid(unsafe_code)]`. CI rejects PyO3 and
extendr dependencies in the core crate and runs Clippy with warnings as
errors.

No user-reachable panics. Outside tests the core denies
`clippy::unwrap_used`, `expect_used`, `panic`, `unreachable`, `todo`, and
`unimplemented` (`crates/core/src/lib.rs`). Each surviving site carries an
`#[expect(..., reason = ...)]` that names the invariant it relies on.
`assert!` stays allowed for invariants the caller already checked. A const
block in `lib.rs` asserts at compile time that the public core types are
`Send + Sync`.

The host boundary is tested with a deliberate panic. Each binding crate
(`crates/python`, `r/src/rust`) has a `test-hooks` cargo feature that adds a
panic entry point. `tests/test_panic_boundary.py` and
`r/tests/testthat/test-panic-boundary.R` call it in a child process. Python
receives PyO3's `PanicException` and the next call in the same process
succeeds; R (extendr 0.9) receives a plain `simpleError` and the session
continues. `pixi run test` and
`test-all` build the Python hook through `build-dev`, and `r-install` builds
the R one; the test tasks set `PEDIGREE_GRAPH_REQUIRE_TEST_HOOKS=1` so a
missing hook fails. Published wheels and the R tarball never carry the
feature: `tools/r_build_tarball.sh` refuses to run with `PG_CARGO_FEATURES`
set.

Potentially large buffers use fallible reservation and surface as the
structured error `allocation_failed` (fields `operation`,
`requested_elements`, `dtype`; Python `ResourceError`) instead of aborting.
Subprocess tests force each allocation family to fail through a private test
seam. No default memory budget is set.

### Structured errors

The core has one `Error` enum whose variants carry ids, rows, capacities,
and operation context, each with a class, a stable code, and fields
(`crates/core/src/error.rs`). Python maps the classes to the exceptions in
ADR 0006. In R, failures cross the boundary as data and become classed
conditions, so nothing unwinds through R's longjmp.

### Python packaging

Maturin is the build backend (`pyproject.toml` `[tool.maturin]`). It builds
`pedigree_graph._native` from `crates/python`, as `abi3-py313` wheels that
ship `py.typed` and the `_native.pyi` stubs. CI installs the built wheel
outside the checkout and runs the tests against it. There is no production
Python fallback for a migrated operation.

Wheels: CPython 3.13+ via ABI3 on manylinux x86-64/AArch64, macOS
x86-64/Apple Silicon, and Windows x86-64, plus an sdist that needs Rust. No
PyPy, musllinux, Windows ARM, or 32-bit guarantee.

The release profile uses one codegen unit. Under the default sixteen, adding
unrelated modules moved the generation-summary DP's inlining and cost it 12%
(`docs/pedigree-graph-0.8-migration/gate/15a/NOTES.md`).

### Versioning

`[workspace.package].version` in `Cargo.toml` is authoritative, and maturin
reads the Python version from it. `r/DESCRIPTION` and the R binding crate
(its own workspace, so the source tarball builds without this one) repeat
the version by hand. There is no release tool:
`tests/test_r_version_agreement.py` fails a bump that misses either, and the
publish workflow checks the tag against the workspace version, the R copies,
and every built wheel and sdist. pedigree-graph keeps independent SemVer
outside the simACE/fitACE CalVer family.

### R package

The package is `pedigreegraph`: R package names cannot contain `_` or `-`,
and `pedigree.graph` reads as S3 dispatch. The surface is small:
`pedigree_graph()`, `relationship_pairs()`, `pair_kinship()`,
`kinship_matrix()`, `inbreeding()`, `relationship_categories()`,
`configure_threads()`, and `thread_budget()`. Views, count estimation,
lineage, connectivity, and effective size are not offered. Where R users
are better served than by a literal port, the surface departs from Python:

* `relationship_pairs` returns one long data frame. Columns are `code`,
  `first`, `second` (1-based graph rows) and, by default, `first_id` and
  `second_id`. `code` is a factor whose levels are all 23 registry codes in
  registry order, so every category is present in the object's structure.
  Requested status is `attr(pairs, "requested")`; roles come from
  `relationship_categories()`. A result larger than R's int32 row count is
  refused as a resource error (`pairs_exceed_frame_rows`), not truncated.
* `kinship_matrix` returns a `Matrix::dsCMatrix` that stores the upper
  triangle, built by the core's `kinship_csc_upper`, with ids as dimnames.
* The graph is an ordinary R list, so it survives `saveRDS` and forked
  workers. Kernels trust its columns only through a seal, an xxh3-64 hash
  over every field they read, and refuse an edited graph as
  `graph_modified`.

The source tarball is staged by `tools/r_build_tarball.sh`: the core is
copied in and every crate is vendored. CI checks it with `R CMD check
--as-cran` with the network disabled, and the publish workflow attaches it to
the GitHub release. Parity with Python is held by the goldens in
`r/tests/testthat/golden/` and by byte-identical products on the study
pedigrees (`gate/16b/NOTES.md`). CRAN submission and binary builds are
deferred.

### Migration

The migration ran from 0.8.0, the pure-Python API redesign of ADR 0006 and
the frozen baseline, to 0.10.0, the R package. Each release deleted the
production implementation it replaced. The BFS engine was removed first
(issue #7). Relationship pairs moved to the core in 0.9.0, pairwise kinship
in 0.9.1, the kinship matrices in 0.9.2, and inbreeding and lineage in
0.9.4, when Numba left the runtime dependencies. The replaced Numba kernels
remain as test oracles under `tests/oracle`. Still to come: 1.0.0
stabilisation and the decision whether to publish the crate.

### Gates

* Correctness: the twelve gates listed in the plan (all 23 categories in
  every result, roles/exclusivity/ordering, graph/view conversion under
  reorder and empty views, arbitrary input order, multiplicity without
  overflow, zero-up relationships starting at the individual itself,
  pairwise kinship edge cases, `F = 2·phi − 1`
  within `2^-22`, matrix entries bit-identical to pairwise values (ADR 0009),
  pinned CSC dtypes, partial-metadata rules, structured errors at both host
  boundaries).
* Differential: a readable independent Python oracle, property tests against
  it, large differential tests against the released 0.8.0 baseline,
  Rust-native invariant tests. Replaced Python is never kept as a fallback.
* Performance: the 5% median wall/RSS blocker applies only to
  behaviour-equivalent comparisons against the implementation being
  replaced, same thread budget, fresh interleaved processes, medians with
  uncertainty; block only on a confident >5% regression. Any exception needs
  maintainer sign-off and documentation.
* Cross-repository release gate before every release: pedigree-graph
  (pytest, Ruff, type check, Cargo test/rustfmt/Clippy, testthat, offline
  `R CMD check`), simACE test groups plus workflow smoke, fitACE core and
  every consuming method package, fitACE_epimight integration, and pedsum
  tests plus CLI smoke.

## Consequences

* Consumers get a compile-free import (no Numba warm-up) and one thread
  configuration knob. A consumer that wants more than one thread sets
  `PEDIGREE_GRAPH_THREADS` or calls `configure_threads`.
* Contributors need a Rust toolchain for source installs; binary wheels cover
  the listed platforms.
* No migrated operation has a Python fallback, so a native build failure is a
  hard failure rather than a silent slow path.
* The package left setuptools-scm. Version agreement across Cargo, the R
  package, the tag, and the built distributions is checked by a test and the
  publish workflow.
* R users get a small, correct surface rather than a parity port.

## Alternatives considered

* **Keep Python/Numba and optimise in place** — rejected. It cannot serve an
  R host, and the memory shape of the intermediates is the bottleneck, not
  kernel speed.
* **Port the BFS engine too** — rejected (issue #7). It was experimental and
  would have been adapted to Rust-owned adjacency for no production benefit.
* **Per-call thread arguments** — rejected. Nested pools and mismatched
  budgets between Python orchestration and Rayon are how the old double
  pool arose.
* **Publish the crate from the start** — deferred to 1.0. Publishing freezes
  a core API that ADR 0006's Python surface is still exercising.
* **Pre-accept `Vec<Vec<_>>` DP rows for simplicity** — rejected; the
  prototype had to pass the same gate as everything else, and did.
* **A slab/arena for DP rows** — measured and rejected (see Kinship DP
  storage).
* **Ship a pure-Python fallback wheel** — rejected. Two production
  implementations per operation is the state this migration exists to end.
* **Per-kernel float tolerances across thread budgets** — rejected. Every
  output is bit-identical across budgets.
