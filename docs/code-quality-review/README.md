# Code quality reviews

Two review rounds, both resolved.

**Round 1, May 2026, the PGQ series.** Ten findings, PGQ-001 through PGQ-010,
from the pair coordinate-space mismatch after `from_subsample` to the
experimental BFS engine. Every subject they named is gone from the package
(`from_subsample`, the dense ID remapping, the in-`_core.py` engines,
`_kinship_kernel.py`, `_effective_size.py`, the BFS engine), and the line
budget and coordinate-token guardrails PGQ-010 asked for live in
`tests/test_architecture_guardrails.py` and `pedigree_graph/_view.py`. The
per-finding documents were removed on 2026-09-24; they are in git history up
to `3adf97d`.

**Round 2, September 2026, the v0.8 review.**
[`v0.8-thermo-nuclear-code-quality-review.md`](v0.8-thermo-nuclear-code-quality-review.md)
covers the `v0.8` branch at `ce698ec`. It is a point-in-time record and is not
rewritten, so its status lives here.

| finding | resolution |
|---|---|
| 1, relationship semantics have two independent implementations | the matrix extractor moved to `tests/oracle/` when `relationship_pairs` went to the Rust engine (`96e020e`); [#21](https://github.com/rwaples/pedigree-graph/issues/21) closed 2026-09-19 |
| 2, the Rust core exposes unchecked pedigree state | `70bae7a`, and the residual degree clamp in `a75954a` ([#20](https://github.com/rwaples/pedigree-graph/issues/20)) |
| 3, relationship selector parsing belongs in a shared typed boundary | `2cbb27b` and `a53f57c` |
| 4, sparse-matrix ownership is stateful and not exception-safe | exception safety in `eb2df82`, the residual eager `_Am`/`_Af` pair in `028319a` ([#18](https://github.com/rwaples/pedigree-graph/issues/18)) |
| 5, effective-size orchestration is string-dispatched | warning scope in `eb2df82`; the strategy registry in `dbec1c8` and `791c293` ([#19](https://github.com/rwaples/pedigree-graph/issues/19) closed 2026-09-12) |
| 6, `NeHillResult` encodes two states as a nullable field blob | `74f0818` |
| 7, the benchmark harness is oversized and brittle | both reliability fixes in `eb2df82` (reports publish through `os.replace`, child output goes to temporary files); the suggested module split was not taken, and `benchmarks/_harness.py` is one file |
| 8, public result mappings rely on assertions | `f37e2b7` |
