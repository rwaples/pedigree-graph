# Pedigree Graph Code Quality Review

This document turns the thermo-nuclear code-quality audit into a set of independently addressable issues. The goal is to improve correctness, maintainability, and architectural clarity without changing intended behavior.

## Two review rounds

**Round 1, May 2026, the PGQ series.** PGQ-001 through PGQ-010 below, each an
independently addressable document in this directory.

**Round 2, September 2026, the v0.8 review.** One document,
[`v0.8-thermo-nuclear-code-quality-review.md`](v0.8-thermo-nuclear-code-quality-review.md),
covering the `v0.8` branch at `ce698ec`. Its eight findings are tracked as
GitHub issues rather than as files here.

| finding | status |
|---|---|
| 1, relationship semantics have two independent implementations | open, [#21](https://github.com/rwaples/pedigree-graph/issues/21) |
| 2, the Rust core exposes unchecked pedigree state | landed in `70bae7a`, and the residual degree clamp in `a75954a` ([#20](https://github.com/rwaples/pedigree-graph/issues/20)) |
| 3, relationship selector parsing belongs in a shared typed boundary | landed in `2cbb27b` and `a53f57c` |
| 4, sparse-matrix ownership is stateful and not exception-safe | exception safety landed in `eb2df82`, and the residual eager `_Am`/`_Af` pair in `028319a` ([#18](https://github.com/rwaples/pedigree-graph/issues/18)) |
| 5, effective-size orchestration is string-dispatched | warning scope landed in `eb2df82`; structural half open, [#19](https://github.com/rwaples/pedigree-graph/issues/19) |
| 6, `NeHillResult` encodes two states as a nullable field blob | landed in `74f0818` |
| 7, the benchmark harness is oversized and brittle | both reliability fixes landed in `eb2df82`, reports publish through `os.replace` and child output goes to temporary files; the suggested module split did not, `_harness.py` is still one 1,056-line file |
| 8, public result mappings rely on assertions | landed in `f37e2b7` |

The v0.8 review is a point-in-time record and is not rewritten as findings are
resolved, so this table is where its status lives.

## Suggested order

1. **PGQ-001**: Fix pair coordinate-space mismatch. This is a correctness bug.
2. **PGQ-002**: Tighten constructor/input validation. This protects every downstream path.
3. **PGQ-003 / PGQ-004**: Decompose and unify relationship extraction/counting semantics.
4. **PGQ-005 / PGQ-006**: Clean effective-size typed contracts and module boundaries.
5. **PGQ-007 / PGQ-008**: Simplify and decompose kinship kernel internals.
6. **PGQ-009**: Reassess the experimental BFS engine after the core extraction model is cleaner.

---

## Issues

- [PGQ-001 — Fix pair coordinate-space mismatch after `from_subsample`](pgq-001-fix-pair-coordinate-space-mismatch-after-from-subsample.md)
- [PGQ-002 — Replace loose constructor inputs and dense ID remapping with a validated input model](pgq-002-replace-loose-constructor-inputs-and-dense-id-remapping-with-a-validated-input-model.md)
- [PGQ-003 — Decompose `PedigreeGraph`; move relationship engines out of `_core.py`](pgq-003-decompose-pedigreegraph-move-relationship-engines-out-of-core-py.md)
- [PGQ-004 — Make relationship extraction semantics a single explicit plan instead of duplicated degree branches](pgq-004-make-relationship-extraction-semantics-a-single-explicit-plan-instead-of-duplicated-degree-branches.md)
- [PGQ-005 — Replace stringly typed effective-size payload dicts with explicit typed models](pgq-005-replace-stringly-typed-effective-size-payload-dicts-with-explicit-typed-models.md)
- [PGQ-006 — Split `_effective_size.py` into focused estimator modules](pgq-006-split-effective-size-py-into-focused-estimator-modules.md)
- [PGQ-007 — Simplify or fully support `_assemble_csc()`'s extra contract](pgq-007-simplify-or-fully-support-assemble-csc-s-extra-contract.md)
- [PGQ-008 — Decompose `_kinship_kernel.py` and shrink `_dp_kinship()`'s state machine](pgq-008-decompose-kinship-kernel-py-and-shrink-dp-kinship-s-state-machine.md)
- [PGQ-009 — Reassess the experimental BFS engine after core relationship semantics are centralized](pgq-009-reassess-the-experimental-bfs-engine-after-core-relationship-semantics-are-centralized.md)
- [PGQ-010 — Add architecture guardrails for large files and hidden coordinate/type contracts](pgq-010-add-architecture-guardrails-for-large-files-and-hidden-coordinate-type-contracts.md)
