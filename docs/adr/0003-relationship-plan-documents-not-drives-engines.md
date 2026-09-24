# ADR 0003: The relationship plan documents engine semantics; it does not drive the subtract lists

**Status:** accepted
**Date:** 2026-05-29
**Context:** PGQ-004 (make relationship extraction semantics explicit instead of duplicated across engines)

Revised 2026-09-24 to match 0.10.0; earlier wording in git history.

## Context

When this ADR was written, relationship semantics were duplicated across
three Python engines (matrix, streaming, and BFS). Which codes each engine
computed exactly lived only in prose docstrings and a hand-copied code list
in a test. The registry held labels and kinship but nothing about how each
engine handled each code, so a fix had to be applied, and could drift, in
several places.

## Decision

Keep a `REL_PLAN` layer in `_registry.py`, a `dict[code, EngineSupport]`
that records per-code engine handling beyond the structural
`RelationshipCategory`. Code that needs this handling derives it from the
plan instead of keeping its own list.

Today `EngineSupport` has one field, `estimate_exact`: whether
`close_relative_counts` computes the code. It is true for the six codes MZ,
MO, FO, FS, MHS, and PHS (ADR 0011), and `estimate_exact_codes()` returns
that set. `_streaming_counter.py` reads it to decide which keys of the
result carry a count and which are `None`. The field began as
`streaming_exact`, was renamed `estimate_exact` in 0.8.0, and kept that name
when 0.9.0 replaced the estimator with exact `close_relative_counts`. The
`bfs_diverges_under_inbreeding` field and its helper were removed with the
BFS counter in 0.8.4 (issue #7).

**The plan documents engine semantics; it does not drive the classifier's
control flow.** In particular, the per-code subtract lists stay hand-written
in the engine, not encoded as plan data. They are the `EXCLUSIONS` table in
`crates/core/src/relationships/engine.rs`; for example `1C1R` subtracts
`[MO, FO, GP, GGP, Av, GAv, FS, MHS, PHS, 1C]`. They were the subtract lists
of `_pair_extractor.py` when this was written.

## Considered options

- **Encode the subtract dependency sets as plan data and drive the engine
  from them.** Rejected. The lists are correctness-critical, order-sensitive,
  and entangled with the documented gotchas (booleanise *after* applying
  multiplicity; degree-gating cache population; the `≥ 2 shared ancestors`
  full/half distinction). Turning them into data would trade a real drift
  hazard for a worse correctness hazard, and PGQ-004 scoped this out ("does
  not need to fully generate all engine code on day one"). If a future change
  does data-drive them, it should come with an equivalence test against the
  hand-written lists across the full fixture suite.

## Consequences

- Adding a relationship code means: add it to `RELATIONSHIPS`, add it to
  `REL_PLAN` (an assert in `_registry.py` and
  `tests/test_relationships_registry.py` enforce matching key sets), and
  implement it in the Rust engine, including its `EXCLUSIONS` row. The subtract lists are edited by hand on purpose.
- `REL_PLAN` and its helpers are importable from `_registry` but are not part
  of the public `pedigree_graph` API.
