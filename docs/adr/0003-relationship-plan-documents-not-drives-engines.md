# ADR 0003: The relationship plan documents engine semantics; it does not drive the subtract lists

**Status:** accepted
**Date:** 2026-05-29
**Context:** PGQ-004 (make relationship extraction semantics explicit instead of duplicated across engines)

Revised 2026-09-24 to match 0.10.0; earlier wording in git history.
Amended 2026-11-04 (issue #56): the engine reads each category's path
shape from the core; see "Path shapes are read, not driven".

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

### Path shapes are read, not driven

The core's `Category` owns each category's path shape: `up`, `down` and
`ancestor_count` (issue #56). The engine's call sites read it. A lineal
walk takes its hop count from `up`. An avuncular category reads the paths
at depth `up - 1`, a removed cousin category multiplies `A^down` by
`(A^up)^T`, and second cousins read `up` and `down`. Full against half is
`ancestor_count`'s path-count test (ADR 0018). This does not make the
classifier data-driven. The calls, their order, the degree gates and
`EXCLUSIONS` are still written out by hand, and the engine never loops over
categories to decide what to compute. The shape replaces a literal at a call
that was already there. The walks the engine shares between categories keep
their own constants. The grandparent product feeds both 1C and the 2C rule,
so it walks `GRANDPARENT_DEPTH`, which a compile-time assertion ties to 1C
and GP. The per-depth avuncular caches are filled in order 1 to 4. A wrong
shape in the core now fails the core's own engine tests, and does not wait
for the Python cross-check.

MZ is outside the shape formula. Its shape is `(0, 0, 0)` and its nominal
kinship is 1/2, and the engine finds co-twins from the twin column, not from
a shape.

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
