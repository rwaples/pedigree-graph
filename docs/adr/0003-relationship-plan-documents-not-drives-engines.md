# ADR 0003: The relationship plan documents engine semantics; it does not drive the matrix subtract lists

**Status:** accepted
**Date:** 2026-05-29
**Context:** PGQ-004 (make relationship extraction semantics explicit instead of duplicated across engines)

## Context

Relationship semantics were duplicated across the three engines: the
streaming exact/approximate split and the BFS distinct-vs-paths divergence
lived only in prose docstrings (and a hand-copied code list in the streaming
test). `REL_REGISTRY` held labels and kinship but nothing about how each
engine handles each code, so a fix had to be applied — and could drift —
in several places.

## Decision

Add a `REL_PLAN` layer in `_registry.py` (a `dict[code, EngineSupport]`)
recording, per code, `streaming_exact` and `bfs_diverges_under_inbreeding`,
with helper accessors (`streaming_exact_codes`, `streaming_approximate_codes`,
`bfs_divergent_codes`). Docstrings and tests now derive from this single
source, and a test asserts `set(REL_PLAN) == set(REL_REGISTRY)` plus that all
three engines return exactly the registry key set.

**The plan documents engine semantics; it deliberately does NOT drive engine
control flow.** In particular, the matrix extractor's per-code subtract
dependency lists (e.g. `1C1R` subtracts `[po, gp, GGP, Av, GAv, sib_all, 1C]`)
stay hand-written in `_pair_extractor.py`, not encoded as plan data.

## Considered options

- **Encode the subtract dependency sets as plan data and drive the matrix
  engine from them.** Rejected for now. The lists are correctness-critical,
  order-sensitive, and entangled with the documented gotchas (booleanise
  *after* applying multiplicity; degree-gating cache population; the
  `≥ 2 shared ancestors` full/half distinction). Turning them into data would
  trade a real drift hazard for a worse correctness hazard, and PGQ-004
  explicitly scopes this out ("does not need to fully generate all engine code
  on day one"). If a future change does data-drive them, it should come with
  an equivalence test against the current hand-written lists across the full
  fixture suite.

## Consequences

- Adding a relationship code now means: add to `REL_REGISTRY`, add to
  `REL_PLAN` (the `set(REL_PLAN) == set(REL_REGISTRY)` test enforces this),
  and implement it in each engine. The engine subtract lists are still
  edited by hand — that is intentional, not an oversight.
- `REL_PLAN` and its helpers are importable from `_registry` but are not part
  of the public `pedigree_graph` package API; the surface can be promoted
  later if a downstream consumer needs the precision contract.
