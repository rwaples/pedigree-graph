# ADR 0004: Align `max_degree` cutoffs with `REL_REGISTRY.degree`

**Status:** accepted
**Date:** 2026-06-09
**Context:** relationship extraction API semantics

## Context

`REL_REGISTRY` defines a kinship-distance degree for every relationship code:
`0` for MZ twins, `1` for parent-offspring/full-sib, `2` for half-sibs,
grandparent, and avuncular, `3` for 1st cousins and same-kinship categories,
and so on.

The matrix extractor did not consistently use that meaning. In particular,
`PedigreeGraph.extract_pairs(max_degree=2)` computed `1C` even though `1C` is
registry degree 3. The streaming counter used a different gate for `1C`, and
`max_degree=0` still counted some degree-1 cheap codes. Documentation therefore
had to explain legacy behavior instead of the registry vocabulary.

Downstream simACE stats and plotting defaults had also grown around the legacy
matrix behavior: a default of `2` effectively meant "include 1st cousins" for
matrix extraction even though registry degree semantics say that should be `3`.

## Decision

Make `max_degree` mean exactly:

> include relationship category `code` iff `REL_REGISTRY[code].degree <= max_degree`

for all public pair-count/extraction APIs:

- `PedigreeGraph.extract_pairs`
- `PedigreeGraph.count_pairs`
- `PedigreeGraph.count_pairs_streaming`

Consequences of the cutoff are now:

- `0`: MZ only
- `1`: add mother-offspring, father-offspring, full-sib
- `2`: add maternal/paternal half-sib, grandparent, avuncular
- `3`: add 1st cousins and other degree-3 categories
- `5`: full registry through 2nd cousins

Change the public defaults from `2` to `3` so callers that relied on the old
default still get 1st cousins (and the other degree-3 categories). Explicit
`max_degree=2` now truly excludes 1st cousins.

## Considered options

- **Minimal 1C move only.** Move `1C` from the matrix extractor's degree-2 block
  to degree 3 but leave `max_degree=0` and defaults alone. Rejected: it fixes the
  most visible inconsistency while preserving the underlying ambiguous cutoff.
- **Compatibility flag / legacy mode.** Rejected: it would make every consumer
  choose between two definitions of the same parameter and keep the glossary
  ambiguous.
- **Strict registry alignment with default bump.** Chosen: one definition of
  degree, with defaults adjusted to preserve the old default output shape.

## Consequences

- Explicit `max_degree=2` callers may see fewer relationship categories than
  before: `1C` is no longer included.
- Default callers still include `1C` because the default is now `3`.
- Downstream code that intended "include 1st cousins" should use `max_degree=3`.
  Code that intended a kinship cutoff of 1/8 should keep `max_degree=2`.
- Tests assert that matrix and streaming engines zero every code above the
  registry cutoff.
