# ADR 0004: Align `max_degree` cutoffs with the registry degree

**Status:** accepted
**Date:** 2026-06-09
**Context:** relationship extraction API semantics

Revised 2026-09-24 to match 0.10.0; earlier wording in git history.

## Context

The relationship registry defines a kinship-distance degree for every
relationship code (`RELATIONSHIPS[code].degree`): `0` for MZ twins, `1` for
parent-offspring and full sibs, `2` for half-sibs, grandparent, and
avuncular, `3` for 1st cousins and the other kinship-1/16 categories, and so
on up to `5`.

When this ADR was written the matrix extractor did not use that meaning
consistently. `extract_pairs(max_degree=2)` computed `1C` even though `1C` is
registry degree 3, the streaming counter used a different gate for `1C`, and
`max_degree=0` still counted some degree-1 codes. Documentation had to
explain legacy behaviour instead of the registry vocabulary, and downstream
simACE defaults of `2` effectively meant "include 1st cousins".

## Decision

`max_degree` means exactly:

> include relationship category `code` iff `RELATIONSHIPS[code].degree <= max_degree`

for every public API that takes it: `relationship_pairs` and
`relationship_counts` on `PedigreeGraph` and `PedigreeView`, and
`PedigreeGraph.relationship_kinship_matrix`. `categories_up_to_degree` in
`_registry.py` applies the rule, and `RelationshipSelection.parse` in
`_selection.py` calls it for every endpoint.

The cutoffs are:

- `0`: MZ only
- `1`: add mother-offspring, father-offspring, full-sib
- `2`: add maternal/paternal half-sib, grandparent, avuncular
- `3`: add 1st cousins and the other degree-3 categories
- `5`: the full registry through 2nd cousins

Values outside `[0, MAX_DEGREE]` (`MAX_DEGREE` is 5) raise
`PedigreeValidationError` with code `max_degree_out_of_range`; booleans raise
`TypeError`.

There is no default cutoff. Under ADR 0006 each of these APIs takes exactly
one selector, `max_degree=` or `categories=`, and raises `TypeError` for both
or neither. This ADR first moved the default from `2` to `3` so that default
callers kept 1st cousins; ADR 0006 then removed the default, so every caller
names its cutoff.

## Considered options

- **Minimal 1C move only.** Move `1C` from the matrix extractor's degree-2 block
  to degree 3 but leave `max_degree=0` and defaults alone. Rejected: it fixes the
  most visible inconsistency while preserving the underlying ambiguous cutoff.
- **Compatibility flag / legacy mode.** Rejected: it would make every consumer
  choose between two definitions of the same parameter and keep the glossary
  ambiguous.
- **Strict registry alignment.** Chosen: one definition of degree.

## Consequences

- `max_degree=2` excludes `1C`. Code that intends "include 1st cousins" uses
  `max_degree=3`; code that intends a kinship cutoff of 1/8 uses
  `max_degree=2`.
- Selection is an output filter (ADR 0006): a selected category's closer
  dependencies are still resolved, and unselected categories come back empty
  with `requested=False` (pairs) or `None` (counts).
- `tests/test_relationship_pairs.py` asserts, for every cutoff 0 to 5, that
  a block is requested exactly when its registry degree is at or below the
  cutoff.
