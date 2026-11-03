# ADR 0019: An id is a mother or a father, never both

**Status:** accepted
**Date:** 2026-11-03
**Context:** issue #58 (a parent id in both roles mixes role-specific and
role-blind relationship paths). Amends ADR 0006's input contract: "A child
cannot name the same known individual in both parent roles" and
"Parent-role/sex conflicts do not block construction".

## Context

ADR 0006 refused only one row naming one id as both its mother and its
father (`same_parent_id`). It let an id be the mother of one row and the
father of another, so validation tools could represent imperfect data.

With such an id the relationship definitions disagree with each other, in
the engine and in the path oracle alike. Sib links and the "shares a known
parent" exclusion go by role: sibs are grouped by mother id and by father id
separately. Upward paths ignore role: lineal and cousin paths climb through
either parent (ADR 0018). In the issue's case, P is the mother of a and the
father of b. The engine labels the pair `1C` (nominal kinship 1/16), while
`pair_kinship` returns 1/8, the half sibs' value. Consumers that read
kinship off the label get it wrong. An external parent id (no row) in both
roles splits a half sibship the same way, since sibs are grouped by parent
id per role.

The options were:

- (A) refuse such an id at construction;
- (B) make sib links and the shared-parent exclusion role-blind, which
  needs a rule for a cross-role half sib (neither `MHS` nor `PHS`);
- (C) keep the behaviour and document undefined labels.

## Decision

(A). `graph::build` refuses an id that is the mother of one row and the
father of another, external ids included, with the validation error
`parent_in_both_roles`. It runs right after `same_parent_id`, so the
construction order is: `pedigree_too_large`, `length_mismatch`,
`value_out_of_range`, `duplicate_id`, `same_parent_id`,
`parent_in_both_roles`, `cycle`, the four MZ codes, `birth_year_topology`.

Its fields are `parent_id` (the smallest id in both roles, so the witness
does not depend on row order), `mother_of_row` and `mother_of_id` (the first
row naming it as mother, and that row's id), `father_of_row` and
`father_of_id`, and `parent_count` (how many distinct ids are in both
roles). Rows count from each host's base.

A row whose mother and father are the same id stays `same_parent_id`, the
more specific error. MZ co-twins are two ids, so one may be a mother and the
other a father. Sex is still not checked against role: a parent recorded as
male may be a mother.

## Consequences

- Every relationship category has one meaning on every graph that builds.
  pg-phenotype's MRCA walk no longer needs to match a parent across roles.
- Real data with a sex or role error in its parent columns no longer builds.
  Validation tools must find and repair such ids before construction. pedsum
  tolerates an unsexed id in both roles under `--allow-missing-sex`; with
  this rule its graph construction refuses that input.
- The check marks each resolved parent row's role, one byte a row, while
  the parent columns resolve, and merges the distinct external parent ids of
  the two roles. On sparse ids the merged lookup lists those ids in order as
  it walks; on dense ids they are sorted, O(u log u) for u external parent
  references. `benchmarks/bench_construction.md` measures both: inputs
  without external parents and inputs on sparse ids stay within the 5% gate,
  while dense ids with 60% external parents over 5M rows go from 0.17 s to
  0.35 s.
- The core fixtures `inbred0` to `inbred4` have ids in both roles. The Rust
  engine tests that read them as engine columns keep them; the R goldens and
  `tools/r_parity.py`, which build graphs, no longer use them.
