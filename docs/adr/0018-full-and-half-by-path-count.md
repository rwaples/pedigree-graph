# ADR 0018: a collateral category is full when two paths join the pair

**Status:** accepted; amended 2026-11-05 (issue #61: external parent ids on
cousin paths)
**Date:** 2026-10-30
**Context:** issues #47 (three full/half rules against one documented rule)
and #48 (revisit ADR 0010's category definitions). Supersedes the section
"Categories are the 0.7.1 definitions, with exclusions as a table" of
ADR 0010 and that ADR's consequence that the SciPy matrix engine stays as
the differential oracle. Ships in 0.14.0.

## Context

ADR 0010 kept 0.7.1's full/half rules so the row-streaming rewrite could be
gated on parity with 0.7.1, and said they must not be "fixed" silently. The
engine then decided full against half three ways:

| Categories | Full when |
|---|---|
| sibs, `Av` .. `G3Av` | the two share both parent ids (the sibling index) |
| `1C` | at least two *distinct* shared grandparents |
| `1C1R`, `1C2R`, `2C` | at least two *paths*, each orientation thresholded alone |

while `GLOSSARY.md` gave one rule: full when the connecting ancestors are a
mated pair. Two half connections got opposite labels by shape: double half
first cousins (two unmated shared grandparents) were `1C`, and an aunt who is
a half sib of both of the nephew's parents was `HAv`. pedigree-graph-validate
measured the split against verbalisr 0.7.2 (#47, known difference D1): in
purgeR's `atlas`, 154 of 1,691 sampled related pairs would change label
under the mated-pair rule.

Every consumer reads `nominal_kinship` off the label: simACE's expected
correlations, fitACE's Falconer estimates for `1C` and `Av`, the EPIMIGHT
relatedness table, pedsum's `kinship` column and pg-phenotype's ACE fit; and
fitACE's export, its kinship matrix and PA-FGRS choose their pairs by the
label's degree.

## Decision

### One rule: count the paths

A *path* joins two people through one shared ancestor. Upward it follows
represented parent rows, each route counted once, and ends at an external
parent id (an id with no row) as at a founder; a sib link counts one path
per parent id the two share, external ids included, so full sibs are joined
by two paths and half sibs by one. Every shape sees the same external ids
(amended for issue #61, below). A collateral category is **full** when at
least two paths of its shape join the pair and **half** when exactly one
does:

* the avuncular family at depth k (`Av`/`HAv` k=1, `GAv`/`HGAv` 2,
  `GGAv`/`HGGAv` 3, `G3Av` 4): each k-hop ancestor of the niece or nephew,
  times the parent ids it shares with the aunt or uncle;
* `1C`/`H1C`: the products of path counts through each shared grandparent;
* `1C1R`/`H1C1R`/`1C2R` and `2C`: the same products at their shapes.

Sibs are unchanged: sharing both parent ids is two paths. Lineal codes have
no full/half split.

### Both orientations count

An asymmetric pair can be joined in both orientations: one aunt-to-nephew
path each way, or one half-1C1R path with each member as the junior cousin.
The paths of the two orientations are summed before the threshold, so such a
pair is full. Saturated multiplicity allows the sum (`s(a + b) =
min(s(a) + s(b), 2)`, ADR 0010). A pair joined both ways is reported with the
lower row first, the ADR 0006 tie break, which is what the engine's first arm
already gives.

### Why paths and not mated pairs

The mated-pair rule is the genealogical one and matches verbalisr. It was
rejected because consumers read kinship off the label:

* When exactly two paths through represented ancestors join a pair, its
  kinship is the full label's nominal kinship: double half first cousins,
  `2 × 1/32 = 1/16`, are `1C`; the aunt who is a half sib of both parents,
  `2 × 1/16 = 1/8`, is `Av`. In all 263 pairs #47 found differing, the
  0.13.0 label, which already counted paths or distinct ancestors for those
  categories, was nearer `pair_kinship` than the mated-pair label.
* A pair only ever moves to a *closer* category (`HAv` to `Av`, `H1C` to
  `1C`, `H1C1R` to `1C1R`, and through the closest-category fold, a lineal
  or deeper code to a newly full collateral one). Consumers that select by
  degree gain correctly labelled pairs and never lose one. Under the
  mated-pair rule pairs move deeper, and those with no registered half code
  (`1C2R` and `2C` through one ancestor would be degree 6) fall out of the
  classification.
* It changes the fewest categories: `1C1R`, `1C2R` and `2C` already counted
  paths, and sibs agree under every rule.

The label stays a two-level summary: one path, or two or more. With more
than two paths kinship exceeds the nominal value: double first cousins, or
the children of two half-sib matings within one sibship (four paths through
one grandparent), have kinship 1/8 and are `1C` (nominal 1/16). A path
through an external parent id, at any shape, carries no kinship:
`pair_kinship` follows represented parent rows only. The exact value is
`pair_kinship`.

### What does not change

`EXCLUSIONS`, the first cousin exclusion "shares a known parent id", the 2C
exclusion of pairs sharing a grandparent, sibling grouping by original
parent id, and the closest-category fold.

### One classification oracle

`tests/oracle/relationship_paths.py` states these definitions pair by pair
and is the only classification oracle. The SciPy matrix engine kept under
ADR 0010 (`tests/oracle/relationship_pairs.py`) is deleted: it encoded the
superseded rules, and a second engine changed in lockstep with the first
would share its definition defects. Its block helpers (`canonical_keys`,
`check_exclusive`, `sibling_pairs`) moved to `tests/oracle/pair_utils.py`.

## Evidence

* The engine equals the path oracle block for block, order included, on
  every parity fixture (graph and views, every selector, both executions),
  on generated pedigrees at every degree cutoff, and on 1,200 dense inbred
  random pedigrees of 15 to 44 rows with every code populated. On 400 of
  them, the 0.13.0 engine disagrees with the same oracle in 3,853
  code-by-pedigree cells.
* Folded counts of the Rust parity fixtures that change: `deep_inbred_60g`
  `Av` 441 to 1,205, `GAv` 634 to 2,567, `GGAv` 696 to 3,103, `G3Av` 590 to
  2,717, `1C` 881 to 1,228, with the half codes falling to match; the five
  random inbred pedigrees move by a few to a few dozen pairs per code;
  `external_parents` moves one pair from `HGAv` to `GAv`. Random mating at
  simACE scale moves a little: `sim_d6_n3000` (18,000 rows) `G3Av` 60,471 to
  62,033 and `Av`, `GAv`, `GGAv`, `1C` each by under 1%.
* pedigree-graph-validate, its verbalisr scorer switched to this rule (a
  mated-pair path counts two), passes every scored classification row
  (closest category, counts, per-person burden) on every case, with FAIL 0
  and ERROR 0 overall. Its D1 row, the genealogical rule, now differs on more
  pairs (`atlas` 173 of 1,000 sampled pairs, `deep_chain` 425 of 3,160) and is an
  accepted difference there.

## Consequences

* A breaking change to the relationship contract: pair blocks, counts,
  burden, moments and per-person relatives change on pedigrees with double
  connections, which means inbreeding or relatives joined through two lines.
  Pedigrees without them are unaffected. The pair, structure and pair-kinship
  golden locks, the Rust parity fixtures and the R goldens were regenerated.
* `GLOSSARY.md` "Relationship category" states this rule.
* pedigree-graph-validate's D1 is an intended difference: verbalisr labels by
  genealogy, pedigree-graph by path count.

## Amendment: an external parent id is on every path (issue #61)

As first written, a sib link counted a shared external parent id but the
cousin products walked represented rows only, so an external id joined half
sibs and, through them, avuncular pairs, but never cousins: rows 1 and 2
sharing the external mother 900, with children 3 and 4, gave `MHS` (1, 2),
`HAv` (3, 2) and (4, 1), and no pair (3, 4), which became `H1C` once 900 was
a row. Every cousin shape (`1C`/`H1C`, `1C1R`/`H1C1R`, `1C2R`, `2C`, and the
2C exclusion of pairs sharing a grandparent) now reaches an external id as a
parentless ancestor whose children are the rows naming it. The
classification among the original rows is then the one the pedigree gets
with each external id made a founder row, which
`external_parents_as_founder_rows_add_only_lineal_pairs` checks. Sibs, the
avuncular family and lineal codes already agreed and do not change.

The rejected alternative was to record the gap and restate the founder-row
property as "the original pairs gain collateral categories only through the
new rows". It keeps fewer labels whose nominal kinship exceeds
`pair_kinship` (0 through any external id), but it labels a structure at one
shape and not the next, and leaves real half cousins out of every
degree-selected consumer; this ADR already prefers moving pairs closer to
dropping them.

The engine adds the external hop where the cousin products start
(`Engine::below_ancestors`): when an ancestor `k - 1` up has an external
parent id, the rows naming that id join the first hop below the ancestors
`k` up, and the chain continues from there; otherwise the product is the
represented chain unchanged. Every external id counts, including one that a
single row names, so the per-category sets match the founder-row model
before the closest-category fold as well as after it. The alternative, a
virtual founder row per external id in the parent CSR, would widen every
workspace and accumulator by the number of external ids and make every
lineal and avuncular read skip the virtual rows; it was not built, because
the hop costs little. On `random_300k` (one thread, five calls each, setup
subtracted), user-space instructions changed by +1.23% (`pairs`, degree 3),
+0.84% (`counts`, degree 5), +0.49% (`moments`) and +0.96% (`relatives`).
Wall-time medians on this host could not resolve a few percent: an A/A run
of the unchanged engine moved single cells by -5% to +17%.

## Alternatives considered

* **Mated pair everywhere.** Genealogical and matches verbalisr; rejected
  above.
* **Keep the per-category rules and document them (#47 option 1).** No
  consumer risk, but the same structure keeps opposite labels by shape.
* **Paths per orientation, as removed cousins had them.** Leaves a pair
  joined once each way half, at half its kinship.
* **Keep the SciPy oracle, changed in lockstep.** The issue #29 precedent, but
  it doubles every definition change and checks the engine against a copy of
  itself.
