# pedigree-graph

Sparse-matrix pedigree relationship extraction and kinship computation. A
pedigree is a parent→child DAG of individuals; this context is the vocabulary
for the relationships between individuals and the two coordinate systems used
to name them.

## Language

### Coordinate spaces

**Graph-space**:
An individual's row index within the full pedigree the graph was built over.
_Avoid_: full index, absolute index, internal index

**View-space**:
An individual's row index within an explicitly ordered pedigree view. A view
may reorder or omit individuals from its full pedigree, while relationships
are still resolved through the full pedigree.
_Avoid_: caller-space, subsample index, df index, external index

**Coordinate token**:
The opaque identity a receiver carries for its own coordinate space, which its
results will carry too so they cannot be used against another receiver. It is
instance identity, not value identity: equivalent selections made by separate
`view` calls are distinct receivers with distinct tokens.
_Avoid_: view id, space id, handle

### Pedigree structure

**Structural depth**:
An individual's position in the parent DAG: individuals with no represented
parent have depth 0; every other individual has one plus the greatest depth of
its represented parents. Structural depth is derived from parent relationships
and is independent of input row order.
_Avoid_: generation label, cohort

**Represented founder**:
An individual with no represented mother or father, even when known biological parents have IDs external to the represented rows.
_Avoid_: founder (when biological founder status could be inferred), generation-0 individual

**Represented founder genome**:
A genome node with no represented parental genome node, so parentless MZ co-twins are two represented founders sharing one ancestry source.
_Avoid_: founder row, treating MZ co-twins as independent genetic sources

**Closed represented parentage**:
A pedigree property in which every represented individual has either zero or two represented parents, regardless of whether an unrepresented parent is missing or known by an external ID.
_Avoid_: complete pedigree (biological ancestors may still be outside the graph), complete metadata

**Generation label**:
Optional cohort metadata supplied for an individual. When generation metadata
is absent for the entire pedigree, structural depth supplies the generation
grouping. When only some individuals lack labels, they remain unlabelled and
are excluded explicitly from label-grouped summaries rather than silently
assigned their structural depth. A generation label may be rebased, sparse, or
different from structural depth and never changes pedigree relationships or
kinship.
_Avoid_: structural depth

**Genome node**:
The unit of genetic identity in a pedigree. Every individual belongs to
exactly one genome node, and MZ co-twins share one. Relationships that depend
on identity by descent, such as inbreeding, are properties of the
genome-node pedigree: the pedigree obtained by merging each MZ pair into one
node with the pair's parents.
_Avoid_: collapsed individual, MZ-collapsed row, twin class

### Relationships

**Relationship pair**:
A pair of individuals sharing a relationship category. For an asymmetric
category, the positions have fixed semantic roles (for example
offspring→mother or descendant→ancestor). For a symmetric category, the two
positions have no distinct biological roles. Canonical `(lo, hi)` pair-key
encoding is an internal storage operation and must not erase semantic roles.
_Avoid_: edge, link, tuple; treating internal key order as relationship-role order

**Relationship category**:
A class of relationship identified by a short code (e.g. `FS`, `MHS`, `1C`),
defined by `(up, down, ancestor_count)` — meioses up to the common ancestor(s),
meioses back down, and whether the connecting ancestor is a single individual
(half / lineal) or a mated pair (full). `first` is the pair member at least as
far from the ancestor(s), so `up` counts meioses from `first` up, `down` counts
them from the ancestor(s) down to `second`, and `up >= down` always holds.
_Avoid_: relationship type (when the code is meant), kind, n_ancestors

**Degree**:
The kinship distance of a relationship category — `0` for MZ twins, `1` for
parent-offspring and full sibs, and so on. A degree cutoff includes relationship
categories whose degree is less than or equal to the cutoff.

**Closest category**:
The one relationship category a pair is reported under when it satisfies
several: the lowest degree, then the earliest in registry order. Category
definitions decide membership; closest-category precedence decides reporting.
_Avoid_: fold (as a noun for the rule), exclusivity, dedup, "the exclusions"

**Nominal kinship**:
The kinship coefficient implied by a relationship category's `(up, down,
n_ancestors)` formula, assuming a single relationship path and no inbreeding
or co-coalescence.
_Avoid_: exact kinship, pedigree-expected kinship

**Pedigree-expected kinship**:
The kinship coefficient the package returns for a particular pair of
individuals: the value of the pinned float32 recurrence (ADR 0009) over all
pedigree paths, including inbreeding, MZ co-coalescence (shared genome nodes),
and duplicate relationship paths such as double cousins. Within one graph the
pair and matrix values are bit-identical.
_Avoid_: nominal kinship, exact kinship, pedigree-specific kinship

**Exact rational kinship**:
The dyadic rational a pair's kinship would be with unbounded precision. A
reference-oracle and analysis term; not a public API value.
_Avoid_: using it for what `pair_kinship` returns

### Effective size

**Reference subpopulation**:
The set of individuals whose values are averaged to produce one effective-size
estimate, chosen by the caller rather than implied by the pedigree.
_Avoid_: reference population, target cohort, sample

**Equivalent complete generations**:
The pedigree depth of one individual: the sum over its known ancestors of
`(1/2)^n` for meiotic distance `n`.
_Avoid_: generation number, pedigree completeness, structural depth

**Individual increase in inbreeding**:
The per-generation rate of inbreeding implied by a single individual's own
inbreeding coefficient and its equivalent complete generations.
_Avoid_: delta F, rate of inbreeding

**Self-coancestry**:
An individual's coancestry with itself, one half of one plus its inbreeding
coefficient.
_Avoid_: diagonal kinship, self-kinship

**Group coancestry**:
The average coancestry of a cohort taken over every ordered pair of its genome
nodes, self-coancestries included.
_Avoid_: mean kinship, average coancestry, pairwise coancestry

**Founder contribution**:
The expected fraction of one individual's genome inherited from one represented
founder genome.
_Avoid_: founder representation, ancestry proportion, long-term contribution

**Effective number of founders**:
The number of equally contributing founders that would produce the same founder
diversity as the observed founder contributions.
_Avoid_: founder equivalents, founder genome equivalents

## Relationships

- Every **represented founder** belongs to one **represented founder genome**; two MZ represented founders share the same one.
- **Closed represented parentage** permits represented founders but no individual with exactly one represented parent.
- A **relationship pair** holds two individuals and belongs to one **relationship category**; asymmetric categories define the roles of its two positions, while canonical key ordering remains only a storage/encoding choice.
- When a pair satisfies several **relationship categories**, it belongs to its **closest category**; exact counts and pair lists agree on that assignment.
- Every public row index is expressed in either **graph-space** or **view-space**; the same individual generally has a different index in each.
- A graph query returns graph-space rows, while a view query returns view-space rows. Coordinate space follows the query receiver.
- Structural results derive from **structural depth** alone — kinship, inbreeding, relationship pairs and counts, ancestor and descendant counts — and a supplied **generation label** never enters them. Cohort-indexed results group by **generation label**, falling back to structural depth only when the whole pedigree is unlabelled.
- Every effective-size estimator is reachable directly and through the batch orchestrator. The two paths refuse a graph for the same reason, in the same order, and warn alike; neither is the more permissive way in.
- An effective-size estimate is computed over one **reference subpopulation**; a cohort is one way of choosing one, not the only one.
- **Group coancestry** differs from a mean pairwise kinship by including **self-coancestry** on the diagonal. Both are taken over genome nodes rather than rows, so the diagonal is the whole of the difference.
- A **founder contribution** vector sums to one over the **represented founder genomes**; the **effective number of founders** summarises how evenly it is spread.
- The **effective number of founders** is half the asymptotic effective size only under random mating — a condition a pedigree alone cannot establish.

## Example dialogue

> **Reviewer:** "This relationship pair came from a pedigree view. Can I use its rows against the full graph's matrix?"
> **Author:** "No. Those are **view-space** rows, while the full matrix is indexed in **graph-space**. Query pairwise kinship through the same view, which owns the coordinate conversion."

> **Reviewer:** "The coancestry went up between these two cohorts — can I read an effective size straight off that?"
> **Author:** "Only if you say which coancestry. **Group coancestry** includes each individual's **self-coancestry**, so it moves with inbreeding; the off-diagonal mean does not. And an estimate needs a **reference subpopulation** — say which individuals you averaged over, or the number means nothing."

## Flagged ambiguities

- "index" alone is ambiguous between **graph-space** and **view-space** — always qualify which space, since the same individual differs between them and conflating them caused a kinship-lookup bug (PGQ-001).
- "coancestry" alone is ambiguous between **group coancestry** (diagonal included, over genome nodes) and the off-diagonal mean pairwise kinship — always qualify which, since conflating them let an estimator carry an attribution its source did not support (#15, ADR 0012).
- "contribution" was used for both a **founder contribution** and the long-term contribution of an arbitrary ancestor — resolved: the package's contribution columns are always **represented founder genomes**.
- "Ne" unqualified is forbidden — every reference names its estimator, because the operational definitions are not interchangeable.
