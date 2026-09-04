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

### Pedigree structure

**Structural depth**:
An individual's position in the parent DAG: individuals with no represented
parent have depth 0; every other individual has one plus the greatest depth of
its represented parents. Structural depth is derived from parent relationships
and is independent of input row order.
_Avoid_: generation label, cohort

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

## Relationships

- A **relationship pair** holds two individuals and belongs to one **relationship category**; asymmetric categories define the roles of its two positions, while canonical key ordering remains only a storage/encoding choice.
- Every public row index is expressed in either **graph-space** or **view-space**; the same individual generally has a different index in each.
- A graph query returns graph-space rows, while a view query returns view-space rows. Coordinate space follows the query receiver.

## Example dialogue

> **Reviewer:** "This relationship pair came from a pedigree view. Can I use its rows against the full graph's matrix?"
> **Author:** "No. Those are **view-space** rows, while the full matrix is indexed in **graph-space**. Query pairwise kinship through the same view, which owns the coordinate conversion."

## Flagged ambiguities

- "index" alone is ambiguous between **graph-space** and **view-space** — always qualify which space, since the same individual differs between them and conflating them caused a kinship-lookup bug (PGQ-001).
