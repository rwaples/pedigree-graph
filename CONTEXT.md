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

**Caller-space**:
An individual's row index within the subsample the caller supplied, which may
order or omit individuals differently from the full pedigree.
_Avoid_: subsample index, df index, external index

### Relationships

**Relationship pair**:
An unordered pair of individuals sharing a relationship category, stored
canonically as `(lo, hi)` with `lo < hi`.
_Avoid_: edge, link, tuple

**Relationship category**:
A class of relationship identified by a short code (e.g. `FS`, `MHS`, `1C`),
defined by `(up, down, n_ancestors)` — meioses up to the common ancestor(s),
meioses back down, and whether the connecting ancestor is a single individual
(half / lineal) or a mated pair (full).
_Avoid_: relationship type (when the code is meant), kind

**Degree**:
The kinship distance of a relationship category — `0` for MZ twins, `1` for
parent-offspring and full sibs, and so on.

## Relationships

- A **relationship pair** holds two individuals and belongs to one **relationship category**.
- Every individual index is expressed in either **graph-space** or **caller-space**; the same individual generally has a different index in each.
- A pair returned to a caller is in **caller-space**; the kinship matrix is indexed in **graph-space**. Converting between the two is required whenever both meet.

## Example dialogue

> **Reviewer:** "`extract_pairs` gave me pair `(1, 0)` for the MZ twins — why is its kinship `0.0`?"
> **Author:** "Those indices are **caller-space** — you reversed the subsample. The kinship matrix is **graph-space**, so indexing it with caller indices reads the wrong cell. The pair has to be mapped back to graph-space first."

## Flagged ambiguities

- "index" alone is ambiguous between **graph-space** and **caller-space** — always qualify which space, since the same individual differs between them and conflating them caused a kinship-lookup bug (PGQ-001).
