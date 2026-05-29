# PGQ-004 — Make relationship extraction semantics a single explicit plan instead of duplicated degree branches

**Priority:** high / structural

**Files:**

- `pedigree_graph/_core.py:945-1294`
- `pedigree_graph/_core.py:1326-1586`
- `pedigree_graph/experimental.py:116-575`
- `pedigree_graph/_core.py:110-175` (`REL_REGISTRY`)

### Problem

Relationship logic is duplicated across matrix extraction, streaming counts, and experimental BFS counting. Each engine carries its own degree gates, subtract lists, approximations, and special cases.

`REL_REGISTRY` is the source of truth for labels and kinship, but it does not encode extraction dependencies or subtract semantics.

### Code-quality concern

Duplicated relationship semantics are a drift hazard. Fixes to relationship definitions need to be applied in several engines by hand.

### Preferred remedy

Introduce a relationship plan layer that at least documents and centralizes:

- relationship code;
- degree;
- exact/approximate availability per engine;
- dependencies/subtract sets;
- whether multiplicity or distinct-ancestor semantics apply;
- required sparse powers or sibling matrices.

This does not need to fully generate all engine code on day one. The first win is to remove hidden local dependency rules and make engine differences explicit.

### Acceptance criteria

- Matrix, streaming, and BFS engines refer to shared relationship metadata where practical.
- Approximate streaming codes are explicitly marked as approximate in code, not only in a docstring.
- Tests assert that all engines return exactly the registry key set.
- Adding a new relationship type requires touching one obvious registry/plan area plus engine implementations, not hunting for branch lists.

---
