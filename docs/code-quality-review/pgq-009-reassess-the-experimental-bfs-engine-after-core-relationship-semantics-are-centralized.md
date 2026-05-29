# PGQ-009 — Reassess the experimental BFS engine after core relationship semantics are centralized

**Priority:** medium

**Files:**

- `pedigree_graph/experimental.py:116-575`
- `pedigree_graph/_bfs_kernel.py`
- `tests/test_experimental.py`

### Problem

`count_pairs_bfs()` is a 460-line experimental function that duplicates much of the relationship extraction model. It is intentionally experimental, but it still imports private `PedigreeGraph` helpers and carries its own subtract logic, threading choices, cache behavior, and semantic caveats.

### Code-quality concern

Even experimental code can become permanent debt. The BFS engine currently increases the number of places relationship semantics can drift.

### Preferred remedy

After PGQ-003/PGQ-004, move BFS into a dedicated experimental engine module that consumes shared relationship metadata where practical. Keep its experimental caveats, but isolate it from `PedigreeGraph` internals as much as possible.

### Acceptance criteria

- BFS remains explicitly experimental and not top-level re-exported.
- BFS-specific approximation/difference semantics are encoded and tested.
- Private helper coupling to `PedigreeGraph` is minimized.
- Relationship code set remains synchronized with `REL_REGISTRY` / relationship plan.

---
