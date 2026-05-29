# PGQ-007 — Simplify or fully support `_assemble_csc()`'s extra contract

**Priority:** medium

**Files:**

- `pedigree_graph/_kinship_kernel.py:884-976`
- `pedigree_graph/_kinship_kernel.py:1110-1149`

### Problem

`_assemble_csc()` advertises support for arbitrary `phen_pos` slicing and `to_grm`, but the only current production caller passes `phen_pos = np.arange(n)` and `to_grm=False`.

The function itself notes that non-standard `phen_pos` ordering would require a post-sort, but does not implement one.

### Code-quality concern

This is unsupported generality: extra API surface inside a hot kernel that is not actually exercised. It increases cognitive load and creates a trap for future callers.

### Preferred remedy

Choose one path:

1. **Delete unsupported generality:** make `_assemble_csc()` only assemble the full symmetric kinship CSC. Remove `phen_pos` and `to_grm` until needed.
2. **Fully support and test it:** implement sorting for arbitrary `phen_pos`, add tests for sliced/unsorted `phen_pos`, and test `to_grm=True`.

Given current usage, the first path is probably cleaner.

### Acceptance criteria

- The function contract matches what is actually supported.
- If `phen_pos` remains, unsorted/sliced cases are tested.
- If `to_grm` remains, diagonal/off-diagonal scaling is tested.

---
