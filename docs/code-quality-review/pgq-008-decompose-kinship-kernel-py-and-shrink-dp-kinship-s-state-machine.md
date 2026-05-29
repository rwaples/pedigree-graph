# PGQ-008 — Decompose `_kinship_kernel.py` and shrink `_dp_kinship()`'s state machine

**Priority:** medium / structural

**Files:**

- `pedigree_graph/_kinship_kernel.py`, currently about 1,490 lines
- `pedigree_graph/_kinship_kernel.py:499-853` (`_dp_kinship`, about 355 lines)

### Problem

The kinship kernel combines several concerns:

- depth/EqG utilities;
- slab allocator and free-list logic;
- DP kinship recursion;
- MZ twin fixups;
- theta accumulation;
- CSC assembly;
- Meuwissen-Luo F-only ancestor walk.

Some complexity is justified by numba constraints, but the current function/file size makes maintenance risky.

### Code-quality concern

The code works, but it is difficult to audit. The allocator state machine, row-retirement semantics, and kinship recursion are intertwined in one large hot function.

### Preferred remedy

Decompose at module level first, without changing kernel behavior:

- `_kinship_depth.py` or similar: depth/EqG/topology utilities.
- `_kinship_allocator.py`: `_grow_global`, free-list, row append/retire helpers.
- `_kinship_dp.py`: main DP kernel and theta streaming.
- `_kinship_csc.py`: CSC assembly.
- `_inbreeding_kernel.py`: Meuwissen-Luo F-only kernel.

Then consider whether `_dp_kinship()` itself can be reduced by extracting numba-compatible helper blocks for founder initialization, merge-walk, twin pass, and retirement.

### Acceptance criteria

- No behavior change; parity tests remain green.
- Kernel helpers remain numba-cache compatible.
- The main DP function is easier to scan and has fewer unrelated responsibilities.
- Tests for allocator helpers remain close to allocator code.

---
