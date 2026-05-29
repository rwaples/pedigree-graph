# PGQ-010 — Add architecture guardrails for large files and hidden coordinate/type contracts

**Priority:** medium

**Files:**

- repo-wide
- `pyproject.toml`
- tests/docs as needed

### Problem

The repo now has multiple production files over 1,000 lines and several hidden contracts that are only documented in comments/docstrings.

Large files:

- `pedigree_graph/_core.py` — about 1,982 lines
- `pedigree_graph/_effective_size.py` — about 1,727 lines
- `pedigree_graph/_kinship_kernel.py` — about 1,490 lines

Hidden contracts include:

- pair coordinate space;
- exact vs approximate count semantics;
- path-count vs distinct-ancestor semantics;
- dense vs sparse ID assumptions;
- default all-zero sex behavior.

### Code-quality concern

Without guardrails, future changes will likely keep adding special cases to already large modules.

### Preferred remedy

Add lightweight process and test guardrails:

- document coordinate-space and relationship-engine contracts in developer docs;
- add tests specifically for boundary contracts;
- optionally add a local script or CI check that reports production files over 1,000 lines;
- prefer new focused modules over extending the three oversized files.

### Acceptance criteria

- New contributors can find the relationship/coordinate-space contracts without reading all of `_core.py`.
- Regression tests cover the key hidden invariants.
- Large-file growth is visible during review.
