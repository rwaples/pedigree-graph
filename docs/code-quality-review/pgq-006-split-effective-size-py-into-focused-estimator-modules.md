# PGQ-006 — Split `_effective_size.py` into focused estimator modules

**Priority:** medium-high / decomposition

**Files:**

- `pedigree_graph/_effective_size.py`, currently about 1,727 lines

### Problem

`_effective_size.py` contains public result classes, helper utilities, Hill overlapping-generation logic, founder contribution logic, Caballero-Toro numba plumbing, and orchestration.

### Code-quality concern

The file has crossed a healthy size boundary and mixes several domains. New estimator work will likely keep enlarging it unless there is a clearer module map.

### Preferred remedy

Split by conceptual ownership, for example:

- `_ne_results.py` — result dataclasses and serialization helpers.
- `_ne_family_size.py` — family-size table, variance-family-size, sex-ratio helpers.
- `_ne_founders.py` — founder contribution and LTC helpers.
- `_ne_caballero_toro.py` — CT accumulators and estimator.
- `_ne_hill.py` — Hill overlapping-generation estimator.
- `_effective_size.py` — compatibility facade/re-exports and `compute_all_ne()` orchestration.

### Acceptance criteria

- Public imports from `pedigree_graph` remain stable.
- `_effective_size.py` becomes a thin facade or at least substantially smaller.
- Estimator-specific helpers live near the estimator that owns them.
- Tests continue to import public APIs; private helper tests can be moved to the new modules.

---
