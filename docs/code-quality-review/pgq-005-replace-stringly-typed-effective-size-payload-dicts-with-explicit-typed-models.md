# PGQ-005 — Replace stringly typed effective-size payload dicts with explicit typed models

**Priority:** medium-high

**Files:**

- `pedigree_graph/_effective_size.py:107-197` (`_sex_specific_family_table`)
- `pedigree_graph/_effective_size.py:428-461` (`_sigma2_from_quadrants`)
- `pedigree_graph/_effective_size.py:958-1027` (`_caballero_toro_accumulators`)
- `pedigree_graph/_effective_size.py:1559-1623` (`ne_caballero_toro`)
- `pedigree_graph/_effective_size.py:1631-1720` (`compute_all_ne`)

### Problem

Several internal boundaries pass untyped dictionaries with string keys:

- `dict[int, dict[str, np.ndarray]]` for family-size tables;
- `dict[str, Any]` for Caballero-Toro accumulators;
- `dict[str, Any]` for `compute_all_ne()` results.

### Code-quality concern

These stringly typed contracts hide invariants and make refactors risky. They also force consumers to know magic key names and array shapes.

### Preferred remedy

Introduce typed internal models:

- `FamilySizeEntry`
- `FamilySizeTable`
- `CTAccumulators`
- possibly `NeResults` / `ComputeAllNeResult`

Use frozen dataclasses or `NamedTuple`/`TypedDict` where numba boundaries make dataclasses inconvenient. Keep public serialization via `to_dict()`.

### Acceptance criteria

- No estimator reaches into a `dict[str, Any]` for required fields.
- Array shape/dtype expectations are documented on the typed model.
- Existing result `to_dict()` behavior remains compatible.
- Tests fail clearly if a required accumulator/table field is missing.

---
