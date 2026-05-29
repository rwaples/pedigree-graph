# PGQ-002 — Replace loose constructor inputs and dense ID remapping with a validated input model

**Priority:** high

**Files:**

- `pedigree_graph/_core.py:204-288`
- `pedigree_graph/_core.py:244-270`
- `pedigree_graph/_core.py:1713-1771`

### Problem

`PedigreeGraph.__init__()` accepts a loose dict/DataFrame shape and immediately builds a dense `id_to_row` table sized as `max(id) + 1`.

Risks:

- Sparse high IDs can allocate huge arrays.
- Duplicate IDs silently overwrite earlier rows.
- Negative IDs can behave badly as indices.
- Mismatched column lengths are not validated as a single boundary contract.
- Missing parent IDs silently become `-1` while original parent IDs still drive sibling grouping.
- `from_subsample()` repeats the dense-ID remap pattern for `id_to_df_row`.

### Code-quality concern

This is a weak boundary for the central model object. Pedigree validity is spread across incidental numpy behavior instead of being explicit at construction.

### Preferred remedy

Create a typed validated input boundary, e.g. `PedigreeArrays`, responsible for:

- normalizing DataFrame/dict inputs;
- checking required columns;
- checking equal lengths;
- checking unique nonnegative IDs;
- validating parent/twin IDs;
- validating `sex`, `generation`, and optional `birth_year` shapes/dtypes;
- remapping IDs with a safe sparse strategy (`np.searchsorted` over sorted IDs or a hash map), not a dense `max(id)+1` array unless dense IDs are explicitly required.

### Acceptance criteria

- Constructor raises clear `ValueError`s for duplicate IDs, negative IDs, mismatched lengths, and sparse pathological IDs if unsupported.
- `from_subsample()` uses the same remapping/validation machinery as `__init__()`.
- Existing public constructors continue to work for normal dense-ID pedigrees.
- Tests cover duplicate full-pedigree IDs, missing parents, sparse high IDs, and invalid twins.

---
