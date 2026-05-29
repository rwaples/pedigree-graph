# PGQ-001 — Fix pair coordinate-space mismatch after `from_subsample`

**Priority:** blocker / correctness

**Files:**

- `pedigree_graph/_core.py:1273-1280`
- `pedigree_graph/_core.py:1949-1982`
- tests in `tests/test_pedigree_graph.py`

### Problem

`extract_pairs()` returns pairs in caller/subsample row coordinates after applying `_subsample_remap`, but `compute_pair_kinship()` assumes incoming pairs are full graph-row coordinates and indexes the full graph kinship matrix directly.

This breaks for subsamples whose row order differs from full-pedigree row order. It can also destroy canonical `lo < hi` ordering after remapping.

### Reproduction observed during review

Using a full pedigree where rows `4` and `5` are inbred MZ twins, and a reversed subsample `[id 5, id 4]`:

- `extract_pairs(max_degree=1)` returns `MZ: ([1], [0])` in subsample coordinates.
- `compute_pair_kinship()` indexes `K[1, 0]` and returns `0.0`.
- The correct full-graph value is `K[4, 5] == 0.625`.

### Code-quality concern

The pair coordinate space is implicit. The same tuple shape `(idx1, idx2)` sometimes means graph rows and sometimes caller rows. That hidden state makes the API brittle and forces every consumer to guess what coordinate system it has.

### Preferred remedy

Introduce an explicit pair coordinate boundary. Options:

1. **Minimal safe fix:** keep a private graph-space copy or add an inverse remap in `compute_pair_kinship()` when `_subsample_remap` is set.
2. **Cleaner structural fix:** introduce a small `PairSet` / `PairTable` dataclass with fields like:
   - `idx1`
   - `idx2`
   - `space: Literal["graph", "caller"]`
   - conversion methods `to_graph()` and `to_caller()`

The cleaner version is preferred because it prevents this class of bug from recurring.

### Acceptance criteria

- `compute_pair_kinship()` returns correct values for pairs produced by `extract_pairs()` on reordered `from_subsample()` graphs.
- Remapped pairs are re-canonicalized or documented if pair ordering is intentionally not canonical after remap.
- Tests cover reversed/reordered subsamples, not only ID-filtered subsamples preserving full order.
- Tests include an inbred/MZ pair so the slow exact-kinship path is exercised.

---
