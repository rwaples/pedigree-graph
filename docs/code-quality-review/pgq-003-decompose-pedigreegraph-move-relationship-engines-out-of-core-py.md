# PGQ-003 — Decompose `PedigreeGraph`; move relationship engines out of `_core.py`

**Priority:** high / structural

**Files:**

- `pedigree_graph/_core.py` overall, currently about 1,982 lines
- `pedigree_graph/_core.py:945-1294` (`extract_pairs`, about 350 lines)
- `pedigree_graph/_core.py:1326-1586` (`count_pairs_streaming`, about 261 lines)

### Problem

`PedigreeGraph` currently owns all of these concerns:

- input validation and ID remapping;
- sparse parent matrices;
- relationship registry metadata;
- exact matrix pair extraction;
- scalar streaming pair counts;
- subsample masking/remapping;
- kinship matrix dispatch;
- inbreeding and lineage helpers.

This makes `_core.py` oversized and makes small changes likely to add more ad-hoc branching to a central class.

### Code-quality concern

This is an abstraction boundary smell. `PedigreeGraph` should represent the pedigree graph and provide stable access to graph data/caches. Extraction/counting engines should be separate collaborators.

### Preferred remedy

Split relationship engines out of `PedigreeGraph`:

- `PedigreeGraph`: validated graph data, parent matrices, shared caches.
- `MatrixPairExtractor`: exact pair arrays.
- `StreamingPairCounter`: scalar count formulas.
- `BfsPairCounter` or `experimental/_bfs_counter.py`: experimental count engine.

Public methods can remain as thin compatibility wrappers:

```python
def extract_pairs(...):
    return MatrixPairExtractor(self).extract(...)
```

### Acceptance criteria

- `_core.py` shrinks materially and stops carrying engine-specific implementation detail.
- Public API remains compatible.
- Engine modules have focused tests.
- Pair coordinate conversion from PGQ-001 is centralized, not repeated in each engine.

---
