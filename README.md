# pedigree-graph

Sparse-matrix pedigree relationship extraction and kinship computation.

Builds parent→child CSR adjacency matrices and extracts relationship
categories using sparse matrix algebra (`A @ A.T` for siblings,
`A² @ A²ᵀ` for cousins, etc.).  Each relationship type is parameterised
by `(up, down, n_ancestors)`:

- `up`: meioses from individual A up to common ancestor(s) (canonicalised `up ≤ down`)
- `down`: meioses from common ancestor(s) down to individual B
- `n_ancestors`: 1 (half/lineal) or 2 (full, i.e. mated pair)
- `kinship = n_ancestors × (1/2)^(up + down + 1)`

## Install

```bash
pip install "pedigree-graph @ git+https://github.com/rwaples/pedigree-graph.git@v0.1.0"
```

For development:

```bash
git clone https://github.com/rwaples/pedigree-graph.git
cd pedigree-graph
pip install -e ".[test]"
pytest
```

Requires Python ≥ 3.13.  Runtime deps: `numpy`, `scipy`, `numba`.
Pandas is optional and only needed if you pass DataFrames to the
constructors.

## Usage

```python
import numpy as np
from pedigree_graph import PedigreeGraph, REL_REGISTRY, PAIR_KINSHIP

# Construct from arrays (no pandas needed)
pg = PedigreeGraph.from_arrays(
    ids=np.array([0, 1, 2, 3, 4]),
    mothers=np.array([-1, -1, 0, 0, 0]),
    fathers=np.array([-1, -1, 1, 1, 1]),
)

# Or from a dict of arrays (also pandas-free)
pg = PedigreeGraph({
    "id": np.array([0, 1, 2, 3]),
    "mother": np.array([-1, -1, 0, 0]),
    "father": np.array([-1, -1, 1, 1]),
    "twin":   np.array([-1, -1, -1, -1]),
    "sex":    np.array([0, 1, 0, 1], dtype=np.int8),
    "generation": np.array([0, 0, 1, 1], dtype=np.int32),
})

# Or from a pandas DataFrame
# pg = PedigreeGraph.from_dataframe(df)
# pg = PedigreeGraph(df)   # __init__ accepts both forms

# Extract pairs by relationship type, up to a given degree
pairs = pg.extract_pairs(max_degree=2)
print(pairs["FS"])     # full sibs:  (idx1, idx2)
print(pairs["1C"])     # 1st cousins
print(PAIR_KINSHIP["FS"])  # 0.25
```

## Relationship registry

Codes follow the convention `up_down_n_anc`:

| Code   | Label                         | up | down | n_anc | Kinship | Degree |
|--------|-------------------------------|----|------|-------|---------|--------|
| `MZ`   | MZ twin                       | 0  | 0    | 0     | 0.5     | 0      |
| `MO`   | Mother–offspring              | 1  | 0    | 1     | 0.25    | 1      |
| `FO`   | Father–offspring              | 1  | 0    | 1     | 0.25    | 1      |
| `FS`   | Full sib                      | 1  | 1    | 2     | 0.25    | 1      |
| `MHS`  | Maternal half sib             | 1  | 1    | 1     | 0.125   | 2      |
| `PHS`  | Paternal half sib             | 1  | 1    | 1     | 0.125   | 2      |
| `GP`   | Grandparent                   | 2  | 0    | 1     | 0.125   | 2      |
| `Av`   | Avuncular                     | 1  | 2    | 2     | 0.125   | 2      |
| `1C`   | 1st cousin                    | 2  | 2    | 2     | 0.0625  | 3      |
| ...    | (full registry up to 2nd cousin / kinship 1/64) | | | | | |

See `REL_REGISTRY` for the complete list.

## Experimental engines

The package ships an alternate relationship-counting engine in
`pedigree_graph.experimental` for exploring large-pedigree scaling:

```python
from pedigree_graph import PedigreeGraph
from pedigree_graph.experimental import count_pairs_bfs

pg = PedigreeGraph(df)
counts = count_pairs_bfs(pg)        # dict[str, int] over 23 codes
```

`count_pairs_bfs` uses boolean sparse matmul (set-union semantics) plus
a parallel numba kernel for cousin-style codes.  It is **counts-only**;
there is no pair-array equivalent of `extract_pairs`.

The submodule is **not** re-exported at the top level — callers must
import explicitly via `pedigree_graph.experimental`.  First call emits
a `FutureWarning`.

### Caveats — read before using

1. **Experimental contract.**  API, signature, and semantics may
   change or the function may be removed in any minor release.  No
   deprecation cycle is owed.

2. **Inbred-pedigree counting differs from the matrix engine.**
   On non-inbred pedigrees the BFS counts equal `PedigreeGraph.count_pairs`
   exactly.  On inbred pedigrees, BFS counts *distinct shared
   ancestors* at depth ≥ 2 while the matrix engine counts *paths*
   (multiplicity); the four cousin-style codes
   (`1C1R`, `H1C1R`, `1C2R`, `2C`) may diverge.  See
   `tests/test_experimental.py::test_inbred_with_cousins_cousin_codes_diverge`
   for a hand-built fixture pinning the exact divergence.

3. **`max_degree=5` only.**  Lower values raise `NotImplementedError` —
   use `PedigreeGraph.count_pairs(max_degree=k)` for partial extractions.

4. **No subsample support.**  `PedigreeGraph.from_subsample(...)` graphs
   raise `NotImplementedError`.  Construct directly or use the matrix
   engine.

5. **Threading.**  The numba kernel uses `prange` for cousin-style
   enumeration.  Numba reads `NUMBA_NUM_THREADS` at first JIT
   compilation; the optional `n_threads` kwarg only takes effect on
   the first call in a process.  Set `NUMBA_NUM_THREADS=N` in the
   environment to control threading on all calls.

6. **Performance.**  Scaling claims (BFS faster than matrix above
   ~5M individuals, where the matrix engine OOMs) are unverified at
   the time of v0.2.0.  The matrix engine is faster at n=2M in the
   only head-to-head we have run.  See open issues
   [#2 (numba kernel parallelisation)](https://github.com/rwaples/pedigree-graph/issues/2)
   and [#3 (10M+ scaling)](https://github.com/rwaples/pedigree-graph/issues/3).
   Treat this engine as an experimental scalability spike, not a
   tuned alternative.

## License

MIT — see [LICENSE](LICENSE).
