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

## License

MIT — see [LICENSE](LICENSE).
