# pedigree-graph

Sparse-matrix pedigree relationship extraction and kinship computation.

Builds parent→child CSR adjacency matrices and extracts relationship
categories using sparse matrix algebra (`A @ A.T` for siblings,
`A² @ A²ᵀ` for cousins, etc.).  Each relationship type is parameterised
by `(up, down, n_ancestors)`:

- `up`: meioses from the first member (the junior role: offspring, descendant, niece_nephew, junior_cousin) up to the common ancestor(s); collateral categories are stored `up ≥ down`
- `down`: meioses from common ancestor(s) down to individual B
- `n_ancestors`: 1 (half/lineal) or 2 (full, i.e. mated pair)
- `kinship = n_ancestors × (1/2)^(up + down + 1)`

## Install

```bash
pip install pedigree-graph
```

For development:

```bash
git clone https://github.com/rwaples/pedigree-graph.git
cd pedigree-graph
pip install -e ".[test]"
pytest -m "not slow"   # inner loop, about 2.5 minutes
pytest                 # before a commit, about 13 minutes
```

Five of the nine `slow` tests are the `random_30k` integration gates.  Each runs
the pair-kinship kernel over the whole 30,300-row pedigree, and a single kernel
call there costs 30 to 250 seconds regardless of how many pairs it is asked for,
so the full suite is dominated by a handful of tests.  The other four are the
N=2000 effective-size scaling tests in `tests/test_effective_size_scaling.py`,
which touch neither `random_30k` nor the pair-kinship kernel.

Requires Python ≥ 3.13.  Runtime deps: `numpy`, `scipy`, `numba`.
Pandas is optional and only needed if you pass DataFrames to the
constructors.

## Usage

```python
import numpy as np
from pedigree_graph import RELATIONSHIPS, PedigreeGraph

# Construct from arrays (no pandas needed)
pg = PedigreeGraph.from_arrays(
    ids=np.array([0, 1, 2, 3, 4]),
    mother_ids=np.array([-1, -1, 0, 0, 0]),
    father_ids=np.array([-1, -1, 1, 1, 1]),
)

# Or from a table: a dict of columns (pandas-free), or any pandas/polars frame
pg = PedigreeGraph.from_frame(
    {
        "id": np.array([0, 1, 2, 3]),
        "mother": np.array([-1, -1, 0, 0]),
        "father": np.array([-1, -1, 1, 1]),
        "twin": np.array([-1, -1, -1, -1]),
        "sex": np.array([0, 1, 0, 1], dtype=np.int8),
        "generation": np.array([0, 0, 1, 1], dtype=np.int32),
    }
)
# pg = PedigreeGraph.from_frame(df)

# Pairs by relationship category, up to a given degree; each block holds
# read-only int32 first_rows / second_rows in the category's role orientation
pairs = pg.relationship_pairs(max_degree=3)
first, second = pairs["FS"]  # full sibs
print(len(pairs["1C"]))  # 1st cousins (degree 3)
print(RELATIONSHIPS["FS"].nominal_kinship)  # 0.25

# Pedigree-expected kinship for those pairs, or for any two row arrays
kin = pg.pair_kinship(pairs)  # {code: float32 array}
kin_fs = pg.pair_kinship(first, second)

# Counts without materialising pairs, and the three kinship-matrix families
counts = pg.relationship_counts(max_degree=3)
K = pg.kinship_matrix()  # complete, CSC float32
```

Absent optional columns read as absent: there is no sex default and no
generation fallback.  Effective-size estimators live in
`pedigree_graph.effective_size`; the `FrameLike` protocol in
`pedigree_graph.typing`.  Migrating from 0.7.1: see the old-to-new table in
`CHANGELOG.md`.

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
| `Av`   | Avuncular                     | 2  | 1    | 2     | 0.125   | 2      |
| `1C`   | 1st cousin                    | 2  | 2    | 2     | 0.0625  | 3      |
| ...    | (full registry up to 2nd cousin / kinship 1/64) | | | | | |

See `RELATIONSHIPS` for the complete list; each `RelationshipCategory`
carries `code`, `label`, `degree`, `nominal_kinship`, `up`, `down`,
`ancestor_count`, and the two positional roles.

## Experimental engines

The package ships an alternate relationship-counting engine in
`pedigree_graph.experimental` for exploring large-pedigree scaling:

```python
from pedigree_graph import PedigreeGraph
from pedigree_graph.experimental import count_pairs_bfs

pg = PedigreeGraph.from_frame(df)
counts = count_pairs_bfs(pg)  # dict[str, int] over 23 codes
```

`count_pairs_bfs` uses boolean sparse matmul (set-union semantics) plus
a parallel numba kernel for cousin-style codes.  It is **counts-only**;
there is no pair-array equivalent of `relationship_pairs`.

The submodule is **not** re-exported at the top level — callers must
import explicitly via `pedigree_graph.experimental`.  First call emits
a `FutureWarning`.

### Caveats — read before using

1. **Experimental contract.**  API, signature, and semantics may
   change or the function may be removed in any minor release.  No
   deprecation cycle is owed.

2. **Counts are unfolded, and inbred-pedigree counting differs from the
   matrix engine.**  BFS counts a pair under every category it satisfies,
   where `PedigreeGraph.relationship_counts` keeps only the closest.  On
   non-inbred pedigrees the BFS counts equal the matrix engine's unfolded
   blocks exactly.  On inbred pedigrees, BFS counts *distinct shared
   ancestors* at depth ≥ 2 while the matrix engine counts *paths*
   (multiplicity); the four cousin-style codes
   (`1C1R`, `H1C1R`, `1C2R`, `2C`) may diverge.  See
   `tests/test_experimental.py::test_inbred_with_cousins_cousin_codes_diverge`
   for a hand-built fixture pinning the exact divergence.

3. **`max_degree=5` only.**  Lower values raise `NotImplementedError` —
   use `PedigreeGraph.relationship_counts(max_degree=k)` for partial
   extractions.

4. **No view support.**  Pass a full graph; `PedigreeView` has no BFS
   counterpart.

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

## Architecture

For contributors: [`docs/architecture.md`](docs/architecture.md) maps the
module layout and the hidden contracts (coordinate space, exact vs
approximate counts, path-count vs distinct-ancestor semantics, sparse-ID
handling, default sex), each with its source of truth and regression test.
The relationship/coordinate vocabulary is in [`CONTEXT.md`](CONTEXT.md);
design decisions are in [`docs/adr/`](docs/adr/).

## License

MIT — see [LICENSE](LICENSE).
