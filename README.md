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

Binary wheels cover manylinux x86-64 and AArch64, macOS x86-64 and Apple
Silicon, and Windows x86-64 (one `abi3` wheel per platform for CPython >= 3.13).
Elsewhere pip builds the sdist, which needs a Rust toolchain (`rustc >= 1.85`).

For development the locked pixi environment carries Python, Rust, and maturin
and installs the package editable (the extension module is rebuilt with
`pixi run maturin develop --release` after a Rust change):

```bash
git clone https://github.com/rwaples/pedigree-graph.git
cd pedigree-graph
pixi install --locked
pixi run pytest -m "not slow"   # inner loop, about 2.5 minutes
pixi run pytest                 # before a commit, about 13 minutes
pixi run cargo test --release   # the Rust core
```

Five of the nine `slow` tests are the `random_30k` integration gates.  Each runs
the pair-kinship kernel over the whole 30,300-row pedigree, and a single kernel
call there costs 30 to 250 seconds regardless of how many pairs it is asked for,
so the full suite is dominated by a handful of tests.  The other four are the
N=2000 effective-size scaling tests in `tests/test_effective_size_scaling.py`,
which touch neither `random_30k` nor the pair-kinship kernel.

Requires Python ≥ 3.13.  Runtime deps: `numpy`, `scipy`, and `numba` (still
used by the inbreeding walk, the equivalent-generation kernel and the lineage
kernels); construction, relationships, topology, pairwise kinship, the
kinship matrices and the generation kinship summary are compiled Rust
(`pedigree_graph._native`, sources under `crates/`).
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
# read-only int32 first_rows / second_rows in the category's role orientation.
# execution="memory" returns the same blocks at the lowest peak memory
# (the result plus engine state) for about twice the wall time.
pairs = pg.relationship_pairs(max_degree=3)
first, second = pairs["FS"]  # full sibs
print(len(pairs["1C"]))  # 1st cousins (degree 3)
print(RELATIONSHIPS["FS"].nominal_kinship)  # 0.25

# Pedigree-expected kinship for those pairs, or for any two row arrays
kin = pg.pair_kinship(pairs)  # {code: float32 array}
kin_fs = pg.pair_kinship(first, second)

# Exact counts in O(N) memory (no pair lists), and the three kinship-matrix families
counts = pg.relationship_counts(max_degree=3)
K = pg.kinship_matrix()  # complete, CSC float32

# Scalar counts for MZ, MO, FO, FS, MHS and PHS only, also exact
close = pg.close_relative_counts()  # no degree/category selector
assert close["FS"] == counts["FS"]
assert close["GP"] is None  # not computed, not zero
```

Absent optional columns read as absent: there is no sex default and no
generation fallback.  Effective-size estimators live in
`pedigree_graph.effective_size`; the `FrameLike` protocol in
`pedigree_graph.typing`.  Migrating from 0.7.1: see the old-to-new table in
`CHANGELOG.md`.

`close_relative_counts()` is full-graph only. Its `requested` and `exact`
sets contain the six codes above; all other registry keys map to `None`.
It counts parent edges and sibling groups without pair lists or adjacency
powers. For grandparents, avuncular pairs, cousins, or a pedigree view, use
`relationship_counts()` instead. The former `estimate_relationship_counts`
method and the result fields `approximate` and `clamped` have been removed.

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

## Architecture

For contributors: [`docs/architecture.md`](docs/architecture.md) maps the
module layout and the hidden contracts (coordinate space, exact count
coverage, sparse-ID handling, default sex), each with its source
of truth and regression test.
The relationship/coordinate vocabulary is in [`CONTEXT.md`](CONTEXT.md);
design decisions are in [`docs/adr/`](docs/adr/).

## License

MIT — see [LICENSE](LICENSE).
