"""Compare the R package with the Python package beyond the goldens.

The goldens (``tools/r_golden.py``) ship in the R tarball and so cover only
small fixtures.  This gate runs the larger core fixtures and the simACE study
pedigrees through both hosts and compares every product byte for byte:
degree-``d`` relationship pairs (registry index, 1-based rows), inbreeding,
pairwise kinship of the pairs up to degree 3, and the stored upper triangle
of the kinship matrix, in ``(column, row)`` order.  R writes raw
little-endian arrays (``tools/r_parity.R``); this script writes the same
arrays from Python, compares them and prints both SHA-256s.

Needs the ``r`` environment with the package installed
(``pixi run -e r r-install``) and the simACE study results.  Launch from
the simACE umbrella root, as the other gate tools::

    pixi run --manifest-path external/pedigree-graph/pixi.toml python external/pedigree-graph/tools/r_parity.py [cell ...]
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import polars as pl
import scipy.sparse as sp

from pedigree_graph import PedigreeGraph
from pedigree_graph.relationships import RELATIONSHIPS

REPO = Path(__file__).resolve().parents[1]
UMBRELLA = REPO.parents[1]
FIXTURES = REPO / "crates" / "core" / "tests" / "fixtures"

# name -> (source, max_degree, kinship matrix, pairwise kinship).  A source is
# a core fixture name or a parquet path under the umbrella.  The matrix cells
# are the ones slice 14's gate ran; baseline100K and the 50k pedigree have
# no complete matrix there and no pairwise walk here (minutes and GiB each).
CELLS: dict[str, tuple[str, int, bool, bool]] = {
    **{
        name: (name, 5, True, True)
        for name in (
            "deep_inbred_60g",
            "inbred1",
            "inbred2",
            "inbred3",
            "inbred4",
            "random_1k",
            "small_pedigree",
            "sim_d2_n3000",
            "sim_d3_n3000",
            "sim_d6_n3000",
            "random_30k",
        )
    },
    "dev_mean_n10k": ("results/dev/dev_mean_n10k/rep1/pedigree.parquet", 5, True, True),
    "dev_cont_n10k": ("results/dev/dev_cont_n10k/rep1/pedigree.parquet", 3, True, True),
    "baseline10K": ("results/base/baseline10K/rep1/pedigree.parquet", 3, True, True),
    "dev_laplace_am_strong_50k": ("results/dev/dev_laplace_am_strong_50k/rep1/pedigree.parquet", 3, False, False),
    "baseline100K": ("results/base/baseline100K/rep1/pedigree.parquet", 3, False, False),
}


def _fixture_frame(name: str) -> pl.DataFrame:
    spec = importlib.util.spec_from_file_location("r_golden", REPO / "tools" / "r_golden.py")
    assert spec is not None
    assert spec.loader is not None
    golden = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(golden)
    return golden._frame(pl.read_csv(FIXTURES / f"{name}.tsv", separator="\t"))


def _frame(source: str) -> pl.DataFrame:
    if source.endswith(".parquet"):
        return pl.read_parquet(UMBRELLA / source, columns=["id", "mother", "father", "twin"])
    return _fixture_frame(source)


def _python_products(frame: pl.DataFrame, max_degree: int, matrix: bool, pairwise: bool) -> dict[str, bytes]:
    graph = PedigreeGraph.from_frame({c: frame[c].to_numpy() for c in frame.columns})
    codes = list(RELATIONSHIPS)
    pairs = graph.relationship_pairs(max_degree=max_degree)
    code = np.concatenate([np.full(len(b), codes.index(b.code) + 1, np.int32) for b in pairs.values()])
    first = np.concatenate([b.first_rows for b in pairs.values()]).astype(np.int32) + 1
    second = np.concatenate([b.second_rows for b in pairs.values()]).astype(np.int32) + 1
    out = {
        "pairs_code.bin": code.astype("<i4").tobytes(),
        "pairs_first.bin": first.astype("<i4").tobytes(),
        "pairs_second.bin": second.astype("<i4").tobytes(),
        "inbreeding.bin": np.asarray(graph.inbreeding(), dtype="<f8").tobytes(),
    }
    if pairwise:
        degree = np.array([RELATIONSHIPS[c].degree for c in codes])
        close = degree[code - 1] <= 3
        values = graph.pair_kinship(first[close] - 1, second[close] - 1)
        out["pair_kinship.bin"] = np.asarray(values, dtype="<f8").tobytes()
    if matrix:
        upper = sp.triu(graph.kinship_matrix(), format="coo")
        order = np.lexsort((upper.row, upper.col))
        out["kinship_i.bin"] = (upper.row[order] + 1).astype("<i4").tobytes()
        out["kinship_j.bin"] = (upper.col[order] + 1).astype("<i4").tobytes()
        out["kinship_x.bin"] = upper.data[order].astype("<f8").tobytes()
    return out


def _run(name: str, tmp: Path) -> list[tuple[str, str, int, str, str, bool]]:
    source, max_degree, matrix, pairwise = CELLS[name]
    frame = _frame(source)
    work = tmp / name
    work.mkdir()
    pedigree = work / "pedigree.tsv"
    frame.write_csv(pedigree, separator="\t")
    start = time.perf_counter()
    want = _python_products(frame, max_degree, matrix, pairwise)
    py_seconds = time.perf_counter() - start
    subprocess.run(
        [
            "pixi",
            "run",
            "--manifest-path",
            str(REPO / "pixi.toml"),
            "-e",
            "r",
            "Rscript",
            str(REPO / "tools" / "r_parity.R"),
            str(pedigree),
            str(work),
            str(max_degree),
            str(int(matrix)),
            str(int(pairwise)),
        ],
        check=True,
    )
    print(f"# {name}: {frame.height} rows, Python {py_seconds:.1f} s; R seconds:", flush=True)
    print((work / "r_seconds.tsv").read_text().strip(), flush=True)
    rows = []
    for product, py_bytes in want.items():
        r_bytes = (work / product).read_bytes()
        rows.append(
            (
                name,
                product.removesuffix(".bin"),
                len(py_bytes),
                hashlib.sha256(py_bytes).hexdigest()[:16],
                hashlib.sha256(r_bytes).hexdigest()[:16],
                py_bytes == r_bytes,
            )
        )
    return rows


def main() -> int:
    """Run the named cells (default all); exit 1 if any product differs."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("cells", nargs="*", help=f"cells to run (default all): {', '.join(CELLS)}")
    args = parser.parse_args()
    names = args.cells or list(CELLS)
    unknown = sorted(set(names) - set(CELLS))
    if unknown:
        parser.error(f"unknown cells: {unknown}")
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        for name in names:
            results += _run(name, Path(tmp))
    print("cell\tproduct\tbytes\tpython_sha256\tr_sha256\tidentical")
    for row in results:
        print("\t".join(str(v) for v in row))
    mismatched = [r for r in results if not r[-1]]
    print(f"{len(results) - len(mismatched)}/{len(results)} products identical")
    return 1 if mismatched else 0


if __name__ == "__main__":
    sys.exit(main())
