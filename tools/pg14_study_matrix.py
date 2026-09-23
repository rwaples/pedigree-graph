"""Freeze and replay the three kinship-matrix products on the simACE study pedigrees (slice 14, gate 14a).

The pedigrees live under the simACE umbrella's ``results/`` (plus the
``random_30k`` parity fixture), so this is a one-off gate artifact rather than
a test.  Each product runs on its own freshly built graph in a child process,
so ``mean_kinship_by_generation`` never takes the cached-matrix route and a
product that outgrows the timeout is recorded rather than blocking the rest.
Two roles:

``capture`` runs under the simACE pixi env, which is locked to the 0.9.1 PyPI
wheel until the relock, and writes ``capture.json``: per product the nnz and
the SHA-256 of ``indptr``, ``indices`` and the uint32 view of ``data`` for
``kinship_matrix()`` and ``approximate_kinship_matrix(0.001)``, and the bytes
of ``mean_kinship_by_generation()``::

    cd external/pedigree-graph
    pixi run --manifest-path ../../pixi.toml python tools/pg14_study_matrix.py capture

``compare`` runs under this repo's env against the source build, recomputes
the same products and writes ``comparison.json`` with a per-product verdict::

    pixi run python tools/pg14_study_matrix.py compare

Only hashes are stored, so the record is small and a match means the three
arrays are byte-identical.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
UMBRELLA = REPO.parent.parent
STORE = REPO / "docs" / "pedigree-graph-0.8-migration" / "gate" / "14a" / "study"
THRESHOLD = 0.001

PRODUCTS = ("kinship_matrix", "approximate_kinship_matrix", "mean_kinship_by_generation")

# (source, products): a parquet path relative to the umbrella, or the parity
# fixture name.  The 536k pedigree gets the summary only (plan, "Study
# pedigrees"): its complete matrix is out of reach.
PEDIGREES: dict[str, tuple[str, tuple[str, ...]]] = {
    "dev_mean_n10k": ("results/dev/dev_mean_n10k/rep1/pedigree.parquet", PRODUCTS),
    "dev_cont_n10k": ("results/dev/dev_cont_n10k/rep1/pedigree.parquet", PRODUCTS),
    "baseline10K": ("results/base/baseline10K/rep1/pedigree.parquet", PRODUCTS),
    "random_30k": ("parity:random_30k", PRODUCTS),
    "baseline100K": ("results/base/baseline100K/rep1/pedigree.parquet", ("mean_kinship_by_generation",)),
}


def _sha(array) -> str:
    import numpy as np

    c = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str(c.dtype).encode())
    h.update(str(c.shape).encode())
    h.update(c.tobytes())
    return h.hexdigest()


def _input_sha(source: str) -> str:
    if source.startswith("parity:"):
        sys.path.insert(0, str(REPO / "tests" / "parity"))
        import pedigrees

        name = source.partition(":")[2]
        return pedigrees.input_hash(pedigrees.build_random(name, pedigrees.LARGE_FIXTURES[name]))
    return hashlib.sha256((UMBRELLA / source).read_bytes()).hexdigest()


def _graph(source: str):
    from pedigree_graph import PedigreeGraph

    if source.startswith("parity:"):
        sys.path.insert(0, str(REPO / "tests" / "parity"))
        import pedigrees

        name = source.partition(":")[2]
        fx = pedigrees.build_random(name, pedigrees.LARGE_FIXTURES[name])
        return PedigreeGraph.from_frame(
            {key: fx[key] for key in ("mother", "father", "twin", "sex") if key in fx} | {"id": fx["ids"]}
        )
    import polars as pl

    df = pl.read_parquet(UMBRELLA / source, columns=["id", "mother", "father", "twin", "sex"])
    return PedigreeGraph.from_frame(df)


def _version() -> dict[str, str]:
    import importlib.metadata

    import pedigree_graph
    from pedigree_graph import _native

    return {
        "package_version": importlib.metadata.version("pedigree-graph"),
        "core_version": _native.core_version(),
        "package_file": pedigree_graph.__file__,
    }


def _product(source: str, product: str) -> dict:
    """Child role: one product on a fresh graph, as a JSON-ready record."""
    import numpy as np

    graph = _graph(source)
    started = time.perf_counter()
    if product == "mean_kinship_by_generation":
        summary = graph.mean_kinship_by_generation()
        record = {
            "generations": summary.generations.tolist(),
            "mean_kinship_bits": np.ascontiguousarray(summary.mean_kinship).view(np.uint64).tolist(),
            "pair_counts": summary.pair_counts.tolist(),
            "unlabelled_individual_count": int(summary.unlabelled_individual_count),
        }
    else:
        if product == "kinship_matrix":
            matrix = graph.kinship_matrix()
        else:
            matrix = graph.approximate_kinship_matrix(min_propagated_kinship=THRESHOLD)
        assert matrix.data.dtype == np.float32
        record = {
            "nnz": int(matrix.nnz),
            "indptr_sha256": _sha(matrix.indptr),
            "indices_sha256": _sha(matrix.indices),
            "data_bits_sha256": _sha(matrix.data.view(np.uint32)),
        }
    record["n"] = int(graph.n_individuals)
    record["seconds"] = round(time.perf_counter() - started, 1)
    return record


def _run_product(source: str, product: str, timeout_s: float) -> dict:
    """Spawn the child role; a timeout or crash becomes a recorded outcome."""
    command = [sys.executable, str(Path(__file__).resolve()), "child", "--source", source, "--product", product]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout_s, check=False)
    except subprocess.TimeoutExpired:
        return {"outcome": "timed_out", "timeout_s": timeout_s}
    if result.returncode != 0:
        return {"outcome": "failed", "stderr": result.stderr[-2000:]}
    record = json.loads(result.stdout.strip().splitlines()[-1])
    record["outcome"] = "completed"
    return record


def capture(store: Path, timeout_s: float, only: list[str] | None) -> None:
    """Freeze every study product under the installed package."""
    store.mkdir(parents=True, exist_ok=True)
    manifest = {"role": "capture", **_version(), "threshold": THRESHOLD, "pedigrees": {}}
    for name, (source, products) in PEDIGREES.items():
        if only and name not in only:
            continue
        entry: dict = {"source": source, "input_sha256": _input_sha(source), "products": {}}
        for product in products:
            record = _run_product(source, product, timeout_s)
            entry["products"][product] = record
            print(f"{name}/{product}: {record['outcome']} {record.get('seconds', '')}", flush=True)
        manifest["pedigrees"][name] = entry
        (store / "capture.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"wrote {store / 'capture.json'}")


def _same(expected: dict, got: dict) -> bool:
    keys = [k for k in expected if k not in ("seconds", "outcome", "timeout_s", "stderr")]
    return all(expected[k] == got.get(k) for k in keys)


def compare(store: Path, timeout_s: float, only: list[str] | None) -> int:
    """Replay every captured product and report; ``0`` when all bytes match."""
    captured = json.loads((store / "capture.json").read_text())
    report = {
        "role": "compare",
        **_version(),
        "captured_with": {k: captured[k] for k in ("package_version", "core_version")},
        "pedigrees": {},
    }
    failures = 0
    for name, entry in captured["pedigrees"].items():
        if only and name not in only:
            continue
        source = entry["source"]
        assert _input_sha(source) == entry["input_sha256"], f"{name}: input changed"
        result: dict = {"source": source, "products": {}}
        for product, expected in entry["products"].items():
            if expected["outcome"] != "completed":
                result["products"][product] = {"verdict": "no_baseline", "baseline_outcome": expected["outcome"]}
                print(f"{name}/{product}: no baseline ({expected['outcome']})", flush=True)
                continue
            got = _run_product(source, product, timeout_s)
            if got["outcome"] != "completed":
                verdict = got["outcome"]
                failures += 1
            elif _same(expected, got):
                verdict = "byte_identical"
            else:
                verdict = "differs"
                failures += 1
            result["products"][product] = {"verdict": verdict, **got}
            print(f"{name}/{product}: {verdict} {got.get('seconds', '')}", flush=True)
        report["pedigrees"][name] = result
        report["byte_identical"] = failures == 0
        (store / "comparison.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {store / 'comparison.json'}; byte_identical={failures == 0}")
    return 0 if failures == 0 else 1


def main() -> None:
    """Dispatch ``capture``, ``compare`` or the ``child`` role."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("role", choices=("capture", "compare", "child"))
    ap.add_argument("--store", type=Path, default=STORE)
    ap.add_argument("--timeout", type=float, default=3600.0, help="seconds per product")
    ap.add_argument("--only", nargs="*", help="pedigree names to run")
    ap.add_argument("--source")
    ap.add_argument("--product", choices=PRODUCTS)
    args = ap.parse_args()
    if args.role == "child":
        print(json.dumps(_product(args.source, args.product), sort_keys=True))
    elif args.role == "capture":
        capture(args.store, args.timeout, args.only)
    else:
        sys.exit(compare(args.store, args.timeout, args.only))


if __name__ == "__main__":
    main()
