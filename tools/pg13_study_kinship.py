"""Freeze and replay ``pair_kinship`` bits on the four simACE study pedigrees (slice 13, gate 13a).

The pedigrees live under the simACE umbrella's ``results/`` and are not repo
fixtures, so this is a one-off gate artifact rather than a test.  Two roles:

``capture`` runs under the simACE pixi env, which is locked to the 0.9.0 PyPI
wheel until the relock, and writes one ``.npz`` per pedigree: the SHA-256 of
every degree-3 pair block (degree 5 as well for the 20k pedigree), the
``pair_kinship`` values as uint32 bit views, and the self-pair values::

    cd external/pedigree-graph
    pixi run --manifest-path ../../pixi.toml python tools/pg13_study_kinship.py capture

``compare`` runs under this repo's env against the source build, recomputes
the same queries in both endpoint orders, asserts the pair hashes match so the
comparison is over identical pairs, and writes ``comparison.json`` with the
count of differing elements and the max ULP distance per query::

    pixi run python tools/pg13_study_kinship.py compare

The ``.npz`` files stay local (gitignored); ``comparison.json`` is the record.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
UMBRELLA = REPO.parent.parent
STORE = REPO / "docs" / "pedigree-graph-0.8-migration" / "gate" / "13a" / "study"

PEDIGREES = {
    "dev_mean_n10k": ("results/dev/dev_mean_n10k/rep1/pedigree.parquet", (3, 5)),
    "baseline10K": ("results/base/baseline10K/rep1/pedigree.parquet", (3,)),
    "dev_laplace_am_strong_50k": ("results/dev/dev_laplace_am_strong_50k/rep1/pedigree.parquet", (3,)),
    "baseline100K": ("results/base/baseline100K/rep1/pedigree.parquet", (3,)),
}


def _sha(*arrays: np.ndarray) -> str:
    h = hashlib.sha256()
    for arr in arrays:
        c = np.ascontiguousarray(arr)
        h.update(str(c.dtype).encode())
        h.update(str(c.shape).encode())
        h.update(c.tobytes())
    return h.hexdigest()


def _graph(path: Path):
    import polars as pl

    from pedigree_graph import PedigreeGraph

    df = pl.read_parquet(path, columns=["id", "mother", "father", "twin", "sex"])
    return PedigreeGraph.from_frame(df)


def _bits(values: np.ndarray) -> np.ndarray:
    assert values.dtype == np.float32
    return values.view(np.uint32)


def _version() -> dict[str, str]:
    import importlib.metadata

    import pedigree_graph
    from pedigree_graph import _native

    return {
        "package_version": importlib.metadata.version("pedigree-graph"),
        "core_version": _native.core_version(),
        "package_file": pedigree_graph.__file__,
    }


def capture(store: Path) -> None:
    """Freeze the queries of every study pedigree under the installed package."""
    store.mkdir(parents=True, exist_ok=True)
    manifest = {"role": "capture", **_version(), "pedigrees": {}}
    for name, (relative, degrees) in PEDIGREES.items():
        path = UMBRELLA / relative
        started = time.perf_counter()
        graph = _graph(path)
        arrays: dict[str, np.ndarray] = {}
        entry: dict = {
            "path": relative,
            "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "n": graph.n_individuals,
            "queries": {},
        }
        for degree in degrees:
            pairs = graph.relationship_pairs(max_degree=degree)
            values = graph.pair_kinship(pairs)
            for code, block in pairs.items():
                if not len(block):
                    continue
                key = f"d{degree}/{code}"
                arrays[f"{key}/first"] = block.first_rows
                arrays[f"{key}/second"] = block.second_rows
                arrays[f"{key}/values"] = _bits(values[code])
                entry["queries"][key] = {"pairs": len(block), "pair_sha256": _sha(block.first_rows, block.second_rows)}
        rows = np.arange(graph.n_individuals, dtype=np.int32)
        arrays["self/first"] = rows
        arrays["self/second"] = rows
        arrays["self/values"] = _bits(graph.pair_kinship(rows, rows))
        entry["queries"]["self"] = {"pairs": int(rows.size), "pair_sha256": _sha(rows, rows)}
        np.savez_compressed(store / f"{name}.npz", **arrays)
        entry["seconds"] = round(time.perf_counter() - started, 1)
        manifest["pedigrees"][name] = entry
        print(f"{name}: n={graph.n_individuals} {entry['seconds']}s", flush=True)
    (store / "capture.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"wrote {store / 'capture.json'}")


def compare(store: Path) -> int:
    """Replay the frozen queries and report bit differences; ``0`` when none."""
    captured = json.loads((store / "capture.json").read_text())
    report = {
        "role": "compare",
        **_version(),
        "captured_with": {k: captured[k] for k in ("package_version", "core_version")},
        "pedigrees": {},
    }
    failures = 0
    for name, entry in captured["pedigrees"].items():
        path = UMBRELLA / entry["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["input_sha256"], f"{name}: input changed"
        started = time.perf_counter()
        graph = _graph(path)
        assert graph.n_individuals == entry["n"]
        result: dict = {"n": entry["n"], "queries": {}}
        with np.load(store / f"{name}.npz") as npz:
            for key, query in entry["queries"].items():
                first, second, expected = npz[f"{key}/first"], npz[f"{key}/second"], npz[f"{key}/values"]
                assert _sha(first, second) == query["pair_sha256"], (
                    f"{name} {key}: stored pairs differ from the manifest"
                )
                outcome = {"pairs": int(first.size), "pair_sha256_matches": True}
                if key != "self":
                    degree, code = key.split("/")
                    block = graph.relationship_pairs(max_degree=int(degree[1:]))[code]
                    outcome["pair_sha256_matches"] = _sha(block.first_rows, block.second_rows) == query["pair_sha256"]
                for order, (a, b) in (("forward", (first, second)), ("reverse", (second, first))):
                    got = _bits(graph.pair_kinship(a, b))
                    ulp = np.abs(got.astype(np.int64) - expected.astype(np.int64))
                    outcome[order] = {"differing": int((ulp > 0).sum()), "max_ulp": int(ulp.max()) if ulp.size else 0}
                    if outcome[order]["differing"]:
                        failures += 1
                if not outcome["pair_sha256_matches"]:
                    failures += 1
                result["queries"][key] = outcome
        result["seconds"] = round(time.perf_counter() - started, 1)
        report["pedigrees"][name] = result
        worst = max((q[o]["differing"] for q in result["queries"].values() for o in ("forward", "reverse")), default=0)
        print(f"{name}: n={entry['n']} worst differing={worst} {result['seconds']}s", flush=True)
    report["bit_identical"] = failures == 0
    (store / "comparison.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {store / 'comparison.json'}; bit_identical={failures == 0}")
    return 0 if failures == 0 else 1


def main() -> None:
    """Dispatch ``capture`` or ``compare``."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("role", choices=("capture", "compare"))
    ap.add_argument("--store", type=Path, default=STORE)
    args = ap.parse_args()
    if args.role == "capture":
        capture(args.store)
    else:
        sys.exit(compare(args.store))


if __name__ == "__main__":
    main()
