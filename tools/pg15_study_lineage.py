"""Freeze and replay inbreeding, lineage, EqG and Ne on the simACE study pedigrees (slice 15, gate 15a).

The pedigrees live under the simACE umbrella's ``results/`` (plus the
``random_30k`` parity fixture), so this is a one-off gate artifact rather than
a test.  Each product runs on its own freshly built graph in a child process,
so no product reuses a prerequisite another one computed, and a product that
outgrows the timeout is recorded rather than blocking the rest.

Products per pedigree:

* ``distinct_ancestor_counts`` and ``descendant_path_counts``: SHA-256 only,
  since the plan holds counts byte-identical;
* ``inbreeding`` and ``eqg`` (Maignel's equivalent complete generations): the
  float64 vector, stored in ``.npz`` beside its SHA-256;
* ``effective_sizes``: every field of the eight Ne records from
  ``estimate_effective_sizes`` (``ne_coancestry`` excluded on
  ``baseline100K``), numeric fields in ``.npz`` and the rest (refusal
  reasons, ``None``) in the JSON record.

Two roles:

``capture`` runs under the simACE pixi env, which is locked to the 0.9.3 PyPI
wheel until the relock, and writes ``capture.json`` plus ``capture/*.npz``::

    cd external/pedigree-graph
    pixi run --manifest-path ../../pixi.toml python tools/pg15_study_lineage.py capture

``compare`` runs under this repo's env against the source build, recomputes
the same products, writes ``comparison.json`` plus ``compare/*.npz``, and
reports per product whether the bytes matched and, for floats, the maximum
absolute and relative difference against ``rtol 1e-9, atol 1e-12``::

    pixi run python tools/pg15_study_lineage.py compare

EqG has no public method.  The capture computes it as 0.9.3's
``ne_individual_delta_f`` does (``_kinship_depth._compute_eqg`` on the graph's
parent rows and depth); compare calls ``_native.equivalent_generations(graph._built,
graph.depth)`` (plan D12) when the build has it and falls back to the 0.9.3
kernel otherwise, recording which one ran.
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
STORE = REPO / "docs" / "pedigree-graph-0.8-migration" / "gate" / "15a" / "study"
RTOL = 1e-9
ATOL = 1e-12

COUNTS = ("distinct_ancestor_counts", "descendant_path_counts")
PRODUCTS = (*COUNTS, "inbreeding", "eqg", "effective_sizes")

# (source, estimators excluded): a parquet path relative to the umbrella, or
# the parity fixture name.  ne_coancestry is excluded on the 536k pedigree
# (plan D13).
PEDIGREES: dict[str, tuple[str, tuple[str, ...]]] = {
    "dev_mean_n10k": ("results/dev/dev_mean_n10k/rep1/pedigree.parquet", ()),
    "dev_cont_n10k": ("results/dev/dev_cont_n10k/rep1/pedigree.parquet", ()),
    "baseline10K": ("results/base/baseline10K/rep1/pedigree.parquet", ()),
    "random_30k": ("parity:random_30k", ()),
    "baseline100K": ("results/base/baseline100K/rep1/pedigree.parquet", ("ne_coancestry",)),
}


def _sha(array) -> str:
    import numpy as np

    c = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str(c.dtype).encode())
    h.update(str(c.shape).encode())
    h.update(c.tobytes())
    return h.hexdigest()


def _pedigrees():
    sys.path.insert(0, str(REPO / "tests" / "parity"))
    import pedigrees

    return pedigrees


def _parity(source: str) -> dict:
    pedigrees = _pedigrees()
    name = source.partition(":")[2]
    return pedigrees.build_random(name, pedigrees.LARGE_FIXTURES[name])


def _input_sha(source: str) -> str:
    if source.startswith("parity:"):
        return _pedigrees().input_hash(_parity(source))
    return hashlib.sha256((UMBRELLA / source).read_bytes()).hexdigest()


def _graph(source: str):
    from pedigree_graph import PedigreeGraph

    if source.startswith("parity:"):
        fx = _parity(source)
        return PedigreeGraph.from_frame(
            {key: fx[key] for key in ("mother", "father", "twin", "sex") if key in fx} | {"id": fx["ids"]}
        )
    import polars as pl
    import pyarrow.parquet as pq

    path = UMBRELLA / source
    present = set(pq.read_schema(path).names)
    columns = [c for c in ("id", "mother", "father", "twin", "sex", "generation") if c in present]
    return PedigreeGraph.from_frame(pl.read_parquet(path, columns=columns))


def _version() -> dict[str, str]:
    import importlib.metadata

    import pedigree_graph
    from pedigree_graph import _native

    return {
        "package_version": importlib.metadata.version("pedigree-graph"),
        "core_version": _native.core_version(),
        "package_file": pedigree_graph.__file__,
    }


def _eqg(graph) -> tuple[object, str]:
    import numpy as np

    from pedigree_graph import _native

    if hasattr(_native, "equivalent_generations"):
        return _native.equivalent_generations(graph._built, graph.depth), "_native.equivalent_generations"
    from pedigree_graph._kinship_depth import _compute_eqg

    eqg = _compute_eqg(
        np.asarray(graph.mother_rows), np.asarray(graph.father_rows), np.asarray(graph.depth), graph.n_individuals
    )
    return eqg, "_kinship_depth._compute_eqg"


def _flatten(prefix: str, value: object, arrays: dict, exact: dict) -> None:
    """Split one record into numeric arrays (for the ``.npz``) and everything else (for the JSON)."""
    from collections.abc import Mapping
    from dataclasses import fields, is_dataclass

    import numpy as np

    if isinstance(value, np.ndarray) and value.dtype.kind in "biuf":
        arrays[prefix] = np.ascontiguousarray(value)
    elif isinstance(value, bool | np.bool_):
        exact[prefix] = bool(value)
    elif isinstance(value, int | float | np.integer | np.floating):
        arrays[prefix] = np.asarray(value)
    elif is_dataclass(value):
        for item in fields(value):
            _flatten(f"{prefix}/{item.name}", getattr(value, item.name), arrays, exact)
    elif isinstance(value, Mapping):
        for key in value:
            _flatten(f"{prefix}/{key}", value[key], arrays, exact)
    elif hasattr(value, "_asdict"):
        _flatten(prefix, value._asdict(), arrays, exact)
    elif value is None or isinstance(value, str):
        exact[prefix] = value
    else:
        exact[prefix] = repr(value)


def _product(source: str, product: str, excluded: list[str], npz: Path) -> dict:
    """Child role: one product on a fresh graph, as a JSON-ready record plus an ``.npz``."""
    import numpy as np

    graph = _graph(source)
    started = time.perf_counter()
    arrays: dict = {}
    exact: dict = {}
    if product in COUNTS:
        counts = getattr(graph, product)()
        seconds = time.perf_counter() - started
        record: dict = {"sha256": _sha(counts), "dtype": str(counts.dtype), "max": int(counts.max(initial=0))}
    elif product == "inbreeding":
        arrays["F"] = np.asarray(graph.inbreeding())
        seconds = time.perf_counter() - started
        record = {}
    elif product == "eqg":
        eqg, path = _eqg(graph)
        arrays["eqg"] = np.asarray(eqg)
        seconds = time.perf_counter() - started
        record = {"eqg_path": path}
    else:
        from pedigree_graph.effective_size import ALL_EFFECTIVE_SIZE_ESTIMATORS, estimate_effective_sizes

        selected = [name for name in ALL_EFFECTIVE_SIZE_ESTIMATORS if name not in excluded]
        results = estimate_effective_sizes(graph, selected)
        seconds = time.perf_counter() - started
        for name, result in results.items():
            _flatten(name, result, arrays, exact)
        record = {"estimators": selected, "exact": exact}
    if arrays:
        npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(npz, **arrays)
        record["npz"] = npz.name
        record["arrays_sha256"] = {key: _sha(array) for key, array in arrays.items()}
    record["n"] = int(graph.n_individuals)
    record["seconds"] = round(seconds, 2)
    return record


def _run_product(name: str, product: str, npz_dir: Path, timeout_s: float) -> dict:
    """Spawn the child role; a timeout or crash becomes a recorded outcome."""
    source, excluded = PEDIGREES[name]
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "child",
        "--source",
        source,
        "--product",
        product,
        "--npz",
        str(npz_dir / f"{name}.{product}.npz"),
        "--exclude",
        *excluded,
    ]
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
    path = store / "capture.json"
    manifest = json.loads(path.read_text()) if only and path.exists() else {"pedigrees": {}}
    manifest |= {"role": "capture", **_version(), "rtol": RTOL, "atol": ATOL}
    for name, (source, _excluded) in PEDIGREES.items():
        if only and name not in only:
            continue
        entry: dict = {"source": source, "input_sha256": _input_sha(source), "products": {}}
        for product in PRODUCTS:
            record = _run_product(name, product, store / "capture", timeout_s)
            entry["products"][product] = record
            print(f"{name}/{product}: {record['outcome']} {record.get('seconds', '')}", flush=True)
        manifest["pedigrees"][name] = entry
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"wrote {path}")


def _diff_arrays(expected_npz: Path, got_npz: Path) -> dict:
    """Per array: byte identity, and for floats the max abs/rel difference against the tolerance."""
    import numpy as np

    with np.load(expected_npz) as expected_file, np.load(got_npz) as got_file:
        expected = {key: expected_file[key] for key in expected_file.files}
        got = {key: got_file[key] for key in got_file.files}
    report: dict = {"missing": sorted(expected.keys() - got.keys()), "extra": sorted(got.keys() - expected.keys())}
    arrays: dict = {}
    for key in sorted(expected.keys() & got.keys()):
        a, b = expected[key], got[key]
        entry: dict = {"bytes_identical": a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()}
        if a.shape != b.shape:
            entry |= {"within_tolerance": False, "shape": [list(a.shape), list(b.shape)]}
        elif a.dtype.kind == "f" or b.dtype.kind == "f":
            a64, b64 = a.astype(np.float64), b.astype(np.float64)
            same_nan = bool(np.array_equal(np.isnan(a64), np.isnan(b64)))
            finite = ~np.isnan(a64) & ~np.isnan(b64)
            diff = np.abs(a64[finite] - b64[finite])
            scale = np.abs(a64[finite])
            nonzero = scale > 0
            entry |= {
                "max_abs_diff": float(diff.max(initial=0.0)),
                "max_rel_diff": float((diff[nonzero] / scale[nonzero]).max(initial=0.0)),
                "within_tolerance": same_nan and bool(np.allclose(b64, a64, rtol=RTOL, atol=ATOL, equal_nan=True)),
            }
        else:
            entry["within_tolerance"] = bool(np.array_equal(a, b))
        arrays[key] = entry
    report["arrays"] = arrays
    report["bytes_identical"] = (
        not report["missing"] and not report["extra"] and all(e["bytes_identical"] for e in arrays.values())
    )
    report["within_tolerance"] = (
        not report["missing"] and not report["extra"] and all(e["within_tolerance"] for e in arrays.values())
    )
    report["max_abs_diff"] = max((e.get("max_abs_diff", 0.0) for e in arrays.values()), default=0.0)
    report["max_rel_diff"] = max((e.get("max_rel_diff", 0.0) for e in arrays.values()), default=0.0)
    return report


def _verdict(product: str, expected: dict, got: dict, store: Path) -> dict:
    if product in COUNTS:
        identical = expected["sha256"] == got["sha256"]
        return {"verdict": "byte_identical" if identical else "differs", "bytes_identical": identical}
    diff = _diff_arrays(store / "capture" / expected["npz"], store / "compare" / got["npz"])
    exact_same = expected.get("exact", {}) == got.get("exact", {})
    if not exact_same:
        diff["exact_differs"] = {
            key: [expected.get("exact", {}).get(key), got.get("exact", {}).get(key)]
            for key in expected.get("exact", {}).keys() | got.get("exact", {}).keys()
            if expected.get("exact", {}).get(key) != got.get("exact", {}).get(key)
        }
    if diff["bytes_identical"] and exact_same:
        verdict = "byte_identical"
    elif diff["within_tolerance"] and exact_same:
        verdict = "within_tolerance"
    else:
        verdict = "differs"
    return {"verdict": verdict, **diff}


def compare(store: Path, timeout_s: float, only: list[str] | None) -> int:
    """Replay every captured product and report; ``0`` when all hold D2's tolerance."""
    captured = json.loads((store / "capture.json").read_text())
    report = {
        "role": "compare",
        **_version(),
        "captured_with": {k: captured[k] for k in ("package_version", "core_version")},
        "rtol": RTOL,
        "atol": ATOL,
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
            got = _run_product(name, product, store / "compare", timeout_s)
            if got["outcome"] != "completed":
                outcome = {"verdict": got["outcome"]}
            else:
                outcome = _verdict(product, expected, got, store)
            failures += outcome["verdict"] not in ("byte_identical", "within_tolerance")
            summary = {k: got[k] for k in ("seconds", "eqg_path", "stderr") if k in got}
            result["products"][product] = {**outcome, **summary, "baseline_seconds": expected.get("seconds")}
            detail = (
                f" max_abs={outcome['max_abs_diff']:.3g} max_rel={outcome['max_rel_diff']:.3g}"
                if "max_abs_diff" in outcome
                else ""
            )
            print(f"{name}/{product}: {outcome['verdict']}{detail} {got.get('seconds', '')}", flush=True)
        report["pedigrees"][name] = result
        report["all_within_tolerance"] = failures == 0
        (store / "comparison.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {store / 'comparison.json'}; all_within_tolerance={failures == 0}")
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
    ap.add_argument("--npz", type=Path)
    ap.add_argument("--exclude", nargs="*", default=[])
    args = ap.parse_args()
    if args.role == "child":
        print(json.dumps(_product(args.source, args.product, args.exclude, args.npz), sort_keys=True))
    elif args.role == "capture":
        capture(args.store, args.timeout, args.only)
    else:
        sys.exit(compare(args.store, args.timeout, args.only))


if __name__ == "__main__":
    main()
