#!/usr/bin/env python
"""Measure what the pedigree-graph 0.7.1 -> 0.8.0 migration changes for consumers.

Slice 8 of the 0.8.0 plan migrates simACE, fitACE, fitACE_epimight, and pedsum.
0.8 corrects values and re-classifies pairs, so byte parity is not the gate;
the gate is a recorded, rerunnable comparison. This tool takes one snapshot of
consumer-visible outputs under whichever pedigree-graph the environment
resolves, and compares two snapshots.

Three groups of cases:

- ``pipeline``: the smoke scenario ``results/test/small_test`` after a full
  simACE + fitACE build. Reads the per-rep YAML reports and fitACE result
  tables. The build is the caller's job; this tool only reads.
- ``library``: pedigree-graph calls that consumers route through, on the smoke
  scenario's full pedigree and, with ``--large``, on the ``random_30k`` parity
  fixture. Branches on the installed API so the same case runs under both
  versions.
- ``pedsum``: ``pedsum summarize`` on its example pedigree and on a
  parent-offspring-incest fixture, run in pedsum's own pixi env; reads the slim
  summary. ``--source`` routes that subprocess to the source checkout.

Usage::

    pixi run python external/pedigree-graph/tools/pg08_migration_diff.py snapshot --out DIR [--large] [--source]
    pixi run python external/pedigree-graph/tools/pg08_migration_diff.py compare BASELINE_DIR MIGRATED_DIR [--out report.md]
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml

PG_SOURCE = Path(__file__).resolve().parents[1]
ROOT = PG_SOURCE.parents[1]  # the simACE umbrella checkout this repo sits inside
SMOKE = ROOT / "results" / "test" / "small_test"
PEDSUM = ROOT / "external" / "pedsum"
REPS = ("rep1", "rep2", "rep3")
APPROX_THRESHOLD = 0.001
FITACE_TABLES = (
    "exports/inbreeding.tsv",
    "pcgc/fit.vc.tsv",
    "iter_reml_fp32/fit.vc.tsv",
    "iter_reml_fp64/fit.vc.tsv",
)


def _sha(*arrays: np.ndarray) -> str:
    h = hashlib.sha256()
    for arr in arrays:
        c = np.ascontiguousarray(arr)
        h.update(str(c.dtype).encode())
        h.update(str(c.shape).encode())
        h.update(c.tobytes())
    return h.hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    """Leaves of nested dicts and lists keyed by dotted path."""
    out: dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(_flatten(v, f"{prefix}{k}."))
    elif isinstance(obj, list):
        out[f"{prefix}len"] = len(obj)
        for i, v in enumerate(obj):
            out.update(_flatten(v, f"{prefix}{i}."))
    else:
        out[prefix.rstrip(".")] = obj
    return out


def _library_version() -> dict[str, str]:
    import pedigree_graph

    api = "0.8" if hasattr(pedigree_graph.PedigreeGraph, "relationship_pairs") else "0.7.1"
    return {"file": pedigree_graph.__file__, "api": api}


# --- pipeline --------------------------------------------------------------


def _table(path: Path) -> dict[str, Any]:
    df = pl.read_csv(path, separator="\t", infer_schema_length=10_000)
    return {"n_rows": df.height, "columns": {c: df[c].to_list() for c in df.columns}}


def _pairwise_relatedness(path: Path) -> dict[str, Any]:
    df = pl.read_csv(path, separator="\t", infer_schema_length=10_000)
    code_col = next(c for c in ("rel_code", "relationship", "code") if c in df.columns)
    kin_col = next(c for c in ("kinship", "phi") if c in df.columns)
    by_code = df.group_by(code_col).agg(pl.len().alias("n"), pl.col(kin_col).sum().alias("kin_sum")).sort(code_col)
    return {
        "n_rows": df.height,
        "by_code": {r[code_col]: {"n": r["n"], "kin_sum": r["kin_sum"]} for r in by_code.iter_rows(named=True)},
    }


def _effective_size_yaml(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text())
    out: dict[str, Any] = {}
    for name, rec in raw.items():
        if not isinstance(rec, dict):
            out[name] = rec
            continue
        out[name] = {
            "ne": rec.get("ne"),
            "expected": rec.get("expected"),
            "reason": rec.get("reason"),
            "array_lengths": {k: len(v) for k, v in rec.items() if isinstance(v, list)},
            "keys": sorted(rec),
        }
    return out


def snapshot_pipeline() -> dict[str, Any]:
    """Read the smoke scenario's per-rep report and fitACE result tables."""
    out: dict[str, Any] = {}
    for rep in REPS:
        d = SMOKE / rep
        if not d.is_dir():
            continue
        r: dict[str, Any] = {}
        report = yaml.safe_load((d / "report.yaml").read_text())
        sample = report["observed"]["analysis_sample"]
        r["relationship_pair_counts"] = sample["relationship_pair_counts"]
        r["pedigree_relationship_pair_counts"] = report["observed"]["analysis_pedigree"]["relationship_pair_counts"]
        r["liability_correlations"] = _flatten(sample["liability_correlations"])
        r["tetrachoric"] = _flatten(sample["tetrachoric"])
        r["heritability"] = _flatten(report["estimators"]["heritability"])
        r["half_sibs"] = _flatten(report["truth"]["recorded_pedigree"]["family_structure"]["half_sibs"])
        r["quality_checks"] = _flatten(report["quality_checks"])
        r["effective_size"] = _effective_size_yaml(d / "effective_size.yaml")
        pr = d / "exports" / "pairwise_relatedness.tsv"
        if pr.exists():
            r["pairwise_relatedness"] = _pairwise_relatedness(pr)
        grm = d / "grm" / "A.grm.sp.bin"
        if grm.exists():
            r["grm"] = {
                "bytes": grm.stat().st_size,
                "sha": _file_sha(grm),
                "n_ids": len((d / "grm" / "A.grm.id").read_text().splitlines()),
            }
        for rel in FITACE_TABLES:
            p = d / rel
            if p.exists():
                r[rel] = _table(p)
        for p in sorted((d / "pafgrs").glob("metrics*.tsv")):
            r[f"pafgrs/{p.name}"] = _table(p)
        out[rep] = r
    return out


# --- library ---------------------------------------------------------------


def _load_parity_pedigrees():
    spec = importlib.util.spec_from_file_location("parity_pedigrees", PG_SOURCE / "tests" / "parity" / "pedigrees.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _upper(K) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coo = K.tocoo()
    keep = coo.row <= coo.col
    r, c, v = coo.row[keep].astype(np.int64), coo.col[keep].astype(np.int64), coo.data[keep].astype(np.float64)
    order = np.lexsort((c, r))
    return r[order], c[order], v[order]


def _support_key(r: np.ndarray, c: np.ndarray, n: int) -> np.ndarray:
    return r * n + c


def _build_graph(ids, mother, father, sex, twin, api: str):
    from pedigree_graph import PedigreeGraph

    if api == "0.8":
        return PedigreeGraph.from_arrays(ids=ids, mother_ids=mother, father_ids=father, twin_ids=twin, sex=sex)
    return PedigreeGraph.from_arrays(ids=ids, mothers=mother, fathers=father, twins=twin, sex=sex)


def _library_case(name: str, ids, mother, father, sex, twin, api: str) -> dict[str, Any]:
    pg = _build_graph(ids, mother, father, sex, twin, api)
    n = len(ids)
    case: dict[str, Any] = {"n": n}
    if api == "0.8":
        pairs = pg.relationship_pairs(max_degree=3)
        case["pairs_deg3"] = {code: len(block) for code, block in pairs.items()}
        est = pg.estimate_relationship_counts(max_degree=5)
        case["estimate_deg5"] = {code: est[code] for code in est}
        case["estimate_exact_codes"] = sorted(est.exact)
        deg2 = pg.relationship_kinship_matrix(max_degree=2)
        approx = pg.approximate_kinship_matrix(min_propagated_kinship=APPROX_THRESHOLD)
        complete = pg.kinship_matrix()
        f = pg.inbreeding()
        desc = pg.descendant_path_counts()
    else:
        pairs = pg.extract_pairs(max_degree=3)
        case["pairs_deg3"] = {code: len(a) for code, (a, _) in pairs.items()}
        est = pg.count_pairs_streaming(max_degree=5)
        case["estimate_deg5"] = {code: int(v) for code, v in est.items()}
        case["estimate_exact_codes"] = None
        deg2 = pg.kinship_matrix(max_degree=2)
        approx = pg.kinship_matrix(min_kinship=APPROX_THRESHOLD)
        complete = pg.kinship_matrix(min_kinship=0.0)
        f = pg.compute_inbreeding()
        desc = pg.compute_n_descendants()
    r2, c2, v2 = _upper(deg2)
    ra, ca, va = _upper(approx)
    rc, cc, vc = _upper(complete)
    case["degree2_matrix"] = {"upper_nnz": len(r2), "support_sha": _sha(r2, c2), "value_sum": float(v2.sum())}
    complete_key = _support_key(rc, cc, n)
    approx_key = _support_key(ra, ca, n)
    shared = np.isin(approx_key, complete_key)
    vc_on_shared = vc[np.searchsorted(complete_key, approx_key[shared])]
    case["approx_matrix"] = {
        "threshold": APPROX_THRESHOLD,
        "upper_nnz": len(ra),
        "support_sha": _sha(ra, ca),
        "value_sum": float(va.sum()),
        "max_abs_diff_vs_complete_on_support": float(np.max(np.abs(va[shared] - vc_on_shared)))
        if shared.any()
        else 0.0,
        "complete_entries_absent": int(len(rc) - shared.sum()),
    }
    case["complete_matrix"] = {
        "upper_nnz": len(rc),
        "value_sum": float(vc.sum()),
        "value_sha": _sha(vc.astype(np.float32)),
    }
    case["inbreeding"] = {"sum": float(np.sum(f)), "max": float(np.max(f)), "n_positive": int(np.count_nonzero(f > 0))}
    case["descendant_counts"] = {"dtype": str(desc.dtype), "max": int(desc.max()), "sum": int(desc.sum())}
    return case


def snapshot_library(large: bool) -> dict[str, Any]:
    """Run the library cases consumers route through, under the resolved API."""
    api = _library_version()["api"]
    out: dict[str, Any] = {}
    df = pl.read_parquet(SMOKE / "rep1" / "pedigree.full.parquet")
    out["small_test_rep1_full"] = _library_case(
        "small_test",
        df["id"].to_numpy(),
        df["mother"].to_numpy(),
        df["father"].to_numpy(),
        df["sex"].to_numpy(),
        df["twin"].to_numpy(),
        api,
    )
    if large:
        peds = _load_parity_pedigrees()
        fx = peds.build_random("random_30k", dict(peds.LARGE_FIXTURES["random_30k"]))
        out["random_30k"] = _library_case(
            "random_30k", fx["ids"], fx["mother"], fx["father"], fx["sex"], fx["twin"], api
        )
        out["random_30k"]["input_hash"] = peds.input_hash(fx)
    return out


# --- pedsum ----------------------------------------------------------------

INCEST_FIXTURE = """id\tsex\tmother\tfather\tgeneration
1\tF\t-1\t-1\t0
2\tM\t-1\t-1\t0
3\tF\t1\t2\t1
4\tM\t1\t2\t1
5\tF\t3\t2\t2
6\tM\t3\t2\t2
7\tF\t3\t4\t2
8\tM\t5\t6\t3
9\tF\t7\t6\t3
"""


def _run_pedsum(input_tsv: Path, out_dir: Path, source: bool) -> dict[str, Any]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("PIXI_")}
    cmd = ["pixi", "run"]
    if source:
        env["PYTHONPATH"] = str(PG_SOURCE)
        cmd.append("--frozen")
    cmd += [
        "python",
        "-c",
        "import sys; from pedsum.cli import main; sys.exit(main(sys.argv[1:]))",
        "summarize",
        "--in",
        str(input_tsv),
        "--out",
        str(out_dir),
        "--effective-size",
        "--ne-coancestry",
        "--per-individual-pairs",
        "-q",
    ]
    subprocess.run(cmd, cwd=PEDSUM, env=env, check=True)
    slim = yaml.safe_load((out_dir / "summary.yaml").read_text())
    extra_lines = len((out_dir / "summary.extra.yaml").read_text().splitlines())
    return {"slim": _flatten(slim), "extra_yaml_lines": extra_lines}


def snapshot_pedsum(source: bool) -> dict[str, Any]:
    """Run pedsum summarize in its own env on the example and incest fixtures."""
    out: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="pg08_pedsum_") as tmp:
        tmpd = Path(tmp)
        out["example_pedigree"] = _run_pedsum(PEDSUM / "example_pedigree.tsv", tmpd / "example", source)
        incest = tmpd / "incest.tsv"
        incest.write_text(INCEST_FIXTURE)
        out["parent_offspring_incest"] = _run_pedsum(incest, tmpd / "incest", source)
    return out


# --- compare ---------------------------------------------------------------


def _leaf_equal(a: Any, b: Any) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        if a is None or b is None:
            return a is b
        if isinstance(a, bool) or isinstance(b, bool):
            return a == b
        try:
            fa, fb = float(a), float(b)
        except (TypeError, ValueError):
            return a == b
        if math.isnan(fa) and math.isnan(fb):
            return True
        return math.isclose(fa, fb, rel_tol=1e-9, abs_tol=1e-12)
    return a == b


def compare(baseline: dict[str, Any], migrated: dict[str, Any]) -> list[tuple[str, Any, Any]]:
    """Leaves that differ between two snapshots, as (path, baseline, migrated)."""
    fa, fb = _flatten(baseline), _flatten(migrated)
    rows = []
    for key in sorted(set(fa) | set(fb)):
        a, b = fa.get(key, "<absent>"), fb.get(key, "<absent>")
        if not _leaf_equal(a, b):
            rows.append((key, a, b))
    return rows


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def render(rows: list[tuple[str, Any, Any]], baseline_meta: dict, migrated_meta: dict) -> str:
    """Markdown table of the changed leaves."""
    lines = [
        "# pedigree-graph 0.8 migration diff",
        "",
        f"- baseline: `{baseline_meta['library']['file']}` (api {baseline_meta['library']['api']})",
        f"- migrated: `{migrated_meta['library']['file']}` (api {migrated_meta['library']['api']})",
        f"- changed leaves: {len(rows)}",
        "",
        "| path | baseline | migrated |",
        "|---|---|---|",
    ]
    for key, a, b in rows:
        lines.append(f"| `{key}` | {_fmt(a)} | {_fmt(b)} |")
    return "\n".join(lines) + "\n"


# --- cli -------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("snapshot", help="record consumer-visible outputs under the resolved pedigree-graph")
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--large", action="store_true", help="also run the random_30k library case")
    s.add_argument("--source", action="store_true", help="route the pedsum subprocess to the source checkout")
    s.add_argument("--skip", nargs="*", default=(), choices=("pipeline", "library", "pedsum"))
    c = sub.add_parser("compare", help="diff two snapshot directories")
    c.add_argument("baseline", type=Path)
    c.add_argument("migrated", type=Path)
    c.add_argument("--out", type=Path)
    args = ap.parse_args(argv)

    if args.cmd == "snapshot":
        args.out.mkdir(parents=True, exist_ok=True)
        meta = {"library": _library_version(), "argv": sys.argv[1:]}
        (args.out / "meta.json").write_text(json.dumps(meta, indent=2))
        for name in ("pipeline", "library", "pedsum"):
            if name in args.skip:
                continue
            print(f"[{name}] under {meta['library']['api']}", flush=True)
            if name == "pipeline":
                data = snapshot_pipeline()
            elif name == "library":
                data = snapshot_library(args.large)
            else:
                data = snapshot_pedsum(args.source)
            (args.out / f"{name}.json").write_text(json.dumps(data, indent=2, sort_keys=True))
        return 0

    base_meta = json.loads((args.baseline / "meta.json").read_text())
    mig_meta = json.loads((args.migrated / "meta.json").read_text())
    rows: list[tuple[str, Any, Any]] = []
    for name in ("pipeline", "library", "pedsum"):
        pa, pb = args.baseline / f"{name}.json", args.migrated / f"{name}.json"
        if pa.exists() and pb.exists():
            rows += [
                (f"{name}.{k}", a, b) for k, a, b in compare(json.loads(pa.read_text()), json.loads(pb.read_text()))
            ]
    text = render(rows, base_meta, mig_meta)
    if args.out:
        args.out.write_text(text)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
