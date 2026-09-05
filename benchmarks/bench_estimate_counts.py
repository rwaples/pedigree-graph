"""A/B benchmark for the scalar relationship-count path (slice 4c gate).

Builds ``random_30k`` (the parity parameters) and ``random_300k`` (the release
parameters) with ``tests/parity/pedigrees.build_random``, then times the count
call on a fresh graph in a child process per run.  Wall time is the count call
alone, measured inside the child; peak RSS is the child's ``ru_maxrss`` from
``os.wait4`` and therefore includes loading the arrays and building the graph,
which both conditions share.

    python benchmarks/bench_estimate_counts.py run --method count_pairs_streaming --tag baseline-1 --out DIR
    python benchmarks/bench_estimate_counts.py run --method estimate_relationship_counts --tag new-1 --out DIR
    python benchmarks/bench_estimate_counts.py report DIR

``report`` prints the median wall and peak RSS per condition and pedigree with
the new/baseline ratio, and exits 1 when that ratio exceeds 1.05 on either
metric for either pedigree.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests" / "parity"))

PEDIGREES = {
    "random_30k": {"seed": 30_000, "n_founders": 1_500, "n_generations": 8, "per_generation": 3_600},
    "random_300k": {"seed": 300_000, "n_founders": 15_000, "n_generations": 8, "per_generation": 36_000},
}
METHODS = ("count_pairs_streaming", "estimate_relationship_counts")
BASELINE, DEFAULT = METHODS
GATE = 1.05
GATED = (DEFAULT,)
MAX_DEGREE = 5


def _child(npz: str, method: str) -> None:
    import warnings

    from pedigree_graph import PedigreeGraph

    with np.load(npz) as data:
        columns = {name: data[name] for name in data.files}
    graph = PedigreeGraph(columns)
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if method == BASELINE:
            counts = graph.count_pairs_streaming(max_degree=MAX_DEGREE)
        else:
            counts = dict(graph.estimate_relationship_counts(max_degree=MAX_DEGREE))
    wall = time.perf_counter() - t0
    print(json.dumps({"wall": wall, "FS": counts["FS"], "GP": counts["GP"]}))


def _materialise(name: str, directory: Path) -> Path:
    import pedigrees

    path = directory / f"{name}.npz"
    if not path.exists():
        fx = pedigrees.build_random(name, PEDIGREES[name])
        np.savez(path, id=fx["ids"], mother=fx["mother"], father=fx["father"], twin=fx["twin"], sex=fx["sex"])
    return path


def _run(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = Path(tempfile.gettempdir()) / "bench_estimate_counts"
    cache.mkdir(exist_ok=True)
    records = []
    for name in args.pedigrees:
        npz = _materialise(name, cache)
        for run in range(args.runs):
            proc = subprocess.Popen(
                [sys.executable, __file__, "child", str(npz), args.method],
                stdout=subprocess.PIPE,
                text=True,
            )
            assert proc.stdout is not None
            stdout = proc.stdout.read()
            proc.stdout.close()
            _, status, usage = os.wait4(proc.pid, 0)
            proc.returncode = os.waitstatus_to_exitcode(status)
            if proc.returncode != 0:
                raise RuntimeError(f"child failed: {stdout}")
            record = json.loads(stdout)
            record.update(pedigree=name, method=args.method, tag=args.tag, run=run, rss_kib=usage.ru_maxrss)
            records.append(record)
            print(json.dumps(record), flush=True)
    (out / f"{args.tag}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))


def _report(args: argparse.Namespace) -> None:
    records = [
        json.loads(line) for path in sorted(Path(args.dir).glob("*.jsonl")) for line in path.read_text().splitlines()
    ]
    print(f"{'pedigree':12s} {'metric':8s} {'condition':42s} {'median':>10s} {'n':>3s} {'ratio':>7s}")
    failed = False
    for name in PEDIGREES:
        for metric, key, scale in (("wall_s", "wall", 1.0), ("rss_mib", "rss_kib", 1 / 1024)):
            medians = {}
            counts = {}
            for method in METHODS:
                values = [r[key] * scale for r in records if r["pedigree"] == name and r["method"] == method]
                if values:
                    medians[method] = statistics.median(values)
                    counts[method] = len(values)
            if BASELINE not in medians:
                continue
            for method in METHODS:
                if method not in medians:
                    continue
                ratio = medians[method] / medians[BASELINE]
                blocked = method in GATED and ratio > GATE
                failed |= blocked
                print(
                    f"{name:12s} {metric:8s} {method:42s} {medians[method]:10.3f} {counts[method]:3d} {ratio:7.3f}"
                    f"{'  BLOCK' if blocked else ''}"
                )
    sys.exit(1 if failed else 0)


def main() -> None:
    """Dispatch the child, run, and report entry points."""
    if len(sys.argv) > 1 and sys.argv[1] == "child":
        _child(sys.argv[2], sys.argv[3])
        return
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--method", choices=METHODS, required=True)
    run.add_argument("--tag", required=True)
    run.add_argument("--out", required=True)
    run.add_argument("--runs", type=int, default=5)
    run.add_argument("--pedigrees", nargs="+", default=list(PEDIGREES), choices=list(PEDIGREES))
    run.set_defaults(func=_run)
    report = sub.add_parser("report")
    report.add_argument("dir")
    report.set_defaults(func=_report)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
