r"""Slice 12 stage B: qualify the surviving pair executions against the PyPI 0.8.4 matrix engine.

Cells are fixture x receiver (graph, seeded reordered half view) x degree
(3, 5) x threads (1, 6); arms are the three Rust executions through
``target/release/pgr-bench-pairs`` and ``relationship_pairs`` on the 0.8.4
wheel through ``_pair_baseline_child.py`` under ``--baseline-python``.
Every arm runs in a fresh process, repetitions interleaved so host drift
lands on every arm alike.  Before any timing, each cell's blocks from every
Rust execution are compared element for element with the baseline's dump; a
difference aborts the run.

    pixi run cargo build --release
    pixi run python benchmarks/bench_pair_qualification.py --repeat 5 \
        --baseline-python <venv>/bin/python \
        --out benchmarks/reports/pair_qualification.json

Peak RSS is each process's ``VmHWM`` after the call minus before it, so the
baseline is charged for its sparse products and the executions for their
chunks and blocks, and neither for building the graph.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from _harness import Environment
from _pair_common import CODES, load_dump
from _pair_fixtures import BINARY, EXECUTIONS, HERE, Inputs

BASELINE = "python_0.8.4"


def run_arm(
    arm: str, inputs: Inputs, receiver: str, degree: int, threads: int, baseline_python: Path, dump: Path | None
) -> dict:
    """One fresh-process run of *arm*, as its JSON record plus the engine RSS delta."""
    if arm == BASELINE:
        cmd = [
            str(baseline_python), str(HERE / "_pair_baseline_child.py"), "--columns", str(inputs.columns),
            "--receiver", receiver, "--degree", str(degree), "--threads", str(threads),
        ]  # fmt: skip
    else:
        cmd = [str(BINARY), str(inputs.tsv), "--execution", arm, "--threads", str(threads), "--max-degree", str(degree)]
        if receiver == "view":
            cmd += ["--view", str(inputs.view)]
    if dump is not None:
        cmd += ["--dump", str(dump)]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise SystemExit(f"{' '.join(cmd)} failed:\n{proc.stderr[-2000:]}")
    record = json.loads(proc.stdout)
    record["engine_rss_mib"] = round(record["peak_rss_mib"] - record["baseline_rss_mib"], 1)
    return record


def check_elementwise(name: str, inputs: Inputs, receiver: str, degree: int, baseline_python: Path, work: Path) -> int:
    """Every execution's dump must equal the baseline's, block by block; return the pair count."""
    base_dir = work / "dump" / name / receiver / str(degree) / BASELINE
    base_record = run_arm(BASELINE, inputs, receiver, degree, 2, baseline_python, base_dir)
    base = load_dump(base_dir)
    for execution in EXECUTIONS:
        out_dir = work / "dump" / name / receiver / str(degree) / execution
        run_arm(execution, inputs, receiver, degree, 2, baseline_python, out_dir)
        got = load_dump(out_dir)
        bad = [
            code
            for code in CODES
            if not (np.array_equal(got[code][0], base[code][0]) and np.array_equal(got[code][1], base[code][1]))
        ]
        if bad:
            raise SystemExit(f"{name}/{receiver}/degree {degree}/{execution} differs from 0.8.4 in {bad}")
    print(f"element-for-element ok: {name}/{receiver}/degree {degree}, {base_record['pairs']:,} pairs", flush=True)
    return base_record["pairs"]


def summarise(runs: list[dict]) -> list[dict]:
    """Medians and ranges per (fixture, receiver, degree, threads, arm), with baseline ratios."""
    cells: dict[tuple, list[dict]] = {}
    for run in runs:
        cells.setdefault(
            (run["fixture"], run["receiver"], run["max_degree"], run["threads"], run["execution"]), []
        ).append(run)
    rows = []
    for (fixture, receiver, degree, threads, arm), group in sorted(cells.items()):
        secs = [r["seconds"] for r in group]
        rss = [r["engine_rss_mib"] for r in group]
        rows.append(
            {
                "fixture": fixture,
                "receiver": receiver,
                "degree": degree,
                "threads": threads,
                "arm": arm,
                "runs": len(group),
                "pairs": group[0]["pairs"],
                "seconds_median": round(statistics.median(secs), 3),
                "seconds_min": min(secs),
                "seconds_max": max(secs),
                "engine_rss_mib_median": round(statistics.median(rss), 1),
                "engine_rss_mib_max": max(rss),
            }
        )
    by_cell = {(r["fixture"], r["receiver"], r["degree"], r["threads"], r["arm"]): r for r in rows}
    for r in rows:
        base = by_cell.get((r["fixture"], r["receiver"], r["degree"], r["threads"], BASELINE))
        if base is not None and base["seconds_median"] > 0:
            r["wall_vs_baseline"] = round(r["seconds_median"] / base["seconds_median"], 3)
            r["rss_vs_baseline"] = round(r["engine_rss_mib_median"] / max(base["engine_rss_mib_median"], 0.1), 3)
    return rows


def render(rows: list[dict]) -> str:
    """The cells as a markdown table."""
    lines = [
        "| fixture | receiver | degree | threads | arm | pairs | wall median (s) | wall range | wall / 0.8.4 | engine RSS median (MiB) | RSS / 0.8.4 |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    lines.extend(
        f"| {r['fixture']} | {r['receiver']} | {r['degree']} | {r['threads']} | {r['arm']} | {r['pairs']:,} | "
        f"{r['seconds_median']:.3f} | {r['seconds_min']:.3f} to {r['seconds_max']:.3f} | {r.get('wall_vs_baseline', float('nan')):.3f} | "
        f"{r['engine_rss_mib_median']:.1f} | {r.get('rss_vs_baseline', float('nan')):.3f} |"
        for r in rows
    )
    return "\n".join(lines)


def main() -> None:
    """Dump, prove equality with 0.8.4, run the cells, write the report."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--baseline-python", type=Path, required=True, help="interpreter with pedigree-graph 0.8.4")
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--fixtures", nargs="+", default=["random_30k", "random_300k"])
    ap.add_argument("--receivers", nargs="+", default=["graph", "view"], choices=["graph", "view"])
    ap.add_argument("--degrees", nargs="+", type=int, default=[3, 5])
    ap.add_argument("--threads", nargs="+", type=int, default=[1, 6])
    ap.add_argument("--arms", nargs="+", default=[BASELINE, *EXECUTIONS])
    ap.add_argument("--work-dir", type=Path, default=None)
    args = ap.parse_args()
    if not BINARY.exists():
        raise SystemExit(f"{BINARY} missing: run `pixi run cargo build --release`")
    work = args.work_dir or Path(tempfile.mkdtemp(prefix="pair-qualification-"))
    work.mkdir(parents=True, exist_ok=True)

    inputs = {name: Inputs(name, work) for name in args.fixtures}
    for name in args.fixtures:
        inputs[name].graph = None
    equality = {}
    for name in args.fixtures:
        for receiver in args.receivers:
            for degree in args.degrees:
                equality[f"{name}/{receiver}/{degree}"] = check_elementwise(
                    name, inputs[name], receiver, degree, args.baseline_python, work
                )

    runs: list[dict] = []
    for rep in range(args.repeat):
        for name in args.fixtures:
            for receiver in args.receivers:
                for degree in args.degrees:
                    for threads in args.threads:
                        for arm in args.arms:
                            record = run_arm(arm, inputs[name], receiver, degree, threads, args.baseline_python, None)
                            record.update({"fixture": name, "receiver": receiver, "repeat": rep})
                            runs.append(record)
                            print(
                                f"[{rep}] {name}/{receiver}/d{degree}/t{threads}/{arm}: {record['seconds']:.3f}s, "
                                f"engine +{record['engine_rss_mib']:.1f} MiB",
                                flush=True,
                            )
    digests: dict[tuple, dict] = {}
    for run in runs:
        key = (run["fixture"], run["receiver"], run["max_degree"])
        if key in digests and digests[key] != run["blocks"]:
            raise SystemExit(f"digest mismatch within {key}: {run['execution']} at {run['threads']} threads")
        digests.setdefault(key, run["blocks"])

    rows = summarise(runs)
    report = {
        "benchmark": "pair_qualification",
        "environment": Environment.capture(Path(__file__)).as_dict(),
        "arguments": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "elementwise_equal_pairs": equality,
        "cells": rows,
        "runs": runs,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    table = render(rows)
    args.out.with_suffix(".md").write_text(table + "\n")
    print(table)


if __name__ == "__main__":
    main()
