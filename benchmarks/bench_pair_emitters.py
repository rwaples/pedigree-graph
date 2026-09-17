r"""Slice 12 stage A: screen the three experimental Rust pair executions.

The arms are not Python callables but ``target/release/pgr-bench-pairs``, so
this driver spawns that binary in fresh, interleaved processes instead of
declaring a :class:`_harness.Suite`; it still records the harness
:class:`_harness.Environment` and keeps every raw run in the JSON report.
Peak RSS is the child's ``VmHWM`` read by the binary itself, before and
after the engine call, so the engine's own peak above the loaded columns is
what the table reports.

    pixi run cargo build --release
    pixi run python benchmarks/bench_pair_executions.py --repeat 3 \\
        --out benchmarks/reports/pair_executions_screening.json

Before timing anything the driver proves the executions agree with the Python
matrix oracle: for the ``--parity`` fixtures it computes
``relationship_pairs(max_degree=5)`` on the graph and on a deterministic
half view and compares per-block counts and digests with every execution's
output.  The digest is ``PairBlock::digest`` re-expressed in wrapping
``uint64`` NumPy arithmetic (``_pair_common.digest``).  A mismatch aborts the run.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import tempfile
from pathlib import Path

from _harness import Environment
from _pair_common import block_summary
from _pair_fixtures import BINARY, EXECUTIONS, Inputs

from pedigree_graph import RELATIONSHIPS


def run_binary(tsv: Path, execution: str, threads: int, view: Path | None, degree: int) -> dict:
    """One fresh-process run of the benchmark binary, as its JSON record."""
    cmd = [str(BINARY), str(tsv), "--execution", execution, "--threads", str(threads), "--max-degree", str(degree)]
    if view is not None:
        cmd += ["--view", str(view)]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    record = json.loads(proc.stdout)
    record["engine_rss_mib"] = round(record["peak_rss_mib"] - record["baseline_rss_mib"], 1)
    return record


def check_parity(name: str, inputs: Inputs) -> dict[str, dict[str, list[int]]]:
    """Every execution's blocks must equal the in-tree oracle's on the graph and on the view."""
    graph = inputs.graph
    expected = {
        "graph": block_summary(graph.relationship_pairs(max_degree=5)),
        "view": block_summary(graph.view(rows=inputs.view_rows).relationship_pairs(max_degree=5)),
    }
    for receiver, want in expected.items():
        for execution in EXECUTIONS:
            got = run_binary(inputs.tsv, execution, 2, inputs.view if receiver == "view" else None, 5)["blocks"]
            bad = [code for code in RELATIONSHIPS if got[code] != want[code]]
            if bad:
                detail = ", ".join(f"{c}: rust {got[c]} oracle {want[c]}" for c in bad)
                raise SystemExit(f"{name}/{receiver}/{execution} disagrees with the oracle: {detail}")
        print(f"parity ok: {name}/{receiver}, {sum(v[0] for v in want.values())} pairs", flush=True)
    return expected


def summarise(runs: list[dict]) -> list[dict]:
    """Medians and ranges per (fixture, receiver, threads, execution) cell."""
    cells: dict[tuple, list[dict]] = {}
    for run in runs:
        cells.setdefault((run["fixture"], run["receiver"], run["threads"], run["execution"]), []).append(run)
    rows = []
    for (fixture, receiver, threads, execution), group in sorted(cells.items()):
        secs = [r["seconds"] for r in group]
        rss = [r["engine_rss_mib"] for r in group]
        peak = [r["peak_rss_mib"] for r in group]
        rows.append(
            {
                "fixture": fixture,
                "receiver": receiver,
                "threads": threads,
                "execution": execution,
                "runs": len(group),
                "pairs": group[0]["pairs"],
                "seconds_median": round(statistics.median(secs), 3),
                "seconds_min": min(secs),
                "seconds_max": max(secs),
                "engine_rss_mib_median": round(statistics.median(rss), 1),
                "engine_rss_mib_max": max(rss),
                "peak_rss_mib_median": round(statistics.median(peak), 1),
            }
        )
    return rows


def render(rows: list[dict]) -> str:
    """The cells as a markdown table."""
    lines = [
        "| fixture | receiver | threads | execution | pairs | wall median (s) | wall range | engine RSS median (MiB) | engine RSS max | process peak (MiB) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    lines.extend(
        f"| {r['fixture']} | {r['receiver']} | {r['threads']} | {r['execution']} | {r['pairs']:,} | "
        f"{r['seconds_median']:.3f} | {r['seconds_min']:.3f} to {r['seconds_max']:.3f} | "
        f"{r['engine_rss_mib_median']:.1f} | {r['engine_rss_mib_max']:.1f} | {r['peak_rss_mib_median']:.1f} |"
        for r in rows
    )
    return "\n".join(lines)


def main() -> None:
    """Dump, check parity, run the cells, write the report."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--fixtures", nargs="+", default=["random_30k", "random_300k"])
    ap.add_argument("--parity", nargs="*", default=["random_30k"], help="fixtures checked against the Python oracle")
    ap.add_argument("--threads", nargs="+", type=int, default=[1, 6])
    ap.add_argument("--executions", nargs="+", default=list(EXECUTIONS))
    ap.add_argument("--receivers", nargs="+", default=["graph"], choices=["graph", "view"])
    ap.add_argument("--degree", type=int, default=5)
    ap.add_argument("--work-dir", type=Path, default=None, help="where fixture dumps go (default: a temp dir)")
    args = ap.parse_args()
    if not BINARY.exists():
        raise SystemExit(f"{BINARY} missing: run `pixi run cargo build --release`")

    work = args.work_dir or Path(tempfile.mkdtemp(prefix="pair-executions-"))
    work.mkdir(parents=True, exist_ok=True)
    inputs = {name: Inputs(name, work) for name in args.fixtures}
    parity = {name: check_parity(name, inputs[name]) for name in args.fixtures if name in args.parity}
    for name in args.fixtures:
        inputs[name].graph = None

    runs: list[dict] = []
    for rep in range(args.repeat):
        for name in args.fixtures:
            for receiver in args.receivers:
                view = inputs[name].view if receiver == "view" else None
                for threads in args.threads:
                    for execution in args.executions:
                        record = run_binary(inputs[name].tsv, execution, threads, view, args.degree)
                        record.update({"fixture": name, "receiver": receiver, "repeat": rep})
                        runs.append(record)
                        print(
                            f"[{rep}] {name}/{receiver}/t{threads}/{execution}: {record['seconds']:.3f}s, "
                            f"engine +{record['engine_rss_mib']:.1f} MiB, pairs {record['pairs']:,}",
                            flush=True,
                        )
    digests: dict[tuple, dict] = {}
    for run in runs:
        key = (run["fixture"], run["receiver"])
        if key in digests and digests[key] != run["blocks"]:
            raise SystemExit(f"digest mismatch within {key}: {run['execution']} at {run['threads']} threads")
        digests.setdefault(key, run["blocks"])

    rows = summarise(runs)
    report = {
        "benchmark": "pair_executions_screening",
        "environment": Environment.capture(Path(__file__)).as_dict(),
        "arguments": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "parity": parity,
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
