r"""Slice 12 stage A: screen the three experimental Rust pair executions.

The arms are not Python callables but ``target/release/pgr-bench-pairs``, so
this driver spawns that binary in fresh, interleaved processes instead of
declaring a :class:`_harness.Suite`; it still records the harness
:class:`_harness.Environment` and keeps every raw run in the JSON report.
Peak RSS is the child's ``VmHWM`` read by the binary itself, before and
after the engine call, so the engine's own peak above the loaded columns is
what the table reports.

    pixi run cargo build --release
    pixi run python benchmarks/bench_pair_emitters.py --repeat 3 \\
        --out benchmarks/reports/pair_executions_screening.json

``--baseline-binary`` gates this checkout's binary (or ``--binary``) against
another checkout's: every cell runs both, interleaved, and each candidate
cell gets a :func:`_harness.compare` verdict on wall and engine RSS against
its base twin, at :data:`_harness.BUILD_GATE_REPEATS` repetitions by
default.  Both binaries must print the same block digests.  Every run
records the binary it ran and its hash, so a report cannot hide which build
it timed.  Exits 1 on a block or a digest mismatch.

Before timing anything the driver proves the executions agree with the Python
matrix oracle: for the ``--parity`` fixtures it computes
``relationship_pairs(max_degree=5)`` on the graph and on a deterministic
half view and compares per-block counts and digests with every execution's
output.  The digest is ``PairBlock::digest`` re-expressed in wrapping
``uint64`` NumPy arithmetic (``_pair_common.digest``).  A mismatch aborts the run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import tempfile
from pathlib import Path

from _harness import BUILD_GATE_REPEATS, Environment, Verdict, compare
from _pair_common import block_summary
from _pair_fixtures import BINARY, EXECUTIONS, Inputs

from pedigree_graph import RELATIONSHIPS

GATED = (("seconds", "wall"), ("engine_rss_mib", "engine RSS"))
"""The per-run metrics a candidate cell is gated on against its base twin."""


def binary_identity(binary: Path) -> dict[str, str]:
    """The path, hash and source tree ``git describe`` of one benchmark binary."""
    describe = subprocess.run(
        ["git", "-C", str(binary.parent), "describe", "--tags", "--always", "--dirty", "--abbrev=12"],
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "binary": str(binary),
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest()[:16],
        "binary_git": describe.stdout.strip() if describe.returncode == 0 else "",
    }


def run_binary(binary: Path, tsv: Path, execution: str, threads: int, view: Path | None, degree: int) -> dict:
    """One fresh-process run of a benchmark binary, as its JSON record."""
    cmd = [str(binary), str(tsv), "--execution", execution, "--threads", str(threads), "--max-degree", str(degree)]
    if view is not None:
        cmd += ["--view", str(view)]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    record = json.loads(proc.stdout)
    record["engine_rss_mib"] = round(record["peak_rss_mib"] - record["baseline_rss_mib"], 1)
    return record


def check_parity(name: str, inputs: Inputs, binaries: dict[str, Path]) -> dict[str, dict[str, list[int]]]:
    """Every binary's every execution must return the in-tree oracle's blocks on the graph and on the view."""
    graph = inputs.graph
    expected = {
        "graph": block_summary(graph.relationship_pairs(max_degree=5)),
        "view": block_summary(graph.view(rows=inputs.view_rows).relationship_pairs(max_degree=5)),
    }
    for receiver, want in expected.items():
        for arm, binary in binaries.items():
            for execution in EXECUTIONS:
                view = inputs.view if receiver == "view" else None
                got = run_binary(binary, inputs.tsv, execution, 2, view, 5)["blocks"]
                bad = [code for code in RELATIONSHIPS if got[code] != want[code]]
                if bad:
                    detail = ", ".join(f"{c}: rust {got[c]} oracle {want[c]}" for c in bad)
                    raise SystemExit(f"{name}/{receiver}/{execution} ({arm}) disagrees with the oracle: {detail}")
        print(f"parity ok: {name}/{receiver}, {sum(v[0] for v in want.values())} pairs", flush=True)
    return expected


def _cell(run: dict) -> tuple:
    return (run["fixture"], run["receiver"], run["threads"], run["execution"])


def summarise(runs: list[dict], min_repeats: int) -> list[dict]:
    """Medians and ranges per (fixture, receiver, threads, execution, arm) cell, and each candidate's verdict."""
    cells: dict[tuple, list[dict]] = {}
    for run in runs:
        cells.setdefault((*_cell(run), run["arm"]), []).append(run)
    rows = []
    for (fixture, receiver, threads, execution, arm), group in sorted(cells.items()):
        secs = [r["seconds"] for r in group]
        rss = [r["engine_rss_mib"] for r in group]
        peak = [r["peak_rss_mib"] for r in group]
        base = cells.get((fixture, receiver, threads, execution, "base")) if arm == "candidate" else None
        verdict = None
        if base is not None:
            verdicts = {compare([r[k] for r in group], [r[k] for r in base], min_repeats) for k, _ in GATED}
            verdict = next(v for v in (Verdict.BLOCK, Verdict.INCONCLUSIVE, Verdict.PASS) if v in verdicts)
        rows.append(
            {
                "fixture": fixture,
                "receiver": receiver,
                "threads": threads,
                "execution": execution,
                "arm": arm,
                "binary_sha256": group[0]["binary_sha256"],
                "verdict": None if verdict is None else str(verdict),
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
        "| fixture | receiver | threads | execution | arm (binary) | verdict | pairs | wall median (s) | wall range | engine RSS median (MiB) | engine RSS max | process peak (MiB) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    lines.extend(
        f"| {r['fixture']} | {r['receiver']} | {r['threads']} | {r['execution']} | "
        f"{r['arm']} (`{r['binary_sha256']}`) | {r['verdict'] or ''} | {r['pairs']:,} | "
        f"{r['seconds_median']:.3f} | {r['seconds_min']:.3f} to {r['seconds_max']:.3f} | "
        f"{r['engine_rss_mib_median']:.1f} | {r['engine_rss_mib_max']:.1f} | {r['peak_rss_mib_median']:.1f} |"
        for r in rows
    )
    return "\n".join(lines)


def main() -> None:
    """Dump, check parity, run the cells, write the report."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--repeat", type=int, help=f"default 3, or {BUILD_GATE_REPEATS} with --baseline-binary")
    ap.add_argument("--binary", type=Path, default=BINARY, help="the candidate binary (default: this checkout's)")
    ap.add_argument("--baseline-binary", type=Path, help="gate the candidate against this binary (another checkout's)")
    ap.add_argument("--fixtures", nargs="+", default=["random_30k", "random_300k"])
    ap.add_argument("--parity", nargs="*", default=["random_30k"], help="fixtures checked against the Python oracle")
    ap.add_argument("--threads", nargs="+", type=int, default=[1, 6])
    ap.add_argument("--executions", nargs="+", default=list(EXECUTIONS))
    ap.add_argument("--receivers", nargs="+", default=["graph"], choices=["graph", "view"])
    ap.add_argument("--degree", type=int, default=5)
    ap.add_argument("--work-dir", type=Path, default=None, help="where fixture dumps go (default: a temp dir)")
    args = ap.parse_args()
    binaries = {"candidate": args.binary.resolve()}
    if args.baseline_binary is not None:
        binaries = {"base": args.baseline_binary.resolve(), **binaries}
    for binary in binaries.values():
        if not binary.exists():
            raise SystemExit(f"{binary} missing: run `pixi run cargo build --release` in its checkout")
    builds = {arm: binary_identity(binary) for arm, binary in binaries.items()}
    same_build = len({build["binary_sha256"] for build in builds.values()}) < len(builds)
    repeat = args.repeat or (BUILD_GATE_REPEATS if args.baseline_binary is not None else 3)
    if args.baseline_binary is not None and repeat < BUILD_GATE_REPEATS:
        raise SystemExit(f"a build gate needs --repeat >= {BUILD_GATE_REPEATS}, got {repeat}")

    work = args.work_dir or Path(tempfile.mkdtemp(prefix="pair-executions-"))
    work.mkdir(parents=True, exist_ok=True)
    inputs = {name: Inputs(name, work) for name in args.fixtures}
    parity = {name: check_parity(name, inputs[name], binaries) for name in args.fixtures if name in args.parity}
    for name in args.fixtures:
        inputs[name].graph = None

    runs: list[dict] = []
    for rep in range(repeat):
        for name in args.fixtures:
            for receiver in args.receivers:
                view = inputs[name].view if receiver == "view" else None
                for threads in args.threads:
                    for execution in args.executions:
                        # Alternate which arm runs first, so a warm second run favours neither.
                        order = list(binaries.items())
                        for arm, binary in order if rep % 2 == 0 else reversed(order):
                            record = run_binary(binary, inputs[name].tsv, execution, threads, view, args.degree)
                            record.update({"fixture": name, "receiver": receiver, "repeat": rep, "arm": arm})
                            record.update(builds[arm])
                            runs.append(record)
                            print(
                                f"[{rep}] {name}/{receiver}/t{threads}/{execution}/{arm}: {record['seconds']:.3f}s, "
                                f"engine +{record['engine_rss_mib']:.1f} MiB, pairs {record['pairs']:,}",
                                flush=True,
                            )
    digests: dict[tuple, dict] = {}
    for run in runs:
        key = (run["fixture"], run["receiver"])
        if key in digests and digests[key] != run["blocks"]:
            raise SystemExit(
                f"digest mismatch within {key}: {run['execution']} at {run['threads']} threads ({run['arm']})"
            )
        digests.setdefault(key, run["blocks"])

    rows = summarise(runs, BUILD_GATE_REPEATS)
    report = {
        "benchmark": "pair_executions_screening",
        "environment": Environment.capture(Path(__file__)).as_dict(),
        "builds": builds,
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
    blocked = [row for row in rows if row["verdict"] == Verdict.BLOCK]
    for row in blocked:
        print(f"BLOCK: {_cell(row)} candidate regressed beyond the gate with disjoint ranges")
    for row in rows:
        if row["verdict"] == Verdict.INCONCLUSIVE:
            print(f"inconclusive: {_cell(row)} (overlapping ranges past the gate)")
    if same_build:
        print(f"WRONG BUILD: base and candidate are the same binary: {builds}")
    raise SystemExit(1 if blocked or same_build else 0)


if __name__ == "__main__":
    main()
