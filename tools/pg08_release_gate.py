#!/usr/bin/env python
"""Run a pedigree-graph release gate across the family and record evidence.

Written for the 0.8.0 release and reused for every one since; 0.9.0 is the
current subject.  All thirteen family check units of ``tools/family_repos.py``
are covered, which ``tests/test_consumer_gate_covers_family.py`` enforces.

Each unit runs from its own pixi manifest with ``--frozen`` (the consumer locks
still pin the previous pedigree-graph until the new wheel is on PyPI, so the
``--routing`` argument, not the lock, decides which build is under test).
Consumers are routed to a pedigree-graph build via
``PYTHONPATH`` and every unit starts with an assertion that the routed
``pedigree_graph.__file__`` lives where the run says it does::

    # 9a: consumers import the source checkout
    pixi run --frozen python external/pedigree-graph/tools/pg08_release_gate.py run --stage 9a --routing source

    # 9b: consumers import an installed wheel (pip install --target <dir>)
    pixi run --frozen python external/pedigree-graph/tools/pg08_release_gate.py run --stage 9b --routing /path/to/wheel-site

    # 9c: consumers import the locked env (no PYTHONPATH)
    pixi run python external/pedigree-graph/tools/pg08_release_gate.py run --stage 9c --routing locked

    pixi run python external/pedigree-graph/tools/pg08_release_gate.py run --stage 9a --unit fitACE_tetraher --slow
    pixi run python external/pedigree-graph/tools/pg08_release_gate.py list

One JSON record per unit goes to ``external/pedigree-graph/docs/pedigree-graph-0.8-migration/gate/<stage>/``
with the full per-step logs beside it; reruns overwrite only the units they ran.
``summary`` prints one line per recorded unit.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

# This gate lives in pedigree-graph but exercises its consumers, so it runs from
# a simACE umbrella checkout (external/pedigree-graph/tools/<this file>) and
# borrows the family membership list from the umbrella's tools/family_repos.py.
PG_SOURCE = Path(__file__).resolve().parents[1]
UMBRELLA = PG_SOURCE.parents[1]
sys.path.insert(0, str(UMBRELLA / "tools"))
from family_repos import ROOT  # noqa: E402

assert ROOT == UMBRELLA, f"family_repos.ROOT {ROOT} is not this checkout's umbrella {UMBRELLA}"
EVIDENCE = PG_SOURCE / "docs" / "pedigree-graph-0.8-migration" / "gate"
SMOKE = "results/test/small_test"
LOG_TAIL = 12


@dataclass(frozen=True)
class Step:
    """One command of a unit, run through the unit's pixi manifest."""

    name: str
    argv: tuple[str, ...]
    slow: bool = False
    """Run only with ``--slow``."""


@dataclass(frozen=True)
class Unit:
    """One family check unit: where it runs, which manifest, which steps."""

    label: str
    cwd: Path
    manifest: Path
    steps: tuple[Step, ...]
    routed: bool = True
    """``False`` where no step imports ``pedigree_graph``, so the routing assertion has
    nothing to assert: pedigree-graph itself (own editable manifest), the
    ``ace_iter_reml`` C++ binaries, and the ``tetraher_simace`` LDAK fork."""


def _pytest(*paths: str, extra: tuple[str, ...] = ()) -> tuple[str, ...]:
    return ("pytest", "-q", "-p", "no:cacheprovider", *extra, *paths)


def _fitace(label: str, subdir: str) -> Unit:
    return Unit(
        label,
        ROOT / "fitACE",
        ROOT / "fitACE" / "pixi.toml",
        (Step("pytest", _pytest(f"{subdir}/tests")),),
    )


def units() -> tuple[Unit, ...]:
    """The gate's units in run order (pedigree-graph first, consumers after)."""
    fitace = ROOT / "fitACE"
    pedsum = ROOT / "external" / "pedsum"
    smoke_ped = ROOT / SMOKE / "rep1" / "pedigree.parquet"
    return (
        Unit(
            "pedigree-graph",
            PG_SOURCE,
            PG_SOURCE / "pixi.toml",
            (
                Step("ruff", ("ruff", "check")),
                Step("format", ("ruff", "format", "--check")),
                Step("ty", ("ty", "check")),
                Step("pytest", _pytest("tests", extra=("-m", "not slow"))),
                Step("pytest-slow", _pytest("tests", extra=("-m", "slow")), slow=True),
            ),
            routed=False,
        ),
        Unit(
            "simACE",
            ROOT,
            ROOT / "pixi.toml",
            (
                Step("ruff", ("ruff", "check")),
                Step("format", ("ruff", "format", "--check")),
                Step("test", ("test",)),
                Step(
                    "smoke",
                    (
                        "snakemake",
                        "--cores",
                        "4",
                        "--forceall",
                        *(f"{SMOKE}/{t}.done" for t in ("scenario", "validate", "stats", "effective_size")),
                    ),
                ),
                Step("atlas", ("snakemake", "--cores", "4", "-f", f"{SMOKE}/plots/atlas.html")),
            ),
        ),
        Unit(
            "fitACE",
            fitace,
            fitace / "pixi.toml",
            (
                Step("ruff", ("ruff", "check")),
                Step("format", ("ruff", "format", "--check")),
                Step("pytest", _pytest("tests")),
            ),
        ),
        _fitace("fitACE_pcgc", "fitACE_pcgc"),
        _fitace("fitACE_iter_reml", "fitACE_iter_reml"),
        Unit(
            "ace_iter_reml",
            fitace / "fitACE_iter_reml" / "ace_iter_reml",
            fitace / "pixi.toml",
            tuple(
                Step(f"{b}-{t}", (f"./build-{b}/{t}",))
                for b in ("fp32", "fp64")
                for t in ("test_laplace_primitives", "test_mcem_step", "test_tmvn")
            ),
            routed=False,
        ),
        _fitace("fitACE_tetraher", "fitACE_tetraher"),
        Unit(
            "tetraher_simace",
            fitace / "tetraher_simace",
            fitace / "pixi.toml",
            (
                Step("ruff", ("ruff", "check")),
                Step("format", ("ruff", "format", "--check")),
                # LDAK's usage path exits 1, so the binary is probed for what it
                # prints and for the fork's own flag rather than for a zero exit.
                # Numerical equivalence with upstream is the fitACE unit's job
                # (fitACE/tests/tetraher/test_fork_equivalence.py).
                Step("ldak-runs", ("sh", "-c", './ldak6.2.simace 2>&1 | grep -q "LDAK - Software"')),
                # grep -a, not strings: binutils is not guaranteed in the env, and a
                # missing tool would exit 127 and read as "not the fork". Plain
                # grep -q returns 1 on binary input, so -a is load-bearing.
                Step("ldak-is-fork", ("sh", "-c", 'grep -qa -- "--simace-grouping" ldak6.2.simace')),
            ),
            routed=False,
        ),
        _fitace("fitACE_pafgrs", "fitACE_pafgrs"),
        Unit(
            "fitACE_stan",
            fitace / "fitACE_stan",
            fitace / "pixi.toml",
            (
                Step("ruff", ("ruff", "check")),
                Step("format", ("ruff", "format", "--check")),
                Step("import", ("python", "-c", "import fitace_stan, fitace, simace; print(fitace_stan.__version__)")),
            ),
        ),
        _fitace("fitACE_frailty", "fitACE_frailty"),
        Unit(
            "fitACE_epimight",
            fitace / "fitACE_epimight",
            fitace / "pixi.toml",
            (
                Step("pytest", _pytest("tests")),
                Step("pytest-slow", _pytest("tests", extra=("-m", "slow")), slow=True),
            ),
        ),
        Unit(
            "pedsum",
            pedsum,
            pedsum / "pixi.toml",
            (
                Step("ruff", ("ruff", "check")),
                Step("format", ("ruff", "format", "--check")),
                Step("pytest", _pytest("tests")),
                Step(
                    "tsv",
                    (
                        "python",
                        "-c",
                        f"import polars as pl; pl.read_parquet({str(smoke_ped)!r}).write_csv('{{tmp}}/pedigree.tsv', separator='\\t')",
                    ),
                ),
                Step(
                    "cli-smoke",
                    (
                        "python",
                        "pedigree_summary.py",
                        "summarize",
                        "--in",
                        "{tmp}/pedigree.tsv",
                        "--out",
                        "{tmp}/pedsum-smoke",
                    ),
                ),
            ),
        ),
    )


def routing_env(routing: str) -> tuple[dict[str, str], str]:
    """Return the consumer environment and the path prefix the import must resolve under."""
    if routing == "source":
        return {"PYTHONPATH": str(PG_SOURCE)}, str(PG_SOURCE) + os.sep
    if routing == "locked":
        return {}, ".pixi" + os.sep
    site = Path(routing).resolve()
    if not (site / "pedigree_graph").is_dir():
        raise SystemExit(f"--routing {routing}: no pedigree_graph package under {site}")
    return {"PYTHONPATH": str(site)}, str(site) + os.sep


def routing_check(prefix: str) -> Step:
    """A step that exits 3 unless ``pedigree_graph`` imports from under *prefix*."""
    code = (
        "import pedigree_graph as p, sys;"
        f"ok = {prefix!r} in p.__file__;"
        "print(('routed' if ok else 'MISROUTED'), p.__file__);"
        "sys.exit(0 if ok else 3)"
    )
    return Step("routing", ("python", "-c", code))


def run_step(unit: Unit, step: Step, env: dict[str, str], frozen: bool, log_dir: Path, tmp: Path) -> dict:
    """Run one step under ``/usr/bin/time``, log it, and return its evidence record."""
    argv = tuple(a.replace("{tmp}", str(tmp)) for a in step.argv)
    pixi = ["pixi", "run", "--manifest-path", str(unit.manifest)]
    if frozen:
        pixi.append("--frozen")
    stats = tmp / f"{unit.label}-{step.name}.time"
    cmd = ["/usr/bin/time", "-f", "%e %M", "-o", str(stats), *pixi, *argv]
    log = log_dir / f"{step.name}.log"
    started = time.time()
    with log.open("w") as fh:
        rc = subprocess.run(
            cmd, cwd=unit.cwd, env={**os.environ, **env}, stdout=fh, stderr=subprocess.STDOUT, check=False
        ).returncode
    wall = time.time() - started
    max_rss_kib = None
    if stats.exists():
        fields = stats.read_text().split()
        if len(fields) >= 2 and fields[-1].isdigit():
            max_rss_kib = int(fields[-1])
    tail = log.read_text(errors="replace").splitlines()[-LOG_TAIL:]
    return {
        "step": step.name,
        "cmd": shlex.join(pixi + list(argv)),
        "exit": rc,
        "wall_s": round(wall, 1),
        "max_rss_mib": None if max_rss_kib is None else round(max_rss_kib / 1024, 1),
        "log": str(log.relative_to(ROOT)),
        "tail": tail,
    }


def run_unit(unit: Unit, routing: str, stage: str, slow: bool, tmp: Path) -> dict:
    """Run a unit's routing check and steps; write and return its JSON record."""
    env, prefix = routing_env(routing) if unit.routed else ({}, "")
    frozen = routing != "locked"
    steps = [routing_check(prefix)] if unit.routed else []
    steps += [s for s in unit.steps if slow or not s.slow]
    log_dir = EVIDENCE / stage / unit.label
    log_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "unit": unit.label,
        "stage": stage,
        "routing": routing if unit.routed else "own-manifest",
        "cwd": str(unit.cwd.relative_to(ROOT)) or ".",
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "steps": [],
    }
    for step in steps:
        print(f"[{unit.label}] {step.name} ...", end="", flush=True)
        result = run_step(unit, step, env, frozen, log_dir, tmp)
        record["steps"].append(result)
        print(f" exit={result['exit']} wall={result['wall_s']}s rss={result['max_rss_mib']}MiB", flush=True)
        if step.name == "routing" and result["exit"] != 0:
            record["aborted"] = "misrouted"
            break
    record["ok"] = all(s["exit"] == 0 for s in record["steps"]) and "aborted" not in record
    (EVIDENCE / stage / f"{unit.label}.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def summary(stage: str) -> int:
    """Print one line per recorded unit; exit 1 if any failed."""
    rows = sorted((EVIDENCE / stage).glob("*.json"))
    if not rows:
        print(f"no records under {EVIDENCE / stage}")
        return 1
    worst = 0
    for path in rows:
        rec = json.loads(path.read_text())
        if "steps" not in rec:
            continue
        bad = [s["step"] for s in rec["steps"] if s["exit"] != 0]
        wall = sum(s["wall_s"] for s in rec["steps"])
        rss = max((s["max_rss_mib"] or 0) for s in rec["steps"])
        status = "ok" if rec["ok"] else f"FAIL {','.join(bad)}"
        print(f"{rec['unit']:<18} {status:<28} wall={wall:8.1f}s peak_rss={rss:8.1f}MiB routing={rec['routing']}")
        worst |= not rec["ok"]
    return worst


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="print the units and their steps")
    r = sub.add_parser("run", help="run units and write evidence records")
    r.add_argument("--stage", required=True, help="evidence subdirectory, e.g. 9a")
    r.add_argument("--routing", required=True, help="'source', 'locked', or a directory holding an installed wheel")
    r.add_argument("--unit", nargs="*", help="labels to run (default: all)")
    r.add_argument("--slow", action="store_true", help="include the slow-marked steps")
    s = sub.add_parser("summary", help="one line per recorded unit")
    s.add_argument("--stage", required=True)
    args = parser.parse_args(argv)

    if args.command == "list":
        for u in units():
            print(f"{u.label} ({u.cwd.relative_to(ROOT) or '.'})")
            for st in u.steps:
                print(f"    {st.name}{' [slow]' if st.slow else ''}: {shlex.join(st.argv)}")
        return 0
    if args.command == "summary":
        return summary(args.stage)

    selected = units()
    if args.unit:
        unknown = set(args.unit) - {u.label for u in selected}
        if unknown:
            raise SystemExit(f"unknown units: {sorted(unknown)}")
        selected = tuple(u for u in selected if u.label in args.unit)
    with tempfile.TemporaryDirectory(prefix="pg08-gate-") as tmp:
        records = [run_unit(unit, args.routing, args.stage, args.slow, Path(tmp)) for unit in selected]
    failed = [r["unit"] for r in records if not r["ok"]]
    print("failed units:", failed or "none")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
