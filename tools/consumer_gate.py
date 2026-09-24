#!/usr/bin/env python
"""Run pedigree-graph's release gate across the family and record evidence.

All thirteen family check units of simACE's ``tools/family_repos.py`` are
covered, which ``tests/test_consumer_gate_covers_family.py`` enforces.  Each
unit runs from its own pixi manifest; the ``--routing`` argument, not the
consumer locks, decides which pedigree-graph build the consumers import, and
every routed unit starts by asserting where ``pedigree_graph.__file__`` lives.

Before a release, build the candidate and route the consumers through it::

    pixi run --frozen python external/pedigree-graph/tools/consumer_gate.py run --stage 0.11.0-rc --wheel-ref HEAD

``--wheel-ref`` first runs the build stage: a clean worktree at the ref, the
wheel and sdist built there (``cp313-abi3`` tag and version checked), each
installed into its own venv and import-checked, the package's own fast suite
run against the installed wheel from outside the tree, and the wheel
installed with ``pip install --target`` into ``<work>/wheel-site``.  Any
failure there stops the run before the consumer units, which then import from
that site.  After the release is on PyPI and the consumers are relocked::

    pixi run python external/pedigree-graph/tools/consumer_gate.py run --stage 0.11.0 --routing locked

Other routings: ``--routing source`` (the checkout on ``PYTHONPATH``) or
``--routing <dir>`` (an existing ``--target`` install).  ``--unit`` picks
units, ``--slow`` adds slow-marked steps, ``list`` prints the units, and
``summary --stage S`` prints one line per recorded unit.

Records go to ``external/pedigree-graph/docs/release-gates/<stage>/``: one
JSON per unit with its per-step logs beside it, plus ``build.json`` from the
build stage; reruns overwrite only what they ran.  Build products stay under
``target/consumer-gate/<stage>`` (``--work`` overrides).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib
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
EVIDENCE = PG_SOURCE / "docs" / "release-gates"
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
                Step("test-all", ("test-all",)),
                Step("test-rust", ("test-rust",)),
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


# The installed-artifact import check: the package and its native module resolve
# inside the venv's site-packages, ship their typing files, agree on the
# version, and expose every root name and public submodule.
CHECK_INSTALL = """
import importlib.metadata as m, pathlib, sys
import pedigree_graph as p
venv = sys.argv[1]
file = pathlib.Path(p.__file__).resolve()
assert str(file).startswith(str(pathlib.Path(venv).resolve()) + "/"), file
assert "site-packages" in file.parts, file
assert (file.parent / "py.typed").is_file(), "py.typed missing"
assert (file.parent / "_native.pyi").is_file(), "_native.pyi missing"
import pedigree_graph._native as native
assert str(pathlib.Path(native.__file__).resolve()).startswith(str(file.parent) + "/"), native.__file__
version = m.version("pedigree-graph")
assert native.core_version() == version, (native.core_version(), version)
names = [getattr(p, n) for n in p.__all__]
import pedigree_graph.relationships, pedigree_graph.summaries, pedigree_graph.effective_size, pedigree_graph.typing
print("installed", version, file, native.__file__, len(names), "root names")
"""


class BuildFailed(SystemExit):
    """A build-stage check failed; the consumer units do not run."""


def _sh(argv: list[str | Path], **kwargs) -> subprocess.CompletedProcess:
    print("+", shlex.join(str(a) for a in argv), flush=True)
    return subprocess.run([str(a) for a in argv], check=True, **kwargs)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _only(pattern: str, where: Path) -> Path:
    found = sorted(where.glob(pattern))
    if len(found) != 1:
        raise BuildFailed(f"expected one {pattern} under {where}, found {[p.name for p in found]}")
    return found[0]


def _check_install(py: Path, venv: Path, artifact: Path, env: dict[str, str]) -> str:
    shutil.rmtree(venv, ignore_errors=True)
    _sh([py, "-m", "venv", venv], env=env)
    _sh([venv / "bin" / "pip", "install", "--quiet", artifact], env=env)
    # cwd is the work dir: ``python -c`` puts the cwd on sys.path, and from the
    # repository root that would import the source package instead.
    out = _sh(
        [venv / "bin" / "python", "-c", CHECK_INSTALL, venv],
        cwd=venv.parent,
        env=env,
        capture_output=True,
        text=True,
    )
    print(out.stdout, end="")
    return out.stdout.strip()


def build_stage(ref: str, work: Path, stage: str) -> Path:
    """Build and check the wheel at *ref*; return the ``--target`` site the consumers import."""
    py = PG_SOURCE / ".pixi" / "envs" / "default" / "bin" / "python"
    if not py.is_file():
        raise BuildFailed(f"no interpreter at {py} (pixi install in {PG_SOURCE} first)")
    # The pixi env carries cargo and maturin.  The build must not inherit a
    # test-hooks requirement: installed artifacts never carry the feature.
    env = {k: v for k, v in os.environ.items() if k != "PEDIGREE_GRAPH_REQUIRE_TEST_HOOKS"}
    env["PATH"] = f"{py.parent}{os.pathsep}{env.get('PATH', '')}"
    if shutil.which("cargo", path=env["PATH"]) is None:
        raise BuildFailed(f"no cargo on PATH after prepending {py.parent}")

    work.mkdir(parents=True, exist_ok=True)
    clean = work / "clean"
    if clean.exists():
        _sh(["git", "-C", PG_SOURCE, "worktree", "remove", "--force", clean])
    _sh(["git", "-C", PG_SOURCE, "worktree", "add", "--detach", clean, ref])
    head = _sh(["git", "-C", clean, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if _sh(["git", "-C", clean, "status", "--porcelain"], capture_output=True, text=True).stdout:
        raise BuildFailed("clean worktree is dirty")
    version = tomllib.loads((clean / "Cargo.toml").read_text())["workspace"]["package"]["version"]
    print(f"building version {version} from {head}", flush=True)

    dist, build_venv = work / "dist", work / "build-venv"
    shutil.rmtree(dist, ignore_errors=True)
    shutil.rmtree(build_venv, ignore_errors=True)
    _sh([py, "-m", "venv", build_venv], env=env)
    _sh([build_venv / "bin" / "pip", "install", "--quiet", "build"], env=env)
    _sh([build_venv / "bin" / "python", "-m", "build", "--outdir", dist], cwd=clean, env=env)
    wheel, sdist = _only("*.whl", dist), _only("*.tar.gz", dist)
    if not wheel.name.startswith(f"pedigree_graph-{version}-cp313-abi3-"):
        raise BuildFailed(f"wheel {wheel.name} is not pedigree_graph-{version}-cp313-abi3-*")
    if sdist.name != f"pedigree_graph-{version}.tar.gz":
        raise BuildFailed(f"sdist {sdist.name} does not carry version {version}")

    wheel_venv = work / "wheel-venv"
    installs = {
        "wheel": _check_install(py, wheel_venv, wheel, env),
        "sdist": _check_install(py, work / "sdist-venv", sdist, env),
    }

    # The package's own fast suite against the installed wheel.  cwd is the work
    # dir, outside the clean tree, so neither pytest nor the fresh child processes
    # the thread tests spawn can pick up the source package over site-packages;
    # the fixtures resolve relative to the test files.
    _sh([wheel_venv / "bin" / "pip", "install", "--quiet", f"pedigree-graph[test]@file://{wheel}"], env=env)
    log = work / "wheel-pytest.log"
    with log.open("w") as fh:
        pytest_rc = subprocess.run(
            [
                str(wheel_venv / "bin" / "python"),
                *("-m", "pytest", "-q", "-p", "no:cacheprovider", "-m", "not slow"),
                *("--rootdir", str(clean), "-c", str(clean / "pyproject.toml"), str(clean / "tests")),
            ],
            cwd=work,
            env=env,
            stdout=fh,
            stderr=subprocess.STDOUT,
            check=False,
        ).returncode
    pytest_tail = log.read_text(errors="replace").splitlines()[-3:]
    print("\n".join(pytest_tail), flush=True)

    site = work / "wheel-site"
    shutil.rmtree(site, ignore_errors=True)
    _sh([py, "-m", "pip", "install", "--quiet", "--no-deps", "--target", site, wheel], env=env)

    record = {
        "ref": ref,
        "head": head,
        "version": version,
        "wheel": {"file": wheel.name, "sha256": _sha256(wheel)},
        "sdist": {"file": sdist.name, "sha256": _sha256(sdist)},
        "installs": installs,
        "wheel_pytest_exit": pytest_rc,
        "wheel_pytest_tail": pytest_tail,
        "wheel_site": str(site),
    }
    out = EVIDENCE / stage
    out.mkdir(parents=True, exist_ok=True)
    (out / "build.json").write_text(json.dumps(record, indent=2) + "\n")
    if pytest_rc != 0:
        raise BuildFailed(f"the package suite failed against the installed wheel (exit {pytest_rc}); see {log}")
    return site


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
    route = r.add_mutually_exclusive_group(required=True)
    route.add_argument("--routing", help="'source', 'locked', or a directory holding an installed wheel")
    route.add_argument("--wheel-ref", help="build and check the wheel at this git ref, then route through it")
    r.add_argument("--work", type=Path, help="build-stage work dir (default: target/consumer-gate/<stage>)")
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
    routing = args.routing
    if args.wheel_ref:
        work = (args.work or PG_SOURCE / "target" / "consumer-gate" / args.stage).resolve()
        routing = str(build_stage(args.wheel_ref, work, args.stage))
    with tempfile.TemporaryDirectory(prefix="consumer-gate-") as tmp:
        records = [run_unit(unit, routing, args.stage, args.slow, Path(tmp)) for unit in selected]
    failed = [r["unit"] for r in records if not r["ok"]]
    print("failed units:", failed or "none")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
