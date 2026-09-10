"""Sweep pytest-xdist worker counts and schedulers for this repository's own test suite.

Answers GitHub issue #14: how many workers, and which ``--dist`` scheduler, should
``pixi run test`` use.  Nine configurations are crossed with two selections, each in a
fresh process through the harness, so the choice rests on medians rather than on one run.

    python benchmarks/bench_pytest_workers.py --repeat 3 --out benchmarks/reports/pytest_workers.json
    python benchmarks/bench_pytest_workers.py --render benchmarks/reports/pytest_workers.json

This is a sweep, not an A/B gate.  ``serial`` is the ratio baseline and nothing blocks:
picking a configuration is the point, and a parallel arm being slower than serial on a
loaded host is information rather than a regression.

The two selections have different jobs.  ``not_slow`` is the everyday edit-test loop and
is what a worker count should be chosen for.  ``full`` confirms the winner survives the
``slow`` tests, whose long single-test durations are what makes a per-file or per-scope
scheduler strand a worker.

Every cell appends :data:`EXTRA_ARGS`, so cells are comparable with each other and not
with a bare ``pytest`` invocation.  ``-v`` is required to learn which worker ran which
test, and ``--durations=0 --durations-min=0`` to learn what each test cost; both add
output and therefore wall time.

``Measurement.checksum`` is over the six outcome counts, so a configuration that quietly
loses or duplicates tests gets a different checksum from serial and the table prints
**unstable** instead of a faster number.

Two derived facts carry the actual decision.  ``load_spread`` is the busiest worker's
summed test seconds over the idlest worker's, so 1.0 is perfect balance and a large value
means the wall time is one worker's tail.  ``test_seconds_total`` is the sum of every
test's setup, call and teardown; under a per-test scheduler a module-scoped fixture is
rebuilt on every worker, so this total rising against ``serial`` is that rebuild cost made
visible rather than hidden inside the wall time.

The gate reads ``wall_s`` only.  ``peak_rss_mib`` here is the harness child's own
``VmHWM``, and that child is a thin parent that spawns pytest and parses text, so its peak
says nothing about what the workers cost.  ``ru_maxrss_mib`` is recorded as well, but
``wait4`` reports the largest single process in the reaped tree rather than the sum across
workers, so it is not the aggregate a worker count could be chosen on either.
"""

from __future__ import annotations

import ctypes
import functools
import os
import re
import signal
import subprocess
import sys
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, ClassVar, Final

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _harness import (
    PINNED_ENV,
    REPO,
    Arm,
    Fixture,
    Gate,
    Measurement,
    Prepared,
    RunOrder,
    Suite,
    checksum_ints,
    main,
)

EXTRA_ARGS: Final[tuple[str, ...]] = ("-v", "--durations=0", "--durations-min=0")
OUTCOME_KEYS: Final[tuple[str, ...]] = ("passed", "failed", "skipped", "xfailed", "xpassed", "error")
SERIAL_WORKER: Final[str] = "main"
RAN_SUITE: Final[frozenset[pytest.ExitCode]] = frozenset({pytest.ExitCode.OK, pytest.ExitCode.TESTS_FAILED})

_SUMMARY_LINE = re.compile(r"^=+ .* in [\d.]+s.*=+$")
_SUMMARY_PART = re.compile(r"(\d+) (\w+)")
_DURATIONS_HEADER = re.compile(r"^=+ slowest(?: \d+)? durations =+$")
_DURATION_LINE = re.compile(r"^(\d+\.\d+)s (setup|call|teardown)\s+(.+?)\s*$")
_XDIST_LINE = re.compile(r"^\[(gw\d+)\] (?:\[\s*\d+%\] )?([A-Z]+) (.+?)\s*$")
_PLURAL: Final[dict[str, str]] = {"errors": "error", "warnings": "warning"}


class SuiteDidNotRun(RuntimeError):
    """pytest never reached a terminal summary, so this repetition has no comparable wall time."""


@dataclass(frozen=True)
class Selection:
    """A set of tests to run, named so it can be a harness fixture."""

    name: str
    pytest_args: tuple[str, ...]

    # The harness reads ``n_individuals`` off whatever ``fixture.build()`` returns, and
    # what this suite builds is a test selection rather than a pedigree.
    n_individuals: ClassVar[int] = 0


NOT_SLOW: Final[Selection] = Selection("not_slow", ("-m", "not slow"))
FULL: Final[Selection] = Selection("full", ())
WARM_UP: Final[Selection] = Selection("warm_up", ("tests/test_benchmark_contract.py",))


@dataclass(frozen=True)
class WorkerConfig:
    """One xdist worker count and scheduler, plus whether the pytest child is thread-pinned.

    ``dist=None`` is a serial run with no xdist flags, and ``workers`` is then 1.
    """

    name: str
    workers: int
    dist: str | None
    pinned: bool

    @property
    def pytest_args(self) -> tuple[str, ...]:
        """The xdist flags this configuration adds, empty for serial."""
        return () if self.dist is None else ("-n", str(self.workers), "--dist", self.dist)

    @property
    def label(self) -> str:
        """The strategy column of the rendered table."""
        if self.dist is None:
            return "serial (no xdist)"
        return f"`-n {self.workers} --dist {self.dist}`" + ("" if self.pinned else ", unpinned")


CONFIGS: Final[tuple[WorkerConfig, ...]] = (
    WorkerConfig("serial", 1, None, pinned=True),
    WorkerConfig("n2_worksteal", 2, "worksteal", pinned=True),
    WorkerConfig("n3_worksteal", 3, "worksteal", pinned=True),
    WorkerConfig("n4_worksteal", 4, "worksteal", pinned=True),
    WorkerConfig("n6_worksteal", 6, "worksteal", pinned=True),
    WorkerConfig("n8_worksteal", 8, "worksteal", pinned=True),
    WorkerConfig("n6_loadfile", 6, "loadfile", pinned=True),
    WorkerConfig("n6_loadscope", 6, "loadscope", pinned=True),
    WorkerConfig("n6_unpinned", 6, "worksteal", pinned=False),
)

WORKER_PINS: Final[dict[str, str]] = {k: v for k, v in PINNED_ENV.items() if k != "PEDIGREE_GRAPH_THREADS"}


@dataclass(frozen=True)
class PytestOutcome:
    """One pytest run reduced from its terminal output to counts and per-worker seconds."""

    counts: dict[str, int]
    deselected: int
    warnings: int
    worker_of: dict[str, str]
    durations: dict[str, float]
    unmatched_durations: int
    """Durations lines whose nodeid no verbose line claimed."""
    unmatched_verbose: int
    """Verbose lines whose nodeid has no durations entry; the join fails in both directions."""

    @property
    def worker_seconds(self) -> dict[str, float]:
        """Summed test seconds per worker, over every worker that reported a test."""
        totals = dict.fromkeys(self.worker_of.values(), 0.0)
        for nodeid, seconds in self.durations.items():
            worker = self.worker_of.get(nodeid)
            if worker is not None:
                totals[worker] += seconds
        return totals

    @property
    def checksum(self) -> int:
        """Order-independent checksum over the six outcome counts."""
        return checksum_ints(self.counts)

    def facts(self, workers_requested: int) -> dict[str, Any]:
        """The counts and the balance figures, ready to merge into the run record.

        A worker that ran no test never writes a verbose line, so it is invisible in the
        output; ``idle_workers`` counts them against ``workers_requested`` because a healthy
        spread over eleven of twelve workers is the stranding the sweep exists to catch.
        ``load_spread`` is ``None`` rather than infinity when any requested worker has zero
        seconds: the report is JSON and ``Infinity`` is not valid JSON.
        """
        seconds = self.worker_seconds
        idle = max(workers_requested - len(seconds), 0)
        high = max(seconds.values(), default=0.0)
        low = 0.0 if idle else min(seconds.values(), default=0.0)
        return {
            **self.counts,
            "deselected": self.deselected,
            "warnings": self.warnings,
            "workers_requested": workers_requested,
            "workers_seen": len(seconds),
            "idle_workers": idle,
            "test_seconds_total": round(sum(self.durations.values()), 2),
            "worker_seconds_max": round(high, 2),
            "worker_seconds_min": round(low, 2),
            "load_spread": round(high / low, 3) if low else None,
            "unmatched_durations": self.unmatched_durations,
            "unmatched_verbose": self.unmatched_verbose,
        }


def parse_pytest_output(stdout: str) -> PytestOutcome:
    """Reduce one pytest run's terminal output to counts, workers and per-test durations.

    Args:
        stdout: Everything pytest wrote, from a run that used :data:`EXTRA_ARGS`.

    Returns:
        The parsed outcome.

    Raises:
        ValueError: No terminal summary line is present.
    """
    lines = stdout.splitlines()
    summary = next((line for line in reversed(lines) if _SUMMARY_LINE.match(line)), None)
    if summary is None:
        raise ValueError("no pytest summary line in output, so there are no counts to compare")

    tallies: dict[str, int] = {}
    for count, word in _SUMMARY_PART.findall(summary.strip("= ")):
        tallies[_PLURAL.get(word, word)] = int(count)

    worker_of: dict[str, str] = {}
    durations: dict[str, float] = {}
    in_durations = False
    for line in lines:
        if _DURATIONS_HEADER.match(line):
            in_durations = True
            continue
        xdist = _XDIST_LINE.match(line)
        if xdist is not None:
            worker_of[xdist.group(3)] = xdist.group(1)
            continue
        timed = _DURATION_LINE.match(line) if in_durations else None
        if timed is not None:
            durations[timed.group(3)] = durations.get(timed.group(3), 0.0) + float(timed.group(1))

    if not worker_of:
        worker_of = dict.fromkeys(durations, SERIAL_WORKER)
    return PytestOutcome(
        counts={key: tallies.get(key, 0) for key in OUTCOME_KEYS},
        deselected=tallies.get("deselected", 0),
        warnings=tallies.get("warning", 0),
        worker_of=worker_of,
        durations=durations,
        unmatched_durations=sum(1 for nodeid in durations if nodeid not in worker_of),
        unmatched_verbose=sum(1 for nodeid in worker_of if nodeid not in durations),
    )


def _die_with_parent() -> None:
    """Have the kernel SIGKILL the pytest process when the harness child dies.

    The harness enforces ``--timeout`` with ``proc.kill()`` on its own child, which would
    orphan a still-running pytest controller and its workers; an interleaved sweep would
    then time every later cell against a full suite still occupying the cores.
    """
    ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGKILL)  # PR_SET_PDEATHSIG


def _pytest_env(pinned: bool) -> dict[str, str]:
    """Build the pytest child's environment, stripping the harness's own pins first."""
    # PEDIGREE_GRAPH_THREADS is never set for the tests.  Production callers get the
    # package default of 1 unless they configure it, and several tests assert that default
    # or delenv it, so pinning it would measure a configuration nobody runs.
    # PYTEST_ADDOPTS would change what every cell measures without appearing in its
    # recorded pytest_args, so the operator's shell does not get to add flags.
    env = {key: value for key, value in os.environ.items() if key not in PINNED_ENV and key != "PYTEST_ADDOPTS"}
    return env | (WORKER_PINS if pinned else {})


def _collect(selection: Selection) -> tuple[int, str]:
    """Collect a selection without running it, returning its test count and nodeid digest."""
    command = [sys.executable, "-m", "pytest", "--collect-only", "-q", *selection.pytest_args]
    proc = subprocess.run(command, cwd=REPO, capture_output=True, text=True, env=_pytest_env(True), check=False)
    if proc.returncode != pytest.ExitCode.OK:
        raise SuiteDidNotRun(f"collecting {selection.name} exited {proc.returncode}\n{proc.stderr}")
    nodeids = sorted(line.strip() for line in proc.stdout.splitlines() if "::" in line)
    return len(nodeids), sha256("\n".join(nodeids).encode()).hexdigest()[:16]


def _provenance(selection: Selection) -> str:
    """Describe which tests a selection resolves to on this checkout."""
    collected, digest = _collect(selection)
    return f"{collected} tests, nodeids sha256 {digest}"


def _setup(built: object) -> Prepared:
    """Resolve the selection and collect it once, outside the timed region."""
    selection = built if isinstance(built, Selection) else WARM_UP
    collected, digest = _collect(selection)
    return Prepared(selection, {"selection": selection.name, "collected": collected, "selection_sha256": digest})


def _run(config: WorkerConfig, _built: object, selection: Selection) -> Measurement:
    """Run one worker configuration over one selection and reduce its output."""
    command = [sys.executable, "-m", "pytest", *selection.pytest_args, *config.pytest_args, *EXTRA_ARGS]
    proc = subprocess.run(
        command,
        cwd=REPO,
        capture_output=True,
        text=True,
        env=_pytest_env(config.pinned),
        check=False,
        preexec_fn=_die_with_parent,
    )
    try:
        status: pytest.ExitCode | None = pytest.ExitCode(proc.returncode)
    except ValueError:
        status = None
    if status not in RAN_SUITE:
        tail = "\n".join((proc.stderr + "\n" + proc.stdout).splitlines()[-30:])
        raise SuiteDidNotRun(f"{config.name} on {selection.name}: pytest exited {proc.returncode}\n{tail}")

    outcome = parse_pytest_output(proc.stdout)
    return Measurement(
        outcome.checksum,
        {
            "exit_code": proc.returncode,
            **outcome.facts(config.workers),
            "pinned": config.pinned,
            "pytest_args": list(config.pytest_args),
        },
    )


def _selection_fixture(selection: Selection, label: str) -> Fixture:
    """Present a selection as a harness fixture, with its collected nodeids as provenance."""
    return Fixture(
        name=selection.name,
        label=label,
        build=lambda: selection,
        provenance=lambda: _provenance(selection),
    )


def _arm(config: WorkerConfig) -> Arm:
    """Bind one worker configuration into a harness arm."""
    return Arm(name=config.name, run=functools.partial(_run, config), label=config.label, setup=_setup)


SUITE = Suite(
    name="pytest_workers",
    note=Path(__file__).with_suffix(".md"),
    fixtures=(
        _selection_fixture(NOT_SLOW, '`-m "not slow"`'),
        _selection_fixture(FULL, "full suite"),
    ),
    arms=tuple(_arm(config) for config in CONFIGS),
    gate=Gate(baseline="serial", metrics=("wall_s",)),
    order=RunOrder.INTERLEAVED,
    timeout_s=3600.0,
)

if __name__ == "__main__":
    main(SUITE)
