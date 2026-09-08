"""Shared benchmark contract for this repository.

Every benchmark declares a :class:`Suite` and calls :func:`main`.  The harness
owns process spawning, thread pinning, memory measurement, repetition,
aggregation, gating and rendering, so a benchmark script contains only its
inputs and its timed operations.

This module exists because the correct pattern was already in the repository
and was not reused.  ``bench_estimate_counts.py`` had medians, fresh child
processes and the 5% gate; slice 5b still wrote four disposable drivers with a
memory measurement that could not work.  Making the correct path shorter to
write than the wrong one is the point.

It cannot prevent a future script from importing nothing and measuring badly.
Python has no mechanism for that.  ``tests/test_benchmark_contract.py`` catches
the two symptoms that reached tracked artifacts last time: a note citing a
``/tmp`` driver as its method, and a result file with no environment.

Two memory scopes exist here on purpose.  :class:`PeakRss` measures one region
inside a process, which is what attributing cost to a phase requires.
:attr:`ChildOutcome.ru_maxrss_mib` is the whole child process including fixture
construction, which is what ``bench_estimate_counts.py`` documents and compares.
Collapsing them would silently change what that gate means.
"""

from __future__ import annotations

__all__ = [
    "GATE",
    "MIN_CONFIDENT_REPEATS",
    "PINNED_ENV",
    "Arm",
    "Cell",
    "ContractError",
    "Environment",
    "Fixture",
    "Gate",
    "Measurement",
    "Outcome",
    "PeakRss",
    "Prepared",
    "RunOrder",
    "Suite",
    "Verdict",
    "checksum_ints",
    "checksum_matrix_upper",
    "checksum_values",
    "file_fixture",
    "main",
    "parity_fixture",
    "render_markdown",
    "verify_report",
]

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Final, NoReturn

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence

    import scipy.sparse as sp

REPO: Final[Path] = Path(__file__).resolve().parent.parent
HARNESS_PATH: Final[Path] = Path(__file__).resolve()
SCHEMA: Final[str] = "pedigree-graph/benchmark/1"

GATE: Final[float] = 1.05
"""The only definition of the 5% rule.

``docs/adr/0007-rust-core-host-boundary-and-release.md`` states it: block only
on a *confident* >5% median wall or RSS regression, at the same thread budget,
over fresh interleaved processes, with any exception needing maintainer
sign-off.  ``bench_estimate_counts.py`` held the only copy and compared bare
medians, which is the gate without the confidence.
"""

MIN_CONFIDENT_REPEATS: Final[int] = 3
"""Below this, a comparison is :attr:`Verdict.INCONCLUSIVE` rather than a pass."""

PINNED_ENV: Final[dict[str, str]] = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "NUMBA_NUM_THREADS": "1",
    "POLARS_MAX_THREADS": "1",
    "PEDIGREE_GRAPH_THREADS": "1",
}
"""One thread everywhere, which is also ``pedigree_graph``'s own default budget.

``bench_estimate_counts.py`` passed no environment to its children at all, so
its numbers moved with whatever else the host was scheduling.
"""


class ContractError(RuntimeError):
    """A result file or tracked note violates the benchmark contract."""


class RunOrder(StrEnum):
    """Cell-major or repetition-major scheduling."""

    INTERLEAVED = "interleaved"
    """One repetition of every cell, then the next round.  ADR 0007 requires
    this for A/B comparisons: drift in host state hits both arms equally."""

    GROUPED = "grouped"
    """Every repetition of one cell, then the next cell.  A 112-minute cell
    finishes and renders even if the sweep is interrupted."""


class Outcome(StrEnum):
    """Why a cell has the runs it has."""

    COMPLETED = "completed"
    TIMED_OUT = "timed_out"
    UNAVAILABLE = "unavailable"
    """The fixture does not exist on this host, so the cell was never run."""


class Verdict(StrEnum):
    """The result of comparing one cell against the gate baseline."""

    PASS = "pass"
    BLOCK = "block"
    INCONCLUSIVE = "inconclusive"


class PeakRss:
    """Peak resident set size over one timed region, from the kernel.

    Sampling ``/proc/self/statm`` from a Python thread does not work here.  A
    ``@numba.njit`` kernel holds the GIL for its whole call, so the sampler is
    starved.  Measured against a 2.77 s kernel it was scheduled for 1 of an
    expected 553 ticks and reported 358 MiB against a true peak of 759 MiB.
    Three scripts in this repository independently made that mistake.

    Writing ``5`` to ``/proc/self/clear_refs`` resets ``VmHWM`` to current RSS,
    so reading ``VmHWM`` afterwards is the kernel's own peak for the region,
    with no sampling and no dependence on the GIL.
    """

    _STATUS = Path("/proc/self/status")
    _CLEAR_REFS = Path("/proc/self/clear_refs")

    def __init__(self) -> None:
        self.baseline_mib = 0.0
        self.peak_mib = 0.0
        self.wall_s = 0.0
        self._started = 0.0

    @classmethod
    def _field_mib(cls, name: str) -> float:
        for line in cls._STATUS.read_text().splitlines():
            if line.startswith(name):
                return int(line.split()[1]) / 1024.0
        raise RuntimeError(f"{name} missing from /proc/self/status")

    def __enter__(self) -> PeakRss:
        self._CLEAR_REFS.write_text("5\n")  # CLEAR_REFS_MM_HIWATER_RSS
        self.baseline_mib = self._field_mib("VmRSS")
        self._started = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.wall_s = time.perf_counter() - self._started
        self.peak_mib = self._field_mib("VmHWM")

    @property
    def growth_mib(self) -> float:
        return self.peak_mib - self.baseline_mib


def checksum_values(values: np.ndarray) -> int:
    """XOR a flat float32 array as raw uint32 bits."""
    return int(np.bitwise_xor.reduce(values.view(np.uint32), initial=np.uint32(0), dtype=np.uint32))


def checksum_matrix_upper(matrix: sp.csc_matrix) -> int:
    """XOR the upper-triangle float32 bits, so symmetric duplicates cannot cancel."""
    checksum = np.uint32(0)
    for column in range(matrix.shape[1]):
        start, end = matrix.indptr[column], matrix.indptr[column + 1]
        rows = matrix.indices[start:end]
        values = matrix.data[start:end]
        upper = values[rows <= column].view(np.uint32)
        checksum = np.bitwise_xor(checksum, np.bitwise_xor.reduce(upper, initial=np.uint32(0)))
    return int(checksum)


def checksum_ints(values: Mapping[str, int | None]) -> int:
    """Order-independent checksum of a named integer mapping.

    ``None`` is a distinct value, not zero.  ``RelationshipCountResult`` is a
    ``Mapping[str, int | None]`` where ``None`` means "not requested", and a
    checksum that conflated the two would not notice a selector change.
    """
    digest = hashlib.sha256()
    for key in sorted(values):
        value = values[key]
        digest.update(f"{key}={'none' if value is None else int(value)};".encode())
    return int.from_bytes(digest.digest()[:4], "big")


@dataclass(frozen=True, order=True)
class Cell:
    """One fixture crossed with one arm.  The only place ``fixture/arm`` is parsed."""

    fixture: str
    arm: str

    def __str__(self) -> str:
        return f"{self.fixture}/{self.arm}"

    @classmethod
    def parse(cls, text: str) -> Cell:
        fixture, _, arm = text.partition("/")
        if not fixture or not arm:
            raise ValueError(f"cell must be 'fixture/arm', got {text!r}")
        return cls(fixture, arm)


@dataclass(frozen=True)
class Prepared:
    """What an arm's untimed setup hands to its timed run.

    ``payload`` is opaque and passed straight through; ``facts`` are merged into
    the record.  Naming the two halves replaces inferring them from a runtime
    ``isinstance`` check, which silently dropped any non-array payload.
    """

    payload: object = None
    facts: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Measurement:
    """What a timed arm returns.

    ``checksum`` is required, which makes "every config proves correctness, not
    just speed" a structural property rather than a convention.
    """

    checksum: int
    facts: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Fixture:
    """A reproducible input with provenance."""

    name: str
    label: str
    build: Callable[[], Any]
    provenance: Callable[[], str]
    available: Callable[[], bool] = lambda: True


@dataclass(frozen=True)
class Arm:
    """One named timed operation, with optional untimed setup."""

    name: str
    run: Callable[..., Measurement]
    label: str
    setup: Callable[[Any], Prepared] | None = None


@dataclass(frozen=True)
class Gate:
    """The comparison policy.  One field separates an A/B gate from a sweep.

    ``gated`` empty means ratios are computed and shown but nothing blocks.
    ``baseline=None`` means no ratio column at all.
    """

    baseline: str | None
    gated: frozenset[str] = frozenset()
    metrics: tuple[str, ...] = ("wall_s", "peak_rss_mib")
    accepted: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Environment:
    """Everything needed to judge whether two numbers are comparable.

    Captured per record rather than once per sweep.  A shared sidecar cannot
    express a sweep whose rows ran at different commits, which previously had to
    be written into the note by hand.
    """

    git_commit: str
    git_branch: str
    git_dirty: bool
    cpu_model: str
    logical_cpus: int
    cpu_max_mhz: int
    cpu_governor: str
    mem_total_gib: float
    kernel: str
    python: str
    pixi_lock_sha256: str
    harness_sha256: str
    suite_sha256: str
    threads_pinned: int
    rss_method: str = "kernel VmHWM, reset via /proc/self/clear_refs at region start"

    REQUIRED: ClassVar[tuple[str, ...]] = (
        "git_commit",
        "cpu_model",
        "logical_cpus",
        "cpu_max_mhz",
        "kernel",
        "python",
        "pixi_lock_sha256",
        "harness_sha256",
        "threads_pinned",
        "rss_method",
    )

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.as_dict(), sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()[:12]

    def as_dict(self) -> dict[str, Any]:
        return {
            "git_commit": self.git_commit,
            "git_branch": self.git_branch,
            "git_dirty": self.git_dirty,
            "cpu_model": self.cpu_model,
            "logical_cpus": self.logical_cpus,
            "cpu_max_mhz": self.cpu_max_mhz,
            "cpu_governor": self.cpu_governor,
            "mem_total_gib": self.mem_total_gib,
            "kernel": self.kernel,
            "python": self.python,
            "pixi_lock_sha256": self.pixi_lock_sha256,
            "harness_sha256": self.harness_sha256,
            "suite_sha256": self.suite_sha256,
            "threads_pinned": self.threads_pinned,
            "rss_method": self.rss_method,
        }

    @classmethod
    def capture(cls, suite_path: Path) -> Environment:
        def git(*args: str) -> str:
            result = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=False)
            return result.stdout.strip()

        def sha(path: Path) -> str:
            return hashlib.sha256(path.read_bytes()).hexdigest()[:16]

        cpu = next(
            (
                line.split(":", 1)[1].strip()
                for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")
            ),
            "unknown",
        )
        mem_kb = next(
            (
                int(line.split()[1])
                for line in Path("/proc/meminfo").read_text().splitlines()
                if line.startswith("MemTotal")
            ),
            0,
        )
        cpufreq = Path("/sys/devices/system/cpu/cpu0/cpufreq")

        def read(name: str, default: str = "") -> str:
            path = cpufreq / name
            return path.read_text().strip() if path.exists() else default

        # The nameplate model string is not the ceiling.  This host reports
        # "@ 2.60GHz" while cpuinfo_max_freq is 4.5 GHz and scaling_max_freq is
        # 2.6 GHz, so two runs can differ by 1.7x with an identical cpu_model.
        max_khz = read("scaling_max_freq", "0")

        return cls(
            git_commit=git("rev-parse", "HEAD"),
            git_branch=git("rev-parse", "--abbrev-ref", "HEAD"),
            git_dirty=bool(git("status", "--porcelain")),
            cpu_model=cpu,
            logical_cpus=os.cpu_count() or 0,
            cpu_max_mhz=int(max_khz or 0) // 1000,
            cpu_governor=read("scaling_governor", "unknown"),
            mem_total_gib=round(mem_kb / 1024**2, 1),
            kernel=platform.release(),
            python=sys.version.split()[0],
            pixi_lock_sha256=sha(REPO / "pixi.lock"),
            harness_sha256=sha(HARNESS_PATH),
            suite_sha256=sha(suite_path),
            threads_pinned=int(PINNED_ENV["PEDIGREE_GRAPH_THREADS"]),
        )


@dataclass(frozen=True)
class RunRecord:
    """One repetition of one cell in one fresh process."""

    cell: Cell
    wall_s: float
    peak_rss_mib: float
    baseline_rss_mib: float
    ru_maxrss_mib: float
    checksum: int
    n_individuals: int
    facts: Mapping[str, Any]
    environment: str
    started_at: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "cell": str(self.cell),
            "wall_s": self.wall_s,
            "peak_rss_mib": self.peak_rss_mib,
            "baseline_rss_mib": self.baseline_rss_mib,
            "ru_maxrss_mib": self.ru_maxrss_mib,
            "checksum": self.checksum,
            "n_individuals": self.n_individuals,
            "facts": dict(self.facts),
            "environment": self.environment,
            "started_at": self.started_at,
        }


@dataclass(frozen=True)
class CellResult:
    """Every repetition of one cell, plus why it has the runs it has."""

    cell: Cell
    outcome: Outcome
    runs: tuple[RunRecord, ...] = ()
    timeout_s: float | None = None

    def _require_completed(self) -> None:
        if self.outcome is not Outcome.COMPLETED or not self.runs:
            raise ContractError(f"{self.cell} is {self.outcome}, it has no median")

    def median(self, metric: str) -> float:
        self._require_completed()
        return statistics.median(getattr(run, metric) for run in self.runs)

    def spread_pct(self, metric: str) -> float:
        self._require_completed()
        values = [getattr(run, metric) for run in self.runs]
        centre = statistics.median(values)
        return (max(values) - min(values)) / centre * 100.0 if centre else 0.0

    def span(self, metric: str) -> tuple[float, float]:
        self._require_completed()
        values = [getattr(run, metric) for run in self.runs]
        return min(values), max(values)

    @property
    def checksum(self) -> int | None:
        seen = {run.checksum for run in self.runs}
        return seen.pop() if len(seen) == 1 else None

    @property
    def checksum_stable(self) -> bool:
        return len({run.checksum for run in self.runs}) <= 1


@dataclass(frozen=True)
class Suite:
    """The registry: everything a benchmark declares and nothing it does."""

    name: str
    note: Path
    fixtures: tuple[Fixture, ...]
    arms: tuple[Arm, ...]
    cells: tuple[str, ...] = ()
    gate: Gate | None = None
    order: RunOrder = RunOrder.INTERLEAVED
    timeout_s: float = 3600.0

    def fixture(self, name: str) -> Fixture:
        for candidate in self.fixtures:
            if candidate.name == name:
                return candidate
        raise KeyError(f"unknown fixture {name!r}")

    def arm(self, name: str) -> Arm:
        for candidate in self.arms:
            if candidate.name == name:
                return candidate
        raise KeyError(f"unknown arm {name!r}")

    def resolved_cells(self) -> tuple[Cell, ...]:
        """Parse declared cells, or the full product in declaration order.

        A typo fails here, before a sweep starts, rather than hours in.
        """
        if not self.cells:
            return tuple(Cell(f.name, a.name) for f in self.fixtures for a in self.arms)
        resolved = []
        for text in self.cells:
            cell = Cell.parse(text)
            self.fixture(cell.fixture)
            self.arm(cell.arm)
            resolved.append(cell)
        return tuple(resolved)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _parity_module() -> Any:
    if str(REPO / "tests" / "parity") not in sys.path:
        sys.path.insert(0, str(REPO / "tests" / "parity"))
    import pedigrees

    return pedigrees


def parity_fixture(name: str, *, label: str) -> Fixture:
    """A fixture whose parameters come from ``tests/parity/pedigrees.py``.

    There is deliberately no parameters argument.  A benchmark cannot re-declare
    seed and generation counts, which is how a duplicate set drifted from the
    parity definitions before.
    """

    def _params() -> dict[str, Any]:
        pedigrees = _parity_module()
        for table in (pedigrees.RANDOM_FIXTURES, pedigrees.LARGE_FIXTURES, pedigrees.RELEASE_FIXTURES):
            if name in table:
                return table[name]
        raise KeyError(f"{name!r} is not a parity fixture")

    def build() -> Any:
        from pedigree_graph import PedigreeGraph

        pedigrees = _parity_module()
        fx = pedigrees.build_random(name, _params())
        return PedigreeGraph.from_frame(
            {key: fx[key] for key in ("ids", "mother", "father", "twin", "sex") if key in fx} | {"id": fx["ids"]}
        )

    def provenance() -> str:
        pedigrees = _parity_module()
        return pedigrees.input_hash(pedigrees.build_random(name, _params()))

    return Fixture(name=name, label=label, build=build, provenance=provenance)


def file_fixture(name: str, path: Path, *, label: str) -> Fixture:
    """A fixture read from a parquet file, hashed for provenance.

    ``available`` is false when the file is absent, so a machine-local input
    records :attr:`Outcome.UNAVAILABLE` and leaves the rest of the sweep runnable
    instead of crashing it.
    """

    def build() -> Any:
        import pyarrow.parquet as pq

        from pedigree_graph import PedigreeGraph

        table = pq.read_table(path)
        names = set(table.column_names)

        def column(*candidates: str) -> np.ndarray:
            return table[next(n for n in candidates if n in names)].to_numpy()

        return PedigreeGraph.from_frame(
            {
                "id": column("individual_id", "id"),
                "mother": column("mother_id", "mother"),
                "father": column("father_id", "father"),
                "twin": column("twin_id", "twin"),
            }
        )

    def provenance() -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    return Fixture(name=name, label=label, build=build, provenance=provenance, available=path.exists)


_WARM_UP_COLUMNS: Final[dict[str, list[int]]] = {
    "id": [10, 11, 12, 13],
    "mother": [-1, -1, 10, 10],
    "father": [-1, -1, 11, 11],
    "twin": [-1, -1, -1, -1],
}


def _warm_up_graph() -> Any:
    from pedigree_graph import PedigreeGraph

    return PedigreeGraph.from_frame({k: np.array(v, dtype=np.int64) for k, v in _WARM_UP_COLUMNS.items()})


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


def _measure_cell(suite: Suite, cell: Cell, environment: str) -> RunRecord:
    """Child role: warm the arm, build the fixture, run setup, time the arm alone.

    The order matters.  Warming through the arm itself cannot miss a kernel the
    way a hand-listed warm-up can, and it turns a broken arm into a fast failure
    instead of one discovered hours into a sweep.  ``clear_refs`` at region entry
    makes warm-up and setup allocations invisible to ``VmHWM``.
    """
    arm = suite.arm(cell.arm)
    fixture = suite.fixture(cell.fixture)

    warm = _warm_up_graph()
    prepared = arm.setup(warm) if arm.setup is not None else Prepared()
    arm.run(warm, prepared.payload)

    graph = fixture.build()
    prepared = arm.setup(graph) if arm.setup is not None else Prepared()

    started_at = datetime.now(UTC).isoformat(timespec="seconds")
    with PeakRss() as region:
        measurement = arm.run(graph, prepared.payload)

    return RunRecord(
        cell=cell,
        wall_s=region.wall_s,
        peak_rss_mib=region.peak_mib,
        baseline_rss_mib=region.baseline_mib,
        ru_maxrss_mib=0.0,
        checksum=measurement.checksum,
        n_individuals=int(graph.n_individuals),
        facts={**dict(prepared.facts), **dict(measurement.facts)},
        environment=environment,
        started_at=started_at,
    )


@dataclass(frozen=True)
class _ChildOutcome:
    record: dict[str, Any] | None
    ru_maxrss_mib: float
    timed_out: bool


def _spawn(script: Path, cell: Cell, timeout_s: float) -> _ChildOutcome:
    """One fresh pinned child, returning both its record and its whole-process peak.

    ``Popen`` plus ``os.wait4`` rather than ``subprocess.run``: ``run`` reaps the
    child itself, so rusage is never exposed.  That is why one existing script
    has a timeout and no ``ru_maxrss`` and the other has ``ru_maxrss`` and no
    timeout.  This is the first place in the repository that needs both.
    """
    env = {**os.environ, **PINNED_ENV}
    command = [sys.executable, str(script), "--cell", str(cell)]
    proc = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    deadline = time.monotonic() + timeout_s
    usage = None
    while True:
        pid, status, rusage = os.wait4(proc.pid, os.WNOHANG)
        if pid:
            usage = rusage
            break
        if time.monotonic() > deadline:
            proc.kill()
            os.wait4(proc.pid, 0)
            return _ChildOutcome(None, 0.0, True)
        time.sleep(0.05)

    stdout, stderr = proc.communicate()
    exit_code = os.waitstatus_to_exitcode(status)
    if exit_code != 0:
        raise ContractError(f"{cell} child exited {exit_code}\n{stderr}")
    line = stdout.strip().splitlines()[-1]
    return _ChildOutcome(json.loads(line), usage.ru_maxrss / 1024.0, False)


def _schedule(cells: Sequence[Cell], repeat: int, order: RunOrder) -> Iterator[tuple[Cell, int]]:
    if order is RunOrder.INTERLEAVED:
        for index in range(repeat):
            for cell in cells:
                yield cell, index
    else:
        for cell in cells:
            for index in range(repeat):
                yield cell, index


def _verdict(subject: CellResult, baseline: CellResult, metric: str, gated: bool) -> Verdict:
    """ADR 0007's confidence rule, which no code implemented before.

    ``INCONCLUSIVE`` when either side has too few repetitions or the observed
    ranges overlap.  ``BLOCK`` only when the median ratio exceeds the gate *and*
    the ranges are disjoint, so a cell measured twice cannot pass or fail a 5%
    gate on noise.
    """
    if subject.outcome is not Outcome.COMPLETED or baseline.outcome is not Outcome.COMPLETED:
        return Verdict.INCONCLUSIVE
    if min(len(subject.runs), len(baseline.runs)) < MIN_CONFIDENT_REPEATS:
        return Verdict.INCONCLUSIVE
    ratio = subject.median(metric) / baseline.median(metric) if baseline.median(metric) else float("inf")
    if ratio <= GATE:
        return Verdict.PASS
    if not gated:
        return Verdict.PASS
    lo_subject, _ = subject.span(metric)
    _, hi_baseline = baseline.span(metric)
    return Verdict.BLOCK if lo_subject > hi_baseline else Verdict.INCONCLUSIVE


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Report:
    """One sweep's results, with every environment that produced them."""

    suite: str
    cells: tuple[CellResult, ...]
    environments: Mapping[str, Environment]
    schema: str = SCHEMA
    gate: float = GATE

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "suite": self.suite,
            "gate": self.gate,
            "environments": {key: env.as_dict() for key, env in self.environments.items()},
            "cells": [
                {
                    "cell": str(result.cell),
                    "outcome": str(result.outcome),
                    "timeout_s": result.timeout_s,
                    "runs": [run.as_dict() for run in result.runs],
                }
                for result in self.cells
            ],
        }

    def verdicts(self, gate: Gate | None) -> dict[Cell, Verdict]:
        """Verdicts for gated cells only.

        A verdict answers "can this block, and did it".  An ungated arm still
        gets its ratio shown, but carrying a verdict it can never act on would
        be noise.
        """
        if gate is None or gate.baseline is None:
            return {}
        results = {result.cell: result for result in self.cells}
        out: dict[Cell, Verdict] = {}
        for result in self.cells:
            if result.cell.arm == gate.baseline:
                continue
            if result.cell.arm not in gate.gated or str(result.cell) in gate.accepted:
                continue
            baseline = results.get(Cell(result.cell.fixture, gate.baseline))
            if baseline is None:
                continue
            gated = True
            worst = Verdict.PASS
            for metric in gate.metrics:
                verdict = _verdict(result, baseline, metric, gated)
                if verdict is Verdict.BLOCK:
                    worst = Verdict.BLOCK
                elif verdict is Verdict.INCONCLUSIVE and worst is Verdict.PASS:
                    worst = Verdict.INCONCLUSIVE
            out[result.cell] = worst
        return out


def _read_report(path: Path) -> Report:
    payload = json.loads(path.read_text())
    environments = {key: Environment(**value) for key, value in payload["environments"].items()}
    cells = []
    for entry in payload["cells"]:
        cell = Cell.parse(entry["cell"])
        runs = tuple(
            RunRecord(
                cell=cell,
                wall_s=run["wall_s"],
                peak_rss_mib=run["peak_rss_mib"],
                baseline_rss_mib=run["baseline_rss_mib"],
                ru_maxrss_mib=run["ru_maxrss_mib"],
                checksum=run["checksum"],
                n_individuals=run["n_individuals"],
                facts=run["facts"],
                environment=run["environment"],
                started_at=run["started_at"],
            )
            for run in entry["runs"]
        )
        cells.append(CellResult(cell, Outcome(entry["outcome"]), runs, entry["timeout_s"]))
    return Report(payload["suite"], tuple(cells), environments, payload["schema"], payload["gate"])


def verify_report(path: Path) -> Report:
    """Parse a result file, raising :class:`ContractError` on a contract violation.

    This is the enforcement check for result metadata.  A file cannot pass while
    missing an environment, referencing one that is not recorded, leaving a
    required environment field empty, omitting a checksum, or carrying a gate
    other than :data:`GATE`.
    """
    try:
        report = _read_report(path)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ContractError(f"{path}: unreadable benchmark report ({exc})") from exc
    if report.schema != SCHEMA:
        raise ContractError(f"{path}: unknown schema {report.schema!r}")
    if report.gate != GATE:
        raise ContractError(f"{path}: gate {report.gate} does not match GATE {GATE}")
    if not report.environments:
        raise ContractError(f"{path}: no environment recorded")
    for key, environment in report.environments.items():
        for name in Environment.REQUIRED:
            if getattr(environment, name) in ("", None):
                raise ContractError(f"{path}: environment {key} has empty {name}")
    for result in report.cells:
        for run in result.runs:
            if run.environment not in report.environments:
                raise ContractError(f"{path}: {result.cell} references unknown environment {run.environment}")
            if not isinstance(run.checksum, int):
                raise ContractError(f"{path}: {result.cell} has no checksum")
    return report


def _seconds(value: float) -> str:
    return f"{value:,.0f} s ({value / 60:.0f} min)" if value >= 600 else f"{value:,.2f} s"


def render_markdown(suite: Suite, report: Report) -> str:
    """The environment block and results table for the tracked note.

    ``benchmarks/.gitignore`` excludes ``reports/``, so the note is the only
    durable evidence and has to carry environment, repetitions and spread inline.
    Labels come from the registry, so no second table can drift from it.
    """
    counts: dict[str, int] = {}
    for result in report.cells:
        for run in result.runs:
            counts[run.environment] = counts.get(run.environment, 0) + 1
    if not counts:
        return "No results.\n"
    primary_key = max(counts, key=lambda key: counts[key])
    primary = report.environments[primary_key]

    lines = [
        f"- commit `{primary.git_commit[:10]}` on `{primary.git_branch}`"
        + (", working tree dirty" if primary.git_dirty else ""),
        f"- {primary.cpu_model}, {primary.logical_cpus} logical CPUs at up to "
        f"{primary.cpu_max_mhz} MHz ({primary.cpu_governor}), "
        f"{primary.mem_total_gib} GiB RAM, kernel {primary.kernel}",
        f"- Python {primary.python}, pixi lock `{primary.pixi_lock_sha256}`, harness `{primary.harness_sha256}`",
        f"- every backend pinned to {primary.threads_pinned} thread",
        f"- peak RSS is {primary.rss_method}",
    ]
    for key in sorted(k for k in report.environments if k != primary_key):
        other = report.environments[key]
        rows = sorted(str(r.cell) for r in report.cells if any(run.environment == key for run in r.runs))
        lines.append(f"- {', '.join(rows)} ran at commit `{other.git_commit[:10]}`, suite `{other.suite_sha256}`")

    verdicts = report.verdicts(suite.gate)
    table = [
        "| input | strategy | reps | wall (median) | spread | peak RSS (median) | checksum |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for result in report.cells:
        fixture = suite.fixture(result.cell.fixture)
        arm = suite.arm(result.cell.arm)
        if result.outcome is Outcome.TIMED_OUT:
            table.append(
                f"| {fixture.label} | {arm.label} | 0 | did not finish within "
                f"{result.timeout_s:.0f} s | n/a | n/a | n/a |"
            )
            continue
        if result.outcome is Outcome.UNAVAILABLE:
            table.append(f"| {fixture.label} | {arm.label} | 0 | input unavailable on this host | n/a | n/a | n/a |")
            continue
        checksum = f"`{result.checksum}`" if result.checksum_stable else "**unstable**"
        flag = ""
        if verdicts.get(result.cell) is Verdict.BLOCK:
            flag = " **BLOCK**"
        elif verdicts.get(result.cell) is Verdict.INCONCLUSIVE:
            flag = " (inconclusive)"
        table.append(
            f"| {fixture.label} | {arm.label}{flag} | {len(result.runs)} | "
            f"{_seconds(result.median('wall_s'))} | {result.spread_pct('wall_s'):.1f}% | "
            f"{result.median('peak_rss_mib'):,.0f} MiB | {checksum} |"
        )
    return "\n".join(lines) + "\n\n" + "\n".join(table) + "\n"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _drive(
    suite: Suite, script: Path, cells: Sequence[Cell], repeat: int, timeout_s: float, out: Path | None
) -> Report:
    environment = Environment.capture(script)
    fingerprint = environment.fingerprint
    runs: dict[Cell, list[RunRecord]] = {cell: [] for cell in cells}
    outcomes: dict[Cell, Outcome] = {}

    for cell in cells:
        if not suite.fixture(cell.fixture).available():
            outcomes[cell] = Outcome.UNAVAILABLE
            print(f"  {cell}  fixture unavailable on this host, skipped")

    runnable = [cell for cell in cells if cell not in outcomes]
    for cell, index in _schedule(runnable, repeat, suite.order):
        if outcomes.get(cell) is Outcome.TIMED_OUT:
            continue
        print(f"  {cell}  rep {index + 1}/{repeat} ... ", end="", flush=True)
        outcome = _spawn(script, cell, timeout_s)
        if outcome.timed_out:
            outcomes[cell] = Outcome.TIMED_OUT
            print(f"TIMEOUT after {timeout_s:.0f}s")
            continue
        assert outcome.record is not None
        record = RunRecord(
            cell=cell,
            wall_s=outcome.record["wall_s"],
            peak_rss_mib=outcome.record["peak_rss_mib"],
            baseline_rss_mib=outcome.record["baseline_rss_mib"],
            ru_maxrss_mib=outcome.ru_maxrss_mib,
            checksum=outcome.record["checksum"],
            n_individuals=outcome.record["n_individuals"],
            facts=outcome.record["facts"],
            environment=fingerprint,
            started_at=outcome.record["started_at"],
        )
        runs[cell].append(record)
        print(f"{record.wall_s:.2f}s  peak {record.peak_rss_mib:.0f} MiB  (proc {record.ru_maxrss_mib:.0f} MiB)")

        if out is not None:
            _write(suite, out, runs, outcomes, {fingerprint: environment}, timeout_s)

    return _report(suite, runs, outcomes, {fingerprint: environment}, timeout_s)


def _report(
    suite: Suite,
    runs: Mapping[Cell, list[RunRecord]],
    outcomes: Mapping[Cell, Outcome],
    environments: Mapping[str, Environment],
    timeout_s: float,
) -> Report:
    cells = tuple(
        CellResult(
            cell=cell,
            outcome=outcomes.get(cell, Outcome.COMPLETED if records else Outcome.UNAVAILABLE),
            runs=tuple(records),
            timeout_s=timeout_s,
        )
        for cell, records in runs.items()
    )
    return Report(suite.name, cells, environments)


def _write(
    suite: Suite,
    out: Path,
    runs: Mapping[Cell, list[RunRecord]],
    outcomes: Mapping[Cell, Outcome],
    environments: Mapping[str, Environment],
    timeout_s: float,
) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(_report(suite, runs, outcomes, environments, timeout_s).as_dict(), indent=2, sort_keys=True)
    )


def main(suite: Suite) -> NoReturn:
    """The only entry point.  Parent and child roles, selected by ``--cell``.

    Exits 1 on any :attr:`Verdict.BLOCK`, or additionally on any
    :attr:`Verdict.INCONCLUSIVE` under ``--strict``.
    """
    script = Path(sys.argv[0]).resolve()
    parser = argparse.ArgumentParser(description=suite.name)
    parser.add_argument("--cell", help="Child role: measure one cell and print JSON.")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=suite.timeout_s)
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--render", type=Path, help="Render a stored report to markdown.")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    if args.list:
        for cell in suite.resolved_cells():
            print(cell)
        raise SystemExit(0)

    if args.render is not None:
        print(render_markdown(suite, verify_report(args.render)))
        raise SystemExit(0)

    if args.cell:
        cell = Cell.parse(args.cell)
        record = _measure_cell(suite, cell, environment="child")
        print(json.dumps(record.as_dict(), sort_keys=True))
        raise SystemExit(0)

    declared = suite.resolved_cells()
    cells = [Cell.parse(name) for name in args.only] if args.only else list(declared)
    unknown = [str(cell) for cell in cells if cell not in declared]
    if unknown:
        raise SystemExit(f"unknown cells: {unknown}")

    report = _drive(suite, script, cells, args.repeat, args.timeout, args.out)
    print()
    print(render_markdown(suite, report))

    verdicts = report.verdicts(suite.gate)
    blocked = [cell for cell, verdict in verdicts.items() if verdict is Verdict.BLOCK]
    unsure = [cell for cell, verdict in verdicts.items() if verdict is Verdict.INCONCLUSIVE]
    for cell in blocked:
        print(f"BLOCK: {cell} regressed beyond {GATE:.2f}x with disjoint ranges")
    for cell in unsure:
        print(f"inconclusive: {cell} (needs >= {MIN_CONFIDENT_REPEATS} reps and disjoint ranges to decide)")
    raise SystemExit(1 if blocked or (args.strict and unsure) else 0)
