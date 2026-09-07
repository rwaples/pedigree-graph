"""Repeated-run benchmark for the slice 5b matrix exactification strategies.

Regenerates every number in ``benchmarks/matrix_exactification.md``.  The
first round of these measurements ran from disposable ``/tmp`` drivers as
single runs, which the 0.8.0 slice plan forbids as sole evidence.  This driver
is the committed replacement.

Each configuration runs in a fresh subprocess, repeated ``--repeat`` times, and
the report quotes the median with the observed spread.  Peak RSS comes from the
kernel's ``VmHWM``, reset at the start of the timed region.  ``getrusage``
alone cannot do this, because its high-water mark never falls and so reports
setup rather than the phase under test.

Usage::

    python benchmarks/matrix_exactification.py --repeat 3 --out results.json
    python benchmarks/matrix_exactification.py --list
    python benchmarks/matrix_exactification.py --single fitace/fused
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sp

if TYPE_CHECKING:
    from collections.abc import Callable

REPO = Path(__file__).resolve().parent.parent
FITACE_PEDIGREE = Path("/data/Documents/simACE/results/dev/dev_cont_n10k/rep1/pedigree.parquet")
THRESHOLD = 0.001

# The library's own default budget is 1.  Pinning every backend to match keeps
# wall time from drifting with whatever else is scheduled on the host.
PINNED_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "NUMBA_NUM_THREADS": "1",
    "POLARS_MAX_THREADS": "1",
    "PEDIGREE_GRAPH_THREADS": "1",
}


class PeakRss:
    """Measure peak RSS over one timed region using the kernel high-water mark.

    Sampling ``/proc/self/statm`` from a Python thread does not work here.  A
    ``@numba.njit`` kernel holds the GIL for its whole call, so the sampler is
    starved: measured against ``_build_candidate`` on the fitACE pedigree it
    was scheduled for 1 of an expected 553 ticks and reported 358 MiB against
    a true peak of 759 MiB.

    Writing ``5`` to ``/proc/self/clear_refs`` resets ``VmHWM`` to the current
    RSS, so reading ``VmHWM`` afterwards gives the kernel's own peak for the
    region, with no sampling and no dependence on the GIL.
    """

    _STATUS = Path("/proc/self/status")
    _CLEAR_REFS = Path("/proc/self/clear_refs")

    def __init__(self) -> None:
        self.baseline_mib = 0.0
        self.peak_mib = 0.0

    @classmethod
    def _field_mib(cls, name: str) -> float:
        for line in cls._STATUS.read_text().splitlines():
            if line.startswith(name):
                return int(line.split()[1]) / 1024.0
        raise RuntimeError(f"{name} missing from /proc/self/status")

    def __enter__(self) -> PeakRss:
        # CLEAR_REFS_MM_HIWATER_RSS.  Without it VmHWM is a process-lifetime
        # mark and setup would dominate every measurement.
        self._CLEAR_REFS.write_text("5\n")
        self.baseline_mib = self._field_mib("VmRSS")
        return self

    def __exit__(self, *exc: object) -> None:
        self.peak_mib = self._field_mib("VmHWM")


def upper_checksum(matrix: sp.csc_matrix) -> int:
    """XOR the upper-triangle float32 bits, so symmetric duplicates cannot cancel."""
    checksum = np.uint32(0)
    for column in range(matrix.shape[1]):
        start, end = matrix.indptr[column], matrix.indptr[column + 1]
        rows = matrix.indices[start:end]
        values = matrix.data[start:end]
        upper = values[rows <= column].view(np.uint32)
        checksum = np.bitwise_xor(checksum, np.bitwise_xor.reduce(upper, initial=np.uint32(0)))
    return int(checksum)


def values_checksum(values: np.ndarray) -> int:
    """XOR a flat float32 value array as raw uint32 bits."""
    return int(np.bitwise_xor.reduce(values.view(np.uint32), initial=np.uint32(0), dtype=np.uint32))


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def _fitace_graph():
    import pyarrow.parquet as pq

    from pedigree_graph import PedigreeGraph

    table = pq.read_table(FITACE_PEDIGREE)
    names = set(table.column_names)

    def column(*candidates: str) -> np.ndarray:
        return table[next(name for name in candidates if name in names)].to_numpy()

    return PedigreeGraph(
        {
            "id": column("individual_id", "id"),
            "mother": column("mother_id", "mother"),
            "father": column("father_id", "father"),
            "twin": column("twin_id", "twin"),
        }
    )


def _random30k_graph():
    sys.path.insert(0, str(REPO / "tests" / "parity"))
    import pedigrees

    from pedigree_graph import PedigreeGraph

    fixture = pedigrees.build_random("random_30k", pedigrees.LARGE_FIXTURES["random_30k"])
    return PedigreeGraph(
        {
            "id": fixture["ids"],
            "mother": fixture["mother"],
            "father": fixture["father"],
            "twin": fixture["twin"],
            "sex": fixture["sex"],
        }
    )


INPUTS: dict[str, Callable[[], object]] = {
    "fitace": _fitace_graph,
    "random30k": _random30k_graph,
}


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Strategy:
    """One timed operation, with untimed setup kept out of the measurement."""

    name: str
    run: Callable[..., dict]
    setup: Callable[..., dict] | None = None
    note: str = ""


def _build_candidate(graph) -> sp.csc_matrix:
    from pedigree_graph._kinship_dp import _build_kinship_csc

    indptr, indices, propagated = _build_kinship_csc(
        graph.n_individuals,
        graph.mother_rows,
        graph.father_rows,
        graph.twin_rows,
        graph.depth,
        THRESHOLD,
    )
    return sp.csc_matrix((propagated, indices, indptr), shape=(graph.n_individuals,) * 2, copy=False)


def _upper_coordinates(matrix: sp.csc_matrix) -> tuple[np.ndarray, np.ndarray]:
    upper = sp.triu(matrix, k=0, format="coo")
    order = np.lexsort((upper.row, upper.col))
    return upper.row[order].astype(np.int32, copy=False), upper.col[order].astype(np.int32, copy=False)


def _evaluate_chunks(graph, first: np.ndarray, second: np.ndarray, chunks: list[tuple[int, int]]) -> dict:
    from pedigree_graph._kinship_pairwise import _pairwise_kinship_with_stats

    mother, father, twin = graph._topological_parents
    translate = graph._topology.translate
    checksum = 0
    capacity = 0
    for lo, hi in chunks:
        values, stats = _pairwise_kinship_with_stats(
            mother, father, twin, translate(first[lo:hi]), translate(second[lo:hi])
        )
        checksum ^= values_checksum(values)
        capacity = max(capacity, stats["memo_capacity"])
    return {"checksum": checksum, "chunks": len(chunks), "max_memo_capacity": capacity}


def _pairwise_setup(graph) -> dict:
    candidate = _build_candidate(graph)
    first, second = _upper_coordinates(candidate)
    return {"first": first, "second": second, "upper_candidates": len(first)}


def _shared(graph, first, second, **_) -> dict:
    return _evaluate_chunks(graph, first, second, [(0, len(first))])


def _by_pairs(size: int) -> Callable[..., dict]:
    def run(graph, first, second, **_) -> dict:
        chunks = [(i, min(i + size, len(first))) for i in range(0, len(first), size)]
        return _evaluate_chunks(graph, first, second, chunks)

    return run


def _by_columns(width: int) -> Callable[..., dict]:
    def run(graph, first, second, **_) -> dict:
        edges = np.searchsorted(second, np.arange(0, graph.n_individuals + width, width))
        chunks = [(int(a), int(b)) for a, b in itertools.pairwise(edges) if b > a]
        return _evaluate_chunks(graph, first, second, chunks)

    return run


def _candidate_support(graph, **_) -> dict:
    candidate = _build_candidate(graph)
    return {"nnz": int(candidate.nnz), "upper_candidates": (int(candidate.nnz) + graph.n_individuals) // 2}


def _complete_dp(graph, **_) -> dict:
    from pedigree_graph._kinship_dp import _compute_theta_per_gen

    theta = _compute_theta_per_gen(
        graph.n_individuals,
        graph.mother_rows,
        graph.father_rows,
        graph.twin_rows,
        graph.depth,
        0.0,
        labels=graph.depth,
    )
    return {"reduction": float(np.nansum(theta))}


def _fused(graph, **_) -> dict:
    matrix = graph.approximate_kinship_matrix(min_propagated_kinship=THRESHOLD)
    return {
        "nnz": int(matrix.nnz),
        "upper_candidates": (int(matrix.nnz) + graph.n_individuals) // 2,
        "checksum": upper_checksum(matrix),
    }


STRATEGIES: dict[str, Strategy] = {
    "candidate_support": Strategy("candidate_support", _candidate_support, note="propagation-pruned support only"),
    "complete_dp": Strategy("complete_dp", _complete_dp, note="retiring threshold-zero DP, no output CSC"),
    "fused": Strategy("fused", _fused, note="public approximate_kinship_matrix end to end"),
    "pairwise_shared": Strategy("pairwise_shared", _shared, _pairwise_setup, "one shared recurrence memo"),
    "pairwise_columns256": Strategy("pairwise_columns256", _by_columns(256), _pairwise_setup, "256-column chunks"),
    "pairwise_pairs262144": Strategy("pairwise_pairs262144", _by_pairs(262144), _pairwise_setup, "262,144-pair chunks"),
    "pairwise_pairs1048576": Strategy(
        "pairwise_pairs1048576", _by_pairs(1048576), _pairwise_setup, "1,048,576-pair chunks"
    ),
}

# Ordered cheapest first so a partial sweep still yields a usable table.
CONFIGS: list[str] = [
    "fitace/candidate_support",
    "fitace/complete_dp",
    "fitace/fused",
    "random30k/candidate_support",
    "random30k/complete_dp",
    "random30k/fused",
    "fitace/pairwise_shared",
    "fitace/pairwise_pairs1048576",
    "fitace/pairwise_pairs262144",
    "fitace/pairwise_columns256",
    "random30k/pairwise_pairs1048576",
    "random30k/pairwise_shared",
]


def _warm_up() -> None:
    """Pay numba cache-load cost before the timed region."""
    from pedigree_graph import PedigreeGraph

    graph = PedigreeGraph(
        {
            "id": np.array([10, 11, 12, 13], dtype=np.int64),
            "mother": np.array([-1, -1, 10, 10], dtype=np.int64),
            "father": np.array([-1, -1, 11, 11], dtype=np.int64),
            "twin": np.full(4, -1, dtype=np.int64),
        }
    )
    graph.approximate_kinship_matrix(min_propagated_kinship=THRESHOLD)
    graph.kinship_matrix()
    _build_candidate(graph)
    _complete_dp(graph)
    _pairwise_setup(graph)


def run_single(config: str) -> dict:
    """Measure one configuration in this process and return its record."""
    input_name, strategy_name = config.split("/")
    strategy = STRATEGIES[strategy_name]
    _warm_up()
    graph = INPUTS[input_name]()
    context = strategy.setup(graph) if strategy.setup is not None else {}
    timed_context = {k: v for k, v in context.items() if isinstance(v, np.ndarray)}
    scalars = {k: v for k, v in context.items() if not isinstance(v, np.ndarray)}

    with PeakRss() as rss:
        started = time.perf_counter()
        result = strategy.run(graph, **timed_context)
        wall = time.perf_counter() - started

    return {
        "config": config,
        "n_individuals": int(graph.n_individuals),
        "wall_s": wall,
        "peak_rss_mib": rss.peak_mib,
        "baseline_rss_mib": rss.baseline_mib,
        "rss_growth_mib": rss.peak_mib - rss.baseline_mib,
        **scalars,
        **result,
    }


@dataclass
class Aggregate:
    """Repeated runs of one configuration, plus their summary statistics."""

    config: str
    runs: list[dict] = field(default_factory=list)
    timed_out: bool = False
    timeout_s: float | None = None

    def summary(self) -> dict:
        """Return medians, spread, and checksum stability across the runs."""
        if not self.runs:
            return {
                "config": self.config,
                "repeats": 0,
                "timed_out": self.timed_out,
                "timeout_s": self.timeout_s,
            }
        walls = [r["wall_s"] for r in self.runs]
        peaks = [r["peak_rss_mib"] for r in self.runs]
        checksums = {r.get("checksum") for r in self.runs}
        median = statistics.median(walls)
        return {
            "config": self.config,
            "repeats": len(self.runs),
            "n_individuals": self.runs[0]["n_individuals"],
            "wall_median_s": median,
            "wall_min_s": min(walls),
            "wall_max_s": max(walls),
            "wall_spread_pct": (max(walls) - min(walls)) / median * 100.0 if median else 0.0,
            "peak_rss_median_mib": statistics.median(peaks),
            "peak_rss_min_mib": min(peaks),
            "peak_rss_max_mib": max(peaks),
            "rss_growth_median_mib": statistics.median(r["rss_growth_mib"] for r in self.runs),
            "checksum": checksums.pop() if len(checksums) == 1 else None,
            "checksum_stable": len(checksums) <= 1,
            "upper_candidates": self.runs[0].get("upper_candidates"),
            "chunks": self.runs[0].get("chunks"),
            "max_memo_capacity": self.runs[0].get("max_memo_capacity"),
            "nnz": self.runs[0].get("nnz"),
            "timed_out": self.timed_out,
        }


def drive(configs: list[str], repeat: int, timeout_s: float, out: Path | None) -> list[dict]:
    """Run every configuration in fresh pinned subprocesses, writing results as they land."""
    env = {**os.environ, **PINNED_ENV}
    aggregates: list[Aggregate] = []
    for config in configs:
        aggregate = Aggregate(config, timeout_s=timeout_s)
        for index in range(repeat):
            command = [sys.executable, str(Path(__file__).resolve()), "--single", config]
            print(f"  {config}  rep {index + 1}/{repeat} ... ", end="", flush=True)
            started = time.perf_counter()
            try:
                completed = subprocess.run(
                    command, env=env, capture_output=True, text=True, timeout=timeout_s, check=True
                )
            except subprocess.TimeoutExpired:
                aggregate.timed_out = True
                print(f"TIMEOUT after {timeout_s:.0f}s")
                break
            except subprocess.CalledProcessError as exc:
                print("FAILED")
                sys.stderr.write(exc.stderr)
                raise
            record = json.loads(completed.stdout.strip().splitlines()[-1])
            aggregate.runs.append(record)
            print(
                f"{record['wall_s']:.2f}s  peak {record['peak_rss_mib']:.0f} MiB  ({time.perf_counter() - started:.0f}s)"
            )
        aggregates.append(aggregate)
        if out is not None:
            out.write_text(json.dumps([a.summary() for a in aggregates], indent=2, sort_keys=True))
    return [a.summary() for a in aggregates]


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single", help="Run one config in this process and print JSON.")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=9000.0)
    parser.add_argument("--only", nargs="*", help="Restrict the sweep to these configs.")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list:
        for config in CONFIGS:
            print(config)
        return
    if args.single:
        print(json.dumps(run_single(args.single), sort_keys=True))
        return

    configs = args.only or CONFIGS
    unknown = [c for c in configs if c not in CONFIGS]
    if unknown:
        raise SystemExit(f"unknown configs: {unknown}")
    summaries = drive(configs, args.repeat, args.timeout, args.out)
    print(json.dumps(summaries, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
