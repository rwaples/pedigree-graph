"""Fresh-process time/RSS comparison for compact views and burden output.

Run with the pedigree-graph pixi interpreter after ``maturin develop``::

    python tools/pg_tskit_profile.py --repeats 3 --out /tmp/pg-tskit-profile.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests" / "parity"))
sys.path.insert(0, str(REPO / "benchmarks"))

import pedigrees  # noqa: E402
from _harness import PeakRss  # noqa: E402

from pedigree_graph import RELATIONSHIPS, PedigreeGraph, _native  # noqa: E402


def _graph(name: str) -> PedigreeGraph:
    fixture = pedigrees.build_random(name, {**pedigrees.LARGE_FIXTURES, **pedigrees.RELEASE_FIXTURES}[name])
    return PedigreeGraph.from_frame(
        {
            "id": fixture["ids"],
            "mother": fixture["mother"],
            "father": fixture["father"],
            "twin": fixture["twin"],
            "sex": fixture["sex"],
        }
    )


def _view(graph: PedigreeGraph, fraction: float) -> tuple[np.ndarray, int]:
    n = graph.n_individuals
    selected = np.random.default_rng(42).permutation(n)[: max(2, round(n * fraction))]
    mapping = graph.view(rows=selected)._graph_to_view()
    retained = np.zeros(n, dtype=bool)
    stack = selected.tolist()
    while stack:
        row = stack.pop()
        if retained[row]:
            continue
        retained[row] = True
        stack.extend(int(p) for p in (graph.mother_rows[row], graph.father_rows[row], graph.twin_rows[row]) if p >= 0)
    return mapping, int(retained.sum())


def _digest(blocks: dict[str, tuple[np.ndarray, np.ndarray]]) -> str:
    h = hashlib.sha256()
    for code in RELATIONSHIPS:
        h.update(code.encode())
        for array in blocks[code]:
            h.update(array.tobytes())
    return h.hexdigest()


def child(mode: str, fraction: float, threads: int, fixture: str) -> dict:
    """One measured operation after graph construction, in its own process."""
    graph = _graph(fixture)
    view_map, closure = _view(graph, fraction)
    depth = graph.depth
    _native.configure_pool(threads)
    with PeakRss() as region:
        if mode == "burden":
            counts, per_person, same_depth = _native.relationship_burden(graph._built, depth, threads=threads)
        elif mode == "count_full":
            counts = _native.relationship_counts(graph._built, max_degree=5, threads=threads, selected=view_map >= 0)
        elif mode == "count_compact":
            counts = _native.compact_view_counts(graph._built, view_map, max_degree=5, threads=threads)
        else:
            blocks = _native.relationship_pairs(
                graph._built,
                max_degree=5,
                requested=list(RELATIONSHIPS),
                threads=threads,
                execution="speed",
                view_rows=view_map if mode.startswith("view") else None,
                compact=mode == "view_compact",
            )
    if mode == "burden":
        digest = hashlib.sha256(per_person.tobytes() + same_depth.tobytes() + repr(counts).encode()).hexdigest()
        output_bytes = per_person.nbytes + same_depth.nbytes
    elif mode.startswith("count"):
        digest = hashlib.sha256(repr(counts).encode()).hexdigest()
        output_bytes = 0
    else:
        digest = _digest(blocks)
        output_bytes = sum(a.nbytes + b.nbytes for a, b in blocks.values())
    return {
        "mode": mode,
        "fixture": fixture,
        "fraction": fraction,
        "selected": int((view_map >= 0).sum()),
        "closure": closure,
        "wall_s": region.wall_s,
        "peak_rss_mib": region.peak_mib,
        "growth_mib": region.growth_mib,
        "output_mib": output_bytes / 2**20,
        "digest": digest,
    }


def main() -> None:
    """Run each benchmark cell in a fresh process and optionally save JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", action="store_true")
    parser.add_argument(
        "--mode", choices=("view_full", "view_compact", "count_full", "count_compact", "burden", "pairs")
    )
    parser.add_argument("--fraction", type=float, default=0.01)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--fixture", default="random_30k", choices=("random_30k", "random_300k"))
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.child:
        print(json.dumps(child(args.mode, args.fraction, args.threads, args.fixture)))
        return
    rows = []
    for fraction in (0.001, 0.01, 0.1, 0.5):
        for mode in ("view_full", "view_compact"):
            for _ in range(args.repeats):
                command = [
                    sys.executable,
                    __file__,
                    "--child",
                    "--mode",
                    mode,
                    "--fraction",
                    str(fraction),
                    "--threads",
                    str(args.threads),
                    "--fixture",
                    args.fixture,
                ]
                result = subprocess.run(command, capture_output=True, text=True, check=True)
                row = json.loads(result.stdout)
                rows.append(row)
                print(
                    f"{mode:12s} {fraction:5.1%} closure={row['closure']:6d} wall={row['wall_s']:6.3f}s peak={row['peak_rss_mib']:7.1f} MiB",
                    flush=True,
                )
    for mode in ("pairs", "burden"):
        for _ in range(args.repeats):
            result = subprocess.run(
                [
                    sys.executable,
                    __file__,
                    "--child",
                    "--mode",
                    mode,
                    "--threads",
                    str(args.threads),
                    "--fixture",
                    args.fixture,
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            row = json.loads(result.stdout)
            rows.append(row)
            print(f"{mode:12s} wall={row['wall_s']:6.3f}s peak={row['peak_rss_mib']:7.1f} MiB", flush=True)
    if args.out:
        args.out.write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
