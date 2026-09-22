"""``relationship_counts(max_degree=5)`` wall time and process peak RSS, one measurement per process.

Usage: ``python bench_counts.py <pedigree> [threads] [estimate]``.  With
``estimate`` the call is ``estimate_relationship_counts(max_degree=5)``
instead, for the decision-8 comparison.  ``<pedigree>`` is
``random_30k`` (``tests/parity/pedigrees.py``, the parity fixture, needs the
pedigree-graph checkout on ``PEDIGREE_GRAPH_SRC``), ``synthetic_300k`` (the 10b
construction pedigree at 300k rows), or a parquet path with ``id``, ``mother``,
``father`` and optional ``twin`` columns.  Run it in a fresh process per
measurement and interleave the versions under comparison; the 11a gate ran it
from two venvs (PyPI 0.8.2 and the 0.8.3 wheel), five times each.
"""

from __future__ import annotations

import os
import resource
import sys
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np


def synthetic_frame(n: int, gens: int = 12, seed: int = 42) -> dict[str, np.ndarray]:
    """The 10b construction pedigree: parents from the previous generation, disjoint halves."""
    rng = np.random.default_rng(seed)
    per = n // gens
    mother = np.full(n, -1, np.int64)
    father = np.full(n, -1, np.int64)
    for g in range(1, gens):
        lo, hi = g * per, min((g + 1) * per, n)
        plo, phi = (g - 1) * per, g * per
        mid = (plo + phi) // 2
        mother[lo:hi] = rng.integers(plo, mid, hi - lo)
        father[lo:hi] = rng.integers(mid, phi, hi - lo)
    ids = rng.permutation(n) * 3 + 11
    mother_ids = np.where(mother < 0, -1, ids[np.maximum(mother, 0)])
    father_ids = np.where(father < 0, -1, ids[np.maximum(father, 0)])
    perm = rng.permutation(n)
    return {"id": ids[perm], "mother": mother_ids[perm], "father": father_ids[perm]}


def load(name: str) -> dict[str, np.ndarray]:
    """Return the constructor columns of the named pedigree."""
    if name == "synthetic_300k":
        return synthetic_frame(300_000)
    if name == "random_30k":
        sys.path.insert(0, str(Path(os.environ["PEDIGREE_GRAPH_SRC"]) / "tests" / "parity"))
        import pedigrees

        fx = pedigrees.build_random("random_30k", dict(pedigrees.LARGE_FIXTURES["random_30k"]))
        return {"id": fx["ids"], "mother": fx["mother"], "father": fx["father"], "twin": fx["twin"]}
    import polars as pl

    df = pl.read_parquet(name)
    columns = {"id": df["id"].to_numpy(), "mother": df["mother"].to_numpy(), "father": df["father"].to_numpy()}
    if "twin" in df.columns:
        columns["twin"] = df["twin"].to_numpy()
    return columns


def main() -> None:
    """Count on the pedigree named by ``argv[1]`` and print the measurement line."""
    name = sys.argv[1]
    threads = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    estimate = len(sys.argv) > 3 and sys.argv[3] == "estimate"
    os.environ["PEDIGREE_GRAPH_THREADS"] = str(threads)
    from pedigree_graph import PedigreeGraph

    frame = load(name)
    graph = PedigreeGraph.from_frame(frame)
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    t0 = time.perf_counter()
    if estimate:
        counts = graph.estimate_relationship_counts(max_degree=5)
    else:
        counts = graph.relationship_counts(max_degree=5)
    t1 = time.perf_counter()
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    total = sum(v for v in counts.values() if v is not None)
    print(
        f"version={version('pedigree-graph')}\tcall={'estimate' if estimate else 'exact'}\t"
        f"pedigree={Path(name).stem}\tthreads={threads}\t"
        f"n={graph.n_individuals}\tcount_s={t1 - t0:.4f}\trss_before_mib={before:.1f}\t"
        f"maxrss_mib={peak:.1f}\tpairs={total}\tc1r1={counts['1C1R']}"
    )


if __name__ == "__main__":
    main()
