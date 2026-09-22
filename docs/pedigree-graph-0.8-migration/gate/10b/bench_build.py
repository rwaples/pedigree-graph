"""Construction wall time and process peak RSS, one measurement per process.

Usage: ``python bench_build.py <n>``.  Builds a 12-generation synthetic pedigree
in a random row order with ids in a random order, external parents in the
founder generation, sex, and birth years, then constructs the graph through
the public constructor and prints one tab-separated line.  Run it in a fresh
process per measurement and interleave the versions under comparison; the
10b gate ran it from two venvs (PyPI 0.8.1 and the 0.8.2 wheel), five times
each at 300k and 20M rows (``construction-bench.tsv``).
"""

from __future__ import annotations

import resource
import sys
import time
from importlib.metadata import version

import numpy as np

from pedigree_graph import PedigreeGraph


def synthetic_frame(n: int, gens: int = 12, seed: int = 42) -> dict[str, np.ndarray]:
    """Parents from the previous generation, mothers and fathers from disjoint halves."""
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
    return {
        "id": ids[perm],
        "mother": mother_ids[perm],
        "father": father_ids[perm],
        "sex": rng.integers(0, 2, n),
        "birth_year": (1900 + 25 * (np.arange(n) // per) + rng.integers(0, 20, n))[perm],
    }


def main() -> None:
    """Build the pedigree named by ``argv[1]`` rows and print the measurement line."""
    n = int(sys.argv[1])
    frame = synthetic_frame(n)
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    t0 = time.perf_counter()
    graph = PedigreeGraph.from_frame(frame)
    t1 = time.perf_counter()
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(
        f"version={version('pedigree-graph')}\tn={n}\tbuild_s={t1 - t0:.4f}\t"
        f"rss_before_mib={before:.1f}\tmaxrss_mib={peak:.1f}\tchecksum={int(graph.mother_rows.sum())}"
    )


if __name__ == "__main__":
    main()
