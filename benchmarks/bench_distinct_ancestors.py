"""``PedigreeGraph.distinct_ancestor_counts`` across builds (issue #1, slice 15).

Slice 15 moves the retiring sorted-set DP from Numba to the Rust core with
an owned exact-size set per live row (plan D6), and the sweep interleaves two
builds in one run, as ADR 0007 requires for a gated comparison:

* ``wheel``: the 0.9.3 PyPI wheel as the simACE umbrella env installs it, run
  under that env's interpreter (the Numba pool and free lists);
* ``source``: this checkout, built into this repo's env.

The frozen sparse boolean closure the 0.9 DP was first gated against is
retired; its measured rows stay in ``bench_distinct_ancestors.md``.  The width
and depth series of closed-parentage fixtures are where a change of set
storage is most exposed, so every cell is gated on wall and RSS, then the
parity fixtures shared with the other benchmarks and ``baseline100K/rep1``
(unavailable where it has not been generated).  Each record carries a SHA-256
checksum of the int32 counts.

    python benchmarks/bench_distinct_ancestors.py --repeat 3 --out benchmarks/reports/distinct_ancestors.json
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests" / "parity"))

from _harness import (
    WHEEL_INTERPRETER,
    Arm,
    Fixture,
    Gate,
    Measurement,
    RunOrder,
    Suite,
    checksum_array,
    main,
    parity_fixture,
    study_fixture,
)

_WIDTH_FIXTURES = {
    "closed_w200_g8": {"seed": 20_008, "n_founders": 200, "n_generations": 8, "per_generation": 200},
    "closed_w1000_g8": {"seed": 100_008, "n_founders": 1000, "n_generations": 8, "per_generation": 1000},
    "closed_w2000_g8": {"seed": 200_008, "n_founders": 2000, "n_generations": 8, "per_generation": 2000},
}

_DEPTH_FIXTURES = {
    "closed_w128_g8": {"seed": 12_808, "n_founders": 128, "n_generations": 8, "per_generation": 128},
    "closed_w128_g16": {"seed": 12_816, "n_founders": 128, "n_generations": 16, "per_generation": 128},
    "closed_w128_g32": {"seed": 12_832, "n_founders": 128, "n_generations": 32, "per_generation": 128},
    "closed_w128_g60": {"seed": 12_860, "n_founders": 128, "n_generations": 60, "per_generation": 128},
}
_FIXTURES = _WIDTH_FIXTURES | _DEPTH_FIXTURES


def _closed_fixture(name: str) -> Fixture:
    """Build one deterministic closed-parentage scaling fixture."""
    params = _FIXTURES[name]

    def frame() -> dict[str, np.ndarray]:
        import pedigrees

        return pedigrees.deep_inbred_pedigree(**params)

    def build():
        from pedigree_graph import PedigreeGraph

        columns = frame()
        return PedigreeGraph.from_frame(
            {
                "id": columns["ids"],
                "mother": columns["mother"],
                "father": columns["father"],
                "twin": columns["twin"],
                "sex": columns["sex"],
            }
        )

    def provenance() -> str:
        import pedigrees

        return pedigrees.input_hash(frame())

    n = params["n_founders"] + params["n_generations"] * params["per_generation"]
    label = (
        f"`{name}` ({n:,} rows, {params['n_founders']:,} founders, "
        f"{params['n_generations']} generated generations, {params['per_generation']:,}/generation)"
    )
    return Fixture(name=name, label=label, build=build, provenance=provenance)


def _counts(graph, _prepared) -> Measurement:
    """Checksum the counts, and describe the closure they represent."""
    counts = graph.distinct_ancestor_counts()
    total = int(counts.sum(dtype=np.int64))
    return Measurement(
        checksum_array(counts),
        {
            "total_ancestor_links": total,
            "max_ancestors": int(counts.max(initial=0)),
            "mean_ancestors": float(total / len(counts)) if len(counts) else 0.0,
        },
    )


SUITE = Suite(
    name="distinct_ancestors",
    note=Path(__file__).with_suffix(".md"),
    fixtures=(
        _closed_fixture("closed_w200_g8"),
        _closed_fixture("closed_w1000_g8"),
        _closed_fixture("closed_w2000_g8"),
        _closed_fixture("closed_w128_g8"),
        _closed_fixture("closed_w128_g16"),
        _closed_fixture("closed_w128_g32"),
        _closed_fixture("closed_w128_g60"),
        parity_fixture("random_1k", label="`random_1k`"),
        parity_fixture("deep_inbred_60g", label="`deep_inbred_60g`"),
        parity_fixture("random_30k", label="`random_30k`"),
        parity_fixture("random_300k", label="`random_300k`"),
        study_fixture("baseline100K"),
    ),
    arms=(
        Arm("wheel", _counts, label="0.9.3 wheel (simACE env)", interpreter=WHEEL_INTERPRETER),
        Arm("source", _counts, label="source build (this env)"),
    ),
    gate=Gate(baseline="wheel", gated=frozenset({"source"})),
    order=RunOrder.INTERLEAVED,
    timeout_s=3600.0,
)

if __name__ == "__main__":
    main(SUITE)
