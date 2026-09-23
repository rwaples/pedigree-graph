"""``PedigreeGraph.inbreeding()`` across builds: the slice 15 benchmark (ADR 0008).

The genome-node Meuwissen-Luo walk moves from Numba to the Rust core in slice
15, and the sweep interleaves two builds in one run, as ADR 0007 requires for
a gated comparison:

* ``wheel``: the 0.9.3 PyPI wheel as the simACE umbrella env installs it, run
  under that env's interpreter (the Numba walk);
* ``source``: this checkout, built into this repo's env.

``deep_inbred_60g`` is the stress cell on purpose: a 60-generation closed herd
is where the walk's ``touched`` ancestor set grows most.  ``baseline100K/rep1``
is the largest simACE results pedigree and is unavailable where it has not been
generated.  Every record carries the import path and versions of the package
the child measured, and a SHA-256 checksum of the float64 coefficients, so the
cells also say whether the two builds returned the same bits.

    python benchmarks/bench_inbreeding.py --repeat 3 --out benchmarks/reports/inbreeding.json
    python benchmarks/bench_inbreeding.py --render benchmarks/reports/inbreeding.json
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _harness import (
    WHEEL_INTERPRETER,
    Arm,
    Gate,
    Measurement,
    RunOrder,
    Suite,
    checksum_array,
    main,
    parity_fixture,
    study_fixture,
)


def _walk(graph, _prepared) -> Measurement:
    """Checksum every inbreeding coefficient, and surface its scale as readable facts."""
    coefficients = graph.inbreeding()
    return Measurement(
        lambda: checksum_array(coefficients),
        lambda: {"mean_F": float(coefficients.mean()), "max_F": float(coefficients.max())},
    )


SUITE = Suite(
    name="inbreeding",
    note=Path(__file__).with_suffix(".md"),
    fixtures=(
        replace(parity_fixture("deep_inbred_60g", label="`deep_inbred_60g`"), name="inb-60g"),
        replace(parity_fixture("random_300k", label="`random_300k`"), name="inb-300k"),
        replace(study_fixture("baseline100K"), name="inb-536k"),
    ),
    arms=(
        Arm("wheel", _walk, label="0.9.3 wheel (simACE env)", interpreter=WHEEL_INTERPRETER),
        Arm("source", _walk, label="source build (this env)"),
    ),
    gate=Gate(baseline="wheel", gated=frozenset({"source"})),
    order=RunOrder.INTERLEAVED,
    timeout_s=3600.0,
)

if __name__ == "__main__":
    main(SUITE)
