"""Production-scale baseline for ``PedigreeGraph.inbreeding()`` (ADR 0008).

Records what the genome-node Meuwissen-Luo walk costs in wall time and peak RSS
across the four parity pedigrees, which is what the ADR 0007 Rust port ports
against.

    python benchmarks/bench_inbreeding.py --repeat 5 --out benchmarks/reports/inbreeding.json

There is no gate.  The MZ-naive arm this would have been compared against was
deleted with ADR 0008, so what remains is one arm recording a baseline rather
than an A/B.  ``deep_inbred_60g`` is the stress cell on purpose, because a
60-generation closed herd is where the walk's ``touched`` ancestor set grows
most.

Ordering is ``GROUPED`` rather than interleaved.  With one arm there is nothing
to interleave against, and the largest cell may run for a long time, so
finishing a cell before starting the next keeps an interrupted sweep useful.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _harness import Arm, Measurement, RunOrder, Suite, checksum_values, main, parity_fixture


def _walk(graph, _prepared) -> Measurement:
    """Checksum every inbreeding coefficient, and surface its scale as readable facts."""
    coefficients = graph.inbreeding()
    return Measurement(
        checksum_values(coefficients),
        {"mean_F": float(coefficients.mean()), "max_F": float(coefficients.max())},
    )


SUITE = Suite(
    name="inbreeding",
    note=Path(__file__).with_suffix(".md"),
    # Cheapest first, so an interrupted sweep still renders complete cells.
    # With one arm the full product is already this order, so no cells list.
    fixtures=(
        parity_fixture("random_1k", label="`random_1k`"),
        parity_fixture("deep_inbred_60g", label="`deep_inbred_60g`"),
        parity_fixture("random_30k", label="`random_30k`"),
        parity_fixture("random_300k", label="`random_300k`"),
    ),
    arms=(Arm("inbreeding", _walk, label="genome-node Meuwissen-Luo walk"),),
    gate=None,
    order=RunOrder.GROUPED,
    timeout_s=3600.0,
)

if __name__ == "__main__":
    main(SUITE)
