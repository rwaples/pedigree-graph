"""Production-scale baseline for the scalar relationship-count path (ADR 0011).

Records what ``close_relative_counts`` costs in wall time and peak RSS
on the two large parity pedigrees. Only its six exact codes are measured;
this is not the former 23-code estimator workload.

    python benchmarks/bench_estimate_counts.py --repeat 5 --out benchmarks/reports/counts.json

There is no gate.  The ``count_pairs_streaming`` arm this was compared against
was deleted with the 0.7.1 adapters, so what remains is one arm recording a
baseline rather than an A/B; the streaming figures survive in
``benchmarks/relationship_counts_rust.md``.

Ordering is ``GROUPED`` rather than interleaved.  With one arm there is nothing
to interleave against, so finishing a cell before starting the next keeps an
interrupted sweep useful.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _harness import Arm, Measurement, RunOrder, Suite, checksum_ints, main, parity_fixture


def _close_counts(graph, _prepared) -> Measurement:
    """Checksum the six computed codes and report full and maternal half sibs."""
    result = graph.close_relative_counts()
    counts = {code: value for code, value in result.items() if value is not None}
    return Measurement(checksum_ints(counts), {"FS": counts["FS"], "MHS": counts["MHS"]})


SUITE = Suite(
    name="close_relative_counts",
    note=Path(__file__).with_suffix(".md"),
    # Parameters come from tests/parity; re-declaring them here is how the two
    # definitions previously drifted apart.  Cheapest first, so an interrupted
    # sweep still renders complete cells.
    fixtures=(
        parity_fixture("random_30k", label="`random_30k`"),
        parity_fixture("random_300k", label="`random_300k`"),
    ),
    arms=(Arm("close_relative_counts", _close_counts, label="six exact close-relative counts"),),
    gate=None,
    order=RunOrder.GROUPED,
    timeout_s=3600.0,
)

if __name__ == "__main__":
    main(SUITE)
