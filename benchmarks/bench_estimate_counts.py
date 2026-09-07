"""A/B gate for the scalar relationship-count path (slice 4c, ADR 0011).

``estimate_relationship_counts`` must not regress against the 0.7.1 streaming
baseline by more than the family 5% rule.  Ordering is ``INTERLEAVED``, so host
drift lands on both arms equally, which is what ADR 0007 asks of an A/B
comparison.

    python benchmarks/bench_estimate_counts.py --repeat 5 --out benchmarks/reports/counts.json

Exits 1 on a confident regression.  A regression measured over fewer than three
repetitions, or with overlapping ranges, is inconclusive rather than blocking.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _harness import Arm, Gate, Measurement, RunOrder, Suite, checksum_ints, main, parity_fixture

MAX_DEGREE = 5


def _counts_measurement(counts) -> Measurement:
    """Checksum every code, and surface two as readable facts."""
    counts = dict(counts)
    return Measurement(checksum_ints(counts), {"FS": counts["FS"], "GP": counts["GP"]})


def _streaming(graph, _prepared) -> Measurement:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return _counts_measurement(graph.count_pairs_streaming(max_degree=MAX_DEGREE))


def _estimate(graph, _prepared) -> Measurement:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return _counts_measurement(graph.estimate_relationship_counts(max_degree=MAX_DEGREE))


SUITE = Suite(
    name="estimate_counts",
    note=Path(__file__).with_suffix(".md"),
    # Parameters come from tests/parity; re-declaring them here is how the two
    # definitions previously drifted apart.
    fixtures=(
        parity_fixture("random_30k", label="`random_30k`"),
        parity_fixture("random_300k", label="`random_300k`"),
    ),
    arms=(
        Arm("count_pairs_streaming", _streaming, label="0.7.1 streaming baseline"),
        Arm("estimate_relationship_counts", _estimate, label="scalar estimate (default)"),
    ),
    gate=Gate(baseline="count_pairs_streaming", gated=frozenset({"estimate_relationship_counts"})),
    order=RunOrder.INTERLEAVED,
    timeout_s=3600.0,
)

if __name__ == "__main__":
    main(SUITE)
