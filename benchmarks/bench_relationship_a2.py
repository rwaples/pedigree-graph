"""Cost of building ``_A2`` during a half-sibling-only pair query (issue #24).

``MatrixPairExtractor.extract`` used to pre-trigger ``_A2`` for every selection
whose dependency closure reached degree 2. Maternal and paternal half siblings
come from parent-group enumeration, so those selections never consume the
matrix.

    python benchmarks/bench_relationship_a2.py --repeat 5 --out benchmarks/reports/relationship_a2.json

The ``eager_mhs`` arm freezes the old behavior by building ``_A2`` before the
query. The ``selected_mhs`` arm leaves matrix demand to the production query.
Both run the same public selector and must return the same checksum.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _harness import Arm, Gate, Measurement, Suite, checksum_ints, main, parity_fixture


def _mhs_measurement(graph) -> Measurement:
    pairs = graph.relationship_pairs(categories=["MHS"])
    count = len(pairs["MHS"])
    return Measurement(checksum_ints({"MHS": count}), {"MHS": count})


def _eager_mhs(graph, _prepared) -> Measurement:
    _ = graph._A2
    return _mhs_measurement(graph)


def _selected_mhs(graph, _prepared) -> Measurement:
    return _mhs_measurement(graph)


SUITE = Suite(
    name="relationship_a2",
    note=Path(__file__).with_suffix(".md"),
    fixtures=(
        parity_fixture("random_30k", label="`random_30k`"),
        parity_fixture("random_300k", label="`random_300k`"),
    ),
    arms=(
        Arm("eager_mhs", _eager_mhs, label="force `_A2`, then select MHS"),
        Arm("selected_mhs", _selected_mhs, label="select MHS on demand"),
    ),
    gate=Gate(baseline="eager_mhs", gated=frozenset({"selected_mhs"})),
    timeout_s=1800.0,
)

if __name__ == "__main__":
    main(SUITE)
