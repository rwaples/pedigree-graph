"""``relationship_burden`` on the large parity pedigrees: one traversal, O(N) output.

Records what ``PedigreeGraph.relationship_burden`` costs in wall time and peak
RSS.  The checksum covers all three outputs: the per-category counts, the
per-person degree columns and the same-depth pair counts.

    python benchmarks/bench_relationship_burden.py --repeat 5 --out benchmarks/reports/burden.json

Run alone it records a baseline with no gate.  With ``--baseline-python`` it
gates this checkout's build against another checkout's (``_harness.build_pair``).
``random_1k`` is for a quick end-to-end run of the script and is not in the
default sweep.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _harness import Arm, Measurement, RunOrder, Suite, checksum_array, checksum_ints, main, parity_fixture


def _burden(graph, _prepared) -> Measurement:
    result = graph.relationship_burden(progress=False)

    def checksum() -> int:
        digest = hashlib.sha256()
        digest.update(f"counts={checksum_ints(dict(result.category_counts))};".encode())
        digest.update(f"per_person={checksum_array(result.per_person)};".encode())
        digest.update(f"same_depth={checksum_array(result.same_depth_pairs)};".encode())
        return int.from_bytes(digest.digest()[:8], "big")

    return Measurement(checksum, lambda: {"pairs": sum(result.category_counts.values())})


FIXTURES = (
    parity_fixture("random_30k", label="`random_30k`"),
    parity_fixture("random_300k", label="`random_300k`"),
    parity_fixture("random_1k", label="`random_1k`"),
)
ARMS = (Arm("burden", _burden, label="`relationship_burden`, 1 thread"),)

SUITE = Suite(
    name="relationship_burden",
    note=Path(__file__).with_suffix(".md"),
    fixtures=FIXTURES,
    arms=ARMS,
    cells=tuple(f"{fixture.name}/{arm.name}" for fixture in FIXTURES[:2] for arm in ARMS),
    gate=None,
    order=RunOrder.GROUPED,
    timeout_s=3600.0,
)

if __name__ == "__main__":
    main(SUITE)
