"""Release evidence for the 0.8.0 tag on the ``random_300k`` pedigree.

The 0.8.0 plan (slice 9a) records, not gates, what the consumer-facing
operations cost at 300k rows: relationship pairs at the consumer cutoff,
connected components, and pair kinship over the extracted pairs.  Inbreeding
and the scalar count estimate already have ``random_300k`` cells in
``bench_inbreeding.py`` and ``bench_estimate_counts.py``; run those with
``--only random_300k/<arm>`` alongside this suite.

    python benchmarks/bench_release_300k.py --repeat 1 --out benchmarks/reports/release_300k.json

The complete and approximate kinship matrices are deliberately absent.  Both
pay the full coancestry DP pass, which ``matrix_exactification.py`` measured at
6.9 GiB for 30k rows, so they are recorded as unverified at 300k rather than
attempted on a 30 GiB box.  Each cell runs in its own child process (the
harness's ``--cell`` role), which is the one-stage-per-process rule the memory
guard imposes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _harness import Arm, Measurement, Prepared, RunOrder, Suite, checksum_values, main, parity_fixture

MAX_DEGREE = 3


def _pairs_setup(graph) -> Prepared:
    pairs = graph.relationship_pairs(max_degree=MAX_DEGREE)
    return Prepared(payload=pairs, facts={"pairs": sum(len(block) for block in pairs.values())})


def _relationship_pairs(graph, _prepared) -> Measurement:
    pairs = graph.relationship_pairs(max_degree=MAX_DEGREE)
    checksum = 0
    facts = {}
    for code, block in pairs.items():
        facts[f"n_{code}"] = len(block)
        if len(block):
            checksum ^= checksum_values(block.first_rows.astype("int64") * graph.n_individuals + block.second_rows)
    return Measurement(checksum, facts)


def _components(graph, _prepared) -> Measurement:
    labels = graph.connected_component_ids()
    return Measurement(checksum_values(labels), {"components": int(np.unique(labels).size)})


def _pair_kinship(graph, pairs) -> Measurement:
    values = graph.pair_kinship(pairs)
    checksum = 0
    for code in pairs:
        if len(values[code]):
            checksum ^= checksum_values(values[code])
    return Measurement(checksum)


SUITE = Suite(
    name="release_300k",
    note=Path(__file__).with_suffix(".md"),
    fixtures=(parity_fixture("random_300k", label="`random_300k`"),),
    arms=(
        Arm("relationship_pairs", _relationship_pairs, label=f"relationship_pairs(max_degree={MAX_DEGREE})"),
        Arm("connected_component_ids", _components, label="connected_component_ids()"),
        Arm("pair_kinship", _pair_kinship, label=f"degree-{MAX_DEGREE} pair_kinship, fresh graph", setup=_pairs_setup),
    ),
    gate=None,
    order=RunOrder.GROUPED,
    timeout_s=3600.0,
)

if __name__ == "__main__":
    main(SUITE)
