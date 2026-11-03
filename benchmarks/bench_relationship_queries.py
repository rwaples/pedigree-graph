"""Relationship queries on each receiver path, and the calls that walk nothing (issue #52).

Every query operation (pairs, counts, moments, relatives per person) on the
graph, on a 5% view (which takes the compact path on these fixtures), on a
50% view (the full path), and the calls that have nothing to walk: an empty
selection on the graph, and views of zero and one rows.  The ``floor_*`` arms
time ``FLOOR_CALLS`` calls each, so one repetition is well above the timer's
resolution; divide by ``FLOOR_CALLS`` for a per-call figure.

The arms use only the public API, so ``--baseline-python`` gates this
checkout's build against another checkout's unchanged:

    python benchmarks/bench_relationship_queries.py --repeat 5 --strict --baseline-python <pg-base>/.pixi/envs/default/bin/python --out benchmarks/reports/queries.json
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _harness import Arm, Measurement, Prepared, RunOrder, Suite, checksum_array, checksum_ints, main, parity_fixture

if TYPE_CHECKING:
    from collections.abc import Callable

FLOOR_CALLS = 1000
VIEW_SEED = 52
DEGREE = 3


def _rows(n: int, kind: str) -> np.ndarray:
    """The graph rows of one view kind: ``05`` and ``50`` percent, or ``0`` and ``1`` rows."""
    order = np.random.default_rng(VIEW_SEED).permutation(n)
    size = {"05": n // 20, "50": n // 2, "1": 1, "0": 0}[kind]
    return np.sort(order[:size]).astype(np.int64)


def _receiver(kind: str, op: str) -> Callable[[object], Prepared]:
    """Untimed setup: the receiver (the graph or one view of it) and *op*'s per-row inputs sized to it."""

    def setup(graph) -> Prepared:
        receiver = graph if kind == "graph" else graph.view(rows=_rows(graph.n_individuals, kind))
        n = graph.n_individuals if kind == "graph" else len(receiver)
        inputs = {"moments": _moments_inputs, "relatives": _relatives_inputs}.get(op, lambda _: {})(n)
        return Prepared((receiver, inputs), {"rows": n})

    return setup


def _pairs_checksum(result) -> int:
    digest = hashlib.sha256()
    for code in sorted(result):
        block = result[code]
        digest.update(
            f"{code}:{block.requested}:{checksum_array(block.first_rows)}:{checksum_array(block.second_rows)};".encode()
        )
    return int.from_bytes(digest.digest()[:8], "big")


def _moments_inputs(n: int) -> dict:
    rows = np.arange(n)
    return {"first": {"s": (rows % 2).astype(np.int64)}, "values": {"x": (rows % 7).astype(np.float64)}}


def _relatives_inputs(n: int) -> dict:
    return {"thresholds": {"t": ((np.arange(n) % 5).astype(np.float64), 2.0)}}


def _call(op: str, prepared: tuple, selector: dict):
    """One query on the prepared receiver with its prepared inputs."""
    receiver, inputs = prepared
    method = {
        "pairs": receiver.relationship_pairs,
        "counts": receiver.relationship_counts,
        "moments": receiver.relationship_moments,
        "relatives": receiver.relatives_per_person,
    }[op]
    return method(**selector, **inputs, progress=False)


def _checksum(op: str, result) -> int:
    if op == "pairs":
        return _pairs_checksum(result)
    if op == "counts":
        return checksum_ints(dict(result))
    if op == "moments":
        return checksum_array(np.asarray(result.encoded)) ^ result.width
    return checksum_array(result.counts)


def _query(op: str, selector: dict) -> Callable[..., Measurement]:
    def run(_graph, prepared) -> Measurement:
        result = _call(op, prepared, selector)
        return Measurement(lambda: _checksum(op, result))

    return run


def _floor(op: str, selector: dict) -> Callable[..., Measurement]:
    def run(_graph, prepared) -> Measurement:
        result = None
        for _ in range(FLOOR_CALLS):
            result = _call(op, prepared, selector)
        return Measurement(lambda: _checksum(op, result), {"calls": FLOOR_CALLS})

    return run


OPS = ("pairs", "counts", "moments", "relatives")
DEGREES = {"pairs": DEGREE, "counts": 5, "moments": DEGREE, "relatives": DEGREE}

ARMS = (
    *(
        Arm(
            f"{op}_{kind}_d{DEGREES[op]}",
            _query(op, {"max_degree": DEGREES[op]}),
            label=f"`{op}` on the {'graph' if kind == 'graph' else kind.removeprefix('view') + '% view'}, degree {DEGREES[op]}",
            setup=_receiver(kind if kind == "graph" else kind.removeprefix("view"), op),
        )
        for op in OPS
        for kind in ("graph", "view05", "view50")
    ),
    *(
        Arm(
            f"floor_{op}_empty",
            _floor(op, {"categories": []}),
            label=f"`{op}`, empty selection on the graph, x{FLOOR_CALLS}",
            setup=_receiver("graph", op),
        )
        for op in OPS
    ),
    *(
        Arm(
            f"floor_{op}_view{size}",
            _floor(op, {"max_degree": DEGREES[op]}),
            label=f"`{op}` on a {size}-row view, x{FLOOR_CALLS}",
            setup=_receiver(size, op),
        )
        for op in OPS
        for size in ("0", "1")
    ),
)

FIXTURES = (
    parity_fixture("random_30k", label="`random_30k`"),
    parity_fixture("random_300k", label="`random_300k`"),
    parity_fixture("random_1k", label="`random_1k`"),
)

SUITE = Suite(
    name="relationship_queries",
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
