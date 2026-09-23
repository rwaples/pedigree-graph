"""The three kinship-matrix products across builds: the slice 14 benchmark.

``kinship_matrix()``, ``approximate_kinship_matrix(0.001)`` and
``mean_kinship_by_generation()`` share one depth-major DP kernel.  Slice 14
moved it to the Rust core, and the sweep interleaves two builds in one run,
as ADR 0007 requires for a gated comparison:

* ``wheel``: the 0.9.1 PyPI wheel as the simACE umbrella env installs it,
  run under that env's interpreter;
* ``source``: this checkout, built into this repo's env.

The 14a bake-off ran a third arm, the 0.9.1 arena allocator ported as it
was, selected through an environment variable; both went with the record.
Every record carries the import path and versions of the package the child
measured.

Each fixture here is one input crossed with one product, built fresh in the
child so no product ever takes a cached route (the summary walks the complete
matrix when a graph already holds it).  Every arm returns a checksum over the
whole CSC structure, or over the summary arrays, so the cells also show the
three builds returning the same bytes.

    python benchmarks/bench_kinship_matrix.py --repeat 3 --out benchmarks/reports/kinship_matrix.json
    python benchmarks/bench_kinship_matrix.py --render benchmarks/reports/kinship_matrix.json

The slice 14 record is ``docs/pedigree-graph-0.8-migration/gate/14a/NOTES.md``.
"""

from __future__ import annotations

import sys
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _harness import (
    WHEEL_INTERPRETER,
    Arm,
    Fixture,
    Gate,
    Measurement,
    RunOrder,
    Suite,
    main,
    parity_fixture,
    study_fixture,
)

if TYPE_CHECKING:
    from collections.abc import Callable

THRESHOLD = 0.001


@dataclass(frozen=True)
class Query:
    """One graph and the product to time on it; ``n_individuals`` is what the harness records."""

    graph: object
    n_individuals: int
    run: Callable[[], Measurement]


def _crc(*arrays) -> int:
    checksum = 0
    for array in arrays:
        checksum = zlib.crc32(array.tobytes(), checksum)
    return checksum


def _matrix_measurement(matrix) -> Measurement:
    return Measurement(_crc(matrix.indptr, matrix.indices, matrix.data), {"nnz": int(matrix.nnz)})


def _complete(graph) -> Measurement:
    return _matrix_measurement(graph.kinship_matrix())


def _approximate(graph) -> Measurement:
    return _matrix_measurement(graph.approximate_kinship_matrix(min_propagated_kinship=THRESHOLD))


def _summary(graph) -> Measurement:
    summary = graph.mean_kinship_by_generation()
    return Measurement(
        _crc(summary.generations, summary.mean_kinship, summary.pair_counts),
        {"groups": int(summary.generations.shape[0])},
    )


PRODUCTS: dict[str, tuple[str, Callable[[object], Measurement]]] = {
    "km": ("`kinship_matrix()`", _complete),
    "akm": (f"`approximate_kinship_matrix({THRESHOLD})`", _approximate),
    "mkg": ("`mean_kinship_by_generation()`", _summary),
}


def _query_fixture(name: str, base: Fixture, product: str) -> Fixture:
    label, run = PRODUCTS[product]

    def build() -> Query:
        graph = base.build()
        return Query(graph, int(graph.n_individuals), lambda: run(graph))

    return Fixture(
        name=name,
        label=f"{base.label}, {label}",
        build=build,
        provenance=lambda: f"{base.provenance()}/{product}",
        available=base.available,
    )


def _run(subject, _payload) -> Measurement:
    if isinstance(subject, Query):
        return subject.run()
    # The harness warms an arm on a four-row graph before the fixture is
    # built; that graph carries no query, so warm every product's kernel.
    for _label, run in PRODUCTS.values():
        run(subject)
    return Measurement(0, {})


_BASES = {
    "random_1k": parity_fixture("random_1k", label="`random_1k`"),
    "random_30k": parity_fixture("random_30k", label="`random_30k`"),
    **{name: study_fixture(name) for name in ("dev_mean_n10k", "dev_cont_n10k", "baseline10K", "baseline100K")},
}

_CELLS = (
    ("km-1k", "random_1k", "km"),
    ("km-20k", "dev_mean_n10k", "km"),
    ("km-30k", "random_30k", "km"),
    ("km-53k", "baseline10K", "km"),
    ("akm-20k", "dev_cont_n10k", "akm"),
    ("akm-30k", "random_30k", "akm"),
    ("akm-53k", "baseline10K", "akm"),
    ("mkg-30k", "random_30k", "mkg"),
    ("mkg-536k", "baseline100K", "mkg"),
)

SUITE = Suite(
    name="kinship_matrix",
    note=Path(__file__).with_suffix(".md"),
    fixtures=tuple(_query_fixture(name, _BASES[base], product) for name, base, product in _CELLS),
    arms=(
        Arm("wheel", _run, label="0.9.1 wheel (simACE env)", interpreter=WHEEL_INTERPRETER),
        Arm("source", _run, label="source build (this env)"),
    ),
    gate=Gate(baseline="wheel", gated=frozenset({"source"})),
    order=RunOrder.INTERLEAVED,
    timeout_s=3600.0,
)

if __name__ == "__main__":
    main(SUITE)
