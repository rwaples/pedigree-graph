"""Descendant paths and the Ne prerequisites across builds: the slice 15 benchmark.

Slice 15 moves ``descendant_path_counts()``, Maignel's equivalent complete
generations (EqG) and the per-cohort founder-contribution means to the Rust
core.  EqG and the founder means have no public method of their own, so they
are timed through the estimators that consume them: ``ne_individual_delta_f``
(F and EqG) and ``ne_long_term_contributions`` (the founder means).  The
``ne-45k`` cell runs ``estimate_effective_sizes`` with every estimator but
``ne_coancestry``, whose kinship summary slice 14 already moved.

The sweep interleaves two builds in one run, as ADR 0007 requires:

* ``wheel``: the 0.9.3 PyPI wheel as the simACE umbrella env installs it, run
  under that env's interpreter;
* ``source``: this checkout, built into this repo's env.

Each fixture is one input crossed with one product, built fresh in a fresh
child, so no product reuses a prerequisite another computed.  Checksums are
SHA-256 over the int64 counts, or over every field of the result record with
floats as their float64 bytes, so the cells also say where the two builds
returned the same bits.

    python benchmarks/bench_lineage_ne.py --repeat 3 --out benchmarks/reports/lineage_ne.json
    python benchmarks/bench_lineage_ne.py --render benchmarks/reports/lineage_ne.json
"""

from __future__ import annotations

import hashlib
import sys
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

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
    wf_fixture,
)

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True)
class Query:
    """One graph and the product to time on it; ``n_individuals`` is what the harness records."""

    graph: object
    n_individuals: int
    run: Callable[[], Measurement]


def _feed(digest, value: object) -> None:
    """Hash one result field: arrays and floats by their bytes, integers by value, the rest by ``repr``."""
    if isinstance(value, np.ndarray):
        contiguous = np.ascontiguousarray(value)
        digest.update(f"{contiguous.dtype.str}{contiguous.shape};".encode())
        digest.update(contiguous.tobytes())
    elif isinstance(value, float | np.floating):
        digest.update(np.float64(value).tobytes())
    elif isinstance(value, int | np.integer) and not isinstance(value, bool | np.bool_):
        digest.update(str(int(value)).encode())
    elif is_dataclass(value):
        for item in fields(value):
            digest.update(f"{item.name}=".encode())
            _feed(digest, getattr(value, item.name))
    elif isinstance(value, Mapping):
        for key in value:
            digest.update(f"{key}=".encode())
            _feed(digest, value[key])
    elif hasattr(value, "_asdict"):
        _feed(digest, value._asdict())
    else:
        digest.update(repr(value).encode())
    digest.update(b";")


def _record(result) -> Measurement:
    digest = hashlib.sha256()
    _feed(digest, result)
    ne = getattr(result, "ne", None)
    return Measurement(int.from_bytes(digest.digest()[:8], "big"), {"ne": None if ne is None else float(ne)})


def _descendant_paths(graph) -> Measurement:
    counts = graph.descendant_path_counts()
    return Measurement(lambda: checksum_array(counts), lambda: {"max_paths": int(counts.max(initial=0))})


def _individual_delta_f(graph) -> Measurement:
    from pedigree_graph.effective_size import ne_individual_delta_f

    return _record(ne_individual_delta_f(graph))


def _long_term_contributions(graph) -> Measurement:
    from pedigree_graph.effective_size import ne_long_term_contributions

    return _record(ne_long_term_contributions(graph))


def _estimates(graph) -> Measurement:
    from pedigree_graph.effective_size import ALL_EFFECTIVE_SIZE_ESTIMATORS, estimate_effective_sizes

    results = estimate_effective_sizes(graph, [n for n in ALL_EFFECTIVE_SIZE_ESTIMATORS if n != "ne_coancestry"])
    digest = hashlib.sha256()
    _feed(digest, dict(results))
    resolved = sum(getattr(r, "ne", None) is not None for r in results.values())
    return Measurement(int.from_bytes(digest.digest()[:8], "big"), {"n_estimates": resolved})


PRODUCTS: dict[str, tuple[str, Callable[[object], Measurement]]] = {
    "dp": ("`descendant_path_counts()`", _descendant_paths),
    "idf": ("`ne_individual_delta_f` (F and EqG)", _individual_delta_f),
    "ltc": ("`ne_long_term_contributions` (founder means)", _long_term_contributions),
    "ne": ("`estimate_effective_sizes` without `ne_coancestry`", _estimates),
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
    "random_300k": parity_fixture("random_300k", label="`random_300k`"),
    "baseline100K": study_fixture("baseline100K"),
    "wf_n2000_g8": wf_fixture("wf_n2000_g8"),
    "wf_n5000_g8": wf_fixture("wf_n5000_g8"),
}

_CELLS = (
    ("ltc-18k", "wf_n2000_g8", "ltc"),
    ("idf-45k", "wf_n5000_g8", "idf"),
    ("ltc-45k", "wf_n5000_g8", "ltc"),
    ("ne-45k", "wf_n5000_g8", "ne"),
    ("dp-300k", "random_300k", "dp"),
    ("dp-536k", "baseline100K", "dp"),
)

SUITE = Suite(
    name="lineage_ne",
    note=Path(__file__).with_suffix(".md"),
    fixtures=tuple(_query_fixture(name, _BASES[base], product) for name, base, product in _CELLS),
    arms=(
        Arm("wheel", _run, label="0.9.3 wheel (simACE env)", interpreter=WHEEL_INTERPRETER),
        Arm("source", _run, label="source build (this env)"),
    ),
    gate=Gate(baseline="wheel", gated=frozenset({"source"})),
    order=RunOrder.INTERLEAVED,
    timeout_s=3600.0,
)

if __name__ == "__main__":
    main(SUITE)
