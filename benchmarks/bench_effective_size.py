"""Serial ``estimate_effective_sizes`` against the pooled 0.7.1 adapter (slice 6c-3).

ADR 0007 removes the effective-size worker pool from the final path: the eight
estimators and their lazy prerequisites run serially in canonical order.  The
question this suite answers is whether that costs wall time or peak memory
against the ``compute_all_ne`` adapter, which keeps the 0.7.1 execution shape
(every prerequisite built eagerly, then the formulas dispatched serially or
on a ``ThreadPoolExecutor``).

    pixi run python benchmarks/bench_effective_size.py --repeat 5 --out benchmarks/reports/effective_size.json
    pixi run python benchmarks/bench_effective_size.py --render benchmarks/reports/effective_size.json

The arms share every formula (slice 6c-1 routed the adapter through the same
evaluators), so the checksum over the eight scalar estimates must agree across
arms; only the orchestration differs.  ``adapter_pool4`` is the baseline and
``estimate_serial`` the gated subject: a confident regression over the family
5% rule blocks.  ``adapter_serial`` is shown for the cost of the pool itself.

The fixtures are closed-parentage Wright-Fisher pedigrees with dense generation
labels, sex, and birth years, built by ``tests/parity/generate_ne_baseline.py``
(the slice-6c parity generator) so every estimator, Hill's birth-year branch
included, runs on every arm.  The parity corpus is not used because its random
pedigrees carry external parents, which the founder-based estimators now refuse.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests" / "parity"))

from _harness import Arm, Fixture, Gate, Measurement, RunOrder, Suite, checksum_values, main

_FIXTURES = {
    "wf_n2000_g8": {"seed": 31, "n_per_gen": 2000, "n_gens": 8},
    "wf_n5000_g8": {"seed": 37, "n_per_gen": 5000, "n_gens": 8},
}


def _frame(name: str):
    import generate_ne_baseline as gen

    params = _FIXTURES[name]
    return gen.with_birth_years(gen.random_mating(**params))


def _wf_fixture(name: str) -> Fixture:
    params = _FIXTURES[name]

    def build():
        from pedigree_graph import PedigreeGraph

        return PedigreeGraph(_frame(name))

    def provenance() -> str:
        digest = hashlib.sha256()
        frame = _frame(name)
        for column in frame.columns:
            digest.update(column.encode())
            digest.update(np.ascontiguousarray(frame[column].to_numpy()).tobytes())
        return digest.hexdigest()

    label = f"`{name}` (seed {params['seed']}, {params['n_per_gen']} per generation, {params['n_gens']} generations)"
    return Fixture(name=name, label=label, build=build, provenance=provenance)


def _scalars(results) -> Measurement:
    values = np.array([np.nan if r.ne is None else float(r.ne) for _, r in sorted(results.items())], dtype=np.float64)
    return Measurement(
        checksum_values(np.nan_to_num(values, nan=-1.0)), {"n_estimates": int(np.isfinite(values).sum())}
    )


def _adapter_serial(graph, _prepared) -> Measurement:
    from pedigree_graph import compute_all_ne

    return _scalars(compute_all_ne(graph, n_threads=1))


def _adapter_pool4(graph, _prepared) -> Measurement:
    from pedigree_graph import compute_all_ne

    return _scalars(compute_all_ne(graph, n_threads=4))


def _estimate_serial(graph, _prepared) -> Measurement:
    from pedigree_graph.effective_size import estimate_effective_sizes

    return _scalars(estimate_effective_sizes(graph))


SUITE = Suite(
    name="effective_size",
    note=Path(__file__).with_suffix(".md"),
    fixtures=(_wf_fixture("wf_n2000_g8"), _wf_fixture("wf_n5000_g8")),
    arms=(
        Arm("adapter_pool4", _adapter_pool4, label="`compute_all_ne(n_threads=4)`, eager prerequisites + pool"),
        Arm("adapter_serial", _adapter_serial, label="`compute_all_ne(n_threads=1)`, eager prerequisites, serial"),
        Arm("estimate_serial", _estimate_serial, label="`estimate_effective_sizes`, lazy prerequisites, serial"),
    ),
    gate=Gate(baseline="adapter_pool4", gated=frozenset({"estimate_serial"})),
    order=RunOrder.INTERLEAVED,
    timeout_s=3600.0,
)

if __name__ == "__main__":
    main(SUITE)
