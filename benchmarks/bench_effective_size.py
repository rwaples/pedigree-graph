"""Production-scale baseline for serial ``estimate_effective_sizes`` (ADR 0007).

Records what the eight estimators and their lazily built prerequisites cost in
wall time and peak RSS when run serially in canonical order, which is the only
execution shape the 0.8 surface offers.

    pixi run python benchmarks/bench_effective_size.py --repeat 5 --out benchmarks/reports/effective_size.json
    pixi run python benchmarks/bench_effective_size.py --render benchmarks/reports/effective_size.json

There is no gate.  The pooled and serial ``compute_all_ne`` arms this was
compared against were deleted with the 0.7.1 adapters, so what remains is one
arm recording a baseline rather than an A/B; ``benchmarks/bench_effective_size.md``
keeps their measured rows.

Ordering is ``GROUPED`` rather than interleaved.  With one arm there is nothing
to interleave against, and the larger cell runs for close to half an hour, so
finishing a cell before starting the next keeps an interrupted sweep useful.

The fixtures are closed-parentage Wright-Fisher pedigrees with dense generation
labels, sex, and birth years, built by ``tests/parity/generate_ne_baseline.py``
(the slice-6c parity generator) so every estimator, Hill's birth-year branch
included, runs.  The parity corpus is not used because its random pedigrees
carry external parents, which the founder-based estimators refuse.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests" / "parity"))

from _harness import Arm, Fixture, Measurement, RunOrder, Suite, checksum_values, main

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

        return PedigreeGraph.from_frame(_frame(name))

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
    """Checksum the eight scalar estimates, and surface how many resolved.

    An estimator that refused the pedigree carries no ``ne``, so it reads as
    ``nan`` here and drops out of ``n_estimates`` rather than crashing the arm.
    """
    values = np.array(
        [float(ne) if (ne := getattr(r, "ne", None)) is not None else np.nan for _, r in sorted(results.items())],
        dtype=np.float64,
    )
    return Measurement(
        checksum_values(np.nan_to_num(values, nan=-1.0)), {"n_estimates": int(np.isfinite(values).sum())}
    )


def _estimate_serial(graph, _prepared) -> Measurement:
    from pedigree_graph.effective_size import estimate_effective_sizes

    return _scalars(estimate_effective_sizes(graph))


SUITE = Suite(
    name="effective_size",
    note=Path(__file__).with_suffix(".md"),
    # Cheapest first, so an interrupted sweep still renders complete cells.
    # With one arm the full product is already this order, so no cells list.
    fixtures=(_wf_fixture("wf_n2000_g8"), _wf_fixture("wf_n5000_g8")),
    arms=(Arm("estimate_serial", _estimate_serial, label="`estimate_effective_sizes`, lazy prerequisites, serial"),),
    gate=None,
    order=RunOrder.GROUPED,
    timeout_s=3600.0,
)

if __name__ == "__main__":
    main(SUITE)
