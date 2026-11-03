"""``PedigreeGraph.from_arrays`` construction across builds (issue #58).

ADR 0020 adds a validation pass to construction: a byte per row marks the
represented parents' roles, and the distinct external parent ids of each role
are intersected.  Sparse ids put the id index on its sorted layout, whose
lookup lists the external ids in order as it goes; dense ids use a table, and
the external ids are sorted afterwards, so the external-heavy fixtures come in
both.  Gate this checkout's build against another with ``--baseline-python``:

    python benchmarks/bench_construction.py --baseline-python <pg-base>/.pixi/envs/default/bin/python --repeat 5

Each record carries a SHA-256 checksum of the resolved parent rows.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests" / "parity"))

from _harness import (
    Arm,
    Fixture,
    Gate,
    Measurement,
    Prepared,
    RunOrder,
    Suite,
    checksum_array,
    main,
    parity_fixture,
)

_N = 5_000_000
_WINDOW = 1_000


def _synthetic(external_share: float, seed: int, *, sparse: bool) -> dict[str, np.ndarray]:
    """*_N* rows; each parent is missing, an earlier row, or external.

    Even rows are mothers and odd rows fathers.  External mothers and fathers
    come from disjoint pools of ``_N // 5`` ids each, so every id plays one
    parent role.
    """
    rng = np.random.default_rng(seed)
    rows = np.arange(_N, dtype=np.int64)
    ids = rows * 7 + 3 if sparse else rows
    columns = {"ids": ids}
    pool = _N // 5
    for name, parity, external_base in (("mother", 0, 10 * _N), ("father", 1, 20 * _N)):
        earlier = np.maximum(rows - 1 - rng.integers(0, _WINDOW, _N), 0)
        earlier -= (earlier - parity) % 2
        parent = np.where(earlier >= 0, ids[np.maximum(earlier, 0)], -1)
        kind = rng.random(_N)
        parent = np.where(kind < external_share, external_base + rng.integers(0, pool, _N), parent)
        parent[kind > 0.9] = -1
        parent[rows < 2 * _WINDOW] = -1
        columns[name] = parent
    return columns


def _synthetic_fixture(name: str, external_share: float, seed: int, *, sparse: bool = True) -> Fixture:
    def build():
        from pedigree_graph import PedigreeGraph

        columns = _synthetic(external_share, seed, sparse=sparse)
        return PedigreeGraph.from_arrays(ids=columns["ids"], mother_ids=columns["mother"], father_ids=columns["father"])

    def provenance() -> str:
        return f"{name}: seed {seed}, {_N:,} rows, sparse {sparse}, external share {external_share}"

    label = f"`{name}` ({_N:,} rows, {'sparse' if sparse else 'dense'} ids, {external_share:.0%} external parents)"
    return Fixture(name=name, label=label, build=build, provenance=provenance)


def _columns(graph) -> Prepared:
    """The constructor arrays the fixture's graph was built from."""
    return Prepared({"ids": graph.ids, "mother": graph.mother_ids, "father": graph.father_ids})


def _build(_fixture, columns) -> Measurement:
    from pedigree_graph import PedigreeGraph

    graph = PedigreeGraph.from_arrays(ids=columns["ids"], mother_ids=columns["mother"], father_ids=columns["father"])
    return Measurement(lambda: checksum_array(np.concatenate([graph.mother_rows, graph.father_rows])))


SUITE = Suite(
    name="construction",
    note=Path(__file__).with_suffix(".md"),
    fixtures=(
        parity_fixture("random_300k", label="`random_300k`"),
        _synthetic_fixture("rows_5m", external_share=0.0, seed=58),
        _synthetic_fixture("external_5m", external_share=0.6, seed=5858),
        _synthetic_fixture("dense_external_5m", external_share=0.6, seed=5859, sparse=False),
    ),
    arms=(Arm("source", _build, label="source build (this env)", setup=_columns),),
    gate=Gate(baseline=None),
    order=RunOrder.INTERLEAVED,
    timeout_s=1800.0,
)

if __name__ == "__main__":
    main(SUITE)
