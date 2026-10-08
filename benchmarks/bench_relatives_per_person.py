"""``relatives_per_person`` against ``relationship_pairs`` plus ``np.add.at`` on the simACE pedsum pedigrees (issue #33).

    pixi run python benchmarks/bench_relatives_per_person.py --repeat 3 --out /tmp/rpp.json
    pixi run python benchmarks/bench_relatives_per_person.py --render /tmp/rpp.json

Two configurations, each an ``engine`` and a ``pairs`` arm that produce the
same folded int32 arrays, so the two arms of a configuration print the same
checksum.  Both fold the nine registry codes into the eight EPIMIGHT kinds
(``PO FS HS mHS pHS Av 1G 1C``), crediting only the junior member of
``PO``, ``Av`` and ``1G``.

``totals`` is pedsum's ``epimight-input`` skeleton (``pedsum/epimight.py``):
the ``relatives`` column per kind.  ``epimight`` is fitACE_epimight's
``create_input`` with aligned counts: per trait and kind, the total and the
number of relatives affected with onset at or before the person's cutoff.
The ``engine`` arms run one ``relatives_per_person`` call (with two
threshold columns sharing one cutoff array for ``epimight``) and fold with
``RelativesPerPerson.sum``; the ``pairs`` arms run the consumers' degree-3
``relationship_pairs`` extraction and their ``np.add.at`` kernels, copied
here rather than imported.

The trait inputs are synthetic and seeded, drawn in the arm's untimed setup,
so both ``epimight`` arms see identical arrays.  ``--render`` and a sweep
with ``--out`` exit 1 and name the pair when two arms of one configuration
disagree.  ``pedsum_20M`` is declared but not measured by default: pass
``--only pedsum_20M/...`` cells to run it.  ``random_1k`` is the parity
pedigree, for a quick end-to-end run of the script.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _harness import (
    RESULTS,
    Arm,
    Measurement,
    Outcome,
    Prepared,
    RunOrder,
    Suite,
    as_measured,
    candidate_of,
    checksum_array,
    file_fixture,
    main,
    parity_fixture,
    render_markdown,
    verify_report,
)

if TYPE_CHECKING:
    from _harness import Fixture, Report

PEDSUM = {
    "pedsum_2M": ("bench_pedsum/pedsum_2M/rep1/pedigree.full.parquet", "`pedsum_2M/rep1` (2,000,000 rows)"),
    "pedsum_20M": ("bench_pedsum/pedsum_20M/rep1/pedigree.full.parquet", "`pedsum_20M/rep1` (20,000,000 rows)"),
}

KINDS: tuple[tuple[str, tuple[str, ...], bool], ...] = (
    ("PO", ("MO", "FO"), True),
    ("FS", ("FS", "MZ"), False),
    ("HS", ("MHS", "PHS"), False),
    ("mHS", ("MHS",), False),
    ("pHS", ("PHS",), False),
    ("Av", ("Av",), True),
    ("1G", ("GP",), True),
    ("1C", ("1C",), False),
)
"""EPIMIGHT kind, its registry codes, and whether only the junior member is credited (pedsum ``_EPIMIGHT_RELS``)."""

CODES = tuple(dict.fromkeys(code for _, codes, _ in KINDS for code in codes))
PAIRS_MAX_DEGREE = 3
"""The consumers' extraction degree: ``1C`` is the deepest code they read."""

TRAITS = ("trait1", "trait2")
PREVALENCE = {"trait1": 0.1, "trait2": 0.2}
SEED = 33


@dataclass(frozen=True)
class TraitInputs:
    """Per-row inputs of the ``epimight`` configuration, as ``create_input`` holds them."""

    affected: dict[str, np.ndarray]
    onset: dict[str, np.ndarray]
    cutoff: np.ndarray


def _trait_inputs(graph: Any) -> Prepared:
    rng = np.random.default_rng(SEED)
    n = graph.n_individuals
    affected = {trait: rng.random(n) < PREVALENCE[trait] for trait in TRAITS}
    onset = {trait: rng.uniform(0.0, 80.0, n).astype(np.float32) for trait in TRAITS}
    cutoff = rng.uniform(0.0, 100.0, n)
    facts = {f"affected_{trait}": int(affected[trait].sum()) for trait in TRAITS}
    return Prepared(TraitInputs(affected, onset, cutoff), facts)


def _pair_blocks(pairs: Any, codes: tuple[str, ...]) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """``relationship_pair_arrays`` (fitACE) and ``_relationship_pair_blocks`` (pedsum): the non-empty blocks."""
    return tuple((block.first_rows, block.second_rows) for code in codes if len(block := pairs[code]) > 0)


def _count_total_relatives(pair_list, n, unidirectional=False):
    """``count_total_relatives``, identical in pedsum ``epimight.py`` and fitACE_epimight ``pair_extraction.py``."""
    counts = np.zeros(n, dtype=np.int64)
    for idx1, idx2 in pair_list:
        if len(idx1) == 0:
            continue
        np.add.at(counts, idx1, 1)
        if not unidirectional:
            np.add.at(counts, idx2, 1)
    return counts.astype(np.int32, copy=False)


def _count_affected_relatives(pair_list, affected, n, unidirectional=False, onset=None, cutoff=None):
    """fitACE_epimight ``pair_extraction.count_affected_relatives``."""

    def hits(person, relative):
        hit = affected[relative]
        if cutoff is not None:
            hit = hit & (onset[relative] <= cutoff[person])
        return hit.astype(np.int64)

    counts = np.zeros(n, dtype=np.int64)
    for idx1, idx2 in pair_list:
        if len(idx1) == 0:
            continue
        np.add.at(counts, idx1, hits(idx1, idx2))
        if not unidirectional:
            np.add.at(counts, idx2, hits(idx2, idx1))
    return counts.astype(np.int32, copy=False)


def _folded(folds: list[np.ndarray], pairs: Any = None) -> Measurement:
    def facts() -> dict[str, Any]:
        credited = {"folds": len(folds), "credited": sum(int(f.sum(dtype=np.int64)) for f in folds)}
        return credited if pairs is None else credited | {"pairs": sum(len(pairs[code]) for code in pairs)}

    return Measurement(lambda: checksum_array(np.stack(folds)), facts)


def _totals_engine(graph: Any, _: object) -> Measurement:
    result = graph.relatives_per_person(categories=CODES)
    folds = [result.sum(codes).astype(np.int32) for _, codes, _ in KINDS]
    return _folded(folds)


def _totals_pairs(graph: Any, _: object) -> Measurement:
    n = graph.n_individuals
    pairs = graph.relationship_pairs(max_degree=PAIRS_MAX_DEGREE)
    folds = [
        _count_total_relatives(_pair_blocks(pairs, codes), n, unidirectional=directional)
        for _, codes, directional in KINDS
    ]
    return _folded(folds, pairs)


def _epimight_engine(graph: Any, inputs: TraitInputs) -> Measurement:
    thresholds = {
        trait: (np.where(inputs.affected[trait], inputs.onset[trait], np.float64(np.nan)), inputs.cutoff)
        for trait in TRAITS
    }
    result = graph.relatives_per_person(categories=CODES, thresholds=thresholds)
    totals = [result.sum(codes).astype(np.int32) for _, codes, _ in KINDS]
    folds = []
    for trait in TRAITS:
        for (_, codes, _), total in zip(KINDS, totals, strict=True):
            folds += [total, result.sum(codes, trait).astype(np.int32)]
    return _folded(folds)


def _epimight_pairs(graph: Any, inputs: TraitInputs) -> Measurement:
    n = graph.n_individuals
    pairs = graph.relationship_pairs(max_degree=PAIRS_MAX_DEGREE)
    folds = []
    for trait in TRAITS:
        for _, codes, directional in KINDS:
            pair_list = _pair_blocks(pairs, codes)
            diagnosed = _count_affected_relatives(
                pair_list,
                inputs.affected[trait],
                n,
                unidirectional=directional,
                onset=inputs.onset[trait],
                cutoff=inputs.cutoff,
            )
            total = _count_total_relatives(pair_list, n, unidirectional=directional)
            folds += [total, diagnosed]
    return _folded(folds, pairs)


def pedsum_fixture(name: str) -> Fixture:
    """The pedsum pedigree at the moments benchmark's path, unavailable when not generated here."""
    relative, label = PEDSUM[name]
    return file_fixture(name, RESULTS / relative, label=label)


TWELVE = {"PEDIGREE_GRAPH_THREADS": "12"}
ENGINE = "`relatives_per_person` + `sum`"
PAIRS = "`relationship_pairs` + `np.add.at`"
CONFIGURATIONS: dict[str, tuple[Arm, ...]] = {
    "totals": (
        Arm("totals_engine_12t", _totals_engine, label=f"totals: {ENGINE}, 12 threads", env=TWELVE),
        Arm("totals_pairs_12t", _totals_pairs, label=f"totals: {PAIRS}, 12 threads", env=TWELVE),
        Arm("totals_engine_1t", _totals_engine, label=f"totals: {ENGINE}, 1 thread"),
    ),
    "epimight": (
        Arm(
            "epimight_engine_12t",
            _epimight_engine,
            label=f"epimight: {ENGINE}, K=2, 12 threads",
            setup=_trait_inputs,
            env=TWELVE,
        ),
        Arm(
            "epimight_pairs_12t",
            _epimight_pairs,
            label=f"epimight: {PAIRS}, 12 threads",
            setup=_trait_inputs,
            env=TWELVE,
        ),
        Arm("epimight_engine_1t", _epimight_engine, label=f"epimight: {ENGINE}, K=2, 1 thread", setup=_trait_inputs),
    ),
}
"""The arms of each configuration, which must all print the same checksum on one fixture."""

ARMS = tuple(arm for arms in CONFIGURATIONS.values() for arm in arms)
CONFIGURATION = {arm.name: configuration for configuration, arms in CONFIGURATIONS.items() for arm in arms}

SUITE = Suite(
    name="relatives_per_person",
    note=Path(__file__).with_suffix(".md"),
    fixtures=(
        pedsum_fixture("pedsum_2M"),
        pedsum_fixture("pedsum_20M"),
        parity_fixture("random_1k", label="`random_1k` parity pedigree"),
    ),
    arms=ARMS,
    cells=tuple(f"pedsum_2M/{arm.name}" for arm in ARMS),
    gate=None,
    order=RunOrder.INTERLEAVED,
    timeout_s=7200.0,
)


def checksum_mismatches(report: Report) -> list[str]:
    """One line per fixture and configuration whose completed arms disagree or are unstable across repetitions."""
    groups: dict[tuple[str, str], dict[str, int | None]] = {}
    for result in report.cells:
        if result.outcome is Outcome.COMPLETED:
            key = (result.cell.fixture, CONFIGURATION[candidate_of(result.cell.arm)])
            groups.setdefault(key, {})[result.cell.arm] = result.checksum
    return [
        f"{fixture} {configuration}: " + ", ".join(f"{arm}={checksum}" for arm, checksum in sorted(checksums.items()))
        for (fixture, configuration), checksums in sorted(groups.items())
        if None in checksums.values() or len(set(checksums.values())) > 1
    ]


def _main() -> NoReturn:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--render", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--cell")
    known, _ = parser.parse_known_args()
    if known.render is not None:
        report = verify_report(known.render)
        print(render_markdown(as_measured(SUITE, report), report))
    else:
        try:
            main(SUITE)
        except SystemExit as done:
            if known.cell or known.out is None or not known.out.exists() or done.code:
                raise
        report = verify_report(known.out)
    mismatches = checksum_mismatches(report)
    for line in mismatches:
        print(f"CHECKSUM MISMATCH {line}")
    raise SystemExit(1 if mismatches else 0)


if __name__ == "__main__":
    _main()
