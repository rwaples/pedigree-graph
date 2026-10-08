"""The two-build gate: ``build_pair``, per-arm baselines, checksum agreement and build identity."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "benchmarks"))

from _harness import (  # noqa: E402
    BUILD_GATE_REPEATS,
    WHEEL_INTERPRETER,
    Arm,
    Cell,
    CellResult,
    Fixture,
    Gate,
    Measurement,
    Outcome,
    Report,
    RunOrder,
    RunRecord,
    Suite,
    Verdict,
    _schedule,
    as_measured,
    build_pair,
    compare,
    package_facts,
)

BASE_PYTHON = Path("/elsewhere/.pixi/envs/default/bin/python")


def _run(_graph, _payload) -> Measurement:
    return Measurement(0)


def _suite() -> Suite:
    fixture = Fixture("f", "`f`", build=lambda: None, provenance=lambda: "")
    return Suite(
        name="s",
        note=Path("s.md"),
        fixtures=(fixture, Fixture("g", "`g`", build=lambda: None, provenance=lambda: "")),
        arms=(
            Arm("wheel", _run, label="wheel", interpreter=WHEEL_INTERPRETER),
            Arm("one", _run, label="one", env={"PEDIGREE_GRAPH_THREADS": "12"}),
            Arm("two", _run, label="two"),
        ),
        cells=("f/one", "f/wheel", "g/two"),
        order=RunOrder.GROUPED,
    )


def _record(cell: Cell, wall: float, checksum: int = 7, native: str = "aaaa") -> RunRecord:
    return RunRecord(
        cell=cell,
        wall_s=wall,
        peak_rss_mib=100.0,
        baseline_rss_mib=0.0,
        ru_maxrss_mib=0.0,
        checksum=checksum,
        n_individuals=1,
        facts={"native_sha256": native, "package_git": "v1", "core_version": "1", "native_file": f"/{native}.so"},
        environment="e",
        started_at="",
    )


def _report(base: list[RunRecord], candidate: list[RunRecord]) -> Report:
    return Report(
        "s",
        (
            CellResult(Cell("f", "one@base"), Outcome.COMPLETED, tuple(base)),
            CellResult(Cell("f", "one"), Outcome.COMPLETED, tuple(candidate)),
        ),
        {},
    )


def test_build_pair_twins_every_own_build_arm_and_drops_pinned_ones():
    paired = build_pair(_suite(), BASE_PYTHON)
    arms = {arm.name: arm for arm in paired.arms}
    assert set(arms) == {"one@base", "one", "two@base", "two"}
    assert arms["one@base"].interpreter == BASE_PYTHON
    assert arms["one"].interpreter is None
    assert arms["one@base"].env == arms["one"].env == {"PEDIGREE_GRAPH_THREADS": "12"}
    assert arms["one@base"].run is arms["one"].run
    assert [str(cell) for cell in paired.resolved_cells()] == ["f/one@base", "f/one", "g/two@base", "g/two"]
    assert paired.order is RunOrder.INTERLEAVED
    assert paired.gate is not None
    assert paired.gate.baseline_of("one") == "one@base"
    assert paired.gate.baseline_of("one@base") is None
    assert paired.gate.min_repeats == BUILD_GATE_REPEATS


def test_compare_needs_the_repetitions_and_disjoint_ranges():
    assert compare([1.0] * 4, [1.0] * 4, min_repeats=5) is Verdict.INCONCLUSIVE
    assert compare([1.0] * 5, [1.0] * 5, min_repeats=5) is Verdict.PASS
    assert compare([2.0] * 5, [1.0] * 5, min_repeats=5) is Verdict.BLOCK
    assert compare([2.0, 2.0, 2.0, 2.0, 0.5], [1.0] * 5, min_repeats=5) is Verdict.INCONCLUSIVE


def test_a_slower_candidate_blocks_against_its_own_base_twin():
    gate = build_pair(_suite(), BASE_PYTHON).gate
    report = _report(
        [_record(Cell("f", "one@base"), 1.0, native="base")] * 5, [_record(Cell("f", "one"), 2.0, native="cand")] * 5
    )
    assert report.verdicts(gate) == {Cell("f", "one"): Verdict.BLOCK}
    assert report.build_conflicts(gate) == []


def test_a_candidate_with_another_checksum_is_a_mismatch_whatever_its_speed():
    gate = build_pair(_suite(), BASE_PYTHON).gate
    report = _report(
        [_record(Cell("f", "one@base"), 1.0, native="base")],
        [_record(Cell("f", "one"), 1.0, checksum=8, native="cand")],
    )
    assert report.verdicts(gate) == {Cell("f", "one"): Verdict.MISMATCH}


def test_a_baseline_that_imported_the_candidate_build_is_reported():
    gate = build_pair(_suite(), BASE_PYTHON).gate
    report = _report([_record(Cell("f", "one@base"), 1.0)], [_record(Cell("f", "one"), 1.0)])
    (problem,) = report.build_conflicts(gate)
    assert "one and its baseline one@base imported the same native library" in problem


def test_a_report_from_before_build_identity_has_no_conflicts():
    old = replace(_record(Cell("f", "one"), 1.0), facts={})
    report = _report([replace(old, cell=Cell("f", "one@base"))], [old])
    assert report.build_conflicts(build_pair(_suite(), BASE_PYTHON).gate) == []


def test_a_rebuild_mid_sweep_is_reported():
    base = [_record(Cell("f", "one@base"), 1.0, native="x"), _record(Cell("f", "one@base"), 1.0, native="y")]
    report = _report(base, [])
    assert report.build_conflicts(None) == ["one@base imported 2 different builds"]


def test_a_source_tree_turning_dirty_mid_sweep_is_not_a_rebuild():
    first = _record(Cell("f", "one@base"), 1.0)
    dirty = replace(first, facts={**first.facts, "package_git": "v1-dirty"})
    assert _report([first, dirty], []).build_conflicts(None) == []


def test_an_ab_gate_of_two_arms_in_one_build_is_not_a_wrong_build():
    gate = Gate(baseline="one@base", gated=frozenset({"one"}))
    report = _report([_record(Cell("f", "one@base"), 1.0)], [_record(Cell("f", "one"), 1.0)])
    assert report.build_conflicts(gate) == []


def test_a_stored_two_build_report_renders_against_the_paired_suite():
    report = _report([_record(Cell("f", "one@base"), 1.0)], [_record(Cell("f", "one"), 1.0)])
    paired = as_measured(_suite(), report)
    assert {arm.name for arm in paired.arms} >= {"one@base", "one"}
    assert as_measured(paired, report) is paired


def test_package_facts_name_the_imported_native_library():
    facts = package_facts()
    assert Path(facts["native_file"]).name.startswith("_native")
    assert len(facts["native_sha256"]) == 16
    assert facts["python"] == sys.executable


def test_interleaved_rounds_alternate_which_arm_runs_first():
    base, candidate = Cell("f", "a@base"), Cell("f", "a")
    order = [cell for cell, _ in _schedule([base, candidate], 4, RunOrder.INTERLEAVED)]
    assert order == [base, candidate, candidate, base, base, candidate, candidate, base]
