"""Mechanical checks for the benchmark contract.

The 0.8.0 slice plan already said in prose that ``/tmp`` scripts must not be the
sole evidence for a benchmark number.  A slice then wrote four disposable
drivers into ``/tmp`` and cited them from a tracked note, and review did not
catch it.  These tests are that rule encoded, because the prose form did not
hold.

Neither check can stop someone writing a fifth throwaway driver; Python has no
way to require an import.  They catch the two symptoms that reached tracked
artifacts: a note whose method lives in ``/tmp``, and a result file that cannot
say what produced it.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "benchmarks"))

from _harness import GATE, ContractError, Environment, verify_report  # noqa: E402

# A cited /tmp path is only a violation when it is the *method* behind a number.
# A scratch output directory is not, so the pattern requires a script suffix and
# each accepted exception carries a written reason rather than a bare skip.
ACCEPTED_TMP_CITATIONS: dict[tuple[str, str], str] = {
    (
        "docs/adr/0001-dp-kinship-bench-split-rejected.md",
        "/tmp/numba_defaults_probe.py",
    ): (
        "Frozen record of a superseded, rejected decision. The ADR itself says the "
        "probe was kept for one cycle only; rewriting history would misreport it."
    ),
}


def _tracked_markdown() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z", "--", "*.md"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.skip("git unavailable")
    return [REPO / name for name in result.stdout.split("\0") if name]


def find_tmp_citations(text: str) -> list[str]:
    """Return cited ``/tmp`` *scripts*, which is what a method living in /tmp looks like.

    Deliberately not ``/tmp/\\S+``.  ``benchmarks/relationship_counts_rust.md``
    stages generated input under ``/tmp/bench`` with a tracked program, which is
    a destination rather than a provenance, and prose such as "disposable /tmp
    drivers" describes the incident without citing a path.
    """
    import re

    return sorted(set(re.findall(r"/tmp/[\w./-]+\.(?:py|sh|bash)", text)))


def test_no_tracked_note_cites_a_tmp_script():
    """A number whose method lives in /tmp cannot be re-derived by anyone else."""
    offenders: dict[str, list[str]] = {}
    for path in _tracked_markdown():
        relative = path.relative_to(REPO).as_posix()
        cited = [c for c in find_tmp_citations(path.read_text()) if (relative, c) not in ACCEPTED_TMP_CITATIONS]
        if cited:
            offenders[relative] = cited
    assert not offenders, (
        f"tracked note(s) cite a /tmp script as their method: {offenders}. "
        "Commit the driver under benchmarks/ or add a reasoned allowlist entry."
    )


def test_accepted_tmp_citations_are_still_present():
    """Drop an allowlist entry once its citation is gone, so the rule tightens."""
    stale = [
        f"{name}: {citation}"
        for (name, citation), _ in ACCEPTED_TMP_CITATIONS.items()
        if citation not in (REPO / name).read_text()
    ]
    assert not stale, f"remove these obsolete allowlist entries: {stale}"


def test_find_tmp_citations_ignores_scratch_directories_and_prose():
    """The pattern must not flag a destination or a description of the incident."""
    assert find_tmp_citations("pixi run python tests/parity/dump.py --out /tmp/bench") == []
    assert find_tmp_citations("ran from disposable `/tmp` drivers, once each") == []
    assert find_tmp_citations("see /tmp/profile_slice5b.py") == ["/tmp/profile_slice5b.py"]


def test_environment_capture_is_complete():
    """Every required environment field is populated on this host."""
    environment = Environment.capture(REPO / "benchmarks" / "_harness.py")
    empty = [name for name in Environment.REQUIRED if getattr(environment, name) in ("", None)]
    assert not empty, f"environment capture left {empty} empty"


def test_verify_report_rejects_a_result_without_environment(tmp_path):
    """A result file that cannot say what produced it is not evidence."""
    payload = {
        "schema": "pedigree-graph/benchmark/1",
        "suite": "x",
        "gate": GATE,
        "environments": {},
        "cells": [],
    }
    path = tmp_path / "no_env.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ContractError, match="no environment recorded"):
        verify_report(path)


def test_verify_report_rejects_an_unknown_gate(tmp_path):
    """The 5% rule is defined once; a file claiming another gate is not comparable."""
    environment = Environment.capture(REPO / "benchmarks" / "_harness.py")
    payload = {
        "schema": "pedigree-graph/benchmark/1",
        "suite": "x",
        "gate": 1.5,
        "environments": {environment.fingerprint: environment.as_dict()},
        "cells": [],
    }
    path = tmp_path / "bad_gate.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ContractError, match="does not match GATE"):
        verify_report(path)


def test_gate_is_defined_once():
    """No benchmark other than the harness may carry its own 5% literal."""
    offenders = [
        path.name
        for path in sorted((REPO / "benchmarks").glob("*.py"))
        if path.name != "_harness.py" and "1.05" in path.read_text()
    ]
    assert not offenders, f"{offenders} redefine the gate; import GATE from _harness instead"


@pytest.mark.parametrize("path", sorted((REPO / "benchmarks" / "reports").glob("*.json")))
def test_local_result_files_satisfy_the_contract(path):
    """Any result file present on this machine must parse and carry its environment.

    ``benchmarks/.gitignore`` excludes ``reports/``, so this is opportunistic and
    finds nothing in a fresh checkout.  The real guarantee is that the harness
    cannot serialise a report without an environment.
    """
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict) or "schema" not in payload:
        pytest.skip(f"{path.name} predates the contract; regenerate it with the current harness")
    verify_report(path)
