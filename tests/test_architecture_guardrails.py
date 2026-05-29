"""Architecture guardrails (PGQ-010).

Lightweight, in-suite checks that keep the structural improvements from
PGQ-001..009 from silently eroding.  The one rule enforced here: no
production module grows past its line budget unnoticed.  New code should
land in a new focused module, not extend an oversized one.

See ``docs/architecture.md`` for the module map and the per-contract
source-of-truth / regression-test pointers.
"""

from __future__ import annotations

from pathlib import Path

import pedigree_graph

PKG_DIR = Path(pedigree_graph.__file__).parent

# Default hard cap for any production module.  Prefer a new focused module
# over pushing an existing one past this.
DEFAULT_MAX_LINES = 1000

# Reviewed exceptions: filename -> its own cap.  ``_core.py`` is the central
# PedigreeGraph class (already cut from ~1982 to ~1160 lines by PGQ-003);
# the cap keeps it from regrowing.  Prefer extracting read-only collaborators
# (ADR 0002) over growing it further.
ALLOWLIST = {
    "_core.py": 1250,
}


def _line_count(path: Path) -> int:
    with path.open("rb") as fh:
        return sum(1 for _ in fh)


def _production_modules() -> list[Path]:
    return sorted(PKG_DIR.glob("*.py"))


def test_no_module_exceeds_line_budget():
    """Every production module is within its budget (default or allowlisted)."""
    offenders = []
    for path in _production_modules():
        cap = ALLOWLIST.get(path.name, DEFAULT_MAX_LINES)
        n = _line_count(path)
        if n > cap:
            offenders.append(f"{path.name}: {n} lines > cap {cap}")
    assert not offenders, (
        "Production module(s) over the line budget. Split into a focused "
        "module (see docs/architecture.md) or, if genuinely justified, add a "
        "reviewed ALLOWLIST entry:\n  " + "\n  ".join(offenders)
    )


def test_allowlist_entries_are_still_needed():
    """A file allowlisted above the default budget but now within it should be
    dropped from ALLOWLIST, so the guardrail tightens as files shrink."""
    stale = []
    for name in ALLOWLIST:
        path = PKG_DIR / name
        if not path.exists():
            stale.append(f"{name}: allowlisted but no longer exists")
        elif _line_count(path) <= DEFAULT_MAX_LINES:
            stale.append(f"{name}: now within the {DEFAULT_MAX_LINES}-line budget — remove from ALLOWLIST")
    assert not stale, "Stale ALLOWLIST entries:\n  " + "\n  ".join(stale)
