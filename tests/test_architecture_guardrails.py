"""Architecture guardrails (PGQ-010).

Lightweight, in-suite checks that keep the structural improvements from
PGQ-001..009 from silently eroding.  The one rule enforced here: no
production module grows past its line budget unnoticed.  New code should
land in a new focused module, not extend an oversized one.

See ``docs/architecture.md`` for the module map and the per-contract
source-of-truth / regression-test pointers.
"""

from __future__ import annotations

import subprocess
import tokenize
from pathlib import Path

import pytest

import pedigree_graph

PKG_DIR = Path(pedigree_graph.__file__).parent

# Default hard cap for any production module.  Prefer a new focused module
# over pushing an existing one past this.
DEFAULT_MAX_LINES = 1000

# Reviewed exceptions: filename -> its own cap.  Empty since slice 7 deleted
# the 0.7.1 adapters from ``_core.py``; prefer extracting read-only
# collaborators (ADR 0002) over adding an entry.
ALLOWLIST: dict[str, int] = {}


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


# ---------------------------------------------------------------------------
# Namespace freeze (ADR 0006, slice 7)
# ---------------------------------------------------------------------------

ROOT_EXPORTS = (
    "RELATIONSHIPS",
    "MissingMetadataError",
    "PedigreeGraph",
    "PedigreeValidationError",
    "PedigreeView",
    "RelationshipCategory",
    "RelationshipCountResult",
    "RelationshipPairBlock",
    "RelationshipPairs",
    "ResourceError",
    "configure_threads",
)

# Names the 0.7.1 surface exposed and 0.8.0 deleted.  Frozen parity
# generators keep them because they only run against a 0.7.1 checkout.
REMOVED_NAMES = (
    "compute_pair_kinship",
    "compute_inbreeding",
    "compute_all_ne",
    "compute_n_ancestors",
    "compute_n_descendants",
    "per_gen_mean_kinship",
    "from_dataframe",
    "from_subsample",
    "extract_pairs",
    "count_pairs_streaming",
    "REL_REGISTRY",
    "PAIR_KINSHIP",
    "RelType",
    "legacy_defaults",
    "_legacy_view",
    "_kinship_cache",
    "streaming_exact",
)
REPO_DIR = PKG_DIR.parent
SWEPT_DIRS = ("pedigree_graph", "tests", "benchmarks")
FROZEN_GENERATORS = frozenset(
    {
        "tests/parity/generate_baseline.py",
        "tests/parity/dump_relationship_inputs.py",
        "tests/parity/generate_ne_baseline.py",
    }
)


def _swept_files() -> list[Path]:
    files = []
    for name in SWEPT_DIRS:
        files.extend(p for p in (REPO_DIR / name).rglob("*.py") if "__pycache__" not in p.parts)
    return sorted(p for p in files if p.relative_to(REPO_DIR).as_posix() not in FROZEN_GENERATORS)


def _identifier_uses(path: Path, names: frozenset[str]) -> list[str]:
    """Code identifiers in *names*; comments, docstrings, and string keys are prose, not uses."""
    with path.open("rb") as fh:
        tokens = tokenize.tokenize(fh.readline)
        return [
            f"{path.relative_to(REPO_DIR)}:{tok.start[0]}: {tok.string}"
            for tok in tokens
            if tok.type == tokenize.NAME and tok.string in names
        ]


def test_root_exports_are_frozen():
    assert tuple(pedigree_graph.__all__) == ROOT_EXPORTS
    for name in ROOT_EXPORTS:
        assert hasattr(pedigree_graph, name), name


def test_frame_like_lives_in_the_typing_module():
    from pedigree_graph import _frames, typing

    assert typing.FrameLike is _frames.FrameLike
    assert typing.__all__ == ["FrameLike"]
    assert not hasattr(pedigree_graph, "FrameLike")


def test_removed_names_do_not_reappear():
    names = frozenset(REMOVED_NAMES)
    offenders = [use for path in _swept_files() for use in _identifier_uses(path, names)]
    assert not offenders, "0.7.1 names in the 0.8 tree:\n  " + "\n  ".join(offenders)


def test_no_delete_markers_remain():
    marker = "0.8.0-" + "DELETE"
    tracked = subprocess.run(
        ["git", "grep", "-n", marker, "--", ".", ":!CHANGELOG.md"],
        cwd=REPO_DIR,
        capture_output=True,
        text=True,
        check=False,
    )
    if tracked.returncode not in (0, 1):
        pytest.skip(f"git grep unavailable: {tracked.stderr.strip()}")
    assert tracked.returncode == 1, f"{marker} markers remain:\n" + tracked.stdout
