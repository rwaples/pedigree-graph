"""Architecture guardrails.

In-suite checks that keep structural rules from eroding: module line budgets,
the frozen root exports, deleted names staying deleted, generation labels off
the structural path, a numba-free package, and every parallel Rust kernel
paired with a test that its output is bit-identical across thread budgets
(ADR 0007).

See ``docs/architecture.md`` for the module map and the per-contract
source-of-truth / regression-test pointers.
"""

from __future__ import annotations

import ast
import tokenize
from pathlib import Path

import pytest

import pedigree_graph

PKG_DIR = Path(pedigree_graph.__file__).parent

# Default hard cap for any production module.  Prefer a new focused module
# over pushing an existing one past this.
DEFAULT_MAX_LINES = 1000

# Reviewed exceptions: filename -> its own cap.  Empty since 0.8.0 deleted
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
# Namespace freeze (ADR 0006)
# ---------------------------------------------------------------------------

ROOT_EXPORTS = (
    "MAX_DEGREE",
    "RELATIONSHIPS",
    "MissingMetadataError",
    "PedigreeGraph",
    "PedigreeValidationError",
    "PedigreeView",
    "RelationshipBurden",
    "RelationshipCategory",
    "RelationshipCountResult",
    "RelationshipPairBlock",
    "RelationshipPairs",
    "ResourceError",
    "configure_threads",
)

# Names deleted from the public or internal surface, which must not come back.
# Frozen parity generators keep the 0.8.0 ones because they only run against a
# 0.7.1 checkout.  String literals are prose to the sweep, so a test asserting
# an old key is absent from ``ALL_EFFECTIVE_SIZE_ESTIMATORS`` still reads
# naturally.
REMOVED_NAMES = (
    # 0.8.0: the 0.7.1 surface.
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
    # 0.9.0: the Caballero-Toro estimator and its founder-reach weighting
    # fields (issue #15, ADR 0012), and the approximate relationship counter,
    # replaced by exact close-relative counts with no old-name alias (issue
    # #17, ADR 0011).
    "estimate_relationship_counts",
    "ne_caballero_toro",
    "NeCaballeroToroResult",
    "CTAccumulators",
    "_caballero_toro_accumulators",
    "_caballero_toro_from",
    "_ct_ensure_pool_capacity",
    "_ct_merge_to_pool",
    "_ct_accumulators_kernel",
    "mean_self_coancestry_per_gen",
    "n_founders_with_descendants_per_gen",
    "n_iterations",
)
REPO_DIR = PKG_DIR.parent
SWEPT_DIRS = ("pedigree_graph", "tests", "benchmarks")
FROZEN_GENERATORS = frozenset(
    {
        "tests/parity/generate_baseline.py",
        "tests/parity/dump_relationship_inputs.py",
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
    assert not offenders, "deleted names back in the tree:\n  " + "\n  ".join(offenders)


# Every module that reads the ``generation_labels`` property.  The rule this
# pins is the depth-versus-label contract (#22): structural results derive from
# structural depth alone, so every reader here is either the property itself or
# a cohort-side effective-size module.  A structural module joining this set is
# the bug; adding it to the allowlist instead of fixing it defeats the check.
GENERATION_LABEL_READERS = frozenset(
    {
        "_cohorts.py",
        "_ne_common.py",
        "_ne_estimate.py",
        "_ne_metadata.py",
        "_ne_rates.py",
        "_properties.py",
    }
)


def test_generation_labels_stay_off_the_structural_path():
    names = frozenset({"generation_labels"})
    readers = {path.name for path in _production_modules() if _identifier_uses(path, names)}
    assert readers == GENERATION_LABEL_READERS


def test_the_package_does_not_import_numba():
    """Numba left the runtime dependencies in 0.9.4; only the test oracles use it."""
    offenders = []
    for path in sorted(PKG_DIR.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            else:
                continue
            offenders += [f"{path.name}:{node.lineno}" for m in modules if m.split(".")[0] == "numba"]
    assert not offenders, "numba imported by the package:\n  " + "\n  ".join(offenders)


# Every core module that runs work on the Rayon pool, mapped to the test that
# its output is bit-identical at thread budgets 1 and 4 (ADR 0007, "Threads and
# determinism").  A new parallel module fails the first test until it is mapped
# here, and a mapped test that is renamed or deleted fails the second.
# Anchored on this file, not the package: an installed-wheel run imports the
# package from site-packages but still has the checkout's crates and tests.
CHECKOUT = Path(__file__).resolve().parents[1]
CORE_SRC = CHECKOUT / "crates" / "core" / "src"
PARALLEL_MODULES = {
    "relationships/pairs.rs": "tests/test_native_relationship_pairs.py::TestProcessWidePool::"
    "test_blocks_are_identical_under_every_thread_budget",
    "relationships/mod.rs": "tests/test_native_relationship_counts.py::TestSelectorsAndErrors::"
    "test_counts_are_the_same_under_every_thread_budget",
    "relationships/burden.rs": "tests/test_compact_view_and_burden.py::"
    "test_burden_is_bit_identical_under_every_thread_budget",
}
# The pool itself: it builds and installs the Rayon pool and runs no kernel.
POOL_INFRASTRUCTURE = frozenset({"pool.rs"})
PARALLEL_MARKERS = ("rayon", "crate::pool", "par_iter", "par_chunks", "par_sort")


def test_every_parallel_core_module_is_mapped():
    parallel = {
        path.relative_to(CORE_SRC).as_posix()
        for path in CORE_SRC.rglob("*.rs")
        if "bin" not in path.relative_to(CORE_SRC).parts
        and any(marker in path.read_text() for marker in PARALLEL_MARKERS)
    }
    assert parallel == set(PARALLEL_MODULES) | POOL_INFRASTRUCTURE


@pytest.mark.parametrize("test_id", sorted(PARALLEL_MODULES.values()))
def test_every_mapped_cross_budget_test_exists(test_id):
    file, *scope = test_id.split("::")
    nodes = ast.parse((CHECKOUT / file).read_text()).body
    for name in scope:
        match = [n for n in nodes if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == name]
        assert match, f"{test_id}: {name} not found"
        nodes = match[0].body
