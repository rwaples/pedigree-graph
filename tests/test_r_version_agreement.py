"""The R package carries the workspace version (ADR 0007).

``[workspace.package].version`` is the one version anyone edits.  The R
binding crate sits outside that workspace so the R source tarball builds on
its own, and DESCRIPTION is R's, so both repeat it; this test fails a bump
that misses either, as the publish workflow's version check does.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _description_version() -> str:
    text = (REPO / "r" / "DESCRIPTION").read_text()
    match = re.search(r"^Version:\s*(\S+)\s*$", text, flags=re.MULTILINE)
    assert match is not None, "r/DESCRIPTION has no Version: field"
    return match.group(1)


def test_r_description_and_binding_crate_carry_the_workspace_version() -> None:
    workspace = tomllib.loads((REPO / "Cargo.toml").read_text())["workspace"]["package"]["version"]
    binding = tomllib.loads((REPO / "r" / "src" / "rust" / "Cargo.toml").read_text())["package"]["version"]
    assert {"DESCRIPTION": _description_version(), "r/src/rust/Cargo.toml": binding} == {
        "DESCRIPTION": workspace,
        "r/src/rust/Cargo.toml": workspace,
    }
