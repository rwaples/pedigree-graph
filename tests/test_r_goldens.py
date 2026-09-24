"""The R package's committed goldens match what the current package writes."""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _tool():
    spec = importlib.util.spec_from_file_location("r_golden", REPO / "tools" / "r_golden.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_committed_r_goldens_are_current() -> None:
    tool = _tool()
    stale = tool.stale(tool.GOLDEN)
    assert stale == [], f"regenerate with `pixi run python tools/r_golden.py`: {stale}"
