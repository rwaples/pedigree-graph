"""Shared by the slice 12 pair benchmarks and the 0.8.4 baseline child, numpy only."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
from _harness import status_field_mib

if TYPE_CHECKING:
    from pathlib import Path

VIEW_SEED = 12
CODES = (
    "MZ", "MO", "FO", "FS", "MHS", "PHS", "GP", "Av", "GGP", "HAv", "GAv", "1C", "GGGP",
    "HGAv", "GGAv", "H1C", "1C1R", "G3GP", "HGGAv", "G3Av", "H1C1R", "1C2R", "2C",
)  # fmt: skip


def digest(first: np.ndarray, second: np.ndarray) -> int:
    """``PairBlock::digest`` over an aligned block, wrapping in uint64."""
    k1, k2, k3 = np.uint64(0x9E3779B97F4A7C15), np.uint64(0xC2B2AE3D27D4EB4F), np.uint64(0x165667B19E3779F9)
    k = np.arange(1, len(first) + 1, dtype=np.uint64)
    term = k1 * first.astype(np.uint64) + k2 * second.astype(np.uint64) + k3
    return int(np.sum(k * term, dtype=np.uint64))


def half_view_rows(n: int) -> np.ndarray:
    """A fixed-seed reordered selection of half the rows."""
    return np.random.default_rng(VIEW_SEED).permutation(n)[: n // 2]


def peak_rss_mib() -> float:
    """This process's ``VmHWM`` in MiB, through the harness's one reader.

    The pair benchmarks take the high-water mark before and after the call and
    report the difference, so they do not reset it the way
    :class:`_harness.PeakRss` does around a region.
    """
    return status_field_mib("VmHWM:")


def dump_blocks(pairs, directory: Path) -> None:
    """Write every block as ``<code>.first.u32`` / ``<code>.second.u32``, little-endian."""
    directory.mkdir(parents=True, exist_ok=True)
    for code, block in pairs.items():
        (directory / f"{code}.first.u32").write_bytes(np.ascontiguousarray(block.first_rows, dtype="<u4").tobytes())
        (directory / f"{code}.second.u32").write_bytes(np.ascontiguousarray(block.second_rows, dtype="<u4").tobytes())


def load_dump(directory: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Read a dump back as ``{code: (first, second)}``."""
    return {
        code: (
            np.fromfile(directory / f"{code}.first.u32", dtype="<u4"),
            np.fromfile(directory / f"{code}.second.u32", dtype="<u4"),
        )
        for code in CODES
    }


def block_summary(pairs) -> dict[str, list[int]]:
    """``{code: [count, digest]}`` of a ``RelationshipPairs`` result."""
    return {code: [len(block), digest(block.first_rows, block.second_rows)] for code, block in pairs.items()}


def write_json(record: dict) -> None:
    """One JSON object on stdout, the child protocol."""
    print(json.dumps(record), flush=True)
