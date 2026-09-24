"""Compare two ``byte_parity.sh`` output dirs, parsing the files whose hashes differ.

    python tools/compare_floats.py <baseline-dir> <candidate-dir>

Each dir holds the ``manifest.txt`` the probe wrote (``sha256sum`` lines) and
the files it kept.  A file whose hash matches is ``identical``.  One whose hash
differs is parsed on both sides, YAML by tree and TSV or text by tab-separated
cell, and every pair of numbers is compared: integers exactly, floats against
``|b - a| <= atol + rtol * |a|`` with ``rtol 1e-9, atol 1e-12``, with
the maximum relative and absolute difference reported so a stage note can say
"identical" or "within tolerance, max x".  Anything else (strings, ``null``,
row or key structure) must match exactly.  A differing file that was not kept,
or cannot be parsed, is reported as such.

``mean_kinship_by_generation.txt`` writes its means as uint64 bit patterns, so
they compare as integers, that is exactly.  Exits non-zero when any file is
outside tolerance or could not be compared.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

RTOL = 1e-9
ATOL = 1e-12


def _manifest(directory: Path) -> dict[str, str]:
    hashes = {}
    for line in (directory / "manifest.txt").read_text().splitlines():
        digest, _, name = line.partition("  ")
        if name and len(digest) == 64:
            hashes[name] = digest
    return hashes


def _text(directory: Path, name: str) -> str | None:
    path = directory / name
    if path.exists():
        return path.read_text()
    packed = directory / f"{name}.gz"
    if packed.exists():
        return gzip.decompress(packed.read_bytes()).decode()
    return None


def _cell(text: str) -> int | float | str:
    for parse in (int, float):
        try:
            return parse(text)
        except ValueError:
            pass
    return text


def _parse(name: str, text: str) -> object:
    if name.endswith((".yaml", ".yml")):
        import yaml

        return yaml.safe_load(text)
    return [[_cell(cell) for cell in line.split("\t")] for line in text.splitlines()]


class _Diff:
    def __init__(self) -> None:
        self.max_rel = 0.0
        self.max_abs = 0.0
        self.n_floats_differing = 0
        self.mismatches: list[str] = []

    def walk(self, where: str, a: object, b: object) -> None:
        if isinstance(a, dict) and isinstance(b, dict):
            if a.keys() != b.keys():
                self.mismatches.append(f"{where}: keys {sorted(a.keys() ^ b.keys())}")
            for key in a.keys() & b.keys():
                self.walk(f"{where}/{key}", a[key], b[key])
        elif isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                self.mismatches.append(f"{where}: length {len(a)} != {len(b)}")
            for index, (x, y) in enumerate(zip(a, b, strict=False)):
                self.walk(f"{where}[{index}]", x, y)
        elif isinstance(a, float) and isinstance(b, float | int) and not isinstance(b, bool):
            self._float(where, a, float(b))
        elif isinstance(b, float) and isinstance(a, int) and not isinstance(a, bool):
            self._float(where, float(a), b)
        elif type(a) is not type(b) or a != b:
            self.mismatches.append(f"{where}: {a!r} != {b!r}")

    def _float(self, where: str, a: float, b: float) -> None:
        if a == b or (a != a and b != b):
            return
        self.n_floats_differing += 1
        difference = abs(b - a)
        self.max_abs = max(self.max_abs, difference)
        if a != 0:
            self.max_rel = max(self.max_rel, difference / abs(a))
        if not difference <= ATOL + RTOL * abs(a):
            self.mismatches.append(f"{where}: {a!r} vs {b!r} outside tolerance")


def compare(baseline: Path, candidate: Path) -> dict:
    """One verdict per file named in either manifest."""
    expected, got = _manifest(baseline), _manifest(candidate)
    files: dict = {}
    for name in sorted(expected.keys() | got.keys()):
        if name not in expected or name not in got:
            files[name] = {"verdict": "missing", "in_baseline": name in expected, "in_candidate": name in got}
            continue
        if expected[name] == got[name]:
            files[name] = {"verdict": "identical"}
            continue
        texts = _text(baseline, name), _text(candidate, name)
        if None in texts:
            files[name] = {"verdict": "differs_not_kept"}
            continue
        try:
            trees = [_parse(name, text) for text in texts]
        except (UnicodeDecodeError, ValueError) as error:
            files[name] = {"verdict": "differs_unparseable", "error": str(error)}
            continue
        diff = _Diff()
        diff.walk(name, *trees)
        files[name] = {
            "verdict": "differs" if diff.mismatches else "within_tolerance",
            "max_rel_diff": diff.max_rel,
            "max_abs_diff": diff.max_abs,
            "floats_differing": diff.n_floats_differing,
            "mismatches": diff.mismatches[:20],
            "n_mismatches": len(diff.mismatches),
        }
    return {"baseline": str(baseline), "candidate": str(candidate), "rtol": RTOL, "atol": ATOL, "files": files}


def main() -> None:
    """Print the per-file verdicts as JSON; exit 1 unless every file is identical or within tolerance."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("baseline", type=Path)
    ap.add_argument("candidate", type=Path)
    args = ap.parse_args()
    report = compare(args.baseline, args.candidate)
    print(json.dumps(report, indent=2, sort_keys=True))
    ok = all(entry["verdict"] in ("identical", "within_tolerance") for entry in report["files"].values())
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
