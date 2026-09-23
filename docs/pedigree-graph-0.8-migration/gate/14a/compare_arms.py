r"""The three arms of one ``bench_kinship_matrix.py`` report side by side, with ratios against the wheel.

    pixi run python docs/pedigree-graph-0.8-migration/gate/14a/compare_arms.py \\
        benchmarks/reports/kinship_matrix.json

For every fixture the table shows each arm's median wall and peak RSS with the
min..max span and repetition count, then each source arm's ratio to the wheel:
a ratio above 1.05 whose spans do not overlap is BLOCK, above 1.05 with
overlap inconclusive.  A final list says whether the three arms returned the
same checksum on every cell.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

GATE = 1.05
ARMS = ("wheel", "owned", "arena")


def _cells(path: Path) -> dict[str, dict[str, dict[str, list]]]:
    payload = json.loads(path.read_text())
    out: dict[str, dict[str, dict[str, list]]] = {}
    for cell in payload["cells"]:
        fixture, _, arm = cell["cell"].partition("/")
        runs = cell["runs"]
        if not runs:
            continue
        out.setdefault(fixture, {})[arm] = {
            "wall_s": [r["wall_s"] for r in runs],
            "peak_rss_mib": [r["peak_rss_mib"] for r in runs],
            "checksum": sorted({r["checksum"] for r in runs}),
            "package": sorted({r["facts"].get("package_file", "?") for r in runs}),
        }
    return out


def _verdict(subject: list[float], baseline: list[float]) -> str:
    ratio = statistics.median(subject) / statistics.median(baseline)
    if ratio <= GATE:
        return f"{ratio:.3f}"
    if min(subject) > max(baseline):
        return f"{ratio:.3f} BLOCK"
    return f"{ratio:.3f} inconclusive"


def _span(values: list[float]) -> str:
    return f"{statistics.median(values):,.2f} [{min(values):,.2f}..{max(values):,.2f}] (n={len(values)})"


def main() -> None:
    """Print the comparison tables for the report on the command line."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("report", type=Path)
    args = ap.parse_args()
    table = _cells(args.report)
    for metric, unit in (("wall_s", "s"), ("peak_rss_mib", "MiB")):
        print(f"\n## {metric}\n")
        print(
            "| cell | "
            + " | ".join(f"{arm} median ({unit}) [span]" for arm in ARMS)
            + " | "
            + " | ".join(f"{arm}/wheel" for arm in ARMS[1:])
            + " |"
        )
        print("|---|" + "---:|" * (2 * len(ARMS) - 1))
        for fixture, arms in table.items():
            row = [fixture]
            for arm in ARMS:
                values = arms.get(arm, {}).get(metric)
                row.append(_span(values) if values else "n/a")
            base = arms.get("wheel", {}).get(metric)
            for arm in ARMS[1:]:
                values = arms.get(arm, {}).get(metric)
                row.append(_verdict(values, base) if values and base else "n/a")
            print("| " + " | ".join(row) + " |")
    print("\n## checksums\n")
    for fixture, arms in table.items():
        sums = {arm: arms[arm]["checksum"] for arm in ARMS if arm in arms}
        same = len({tuple(v) for v in sums.values()}) == 1
        print(f"- {fixture}: {'identical' if same else 'DIFFER'} {sums}")
    print("\n## packages\n")
    seen: dict[str, set[str]] = {}
    for arms in table.values():
        for arm, data in arms.items():
            seen.setdefault(arm, set()).update(data["package"])
    for arm in ARMS:
        print(f"- {arm}: {sorted(seen.get(arm, []))}")


if __name__ == "__main__":
    main()
