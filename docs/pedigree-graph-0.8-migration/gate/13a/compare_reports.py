r"""Side-by-side medians of several ``bench_pair_kinship.py`` reports, with ratios against the first.

    pixi run python docs/pedigree-graph-0.8-migration/gate/13a/compare_reports.py \\
        0.9.0=benchmarks/reports/pair_kinship_0.9.0.json rows=benchmarks/reports/pair_kinship_rows.json \\
        flat=benchmarks/reports/pair_kinship_flat.json

Each cell prints the median wall and peak RSS per report, the min..max span,
and the ratio of every later report to the first; a ratio above 1.05 whose
spans do not overlap is marked BLOCK, above 1.05 with overlap INCONCLUSIVE.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

GATE = 1.05


def _cells(path: Path) -> dict[str, dict[str, list[float]]]:
    payload = json.loads(path.read_text())
    out = {}
    for cell in payload["cells"]:
        runs = cell["runs"]
        if not runs:
            continue
        out[cell["cell"]] = {
            "wall_s": [r["wall_s"] for r in runs],
            "peak_rss_mib": [r["peak_rss_mib"] for r in runs],
            "checksum": sorted({r["checksum"] for r in runs}),
        }
    return out


def _verdict(subject: list[float], baseline: list[float]) -> str:
    ratio = statistics.median(subject) / statistics.median(baseline)
    if ratio <= GATE:
        return f"{ratio:.3f}"
    if min(subject) > max(baseline):
        return f"{ratio:.3f} BLOCK"
    return f"{ratio:.3f} inconclusive"


def main() -> None:
    """Print the comparison tables for the reports on the command line."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("reports", nargs="+", help="label=path, e.g. 0.9.0=reports/pair_kinship_0.9.0.json")
    args = ap.parse_args()
    labels = [spec.split("=", 1)[0] for spec in args.reports]
    args.reports = [Path(spec.split("=", 1)[1]) for spec in args.reports]
    tables = [_cells(p) for p in args.reports]
    base = tables[0]
    for metric, unit in (("wall_s", "s"), ("peak_rss_mib", "MiB")):
        print(f"\n## {metric}\n")
        header = "| cell | " + " | ".join(f"{label} median ({unit}) [span]" for label in labels)
        header += " | " + " | ".join(f"{label}/{labels[0]}" for label in labels[1:]) + " |"
        print(header)
        print("|---|" + "---:|" * (len(labels) + len(labels) - 1))
        for cell in base:
            row = [cell]
            for table in tables:
                values = table.get(cell, {}).get(metric)
                row.append(
                    f"{statistics.median(values):,.2f} [{min(values):,.2f}..{max(values):,.2f}] (n={len(values)})"
                    if values
                    else "n/a"
                )
            for table in tables[1:]:
                values = table.get(cell, {}).get(metric)
                row.append(_verdict(values, base[cell][metric]) if values else "n/a")
            print("| " + " | ".join(row) + " |")
    print("\n## checksums\n")
    for cell in base:
        sums = {label: table.get(cell, {}).get("checksum") for label, table in zip(labels, tables, strict=True)}
        same = len({tuple(v) for v in sums.values() if v}) == 1
        print(f"- {cell}: {'identical' if same else 'DIFFER'} {sums}")


if __name__ == "__main__":
    main()
