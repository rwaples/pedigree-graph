"""Render the matrix-exactification sweep JSON into the markdown note.

``benchmarks/.gitignore`` excludes ``reports/``, so the tracked note is the only
durable evidence.  It therefore has to carry the environment, the repetition
count, and the observed spread inline rather than pointing at a local file.
Generating the tables from the result JSON keeps the note honest: no figure in
it was typed by hand.

    python benchmarks/render_exactification.py --reports benchmarks/reports
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

LABELS = {
    "candidate_support": "propagation-pruned support only",
    "complete_dp": "complete retiring DP (no output CSC)",
    "fused": "fused capture (public path)",
    "pairwise_shared": "one shared memo",
    "pairwise_columns256": "256-column chunks",
    "pairwise_pairs262144": "262,144-pair chunks",
    "pairwise_pairs1048576": "1,048,576-pair chunks",
}

INPUT_LABELS = {
    "fitace": "fitACE `dev_cont_n10k/rep1`",
    "random30k": "generated `random_30k`",
}


def _seconds(value: float) -> str:
    if value >= 600:
        return f"{value:,.0f} s ({value / 60:.0f} min)"
    return f"{value:,.2f} s"


def _row(summary: dict) -> str:
    input_name, strategy = summary["config"].split("/")
    if summary.get("timed_out") and not summary.get("repeats"):
        return (
            f"| {INPUT_LABELS[input_name]} | {LABELS[strategy]} | 0 | "
            f"did not finish within {summary['timeout_s']:.0f} s | n/a | n/a | n/a |"
        )
    spread = summary["wall_spread_pct"]
    checksum = summary.get("checksum")
    checksum_text = f"`{checksum}`" if checksum is not None else "n/a"
    if not summary.get("checksum_stable", True):
        checksum_text = "**unstable**"
    return (
        f"| {INPUT_LABELS[input_name]} | {LABELS[strategy]} | {summary['repeats']} | "
        f"{_seconds(summary['wall_median_s'])} | {spread:.1f}% | "
        f"{summary['peak_rss_median_mib']:,.0f} MiB | {checksum_text} |"
    )


def render(reports: Path) -> str:
    """Return the environment block and results table for the note."""
    environment = json.loads((reports / "exactification_environment.json").read_text())
    summaries: list[dict] = []
    for path in sorted(reports.glob("exactification_wave*.json")):
        summaries.extend(json.loads(path.read_text()))

    by_config = {s["config"]: s for s in summaries}
    lines: list[str] = []
    lines.append("| input | strategy | reps | wall (median) | spread | peak RSS (median) | upper-value XOR |")
    lines.append("|---|---|---:|---:|---:|---:|---|")
    ordered = (
        "fitace/candidate_support",
        "fitace/complete_dp",
        "fitace/fused",
        "fitace/pairwise_shared",
        "fitace/pairwise_columns256",
        "fitace/pairwise_pairs262144",
        "fitace/pairwise_pairs1048576",
        "random30k/candidate_support",
        "random30k/complete_dp",
        "random30k/fused",
        "random30k/pairwise_pairs1048576",
        "random30k/pairwise_shared",
    )
    lines.extend(_row(by_config[config]) for config in ordered if config in by_config)
    table = "\n".join(lines)

    env_lines = [
        f"- commit `{environment['git_commit'][:10]}` on `{environment['git_branch']}`"
        + (", working tree dirty" if environment["git_dirty"] else ""),
        f"- {environment['cpu_model']}, {environment['logical_cpus']} logical CPUs, "
        f"{environment['mem_total_gib']} GiB RAM, kernel {environment['kernel']}",
        f"- Python {environment['python']}, pixi lock `{environment['pixi_lock_sha256']}`, "
        f"harness `{environment['harness_sha256']}`",
        f"- every backend pinned to {environment['threads_pinned']} thread",
    ]
    return "\n".join(env_lines) + "\n\n" + table + "\n"


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, default=Path("benchmarks/reports"))
    args = parser.parse_args()
    print(render(args.reports))


if __name__ == "__main__":
    main()
