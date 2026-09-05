"""Freeze ``PedigreeGraph.relationship_pairs(max_degree=5)`` as the 0.8 golden lock.

Run against the installed package (the 0.8 branch, not a 0.7.1 worktree)::

    pixi run python tests/parity/generate_relationship_pairs.py --out tests/data/relationship_pairs_v0.8

Small fixtures (motifs, ``random_1k``, ``deep_inbred_60g``, and the shipped
``small_pedigree.parquet``) get their full oriented int32 arrays saved in one
``.npz`` each.  ``random_30k`` gets per-code counts and SHA-256 hashes only.
``manifest.json`` records the generator version, package commit, fixture
parameters, input hashes, and every hash.  Regeneration is a deliberate act:
a changed hash is a changed contract, never a test fix.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pedigrees  # noqa: E402

MAX_DEGREE = 5


def _sha(*arrays: np.ndarray) -> str:
    h = hashlib.sha256()
    for arr in arrays:
        c = np.ascontiguousarray(arr)
        h.update(str(c.dtype).encode())
        h.update(str(c.shape).encode())
        h.update(c.tobytes())
    return h.hexdigest()


def _columns(fx: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {"id": fx["ids"], "mother": fx["mother"], "father": fx["father"], "twin": fx["twin"], "sex": fx["sex"]}


def _capture(pg_mod, fx: dict[str, np.ndarray], *, full_arrays: bool) -> tuple[dict, dict]:
    """Return ``(arrays, summary)``; ``arrays`` is empty when ``full_arrays`` is False."""
    from pedigree_graph._pair_extractor import check_exclusive

    graph = pg_mod.PedigreeGraph.from_frame(_columns(fx))
    result = graph.relationship_pairs(max_degree=MAX_DEGREE)
    check_exclusive(result)
    arrays: dict[str, np.ndarray] = {}
    summary: dict = {"n": graph.n_individuals, "counts": {}, "hashes": {}}
    for code, block in result.items():
        first, second = block
        summary["counts"][code] = len(block)
        summary["hashes"][code] = _sha(first, second)
        if full_arrays:
            arrays[f"pairs/{code}/first"] = first
            arrays[f"pairs/{code}/second"] = second
    return arrays, summary


def _git_commit(root: Path) -> str:
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.run(["git", "-C", str(root), "diff", "--quiet", "HEAD"], check=False).returncode != 0
    return f"{head}-dirty" if dirty else head


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=HERE.parent / "data" / "relationship_pairs_v0.8")
    ap.add_argument("--skip-large", action="store_true")
    args = ap.parse_args()

    import polars as pl

    import pedigree_graph as pg_mod

    package_root = Path(pg_mod.__file__).resolve().parent.parent
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "generator_version": pedigrees.GENERATOR_VERSION,
        "package_commit": _git_commit(package_root),
        "package_version": getattr(pg_mod, "__version__", None),
        "max_degree": MAX_DEGREE,
        "fixtures": {},
    }

    fixtures: list[tuple[str, dict, dict[str, np.ndarray], bool]] = []
    for name, fx in pedigrees.motif_fixtures().items():
        fixtures.append((name, {"kind": "motif"}, fx, True))
    df = pl.read_parquet(HERE.parent / "data" / "small_pedigree.parquet")
    shipped = {
        "ids": df["id"].to_numpy().astype(np.int64),
        "mother": df["mother"].to_numpy().astype(np.int64),
        "father": df["father"].to_numpy().astype(np.int64),
        "twin": df["twin"].to_numpy().astype(np.int64) if "twin" in df.columns else np.full(len(df), -1, np.int64),
        "sex": df["sex"].to_numpy().astype(np.int8),
    }
    fixtures.append(("small_pedigree", {"kind": "shipped", "file": "tests/data/small_pedigree.parquet"}, shipped, True))
    for name, params in pedigrees.RANDOM_FIXTURES.items():
        fixtures.append((name, {"kind": "random", **params}, pedigrees.build_random(name, params), True))
    if not args.skip_large:
        for name, params in pedigrees.LARGE_FIXTURES.items():
            fixtures.append((name, {"kind": "large", **params}, pedigrees.build_random(name, params), False))

    for name, params, fx, full_arrays in fixtures:
        t0 = time.perf_counter()
        arrays, summary = _capture(pg_mod, fx, full_arrays=full_arrays)
        summary["params"] = params
        summary["input_hash"] = pedigrees.input_hash(fx)
        summary["seconds"] = round(time.perf_counter() - t0, 2)
        if full_arrays:
            path = args.out / f"{name}.npz"
            np.savez_compressed(path, **arrays, **{f"input/{k}": v for k, v in fx.items()})
            summary["file"] = path.name
        manifest["fixtures"][name] = summary
        print(f"{name}: n={summary['n']} pairs={sum(summary['counts'].values())} {summary['seconds']}s")

    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out / 'manifest.json'}")


if __name__ == "__main__":
    main()
