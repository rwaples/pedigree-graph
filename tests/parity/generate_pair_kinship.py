"""Freeze ``pair_kinship`` float32 bits at ``v0.9.0`` as the slice 13 golden lock.

Run against the installed package::

    pixi run python tests/parity/generate_pair_kinship.py --out tests/data/pair_kinship_v0.9

For every fixture the generator extracts ``relationship_pairs(max_degree=3)``,
evaluates ``pair_kinship`` over the collection and over every self pair, and
stores the values as ``uint32`` bit views so a later kernel is compared bit
for bit, never within a tolerance.  Small fixtures (motifs, ``random_1k``,
``deep_inbred_60g``, the shipped ``small_pedigree.parquet``) keep the full
arrays in one ``.npz`` each; ``random_30k`` keeps hashes only.  Every value
hash is keyed to the SHA-256 of the pairs it was evaluated on, so a changed
pair extraction cannot masquerade as a kinship change.  Regeneration is a
deliberate contract change, never a test fix.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import pedigrees  # noqa: E402
from generate_relationship_pairs import _columns, _git_commit, _sha  # noqa: E402

MAX_DEGREE = 3


def _bits(values: np.ndarray) -> np.ndarray:
    assert values.dtype == np.float32
    return values.view(np.uint32)


def _capture(pg_mod, fx: dict[str, np.ndarray], *, full_arrays: bool) -> tuple[dict, dict]:
    """Return ``(arrays, summary)``; ``arrays`` is empty when ``full_arrays`` is False."""
    graph = pg_mod.PedigreeGraph.from_frame(_columns(fx))
    pairs = graph.relationship_pairs(max_degree=MAX_DEGREE)
    values = graph.pair_kinship(pairs)
    rows = np.arange(graph.n_individuals, dtype=np.int32)
    self_values = graph.pair_kinship(rows, rows)
    arrays: dict[str, np.ndarray] = {}
    summary: dict = {
        "n": graph.n_individuals,
        "pair_hashes": {},
        "value_hashes": {},
        "self_value_hash": _sha(_bits(self_values)),
    }
    for code, block in pairs.items():
        first, second = block
        summary["pair_hashes"][code] = _sha(first, second)
        summary["value_hashes"][code] = _sha(_bits(values[code]))
        if full_arrays:
            arrays[f"pairs/{code}/first"] = first
            arrays[f"pairs/{code}/second"] = second
            arrays[f"values/{code}"] = _bits(values[code])
    if full_arrays:
        arrays["self_values"] = _bits(self_values)
    return arrays, summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=HERE.parent / "data" / "pair_kinship_v0.9")
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
        "core_version": pg_mod._native.core_version(),
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
        if full_arrays:
            path = args.out / f"{name}.npz"
            np.savez_compressed(path, **arrays, **{f"input/{k}": v for k, v in fx.items()})
            summary["file"] = path.name
        manifest["fixtures"][name] = summary
        print(f"{name}: n={summary['n']} {time.perf_counter() - t0:.2f}s")

    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out / 'manifest.json'}")


if __name__ == "__main__":
    main()
