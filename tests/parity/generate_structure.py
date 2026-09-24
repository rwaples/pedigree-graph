"""Freeze the structural outputs the 0.7.1 baseline locked, captured through the public API.

Run against the installed package::

    pixi run python tests/parity/generate_structure.py

The first capture also passed ``--verify-against`` the 0.7.1
``manifest.json``, which is now only in git history.

Per fixture it captures, with the 0.7.1 dtypes, sort order and ``_sha``
layout: ``depth``, ``n_ancestors`` (``distinct_ancestor_counts``),
``n_descendants`` (``descendant_path_counts`` as int32, or
``n_descendants_overflow`` when a count exceeds int32), ``approx_support``
(``approximate_kinship_matrix(0.001)`` upper-triangle support),
``complete_support`` and ``complete_values`` (``kinship_matrix()``, small
fixtures only), ``subsample/rows`` (the seeded view selection), and the
new-only ``view_pairs/<code>`` (``view(ids=...).relationship_pairs(max_degree=5)``
on that selection).

Small fixtures (motifs, ``small_pedigree``, ``random_1k``,
``deep_inbred_60g``) get their arrays in one ``.npz`` each; ``random_30k``
gets counts and hashes only.  Every key in ``manifest.json`` stores its
``sha256`` and ``v0_7_1_sha256``, the digest the 0.7.1 baseline held for it,
or ``null`` for a key 0.7.1 did not have.  With ``--verify-against`` the
generator refuses to write unless every carried digest, count and input hash
equals the 0.7.1 one, so the switch away from that baseline is proven at
capture time.  Regeneration is a deliberate act: a changed hash is a changed
contract, never a test fix.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import pedigrees  # noqa: E402
from _hashing import APPROX_THRESHOLD, SUBSAMPLE_SEED, _sha, _upper_coo  # noqa: E402

MAX_DEGREE = 5
#: Keys whose digest must equal the 0.7.1 one.  ``view_pairs/*`` is new-only.
CARRIED_KEYS = (
    "depth",
    "n_ancestors",
    "n_descendants",
    "approx_support",
    "complete_support",
    "complete_values",
    "subsample/rows",
)
#: Counts carried from the 0.7.1 manifest's ``counts`` section.
CARRIED_COUNTS = ("approx_support_upper_nnz", "complete_upper_nnz")


def _columns(fx: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {"id": fx["ids"], "mother": fx["mother"], "father": fx["father"], "twin": fx["twin"], "sex": fx["sex"]}


def _capture(pg_mod, fx: dict[str, np.ndarray], *, full_arrays: bool) -> tuple[dict, dict]:
    """Return ``(arrays, summary)``; ``arrays`` is empty when ``full_arrays`` is False.

    ``summary`` holds ``n``, ``counts``, ``hashes`` (key -> sha256), the
    subsample size and, when a descendant count overflows int32,
    ``n_descendants_overflow``.
    """
    graph = pg_mod.PedigreeGraph.from_frame(_columns(fx))
    arrays: dict[str, np.ndarray] = {}
    summary: dict = {"n": graph.n_individuals, "counts": {}, "hashes": {}}

    def put(key: str, *named: tuple[str, np.ndarray]) -> None:
        summary["hashes"][key] = _sha(*(array for _, array in named))
        if full_arrays:
            arrays.update(named)

    put("depth", ("depth", np.asarray(graph.depth, dtype=np.int32)))
    put("n_ancestors", ("n_ancestors", np.asarray(graph.distinct_ancestor_counts(), dtype=np.int32)))
    descendants = graph.descendant_path_counts()
    if descendants.size and int(descendants.max()) > np.iinfo(np.int32).max:
        summary["n_descendants_overflow"] = True
    else:
        put("n_descendants", ("n_descendants", descendants.astype(np.int32)))

    r, c, _ = _upper_coo(graph.approximate_kinship_matrix(min_propagated_kinship=APPROX_THRESHOLD))
    summary["counts"]["approx_support_upper_nnz"] = len(r)
    put("approx_support", ("approx/row", r), ("approx/col", c))
    if full_arrays:
        r, c, v = _upper_coo(graph.kinship_matrix())
        summary["counts"]["complete_upper_nnz"] = len(r)
        put("complete_support", ("complete/row", r), ("complete/col", c))
        put("complete_values", ("complete/val", v))

    keep = pedigrees.subsample_selection(fx, SUBSAMPLE_SEED)
    summary["subsample_n"] = len(keep)
    put("subsample/rows", ("subsample/rows", keep.astype(np.int64)))
    for code, block in graph.view(ids=fx["ids"][keep]).relationship_pairs(max_degree=MAX_DEGREE).items():
        first, second = block
        summary["counts"][f"view_pairs/{code}"] = len(block)
        put(f"view_pairs/{code}", (f"view_pairs/{code}/first", first), (f"view_pairs/{code}/second", second))
    return arrays, summary


def _fixtures(*, skip_large: bool) -> list[tuple[str, dict, dict[str, np.ndarray], bool]]:
    import polars as pl

    fixtures = [(name, {"kind": "motif"}, fx, True) for name, fx in pedigrees.motif_fixtures().items()]
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
        fixtures.append((name, dict(params), pedigrees.build_random(name, params), True))
    if not skip_large:
        for name, params in pedigrees.LARGE_FIXTURES.items():
            fixtures.append((name, dict(params), pedigrees.build_random(name, params), False))
    return fixtures


def _verify(name: str, entry: dict, old: dict | None) -> list[str]:
    """Every way *entry* differs from the 0.7.1 manifest entry *old* on a carried key."""
    if old is None:
        return [f"{name}: not in the 0.7.1 manifest"]
    problems = []
    if entry["input_hash"] != old["input_hash"]:
        problems.append(f"{name}: input_hash differs")
    if entry["n"] != old["n"]:
        problems.append(f"{name}: n {entry['n']} != {old['n']}")
    if entry.get("n_descendants_overflow") != old.get("n_descendants_overflow"):
        problems.append(f"{name}: n_descendants_overflow differs")
    if entry["subsample_n"] != old["subsample"]["n"]:
        problems.append(f"{name}: subsample n differs")
    problems += [
        f"{name}: count {key} differs"
        for key in CARRIED_COUNTS
        if (key in entry["counts"]) != (key in old["counts"]) or entry["counts"].get(key) != old["counts"].get(key)
    ]
    problems += [
        f"{name}: {key} differs from 0.7.1"
        for key, record in entry["keys"].items()
        if key in CARRIED_KEYS and record["sha256"] != record["v0_7_1_sha256"]
    ]
    missing = {key for key in CARRIED_KEYS if key in old["hashes"]} - set(entry["keys"])
    problems += [f"{name}: 0.7.1 key {key} not captured" for key in sorted(missing)]
    return problems


def _git_commit(root: Path) -> str:
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.run(["git", "-C", str(root), "diff", "--quiet", "HEAD"], check=False).returncode != 0
    return f"{head}-dirty" if dirty else head


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=HERE.parent / "data" / "structure_v0.10")
    ap.add_argument("--verify-against", type=Path, help="0.7.1 manifest.json; refuse to write on any carried mismatch")
    ap.add_argument("--skip-large", action="store_true")
    args = ap.parse_args()

    import pedigree_graph as pg_mod

    old_fixtures = json.loads(args.verify_against.read_text())["fixtures"] if args.verify_against else None
    manifest = {
        "generator_version": pedigrees.GENERATOR_VERSION,
        "package_commit": _git_commit(Path(pg_mod.__file__).resolve().parent.parent),
        "package_version": getattr(pg_mod, "__version__", None),
        "max_degree": MAX_DEGREE,
        "approx_threshold": APPROX_THRESHOLD,
        "subsample_seed": SUBSAMPLE_SEED,
        "carried_keys": list(CARRIED_KEYS),
        "fixtures": {},
    }
    outputs: dict[str, dict[str, np.ndarray]] = {}
    problems: list[str] = []
    for name, params, fx, full_arrays in _fixtures(skip_large=args.skip_large):
        t0 = time.perf_counter()
        arrays, summary = _capture(pg_mod, fx, full_arrays=full_arrays)
        old = None if old_fixtures is None else old_fixtures.get(name)
        entry = {
            "n": summary["n"],
            "params": params,
            "input_hash": pedigrees.input_hash(fx),
            "subsample_n": summary["subsample_n"],
            "counts": summary["counts"],
            "keys": {
                key: {
                    "sha256": digest,
                    "v0_7_1_sha256": None if old is None or key not in CARRIED_KEYS else old["hashes"].get(key),
                }
                for key, digest in summary["hashes"].items()
            },
        }
        if "n_descendants_overflow" in summary:
            entry["n_descendants_overflow"] = True
        if old_fixtures is not None:
            problems += _verify(name, entry, old)
        if full_arrays:
            entry["file"] = f"{name}.npz"
            outputs[name] = {**arrays, **{f"input/{k}": v for k, v in fx.items()}}
        manifest["fixtures"][name] = entry
        print(f"{name}: n={entry['n']} keys={len(entry['keys'])} {time.perf_counter() - t0:.2f}s")

    if old_fixtures is not None:
        skipped = set(old_fixtures) - set(manifest["fixtures"])
        problems += [f"{name}: in the 0.7.1 manifest but not captured" for name in sorted(skipped)]
    if problems:
        sys.exit("refusing to write; carried keys differ from 0.7.1:\n  " + "\n  ".join(problems))

    args.out.mkdir(parents=True, exist_ok=True)
    for name, arrays in outputs.items():
        np.savez_compressed(args.out / f"{name}.npz", **arrays)
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out / 'manifest.json'}")


if __name__ == "__main__":
    main()
