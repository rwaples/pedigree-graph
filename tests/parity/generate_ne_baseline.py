"""Freeze the slice-6b effective-size outputs as the ``h = 1`` parity golden.

Run against the ``v0.8`` branch at ``00a3667`` (slice 6b), before the sparse
label rewrite of slice 6c::

    pixi run python tests/parity/generate_ne_baseline.py

Every fixture carries dense ``0..g_max`` generation labels, so slice 6c's gap
formula must reduce to the one-step arithmetic bit for bit on each of them.
``tests/test_ne_h1_parity.py`` replays the same fixtures through the current
estimators and asserts the serialized results are equal.

This module is a generator, not a test: it targets the API of its base commit
``00a3667`` and is never migrated forward.  Only :func:`capture` reaches into
``pedigree_graph``, and it imports there rather than at module scope, so the
test module that reads the fixtures and the frozen output imports nothing from
the package through this file.  The fixtures are deterministic and import
nothing from the test modules, so the golden can be regenerated from any
checkout of the base commit.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
OUT = DATA / "ne_baseline_6b"


def _df(records: list[dict]) -> pl.DataFrame:
    rows = [
        {
            "id": r["id"],
            "mother": r.get("mother", -1),
            "father": r.get("father", -1),
            "twin": r.get("twin", -1),
            "sex": r["sex"],
            "generation": r["generation"],
        }
        for r in records
    ]
    return pl.DataFrame(rows)


def closed_line(n_gens: int = 5) -> pl.DataFrame:
    records = [{"id": 0, "sex": 1, "generation": 0}, {"id": 1, "sex": 0, "generation": 0}]
    next_id, prev_m, prev_f = 2, 0, 1
    for g in range(1, n_gens + 1):
        m, f = next_id, next_id + 1
        records.append({"id": m, "sex": 1, "generation": g, "mother": prev_f, "father": prev_m})
        records.append({"id": f, "sex": 0, "generation": g, "mother": prev_f, "father": prev_m})
        prev_m, prev_f = m, f
        next_id += 2
    return _df(records)


def random_mating(seed: int, n_per_gen: int, n_gens: int) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    n_male = n_per_gen // 2
    n_female = n_per_gen - n_male
    records: list[dict] = []
    prev_male = list(range(n_male))
    prev_female = list(range(n_male, n_male + n_female))
    records.extend({"id": i, "sex": 1, "generation": 0} for i in prev_male)
    records.extend({"id": i, "sex": 0, "generation": 0} for i in prev_female)
    next_id = n_male + n_female
    for g in range(1, n_gens + 1):
        cur_male: list[int] = []
        cur_female: list[int] = []
        for sex, bucket, count in ((1, cur_male, n_male), (0, cur_female, n_female)):
            for _ in range(count):
                f = int(rng.choice(prev_male))
                m = int(rng.choice(prev_female))
                records.append({"id": next_id, "sex": sex, "generation": g, "mother": m, "father": f})
                bucket.append(next_id)
                next_id += 1
        prev_male, prev_female = cur_male, cur_female
    return _df(records)


def skip_gen() -> pl.DataFrame:
    return _df(
        [
            {"id": 0, "sex": 1, "generation": 0},
            {"id": 1, "sex": 0, "generation": 0},
            {"id": 2, "sex": 1, "generation": 0},
            {"id": 3, "sex": 0, "generation": 0},
            {"id": 4, "sex": 1, "generation": 1, "mother": 1, "father": 0},
            {"id": 5, "sex": 0, "generation": 1, "mother": 1, "father": 0},
            {"id": 6, "sex": 1, "generation": 2, "mother": 5, "father": 4},
            {"id": 7, "sex": 0, "generation": 1, "mother": 3, "father": 2},
            {"id": 8, "sex": 1, "generation": 3, "mother": 3, "father": 6},
            {"id": 9, "sex": 0, "generation": 3, "mother": 7, "father": 6},
        ]
    )


def with_birth_years(df: pl.DataFrame, base: int = 1900, step: int = 3) -> pl.DataFrame:
    return df.with_columns((pl.col("generation") * step + base).cast(pl.Int32).alias("birth_year"))


def fixtures() -> dict[str, pl.DataFrame]:
    return {
        "closed_line_5": closed_line(5),
        "wf_n20_g4": random_mating(seed=11, n_per_gen=20, n_gens=4),
        "wf_n60_g6": random_mating(seed=23, n_per_gen=60, n_gens=6),
        "skip_gen": skip_gen(),
        "wf_n40_g5_birth_years": with_birth_years(random_mating(seed=5, n_per_gen=40, n_gens=5)),
        "small_pedigree": pl.read_parquet(DATA / "small_pedigree.parquet"),
    }


def capture(name: str, df: pl.DataFrame) -> dict[str, dict]:
    from pedigree_graph import PedigreeGraph, compute_all_ne

    pg = PedigreeGraph(df)
    hill_kwargs = {"hill_vk_scale": name.endswith("birth_years")}
    return {key: result.to_dict() for key, result in compute_all_ne(pg, **hill_kwargs).items()}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    manifest = {"package_commit": commit, "fixtures": sorted(fixtures())}
    for name, df in fixtures().items():
        payload = capture(name, df)
        (OUT / f"{name}.json").write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"wrote {len(manifest['fixtures'])} fixtures at {commit} to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
