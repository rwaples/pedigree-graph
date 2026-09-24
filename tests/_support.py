"""Helpers shared by several test modules, so no test module imports another."""

from __future__ import annotations

import math
import os
import subprocess
import sys
import textwrap
from dataclasses import fields
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from pedigree_graph import RELATIONSHIPS, PedigreeGraph
from pedigree_graph._topology import structural_depth

sys.path.insert(0, str(Path(__file__).resolve().parent / "parity"))

import pedigrees

# Fresh-interpreter runs: the native pool is built once per process.


def _run_child(*parts: str, **env: str) -> str:
    """Run the dedented *parts* as one script in a fresh interpreter, returning its stdout."""
    code = "\n".join(textwrap.dedent(part) for part in parts)
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={**os.environ, "PEDIGREE_GRAPH_ALLOW_TEST_SEAM": "1", **env},
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


CHILD_PRELUDE = """
    import hashlib, sys
    import numpy as np
    from pedigree_graph import PedigreeGraph, _native
    from pedigree_graph._threads import thread_budget
    n = 400
    rng = np.random.default_rng(7)
    mother = np.full(n, -1); father = np.full(n, -1)
    for i in range(20, n):
        lo = max(0, i - 60)
        mother[i], father[i] = rng.integers(lo, i), rng.integers(lo, i)
        if father[i] == mother[i]:
            father[i] = -1
    graph = PedigreeGraph.from_frame({"id": np.arange(n), "mother": mother, "father": father})
    view = np.where(np.arange(n) % 3 == 0, -1, np.arange(n) // 3 * 2 + np.arange(n) % 3 - 1).astype(np.int32)
"""


# Small hand-built pedigrees with known pairwise kinship.


def _ped_inbred_mz() -> pl.DataFrame:
    # G0: 0,1 founders; G1: 2,3 full-sibs of (0,1); G2: 4,5 MZ twins of (2,3).
    return pl.DataFrame(
        {
            "id": np.arange(6),
            "mother": np.array([-1, -1, 0, 0, 2, 2]),
            "father": np.array([-1, -1, 1, 1, 3, 3]),
            "twin": np.array([-1, -1, -1, -1, 5, 4]),
            "sex": np.array([0, 1, 0, 1, 0, 0]),
            "generation": np.array([0, 0, 1, 1, 2, 2]),
        }
    )


def _ped_double_first_cousins() -> pl.DataFrame:
    # 4,5 full sibs of (0,1); 6,7 full sibs of (2,3); 8=child(4,6),
    # 9=child(5,7), 10=child(4,6).  (8,9) and (9,10) are DOUBLE first cousins
    # (both parent-couples are full-sib pairs) -> phi = 0.125, twice the nominal
    # 1C lookup of 0.0625.
    return pl.DataFrame(
        {
            "id": np.arange(11),
            "mother": np.array([-1, -1, -1, -1, 0, 0, 2, 2, 4, 5, 4]),
            "father": np.array([-1, -1, -1, -1, 1, 1, 3, 3, 6, 7, 6]),
            "twin": np.full(11, -1),
            "sex": np.array([0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0]),
            "generation": np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2]),
        }
    )


def _ped_half_first_cousin_parents() -> pl.DataFrame:
    # 0..4 founders; 5=child(0,1), 6=child(0,2) share founder 0 -> half sibs;
    # 7=child(5,3), 8=child(6,4) -> half-first-cousins (phi=1/32);
    # 9=child(7,8) -> phi(9,7) = 0.5*((1+F_7)/2 + phi(8,7)) = 0.265625.
    # The disproof of threshold-pruning: a sub-threshold (1/32) parental kinship
    # feeds an above-threshold parent-offspring kinship.
    return pl.DataFrame(
        {
            "id": np.arange(10),
            "mother": np.array([-1, -1, -1, -1, -1, 0, 0, 5, 6, 7]),
            "father": np.array([-1, -1, -1, -1, -1, 1, 2, 3, 4, 8]),
            "twin": np.full(10, -1),
            "sex": np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1]),
            "generation": np.array([0, 0, 0, 0, 0, 1, 1, 2, 2, 3]),
        }
    )


def _ped_mz_twins_with_descendants() -> pl.DataFrame:
    # 0,1 founders; 2,3 full sibs of (0,1); 4,5 MZ twins of (2,3);
    # 6,7 unrelated founders (mates); 8=child(4,6), 9=child(5,7).
    # 4 and 5 are genome-identical, so 8 and 9 are half-sib-equivalent with phi
    # elevated above the non-inbred maternal-half-sib value by co-coalescence.
    return pl.DataFrame(
        {
            "id": np.arange(10),
            "mother": np.array([-1, -1, 0, 0, 2, 2, -1, -1, 4, 5]),
            "father": np.array([-1, -1, 1, 1, 3, 3, -1, -1, 6, 7]),
            "twin": np.array([-1, -1, -1, -1, 5, 4, -1, -1, -1, -1]),
            "sex": np.array([0, 1, 0, 1, 0, 0, 1, 1, 0, 0]),
            "generation": np.array([0, 0, 1, 1, 2, 2, 0, 0, 3, 3]),
        }
    )


def _ped_sib_mating() -> pl.DataFrame:
    # 0,1 founders; 2,3 full sibs; 4=child(2,3) (sib-mating, F_4=0.25).
    return pl.DataFrame(
        {
            "id": np.arange(5),
            "mother": np.array([-1, -1, 0, 0, 2]),
            "father": np.array([-1, -1, 1, 1, 3]),
            "twin": np.full(5, -1),
            "sex": np.array([0, 1, 0, 1, 0]),
            "generation": np.array([0, 0, 1, 1, 2]),
        }
    )


_PAIRWISE_FIXTURES = [
    _ped_inbred_mz,
    _ped_double_first_cousins,
    _ped_half_first_cousin_parents,
    _ped_mz_twins_with_descendants,
    _ped_sib_mating,
]


# Parity fixtures and the frozen 0.7.1 baseline.


CODES = tuple(RELATIONSHIPS)


ASYMMETRIC = tuple(code for code, category in RELATIONSHIPS.items() if not category.symmetric)


SYMMETRIC = tuple(code for code, category in RELATIONSHIPS.items() if category.symmetric)


def _fixtures() -> dict[str, dict[str, np.ndarray]]:
    fixtures = dict(pedigrees.motif_fixtures())
    # deep_inbred_60g is here for orientation, not for membership: sixty
    # generations off eight founders is where a pair most easily reaches both
    # arms of an asymmetric product, which is what decides whether the emitted
    # role is discovered or arbitrary (issue #21).
    for name in ("random_1k", "deep_inbred_60g"):
        fixtures[name] = pedigrees.build_random(name, pedigrees.RANDOM_FIXTURES[name])
    return fixtures


FIXTURES = _fixtures()


FIXTURE_NAMES = sorted(FIXTURES)


def _columns(fixture: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        "id": fixture["ids"],
        "mother": fixture["mother"],
        "father": fixture["father"],
        "twin": fixture["twin"],
        "sex": fixture["sex"],
    }


def _graph(name: str) -> PedigreeGraph:
    return PedigreeGraph.from_frame(_columns(FIXTURES[name]))


# Effective-size pedigree builders.


def _df(records: list[dict]) -> pl.DataFrame:
    """Build a pedigree DataFrame from per-row dicts (defaults filled)."""
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


def _build_closed_line(n_gens: int = 5) -> pl.DataFrame:
    """Closed-line full-sib mating: 2 founders, 1 male + 1 female per gen for ``n_gens``."""
    records = [
        {"id": 0, "sex": 1, "generation": 0},
        {"id": 1, "sex": 0, "generation": 0},
    ]
    next_id = 2
    prev_m, prev_f = 0, 1
    for g in range(1, n_gens + 1):
        m = next_id
        records.append({"id": m, "sex": 1, "generation": g, "mother": prev_f, "father": prev_m})
        f = next_id + 1
        records.append({"id": f, "sex": 0, "generation": g, "mother": prev_f, "father": prev_m})
        prev_m, prev_f = m, f
        next_id += 2
    return _df(records)


def _random_mating(n_per_gen: int, n_gens: int, seed: int) -> pl.DataFrame:
    """Closed random-mating pedigree, balanced sex, discrete non-overlapping generations."""
    rng = np.random.default_rng(seed)
    half = n_per_gen // 2
    records: list[dict] = []
    next_id = 0
    previous_m: list[int] = []
    previous_f: list[int] = []
    for g in range(n_gens + 1):
        current_m: list[int] = []
        current_f: list[int] = []
        for j in range(n_per_gen):
            sex = 1 if j < half else 0
            record = {"id": next_id, "sex": sex, "generation": g}
            if g > 0:
                record["mother"] = int(rng.choice(previous_f))
                record["father"] = int(rng.choice(previous_m))
            records.append(record)
            (current_m if sex == 1 else current_f).append(next_id)
            next_id += 1
        previous_m, previous_f = current_m, current_f
    return _df(records)


# Result-shape assertions for the effective-size API.


def _assert_owned_read_only(result: object) -> None:
    for name, value in _array_fields(result).items():
        assert not value.flags.writeable, name
        assert value.flags.c_contiguous, name
        assert value.flags.owndata, name
        with pytest.raises(ValueError, match="read-only"):
            value[...] = 0


def _assert_plain_python(value: object, where: str) -> None:
    if value is None or type(value) in (bool, int, str):
        return
    if type(value) is float:
        assert not math.isnan(value), f"{where} is NaN"
        return
    if type(value) is list:
        for i, item in enumerate(value):
            _assert_plain_python(item, f"{where}[{i}]")
        return
    if type(value) is dict:
        for key, item in value.items():
            assert type(key) is str, f"{where} key {key!r}"
            _assert_plain_python(item, f"{where}.{key}")
        return
    raise AssertionError(f"{where} is {type(value).__name__}, not plain Python")


def _array_fields(result: object) -> dict[str, np.ndarray]:
    found = {}
    for f in fields(result):
        value = getattr(result, f.name)
        if isinstance(value, np.ndarray):
            found[f.name] = value
    return found


# ADR 0008 MZ-twin inbreeding fixtures.


def _mz_frame(ids, mother, father, twin):
    m = np.asarray(mother, dtype=np.int32)
    f = np.asarray(father, dtype=np.int32)
    return pl.DataFrame(
        {
            "id": ids,
            "mother": mother,
            "father": father,
            "twin": twin,
            "sex": [0] * len(ids),
            "generation": structural_depth(np.asarray(m, dtype=np.int32), np.asarray(f, dtype=np.int32)).tolist(),
        }
    )


# (name, mother, father, twin, {row: expected F}); ids are 0..n-1, parents precede children.
ADR_0008_FIXTURES = [
    (
        "mz_ancestry_no_loop",
        [-1, -1, -1, -1, 0, 0, -1, 4],
        [-1, -1, -1, -1, 1, 1, -1, 2],
        [-1, -1, -1, -1, 5, 4, -1, -1],
        {7: 0.0},
    ),
    (
        "mz_only_link",
        [-1, -1, -1, -1, 0, 0, 4, 5, 6],
        [-1, -1, -1, -1, 1, 1, 2, 3, 7],
        [-1, -1, -1, -1, 5, 4, -1, -1, -1],
        {8: 1 / 8},
    ),
    (
        "mz_plus_full_sib_loop",
        [-1, -1, -1, -1, 2, 2, 0, 0, 6, 7, 8],
        [-1, -1, -1, -1, 3, 3, 1, 1, 4, 5, 9],
        [-1, -1, -1, -1, -1, -1, 7, 6, -1, -1, -1],
        {10: 3 / 16},
    ),
    (
        "double_mz_grandparents",
        [-1, -1, -1, -1, 0, 0, 2, 2, 4, 5, 8],
        [-1, -1, -1, -1, 1, 1, 3, 3, 6, 7, 9],
        [-1, -1, -1, -1, 5, 4, 7, 6, -1, -1, -1],
        {10: 1 / 4},
    ),
    (
        "founder_mz_twins_only_link",
        [-1, -1, -1, -1, 0, 1, 4],
        [-1, -1, -1, -1, 2, 3, 5],
        [1, 0, -1, -1, -1, -1, -1],
        {6: 1 / 8},
    ),
    (
        "inbred_twins",
        [-1, -1, 0, 0, -1, 2, 2, 5, 6, 7],
        [-1, -1, 1, 1, -1, 3, 3, 4, 4, 8],
        [-1, -1, -1, -1, -1, 6, 5, -1, -1, -1],
        {5: 1 / 4, 6: 1 / 4, 9: 0.25 * (0.625 + 0.5)},
    ),
    (
        "twins_mate_each_other",
        [-1, -1, 0, 0, 2],
        [-1, -1, 1, 1, 3],
        [-1, -1, 3, 2, -1],
        {4: 0.5},
    ),
]
