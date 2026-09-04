"""Deterministic pedigree inputs shared by the 0.7.1 baseline generator and the
0.8 differential tests.

This module must import nothing from ``pedigree_graph`` so that the same file
runs unchanged inside a ``v0.7.1`` worktree and on the ``v0.8`` branch.

Every fixture is a dict of input-aligned NumPy arrays: ``ids`` (int64),
``mother``/``father``/``twin`` (int64, ``-1`` missing; an ID absent from
``ids`` is an external reference), ``sex`` (int8; 0 female, 1 male). Rows are
topological (parents precede children) because 0.7.1 requires it.
"""

from __future__ import annotations

import hashlib

import numpy as np

GENERATOR_VERSION = 1

EXTERNAL_ID_BASE = 10_000_000


def input_hash(fx: dict[str, np.ndarray]) -> str:
    h = hashlib.sha256()
    for key in ("ids", "mother", "father", "twin", "sex"):
        arr = np.ascontiguousarray(fx[key])
        h.update(key.encode())
        h.update(str(arr.dtype).encode())
        h.update(arr.tobytes())
    return h.hexdigest()


def _fixture(rows: list[tuple[int, int, int, int]], twins: dict[int, int] | None = None) -> dict[str, np.ndarray]:
    """Rows are ``(id, mother_id, father_id, sex)``; ``twins`` maps id -> co-twin id."""
    ids = np.array([r[0] for r in rows], dtype=np.int64)
    mother = np.array([r[1] for r in rows], dtype=np.int64)
    father = np.array([r[2] for r in rows], dtype=np.int64)
    sex = np.array([r[3] for r in rows], dtype=np.int8)
    twin = np.full(len(rows), -1, dtype=np.int64)
    for a, b in (twins or {}).items():
        twin[np.flatnonzero(ids == a)] = b
        twin[np.flatnonzero(ids == b)] = a
    return {"ids": ids, "mother": mother, "father": father, "twin": twin, "sex": sex}


F, M = 0, 1


def motif_fixtures() -> dict[str, dict[str, np.ndarray]]:
    """Hand-built pedigrees, one per relationship motif or edge case.

    IDs are deliberately non-consecutive so that any row/ID confusion shows.
    """
    fx: dict[str, dict[str, np.ndarray]] = {}

    fx["nuclear_full_sibs"] = _fixture(
        [(10, -1, -1, F), (11, -1, -1, M), (12, 10, 11, F), (13, 10, 11, M), (14, 10, 11, F)],
    )
    fx["half_sibs_both_kinds"] = _fixture(
        [
            (20, -1, -1, F),
            (21, -1, -1, M),
            (22, -1, -1, M),
            (23, -1, -1, F),
            (24, 20, 21, F),
            (25, 20, 22, M),
            (26, 23, 21, F),
        ],
    )
    fx["lineal_five_generations"] = _fixture(
        [
            (30, -1, -1, F),
            (31, -1, -1, M),
            (32, 30, 31, F),
            (33, -1, -1, M),
            (34, 32, 33, M),
            (35, -1, -1, F),
            (36, 35, 34, F),
            (37, -1, -1, M),
            (38, 36, 37, M),
            (39, -1, -1, F),
            (40, 39, 38, F),
        ],
    )
    fx["avuncular_and_cousins"] = _fixture(
        [
            (50, -1, -1, F),
            (51, -1, -1, M),
            (52, 50, 51, F),
            (53, 50, 51, M),
            (54, -1, -1, M),
            (55, -1, -1, F),
            (56, 52, 54, F),
            (57, 55, 53, M),
            (58, -1, -1, M),
            (59, -1, -1, F),
            (60, 56, 58, F),
            (61, 59, 57, M),
        ],
    )
    fx["half_cousins"] = _fixture(
        [
            (70, -1, -1, F),
            (71, -1, -1, M),
            (72, -1, -1, M),
            (73, 70, 71, F),
            (74, 70, 72, M),
            (75, -1, -1, M),
            (76, -1, -1, F),
            (77, 73, 75, F),
            (78, 76, 74, M),
        ],
    )
    fx["double_first_cousins"] = _fixture(
        [
            (80, -1, -1, F),
            (81, -1, -1, M),
            (82, -1, -1, F),
            (83, -1, -1, M),
            (84, 80, 81, F),
            (85, 80, 81, M),
            (86, 82, 83, F),
            (87, 82, 83, M),
            (88, 84, 87, F),
            (89, 86, 85, M),
        ],
    )
    fx["mz_twins_with_children"] = _fixture(
        [
            (90, -1, -1, F),
            (91, -1, -1, M),
            (92, 90, 91, F),
            (93, 90, 91, F),
            (94, -1, -1, M),
            (95, -1, -1, M),
            (96, 92, 94, F),
            (97, 93, 95, M),
            (98, 96, 97, F),
        ],
        twins={92: 93},
    )
    fx["founder_mz_twins"] = _fixture(
        [
            (100, -1, -1, M),
            (101, -1, -1, M),
            (102, -1, -1, F),
            (103, -1, -1, F),
            (104, 102, 100, F),
            (105, 103, 101, M),
            (106, 104, 105, F),
        ],
        twins={100: 101},
    )
    fx["twins_mating_loop"] = _fixture(
        [
            (110, -1, -1, F),
            (111, -1, -1, M),
            (112, 110, 111, F),
            (113, 110, 111, F),
            (114, 110, 111, M),
            (115, 112, 114, F),
            (116, 113, 114, M),
            (117, 115, 116, F),
        ],
        twins={112: 113},
    )
    fx["backcross_and_selfing_like"] = _fixture(
        [
            (120, -1, -1, F),
            (121, -1, -1, M),
            (122, 120, 121, F),
            (123, 122, 121, M),
            (124, 122, 123, F),
            (125, 124, 123, M),
            (126, 124, 125, F),
        ],
    )
    fx["external_parents"] = _fixture(
        [
            (130, -1, -1, F),
            (131, EXTERNAL_ID_BASE + 1, EXTERNAL_ID_BASE + 2, M),
            (132, EXTERNAL_ID_BASE + 1, EXTERNAL_ID_BASE + 2, F),
            (133, EXTERNAL_ID_BASE + 1, EXTERNAL_ID_BASE + 3, M),
            (134, 130, 131, F),
            (135, 132, EXTERNAL_ID_BASE + 4, M),
            (136, 134, 135, F),
        ],
    )
    fx["one_parent_known"] = _fixture(
        [(140, -1, -1, F), (141, 140, -1, M), (142, 140, -1, F), (143, -1, 141, M), (144, 142, 143, F)],
    )
    fx["disconnected_components"] = _fixture(
        [
            (150, -1, -1, F),
            (151, -1, -1, M),
            (152, 150, 151, F),
            (160, -1, -1, F),
            (161, -1, -1, M),
            (162, 160, 161, M),
            (163, 160, 161, F),
            (170, -1, -1, M),
        ],
    )
    fx["single_individual"] = _fixture([(180, -1, -1, F)])
    return fx


def random_pedigree(
    seed: int,
    *,
    n_founders: int,
    n_generations: int,
    per_generation: int,
    p_missing_parent: float = 0.05,
    p_external_parent: float = 0.02,
    p_mz_twin: float = 0.02,
    p_skip_generation: float = 0.1,
) -> dict[str, np.ndarray]:
    """Overlapping-generation random pedigree with twins and external parents.

    Deterministic in ``seed`` and the keyword parameters. Parents are drawn
    from the previous generation, or with ``p_skip_generation`` from the one
    before it, so depth and generation disagree for some rows. IDs are a
    seeded permutation scaled to be non-consecutive; row order stays
    topological.
    """
    rng = np.random.default_rng(seed)
    mother: list[int] = []
    father: list[int] = []
    sex: list[int] = []
    twin_of: dict[int, int] = {}
    gen_rows: list[list[int]] = []

    def add(m: int, f: int, s: int) -> int:
        mother.append(m)
        father.append(f)
        sex.append(s)
        return len(mother) - 1

    gen_rows.append([add(-1, -1, int(rng.integers(0, 2))) for _ in range(n_founders)])
    n_external = 0
    for _g in range(1, n_generations + 1):
        rows: list[int] = []
        made = 0
        while made < per_generation:
            pool = gen_rows[-1]
            if len(gen_rows) >= 2 and rng.random() < p_skip_generation:
                pool = gen_rows[-2]
            females = [r for r in pool if sex[r] == 0]
            males = [r for r in pool if sex[r] == 1]
            m = int(rng.choice(females)) if females else -1
            f = int(rng.choice(males)) if males else -1
            if rng.random() < p_missing_parent:
                if rng.random() < 0.5:
                    m = -1
                else:
                    f = -1
            if rng.random() < p_external_parent:
                n_external += 1
                ext = -(n_external)
                if rng.random() < 0.5:
                    m = ext
                else:
                    f = ext
            s = int(rng.integers(0, 2))
            r = add(m, f, s)
            rows.append(r)
            made += 1
            if made < per_generation and rng.random() < p_mz_twin:
                t = add(m, f, s)
                rows.append(t)
                twin_of[r] = t
                twin_of[t] = r
                made += 1
        gen_rows.append(rows)

    n = len(mother)
    id_values = rng.permutation(n).astype(np.int64) * 7 + 3
    ids = id_values

    def to_id(ref: int) -> int:
        if ref == -1:
            return -1
        if ref < -1:
            return EXTERNAL_ID_BASE + (-ref)
        return int(ids[ref])

    mother_ids = np.array([to_id(x) for x in mother], dtype=np.int64)
    father_ids = np.array([to_id(x) for x in father], dtype=np.int64)
    twin_ids = np.full(n, -1, dtype=np.int64)
    for a, b in twin_of.items():
        twin_ids[a] = ids[b]
    return {
        "ids": ids,
        "mother": mother_ids,
        "father": father_ids,
        "twin": twin_ids,
        "sex": np.array(sex, dtype=np.int8),
    }


def deep_inbred_pedigree(
    seed: int, *, n_generations: int, per_generation: int, n_founders: int
) -> dict[str, np.ndarray]:
    """Closed herd: every generation mates only within the previous one."""
    return random_pedigree(
        seed,
        n_founders=n_founders,
        n_generations=n_generations,
        per_generation=per_generation,
        p_missing_parent=0.0,
        p_external_parent=0.0,
        p_mz_twin=0.0,
        p_skip_generation=0.0,
    )


RANDOM_FIXTURES: dict[str, dict] = {
    "random_1k": {"seed": 1001, "n_founders": 60, "n_generations": 6, "per_generation": 160},
    "deep_inbred_60g": {"seed": 1002, "kind": "deep", "n_founders": 8, "n_generations": 60, "per_generation": 12},
}

LARGE_FIXTURES: dict[str, dict] = {
    "random_30k": {"seed": 30_000, "n_founders": 1_500, "n_generations": 8, "per_generation": 3_600},
}

RELEASE_FIXTURES: dict[str, dict] = {
    "random_300k": {"seed": 300_000, "n_founders": 15_000, "n_generations": 8, "per_generation": 36_000},
}


def build_random(name: str, params: dict) -> dict[str, np.ndarray]:
    p = dict(params)
    kind = p.pop("kind", "random")
    if kind == "deep":
        return deep_inbred_pedigree(p.pop("seed"), **p)
    return random_pedigree(p.pop("seed"), **p)


def subsample_selection(fx: dict[str, np.ndarray], seed: int) -> np.ndarray:
    """Roughly half the rows in a seeded shuffled order; the ``from_subsample`` input."""
    n = len(fx["ids"])
    rng = np.random.default_rng(seed)
    keep = np.flatnonzero(rng.random(n) < 0.5)
    if keep.size == 0 and n > 0:
        keep = np.array([0])
    return rng.permutation(keep)
