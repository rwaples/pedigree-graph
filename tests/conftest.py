"""Shared fixtures, parity-pedigree helpers, and Hypothesis strategies for pedigree_graph tests.

The parity helpers below are the one place the ``tests/parity`` fixture tables
are turned into constructor columns, so the modules that build the same motif
and random pedigrees agree on what they built.
"""

import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from hypothesis import strategies as st

from pedigree_graph import PedigreeGraph

_DATA_DIR = Path(__file__).parent / "data"

sys.path.insert(0, str(Path(__file__).resolve().parent / "parity"))

import pedigrees  # noqa: E402

# Pedigree builders are capped small: degree-5 pair extraction is ~quadratic
# and the DP / BFS kernels JIT on first use, so large random pedigrees make the
# property suite slow and flaky.
PEDIGREE_MAX_N = 25


@pytest.fixture
def small_pedigree() -> pl.DataFrame:
    """Snapshot of simace.run_simulation(seed=42, N=1000, G_ped=3, G_sim=3, ...).

    Captured once and shipped as a parquet so tests don't need a runtime
    dependency on simace.  Byte-identical to the upstream fixture. Served as
    a polars frame — the family's primary library; focused pandas coverage
    lives in test_frame_inputs.py.
    """
    return pl.read_parquet(_DATA_DIR / "small_pedigree.parquet")


@st.composite
def pedigree_arrays(draw, *, max_n=PEDIGREE_MAX_N, non_inbred=False, complete=False):
    """Draw ``(ids, mother, father, sex)`` for a valid topological pedigree.

    ``ids`` are ``0..n-1``; each non-founder's parents are drawn from
    strictly earlier rows, so parents precede children (the ``from_arrays``
    contract).  When ``non_inbred`` is set, a mate is chosen constructively
    from earlier individuals whose *closed* ancestor set (ancestors plus
    self) is disjoint from the partner's, so ``compute_inbreeding()`` is
    all-zero by construction (no rejection loops).  When ``complete`` is set,
    every non-founder has *both* parents known (two distinct earlier rows) —
    no half-missing parentage — so founder contributions sum to 1 per cohort;
    such graphs may be inbred.  ``non_inbred`` and ``complete`` are not meant
    to be combined.  Sex is role-consistent: an id used only as a mother is
    female (0), only as a father is male (1), otherwise random.
    """
    n = draw(st.integers(min_value=1, max_value=max_n))
    mother = np.full(n, -1, dtype=np.int64)
    father = np.full(n, -1, dtype=np.int64)
    closed: list[set[int]] = [{i} for i in range(n)]  # ancestors including self

    for i in range(1, n):
        if complete:
            # A non-founder needs two distinct earlier rows, so i < 2 is forced
            # to be a founder; otherwise draw both parents (distinct).
            if i < 2 or not draw(st.booleans()):
                continue  # founder
            m = draw(st.integers(min_value=0, max_value=i - 1))
            f = draw(st.sampled_from([c for c in range(i) if c != m]))
            mother[i] = m
            father[i] = f
            continue  # closed[] is only consumed by the non_inbred branch
        if not draw(st.booleans()):
            continue  # founder
        m = draw(st.integers(min_value=-1, max_value=i - 1))
        if non_inbred and m != -1:
            cands = [c for c in range(i) if c != m and closed[m].isdisjoint(closed[c])]
            f = draw(st.sampled_from([-1, *cands])) if cands else -1
        else:
            f = draw(st.integers(min_value=-1, max_value=i - 1))
            if f == m and f != -1:
                f = -1
        mother[i] = m
        father[i] = f
        anc = {i}
        if m != -1:
            anc |= closed[m]
        if f != -1:
            anc |= closed[f]
        closed[i] = anc

    used_mother = {int(x) for x in mother if x != -1}
    used_father = {int(x) for x in father if x != -1}
    sex = np.empty(n, dtype=np.int8)
    for k in range(n):
        in_m, in_f = k in used_mother, k in used_father
        if in_m and not in_f:
            sex[k] = 0
        elif in_f and not in_m:
            sex[k] = 1
        else:
            sex[k] = draw(st.integers(min_value=0, max_value=1))

    return np.arange(n, dtype=np.int64), mother, father, sex


def _arrays_to_graph(arrays):
    """Build a PedigreeGraph from a ``(ids, mother, father, sex)`` tuple."""
    ids, mother, father, sex = arrays
    return PedigreeGraph.from_arrays(ids=ids, mothers=mother, fathers=father, sex=sex)


def random_pedigree(max_n=PEDIGREE_MAX_N):
    """Strategy of ``PedigreeGraph`` (possibly inbred), with role-consistent sex."""
    return pedigree_arrays(max_n=max_n, non_inbred=False).map(_arrays_to_graph)


def non_inbred_pedigree(max_n=PEDIGREE_MAX_N):
    """Strategy of ``PedigreeGraph`` whose ``compute_inbreeding()`` is all-zero."""
    return pedigree_arrays(max_n=max_n, non_inbred=True).map(_arrays_to_graph)


def complete_parentage_pedigree(max_n=PEDIGREE_MAX_N):
    """Strategy of ``PedigreeGraph`` where every non-founder has both parents known."""
    return pedigree_arrays(max_n=max_n, complete=True).map(_arrays_to_graph)


def relabel_pedigree(arrays, data):
    """Build a graph from *arrays* with id *values* relabelled by a random bijection.

    Draws a permutation and a multiplier from the Hypothesis *data* object and
    maps id ``i`` to ``perm[i] * mult + 3`` (possibly large / sparse), updating
    parent references to match.  Row order — hence topology and kinship — is
    unchanged, so the relabelled graph must produce identical results; this
    exercises the searchsorted ``_map_ids_to_rows`` path on non-trivial ids.
    """
    ids, mother, father, sex = arrays
    perm = np.array(data.draw(st.permutations(range(len(ids)))), dtype=np.int64)
    mult = data.draw(st.sampled_from([1, 1000, 1_000_000]))
    new_ids = perm * mult + 3
    new_mother = np.where(mother == -1, -1, new_ids[mother])
    new_father = np.where(father == -1, -1, new_ids[father])
    return PedigreeGraph.from_arrays(ids=new_ids, mothers=new_mother, fathers=new_father, sex=sex)


def parity_fixtures(*random_names: str) -> dict[str, dict[str, np.ndarray]]:
    """Every motif fixture, plus the named entries of ``pedigrees.RANDOM_FIXTURES``.

    Args:
        random_names: Random-fixture names to build alongside the motifs, e.g.
            ``"random_1k"`` or ``"deep_inbred_60g"``.

    Returns:
        Fixture tables by name.
    """
    fixtures = dict(pedigrees.motif_fixtures())
    for name in random_names:
        fixtures[name] = pedigrees.build_random(name, pedigrees.RANDOM_FIXTURES[name])
    return fixtures


def parity_columns(fixture: dict[str, np.ndarray], birth_year: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Return the constructor columns of one parity *fixture*, dated when *birth_year* is given."""
    columns = {
        "id": fixture["ids"],
        "mother": fixture["mother"],
        "father": fixture["father"],
        "twin": fixture["twin"],
        "sex": fixture["sex"],
    }
    if birth_year is not None:
        columns["birth_year"] = birth_year
    return columns


def kernel_inputs(graph: PedigreeGraph, first: np.ndarray, second: np.ndarray) -> tuple[np.ndarray, ...]:
    """Parent arrays and pair endpoints in the private depth-major order the kinship kernel runs in.

    The kernel peels the greater row, which is the ADR 0009 depth-then-row rule
    only in that order, so a caller reaching past ``pair_kinship`` has to
    translate first.
    """
    mother, father, twin = graph._topological_parents
    return mother, father, twin, graph._topology.translate(first), graph._topology.translate(second)
