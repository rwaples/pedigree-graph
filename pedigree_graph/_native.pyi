"""Type stub for the PyO3 extension module ``pedigree_graph._native`` (ADR 0007)."""

import numpy as np
from numpy.typing import NDArray

def core_version() -> str: ...
def max_degree_max() -> int: ...
def is_topological(mother_rows: NDArray[np.int32], father_rows: NDArray[np.int32], /) -> bool: ...
def structural_depth(mother_rows: NDArray[np.int32], father_rows: NDArray[np.int32], /) -> NDArray[np.int32]: ...
def depth_major_order(depth: NDArray[np.int32], /) -> tuple[NDArray[np.int64], NDArray[np.int64]] | None: ...
def validate_acyclic(
    ids: NDArray[np.int64], mother_rows: NDArray[np.int32], father_rows: NDArray[np.int32], /
) -> None: ...

class BuiltPedigree:
    ids: NDArray[np.int64]
    mother_ids: NDArray[np.int64]
    father_ids: NDArray[np.int64]
    twin_ids: NDArray[np.int64]
    mother_rows: NDArray[np.int32]
    father_rows: NDArray[np.int32]
    twin_rows: NDArray[np.int32]
    sex: NDArray[np.int8] | None
    generation: NDArray[np.int32] | None
    birth_year: NDArray[np.int32] | None
    rows_topological: bool

def build_pedigree(
    ids: NDArray[np.int64],
    mother: NDArray[np.int64],
    father: NDArray[np.int64],
    twin: NDArray[np.int64] | None = None,
    sex: NDArray[np.int64] | None = None,
    generation: NDArray[np.int64] | None = None,
    birth_year: NDArray[np.int64] | None = None,
    /,
    *,
    sex_encoding: str,
    max_rows: int | None = None,
) -> BuiltPedigree: ...
def configure_pool(threads: int, /) -> int: ...
def relationship_counts(
    pedigree: BuiltPedigree,
    /,
    *,
    max_degree: int,
    threads: int,
    selected: NDArray[np.bool_] | None = None,
) -> dict[str, int]: ...
def relationship_pairs(
    pedigree: BuiltPedigree,
    /,
    *,
    max_degree: int,
    requested: list[str],
    threads: int,
    execution: str,
    view_rows: NDArray[np.int32] | None = None,
) -> dict[str, tuple[NDArray[np.int32], NDArray[np.int32]]]: ...
def pair_kinship(
    pedigree: BuiltPedigree,
    depth: NDArray[np.int32],
    first: NDArray[np.int32],
    second: NDArray[np.int32],
    /,
    *,
    layout: str = "rows",
) -> NDArray[np.float32]: ...
def kinship_support_values(
    pedigree: BuiltPedigree,
    depth: NDArray[np.int32],
    indptr: NDArray[np.int64],
    indices: NDArray[np.int32],
    /,
    *,
    layout: str = "rows",
) -> NDArray[np.float32]: ...
def allocation_families() -> list[str]: ...

# Test seam, not public API: refused unless the process was started with
# PEDIGREE_GRAPH_ALLOW_TEST_SEAM=1.
def fail_next_allocation(family: str | None, min_elements: int = 0) -> None: ...

class IdIndex:
    def __init__(self, ids: NDArray[np.int64], /) -> None: ...
    def resolve(self, query: NDArray[np.int64], /) -> NDArray[np.int32]: ...
