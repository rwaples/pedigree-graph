"""Type stub for the PyO3 extension module ``pedigree_graph._native`` (ADR 0007)."""

import numpy as np
from numpy.typing import NDArray

def core_version() -> str: ...
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

class IdIndex:
    def __init__(self, ids: NDArray[np.int64], /) -> None: ...
    def resolve(self, query: NDArray[np.int64], /) -> NDArray[np.int32]: ...
