"""Type stub for the PyO3 extension module ``pedigree_graph._native`` (ADR 0007)."""

from collections.abc import Callable
from typing import Literal

import numpy as np
from numpy.typing import NDArray

# `(phase, rows_done, rows_total)` once per tick; see `run_watched` in lib.rs.
_ProgressCallback = Callable[[Literal["preparing", "walking", "finishing"], int, int | None], None]

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
    progress: _ProgressCallback | None = None,
    tick: float = 1.0,
) -> dict[str, int]: ...
def compact_view_counts(
    pedigree: BuiltPedigree,
    view_rows: NDArray[np.int32],
    /,
    *,
    max_degree: int,
    threads: int,
    progress: _ProgressCallback | None = None,
    tick: float = 1.0,
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
    compact: bool = False,
    progress: _ProgressCallback | None = None,
    tick: float = 1.0,
) -> dict[str, tuple[NDArray[np.int32], NDArray[np.int32]]]: ...
def relationship_moments(
    pedigree: BuiltPedigree,
    /,
    *,
    max_degree: int,
    requested: list[str],
    threads: int,
    labels_first: NDArray[np.int32],
    n_labels_first: int,
    labels_second: NDArray[np.int32],
    n_labels_second: int,
    values: NDArray[np.int64],
    products: list[tuple[int, int, int, int]],
    same: NDArray[np.int64],
    symmetric: str,
    memory_budget_bytes: int,
    view_rows: NDArray[np.int32] | None = None,
    compact: bool = False,
    progress: _ProgressCallback | None = None,
    tick: float = 1.0,
) -> tuple[int, NDArray[np.uint8], int, int, int, list[int], int]: ...
def relatives_per_person(
    pedigree: BuiltPedigree,
    /,
    *,
    max_degree: int,
    requested: list[str],
    threads: int,
    columns: list[tuple[NDArray[np.float64], NDArray[np.float64] | float]],
    view_rows: NDArray[np.int32] | None = None,
    compact: bool = False,
    progress: _ProgressCallback | None = None,
    tick: float = 1.0,
) -> tuple[NDArray[np.uint32], int, int, int, int, list[int]]: ...
def moments_plan(
    *,
    n_categories: int,
    n_labels_first: int,
    n_labels_second: int,
    n_columns: int,
    n_products: int,
    n_same: int,
    threads: int,
    memory_budget_bytes: int,
) -> tuple[int, int]: ...

MOMENTS_MAX_VALUE_COLUMNS: int
MOMENTS_MAX_SAME_KEYS: int

#: ``(shape, n_columns, operands, exponents, width, bytes)``.
_Table = tuple[list[int], int, list[tuple[int, int, int, int]], list[int], int, NDArray[np.uint8]]
#: ``(shape, exponents, width, bytes)``.
_TableOut = tuple[list[int], list[int], int, NDArray[np.uint8]]

def moments_pack(
    factors: list[NDArray[np.int64]], n: int, kind: str
) -> tuple[NDArray[np.int32], int, list[NDArray[np.int64]]]: ...
def moments_quantize(column: NDArray[np.float64], out: NDArray[np.int64], j: int, field: str) -> int: ...
def moments_ratio(numerator: str, denominator: str, shift: int) -> float | None: ...
def moments_table_counts(table_arg: _Table) -> NDArray[np.int64]: ...
def moments_table_sum(table_arg: _Table, axis: int) -> _TableOut: ...
def moments_table_select(table_arg: _Table, axis: int, positions: list[int]) -> _TableOut: ...
def moments_table_merge(a: _Table, b: _Table) -> _TableOut: ...
def moments_table_derive(table_arg: _Table, statistic: str, index: int, name: str) -> NDArray[np.float64]: ...
def relationship_burden(
    pedigree: BuiltPedigree,
    depth: NDArray[np.int32],
    /,
    *,
    threads: int,
    progress: _ProgressCallback | None = None,
    tick: float = 1.0,
) -> tuple[dict[str, int], NDArray[np.uint32], NDArray[np.uint64]]: ...
def pair_kinship(
    pedigree: BuiltPedigree,
    depth: NDArray[np.int32],
    first: NDArray[np.int32],
    second: NDArray[np.int32],
    /,
    *,
    threads: int,
) -> NDArray[np.float32]: ...
def kinship_support_values(
    pedigree: BuiltPedigree,
    depth: NDArray[np.int32],
    indptr: NDArray[np.int64],
    indices: NDArray[np.int32],
    /,
    *,
    threads: int,
) -> NDArray[np.float32]: ...
def kinship_csc(
    pedigree: BuiltPedigree,
    depth: NDArray[np.int32],
    /,
) -> tuple[NDArray[np.int32], NDArray[np.int32], NDArray[np.float32]]: ...
def approximate_kinship_csc(
    pedigree: BuiltPedigree,
    depth: NDArray[np.int32],
    threshold: float,
    /,
) -> tuple[NDArray[np.int32], NDArray[np.int32], NDArray[np.float32]]: ...
def generation_kinship_sums(
    pedigree: BuiltPedigree,
    depth: NDArray[np.int32],
    inbreeding: NDArray[np.float64],
    labels: NDArray[np.int32],
    n_buckets: int,
    /,
) -> NDArray[np.float64]: ...
def inbreeding(pedigree: BuiltPedigree, depth: NDArray[np.int32], /) -> NDArray[np.float64]: ...
def distinct_ancestor_counts(pedigree: BuiltPedigree, depth: NDArray[np.int32] | None, /) -> NDArray[np.int32]: ...
def descendant_path_counts(pedigree: BuiltPedigree, depth: NDArray[np.int32] | None, /) -> NDArray[np.int64]: ...
def equivalent_generations(pedigree: BuiltPedigree, depth: NDArray[np.int32] | None, /) -> NDArray[np.float64]: ...
def founder_contribution_means(
    pedigree: BuiltPedigree,
    depth: NDArray[np.int32],
    cohort: NDArray[np.int32],
    n_cohorts: int,
    founder_column: NDArray[np.int64],
    n_genomes: int,
    /,
) -> NDArray[np.float64]: ...
def allocation_families() -> list[str]: ...

# Test seam, not public API: refused unless the process was started with
# PEDIGREE_GRAPH_ALLOW_TEST_SEAM=1.
def _panic_in_watched_worker_for_test(*, threads: int, tick: float) -> None: ...
def fail_next_allocation(family: str | None, min_elements: int = 0) -> None: ...

class IdIndex:
    def __init__(self, ids: NDArray[np.int64], /) -> None: ...
    def resolve(self, query: NDArray[np.int64], /) -> NDArray[np.int32]: ...
