"""Oracle for ``pedigree_graph._native.build_pedigree``: the Python construction path through 0.8.1.

The semantic half of ``pedigree_graph._input.parse_pedigree_input`` (range,
sex encoding, duplicate and shared-parent ids, id resolution, topology, the
MZ contract, optional-column collapse) plus ``PedigreeGraph._validate_birth_year_topology``,
moved here verbatim when construction went native.  It takes the
same int64 columns the native builder takes and raises the same structured
errors, so the differential test compares codes, fields, and messages.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pedigree_graph._errors import PedigreeValidationError, ResourceError
from tests.oracle import topology

_INT64_MAX = int(np.iinfo(np.int64).max)
_INT32_MAX = int(np.iinfo(np.int32).max)

_RANGES = {
    "id": (0, _INT64_MAX),
    "mother": (-1, _INT64_MAX),
    "father": (-1, _INT64_MAX),
    "twin": (-1, _INT64_MAX),
    "generation": (-1, _INT32_MAX),
    "birth_year": (-1, _INT32_MAX),
}
_SEX_ENCODINGS = {
    "simace": ((-1, 1), (-1, 1), None),
    "plink": ((-1, 2), (0, 2), {-1: -1, 0: -1, 1: 1, 2: 0}),
}


@dataclass(frozen=True)
class Built:
    ids: np.ndarray
    mother_ids: np.ndarray
    father_ids: np.ndarray
    twin_ids: np.ndarray
    mother_rows: np.ndarray
    father_rows: np.ndarray
    twin_rows: np.ndarray
    sex: np.ndarray | None
    generation: np.ndarray | None
    birth_year: np.ndarray | None
    rows_topological: bool


def _out_of_range(field, position, value, minimum, maximum):
    return PedigreeValidationError(
        "value_out_of_range",
        f"{field!r} value at position {position} is outside [{minimum}, {maximum}]",
        field=field,
        position=position,
        value=value,
        minimum=minimum,
        maximum=maximum,
    )


def _check_range(name, values, minimum, maximum, reported):
    if values.size == 0 or (int(values.min()) >= minimum and int(values.max()) <= maximum):
        return
    position = int(np.argmax((values < minimum) | (values > maximum)))
    raise _out_of_range(name, position, int(values[position]), reported[0], reported[1])


def _check_duplicate_ids(ids):
    if ids.size < 2:
        return
    ordered = np.sort(ids)
    repeats = ordered[1:] == ordered[:-1]
    if not repeats.any():
        return
    duplicated = int(ordered[int(np.argmax(repeats))])
    rows = tuple(int(position) for position in np.flatnonzero(ids == duplicated))
    count = int(np.count_nonzero(repeats))
    raise PedigreeValidationError(
        "duplicate_id",
        f"id {duplicated} appears at rows {rows}; {count} row(s) repeat an earlier id",
        id=duplicated,
        rows=rows,
        duplicate_count=count,
    )


def _check_same_parent(ids, mother_ids, father_ids):
    same = (mother_ids == father_ids) & (mother_ids >= 0)
    if not same.any():
        return
    row = int(np.argmax(same))
    raise PedigreeValidationError(
        "same_parent_id",
        f"row {row} names id {int(mother_ids[row])} as both mother and father",
        row=row,
        child_id=int(ids[row]),
        parent_id=int(mother_ids[row]),
    )


def _map_ids_to_rows(target_ids, query_ids):
    order = np.argsort(target_ids, kind="stable")
    sorted_ids = target_ids[order]
    out = np.full(len(query_ids), -1, dtype=np.int32)
    if len(sorted_ids) == 0 or len(query_ids) == 0:
        return out
    sel = np.where(query_ids >= 0)[0]
    if sel.size == 0:
        return out
    q = query_ids[sel]
    pos = np.clip(np.searchsorted(sorted_ids, q), 0, len(sorted_ids) - 1)
    found = sorted_ids[pos] == q
    out[sel[found]] = order[pos[found]].astype(np.int32)
    return out


def _check_mz_pairs(ids, mother_ids, father_ids, twin_rows, sex):
    rows = np.flatnonzero(twin_rows >= 0)
    if rows.size == 0:
        return
    partner = twin_rows[rows].astype(np.int64)

    self_reference = partner == rows
    if self_reference.any():
        row = int(rows[int(np.argmax(self_reference))])
        raise PedigreeValidationError(
            "mz_self_reference", f"row {row} names itself as its MZ co-twin", row=row, id=int(ids[row])
        )

    nonreciprocal = twin_rows[partner] != rows
    if nonreciprocal.any():
        first = int(np.argmax(nonreciprocal))
        row, twin_row = int(rows[first]), int(partner[first])
        raise PedigreeValidationError(
            "mz_nonreciprocal",
            f"the MZ reference at row {row} is not reciprocated by row {twin_row}",
            row=row,
            id=int(ids[row]),
            twin_id=int(ids[twin_row]),
        )

    lower = rows < partner
    rows, partner = rows[lower], partner[lower]

    mismatched = {
        "mother": mother_ids[partner] != mother_ids[rows],
        "father": father_ids[partner] != father_ids[rows],
    }
    parent_mismatch = mismatched["mother"] | mismatched["father"]
    if parent_mismatch.any():
        first = int(np.argmax(parent_mismatch))
        row, twin_row = int(rows[first]), int(partner[first])
        raise PedigreeValidationError(
            "mz_parent_mismatch",
            f"MZ co-twins at rows {row} and {twin_row} do not name the same parents",
            row=row,
            id=int(ids[row]),
            twin_id=int(ids[twin_row]),
            parent_roles=tuple(role for role, flags in mismatched.items() if flags[first]),
        )

    if sex is None:
        return
    both_known = (sex[rows] != -1) & (sex[partner] != -1)
    sex_mismatch = both_known & (sex[rows] != sex[partner])
    if sex_mismatch.any():
        first = int(np.argmax(sex_mismatch))
        row, twin_row = int(rows[first]), int(partner[first])
        raise PedigreeValidationError(
            "mz_sex_mismatch",
            f"MZ co-twins at rows {row} and {twin_row} have different known sexes",
            row=row,
            id=int(ids[row]),
            twin_id=int(ids[twin_row]),
            sex=int(sex[row]),
            twin_sex=int(sex[twin_row]),
        )


def _normalize_optional(values, dtype):
    if values is None or values.size == 0 or bool(np.all(values == -1)):
        return None
    return values.astype(dtype)


def _check_birth_year_topology(ids, mother_rows, father_rows, birth_year):
    for parent_role, parent_arr in (("mother", mother_rows), ("father", father_rows)):
        edge_rows = np.where(parent_arr >= 0)[0]
        if edge_rows.size == 0:
            continue
        parents = parent_arr[edge_rows]
        by_child = birth_year[edge_rows]
        by_parent = birth_year[parents]
        both_known = (by_child >= 0) & (by_parent >= 0)
        if not both_known.any():
            continue
        edge_rows = edge_rows[both_known]
        diffs = (by_child[both_known] - by_parent[both_known]).astype(np.int32)
        violations = diffs < 0
        if not violations.any():
            continue
        first = int(np.argmax(violations))
        child_row = int(edge_rows[first])
        parent_row = int(parent_arr[child_row])
        raise PedigreeValidationError(
            "birth_year_topology",
            f"birth_year topology violation: {parent_role}-child edge at row {child_row} "
            f"has child.birth_year below {parent_role}.birth_year",
            parent_role=parent_role,
            child_row=child_row,
            parent_row=parent_row,
            child_id=int(ids[child_row]),
            parent_id=int(ids[parent_row]),
            child_birth_year=int(birth_year[child_row]),
            parent_birth_year=int(birth_year[parent_row]),
            violation_count=int(violations.sum()),
        )


def build(
    ids,
    mother,
    father,
    twin=None,
    sex=None,
    generation=None,
    birth_year=None,
    *,
    sex_encoding="simace",
    max_rows=_INT32_MAX,
) -> Built:
    """The 0.8.1 construction rules over int64 columns, same signature as the native builder."""
    if sex_encoding not in _SEX_ENCODINGS:
        raise ValueError(f"sex_encoding must be one of {sorted(_SEX_ENCODINGS)}, got {sex_encoding!r}")
    accepted, reported, mapping = _SEX_ENCODINGS[sex_encoding]

    n = len(ids)
    if n > max_rows:
        raise ResourceError(
            "pedigree_too_large",
            f"pedigree has {n:,} rows, exceeding the int32 row-coordinate capacity",
            n_individuals=n,
            maximum=max_rows,
        )
    optional = {"twin": twin, "sex": sex, "generation": generation, "birth_year": birth_year}
    values = {"id": ids, "mother": mother, "father": father} | {k: v for k, v in optional.items() if v is not None}

    for name, column in values.items():
        if name == "sex":
            _check_range("sex", column, accepted[0], accepted[1], reported)
            if mapping is not None:
                table = np.array([mapping[raw] for raw in range(accepted[0], accepted[1] + 1)], dtype=np.int64)
                values["sex"] = table[column - accepted[0]]
            continue
        minimum, maximum = _RANGES[name]
        _check_range(name, column, minimum, maximum, (minimum, maximum))

    _check_duplicate_ids(ids)
    _check_same_parent(ids, mother, father)

    twin_ids = values.get("twin")
    if twin_ids is None:
        twin_ids = np.full(n, -1, dtype=np.int64)
    mother_rows = _map_ids_to_rows(ids, mother)
    father_rows = _map_ids_to_rows(ids, father)
    twin_rows = _map_ids_to_rows(ids, twin_ids)
    rows_topological = topology.is_topological(mother_rows, father_rows)
    if not rows_topological:
        witness = topology.cycle_witness(ids, mother_rows, father_rows)
        if witness is not None:
            raise PedigreeValidationError("cycle", f"parent references form a cycle through ids {witness}", ids=witness)
    _check_mz_pairs(ids, mother, father, twin_rows, values.get("sex"))

    birth_year_stored = _normalize_optional(values.get("birth_year"), np.int32)
    if birth_year_stored is not None:
        _check_birth_year_topology(ids, mother_rows, father_rows, birth_year_stored)

    return Built(
        ids=ids.copy(),
        mother_ids=mother.copy(),
        father_ids=father.copy(),
        twin_ids=twin_ids.copy(),
        mother_rows=mother_rows,
        father_rows=father_rows,
        twin_rows=twin_rows,
        sex=_normalize_optional(values.get("sex"), np.int8),
        generation=_normalize_optional(values.get("generation"), np.int32),
        birth_year=birth_year_stored,
        rows_topological=rows_topological,
    )
