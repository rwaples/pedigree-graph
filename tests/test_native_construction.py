"""Differential and boundary tests for native construction.

``_native.build_pedigree`` must agree with the 0.8.1 Python rules kept in
``tests.oracle.construction`` on every input: the same columns out, or the same
structured error with the same fields and message.  Two generators feed it: a
structured one that builds a valid pedigree and plants at most one defect, so
every code is reached (``test_input.py`` pins each one deterministically),
and a chaos one over small integers, so the precedence between coexisting
defects is exercised.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pedigree_graph import PedigreeValidationError, ResourceError, _native
from pedigree_graph._input import host_columns
from tests.oracle import construction as oracle

_INT32_MAX = int(np.iinfo(np.int32).max)
_DEFECTS = (
    None,
    "id_out_of_range",
    "parent_out_of_range",
    "sex_out_of_range",
    "label_out_of_range",
    "duplicate_id",
    "same_parent",
    "cycle",
    "mz_self",
    "mz_nonreciprocal",
    "mz_parent_mismatch",
    "mz_sex_mismatch",
    "birth_year",
)


def _column(values):
    return np.ascontiguousarray(np.asarray(values, dtype=np.int64))


@st.composite
def structured_input(draw):
    """A valid pedigree in a random row order, then at most one planted defect."""
    n = draw(st.integers(min_value=0, max_value=24))
    encoding = draw(st.sampled_from(["simace", "plink"]))
    # Rows 0..n-1 in a topological order first; parents come from earlier rows
    # or are external (an id no row carries) or missing.
    mother = [-1] * n
    father = [-1] * n
    for i in range(1, n):
        for parents in (mother, father):
            kind = draw(st.sampled_from(["missing", "row", "external"]))
            if kind == "row":
                parents[i] = draw(st.integers(min_value=0, max_value=i - 1))
            elif kind == "external":
                parents[i] = -2 - draw(st.integers(min_value=0, max_value=3))
    # Reciprocal MZ pairs among rows sharing both parents, formed greedily.
    twin = [-1] * n
    if draw(st.booleans()):
        for i in range(n):
            if twin[i] != -1:
                continue
            for j in range(i + 1, n):
                if twin[j] == -1 and mother[j] == mother[i] and father[j] == father[i] and draw(st.booleans()):
                    twin[i], twin[j] = j, i
                    break
    perm = draw(st.permutations(range(n)))
    inverse = [0] * n
    for position, row in enumerate(perm):
        inverse[row] = position
    ids = [row * 7 + 3 for row in range(n)]
    ids = [ids[perm[k]] for k in range(n)]

    def to_id(ref, row_ids=ids):
        if ref == -1:
            return -1
        if ref < -1:
            return 10_000 + (-ref)
        return row_ids[inverse[ref]]

    mother_ids = [to_id(mother[perm[k]]) for k in range(n)]
    father_ids = [to_id(father[perm[k]]) for k in range(n)]
    twin_ids = [to_id(twin[perm[k]]) for k in range(n)]

    columns = {"ids": ids, "mother": mother_ids, "father": father_ids}
    if draw(st.booleans()):
        columns["twin"] = twin_ids
    if draw(st.booleans()):
        raw = draw(
            st.lists(st.integers(min_value=-1, max_value=2 if encoding == "plink" else 1), min_size=n, max_size=n)
        )
        # Co-twins share a sex where both are known.
        for k in range(n):
            partner = twin[perm[k]]
            if partner != -1 and raw[k] != -1:
                raw[inverse[partner]] = raw[k]
        columns["sex"] = raw
    if draw(st.booleans()):
        columns["generation"] = draw(st.lists(st.integers(min_value=-1, max_value=6), min_size=n, max_size=n))
    if draw(st.booleans()):
        depth = [0] * n
        for i in range(n):
            depth[i] = max((depth[p] + 1 for p in (mother[i], father[i]) if p >= 0), default=0)
        years = [-1 if draw(st.booleans()) else 1900 + 25 * depth[perm[k]] + draw(st.integers(0, 20)) for k in range(n)]
        columns["birth_year"] = years

    defect = draw(st.sampled_from(_DEFECTS))
    if n == 0:
        defect = None
    if defect is not None:
        k = draw(st.integers(min_value=0, max_value=n - 1))
    if defect == "id_out_of_range":
        columns["ids"][k] = -1
    elif defect == "parent_out_of_range":
        columns[draw(st.sampled_from(["mother", "father"]))][k] = -2
    elif defect == "sex_out_of_range":
        columns["sex"] = columns.get("sex", [-1] * n)
        columns["sex"][k] = draw(st.sampled_from([-2, 3, 2 if encoding == "simace" else 3]))
    elif defect == "label_out_of_range":
        field = draw(st.sampled_from(["generation", "birth_year"]))
        columns[field] = columns.get(field, [-1] * n)
        columns[field][k] = draw(st.sampled_from([-2, _INT32_MAX + 1]))
    elif defect == "duplicate_id" and n >= 2:
        columns["ids"][k] = columns["ids"][(k + 1) % n]
    elif defect == "same_parent":
        columns["father"][k] = columns["mother"][k] if columns["mother"][k] != -1 else 10_007
        columns["mother"][k] = columns["father"][k]
    elif defect == "cycle" and n >= 2:
        j = draw(st.integers(min_value=0, max_value=n - 1).filter(lambda x: x != k))
        columns["mother"][k] = columns["ids"][j]
        columns["father"][j] = columns["ids"][k]
    elif defect == "mz_self":
        columns["twin"] = columns.get("twin", [-1] * n)
        columns["twin"][k] = columns["ids"][k]
    elif defect == "mz_nonreciprocal" and n >= 2:
        columns["twin"] = columns.get("twin", [-1] * n)
        columns["twin"][k] = columns["ids"][(k + 1) % n]
    elif defect == "mz_parent_mismatch":
        columns["twin"] = columns.get("twin", [-1] * n)
        columns["mother"][k] = 10_009
    elif defect == "mz_sex_mismatch":
        columns["sex"] = columns.get("sex", [-1] * n)
        partner = twin[perm[k]]
        columns["sex"][k] = 1
        if partner != -1:
            columns["sex"][inverse[partner]] = 2 if encoding == "plink" else 0
    elif defect == "birth_year":
        columns["birth_year"] = columns.get("birth_year", [-1] * n)
        columns["birth_year"][k] = 1800
        parent = mother[perm[k]] if mother[perm[k]] >= 0 else father[perm[k]]
        if parent >= 0:
            columns["birth_year"][inverse[parent]] = 1900
    return columns, encoding


@st.composite
def chaos_input(draw):
    """Small random integers in every column, so defects coexist and precedence shows."""
    n = draw(st.integers(min_value=0, max_value=8))
    small = st.integers(min_value=-2, max_value=9)
    columns = {
        "ids": draw(st.lists(st.integers(min_value=-1, max_value=9), min_size=n, max_size=n)),
        "mother": draw(st.lists(small, min_size=n, max_size=n)),
        "father": draw(st.lists(small, min_size=n, max_size=n)),
    }
    for field, bound in (("twin", 9), ("sex", 3), ("generation", 4), ("birth_year", 4)):
        if draw(st.booleans()):
            columns[field] = draw(st.lists(st.integers(min_value=-2, max_value=bound), min_size=n, max_size=n))
    return columns, draw(st.sampled_from(["simace", "plink", "foo"]))


def _outcome(build, columns, encoding):
    args = [_column(columns[name]) for name in ("ids", "mother", "father")]
    args += [
        _column(columns[name]) if name in columns else None for name in ("twin", "sex", "generation", "birth_year")
    ]
    try:
        built = build(*args, sex_encoding=encoding)
    except (PedigreeValidationError, ResourceError) as err:
        return (type(err).__name__, err.code, dict(err.fields), str(err))
    except ValueError as err:
        return ("ValueError", str(err))
    columns_out = {}
    for name in ("ids", "mother_ids", "father_ids", "twin_ids", "mother_rows", "father_rows", "twin_rows"):
        array = getattr(built, name)
        columns_out[name] = (str(array.dtype), array.tolist())
    for name in ("sex", "generation", "birth_year"):
        array = getattr(built, name)
        columns_out[name] = None if array is None else (str(array.dtype), array.tolist())
    columns_out["rows_topological"] = built.rows_topological
    return columns_out


@settings(max_examples=400, deadline=None)
@given(structured_input())
def test_structured_inputs_match_the_oracle(case):
    columns, encoding = case
    assert _outcome(_native.build_pedigree, columns, encoding) == _outcome(oracle.build, columns, encoding)


@settings(max_examples=400, deadline=None)
@given(chaos_input())
def test_chaos_inputs_match_the_oracle(case):
    columns, encoding = case
    assert _outcome(_native.build_pedigree, columns, encoding) == _outcome(oracle.build, columns, encoding)


class TestBoundary:
    def test_lengths_are_checked_natively_too(self):
        with pytest.raises(PedigreeValidationError) as info:
            _native.build_pedigree(_column([0, 1]), _column([-1]), _column([-1, -1]), sex_encoding="simace")
        assert info.value.code == "length_mismatch"
        assert dict(info.value.fields) == {"field": "mother", "expected_length": 2, "actual_length": 1}

    def test_the_row_capacity_is_a_resource_error(self):
        columns = host_columns({"id": [0, 1, 2], "mother": [-1, -1, -1], "father": [-1, -1, -1]})
        with pytest.raises(ResourceError) as info:
            _native.build_pedigree(*columns.as_args(), sex_encoding="simace", max_rows=2)
        assert info.value.code == "pedigree_too_large"
        assert str(info.value) == "pedigree has 3 rows, exceeding the int32 row-coordinate capacity"

    def test_an_unknown_encoding_is_a_plain_value_error(self):
        columns = host_columns({"id": [0], "mother": [-1], "father": [-1]})
        with pytest.raises(ValueError, match="sex_encoding must be one of \\['plink', 'simace'\\], got 'foo'") as info:
            _native.build_pedigree(*columns.as_args(), sex_encoding="foo")
        assert not isinstance(info.value, PedigreeValidationError)

    @pytest.mark.parametrize("kernel", [_native.is_topological, _native.structural_depth])
    def test_parent_rows_outside_the_pedigree_are_rejected_not_indexed(self, kernel):
        rows = np.array([-1, 5], dtype=np.int32)
        with pytest.raises(ValueError, match="mother_rows\\[1\\] = 5"):
            kernel(rows, np.array([-1, -1], dtype=np.int32))
        with pytest.raises(ValueError, match="father_rows\\[0\\] = -2"):
            kernel(np.array([-1, -1], dtype=np.int32), np.array([-2, -1], dtype=np.int32))

    def test_built_arrays_are_owned_and_the_same_object_on_every_access(self):
        columns = host_columns({"id": [0, 1], "mother": [-1, 0], "father": [-1, -1]})
        built = _native.build_pedigree(*columns.as_args(), sex_encoding="simace")
        assert built.ids is built.ids
        assert built.mother_rows.tolist() == [-1, 0]
        columns.ids[0] = 9
        assert built.ids.tolist() == [0, 1]


class TestIdIndex:
    @settings(max_examples=100, deadline=None)
    @given(
        ids=st.lists(st.integers(min_value=0, max_value=50), unique=True),
        query=st.lists(st.integers(min_value=-3, max_value=60)),
    )
    def test_resolve_matches_the_oracle(self, ids, query):
        index = _native.IdIndex(_column(ids))
        got = index.resolve(_column(query))
        assert got.dtype == np.int32
        assert got.tolist() == oracle._map_ids_to_rows(_column(ids), _column(query)).tolist()
