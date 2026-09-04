"""Structural frame-protocol input coverage (FrameLike).

Every constructor accepts dict[str, np.ndarray], pandas, and polars inputs
and produces identical graphs. Pandas coverage here is the focused
compatibility surface (including nullable integer dtypes); polars is the
family's primary frame library; the dict path is the zero-dependency escape
hatch.
"""

import numpy as np
import pandas as pd
import polars as pl
import pytest

from pedigree_graph import FrameLike, PedigreeGraph, PedigreeValidationError

# Two founder couples, two children each, one grandchild generation.
_DATA = {
    "id": np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int64),
    "mother": np.array([-1, -1, -1, -1, 0, 0, 2, 4], dtype=np.int64),
    "father": np.array([-1, -1, -1, -1, 1, 1, 3, 6], dtype=np.int64),
    "twin": np.full(8, -1, dtype=np.int64),
    "sex": np.array([0, 1, 0, 1, 0, 1, 1, 0], dtype=np.int64),
    "generation": np.array([0, 0, 0, 0, 1, 1, 1, 2], dtype=np.int64),
}


def _expected_pairs():
    return PedigreeGraph(dict(_DATA)).extract_pairs(max_degree=2)


def _assert_same_graph(pg: PedigreeGraph) -> None:
    expected = _expected_pairs()
    got = pg.extract_pairs(max_degree=2)
    assert set(got) == set(expected)
    for rel, (i1, i2) in expected.items():
        np.testing.assert_array_equal(got[rel][0], i1)
        np.testing.assert_array_equal(got[rel][1], i2)


class TestEntryPoints:
    @pytest.mark.parametrize("convert", [pd.DataFrame, pl.DataFrame], ids=["pandas", "polars"])
    def test_init_matches_dict_path(self, convert):
        _assert_same_graph(PedigreeGraph(convert(_DATA)))

    @pytest.mark.parametrize("convert", [pd.DataFrame, pl.DataFrame], ids=["pandas", "polars"])
    def test_from_dataframe_matches_dict_path(self, convert):
        _assert_same_graph(PedigreeGraph.from_dataframe(convert(_DATA)))

    @pytest.mark.parametrize("convert", [dict, pd.DataFrame, pl.DataFrame], ids=["dict", "pandas", "polars"])
    def test_from_subsample_matches_full_graph(self, convert):
        sub = {k: v[4:] for k, v in _DATA.items()}
        pg = PedigreeGraph.from_subsample(convert(_DATA), convert(sub))
        pairs = pg.extract_pairs(max_degree=2)
        ref = PedigreeGraph.from_subsample(dict(_DATA), {k: v[4:] for k, v in _DATA.items()})
        ref_pairs = ref.extract_pairs(max_degree=2)
        assert set(pairs) == set(ref_pairs)
        for rel in ref_pairs:
            np.testing.assert_array_equal(pairs[rel][0], ref_pairs[rel][0])
            np.testing.assert_array_equal(pairs[rel][1], ref_pairs[rel][1])


class TestOptionalAndDtypes:
    def test_birth_year_extracted_from_polars(self):
        df = pl.DataFrame({**_DATA, "birth_year": np.array([0, 0, 0, 0, 25, 27, 26, 55], dtype=np.int64)})
        pg = PedigreeGraph(df)
        assert pg.birth_year is not None
        np.testing.assert_array_equal(pg.birth_year, df["birth_year"].to_numpy())

    def test_pandas_nullable_integer_columns_accepted(self):
        df = pd.DataFrame({k: pd.array(v, dtype="Int64") for k, v in _DATA.items()})
        _assert_same_graph(PedigreeGraph(df))

    def test_polars_int32_columns_accepted(self):
        df = pl.DataFrame(_DATA).with_columns(pl.col(c).cast(pl.Int32) for c in _DATA)
        _assert_same_graph(PedigreeGraph(df))

    def test_missing_required_column_reports_uniformly(self):
        for frame in (pd.DataFrame(_DATA).drop(columns=["mother"]), pl.DataFrame(_DATA).drop("mother")):
            with pytest.raises(PedigreeValidationError) as info:
                PedigreeGraph(frame)
            assert info.value.code == "missing_field"
            assert info.value.fields["field"] == "mother"

    def test_optional_columns_may_be_absent(self):
        for frame in (pd.DataFrame(_DATA).drop(columns=["twin"]), pl.DataFrame(_DATA).drop("twin")):
            _assert_same_graph(PedigreeGraph(frame))


def test_framelike_is_exported_protocol():
    assert isinstance(FrameLike, type)
    # Documented contract: structural, no runtime frame-library dependency.
    assert "columns" in FrameLike.__protocol_attrs__ if hasattr(FrameLike, "__protocol_attrs__") else True
