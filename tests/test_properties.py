"""The ADR 0006 read-only property surface.

Pins slice 1c: every array property hands back the graph's own storage with a fixed dtype,
read-only and identical on each access; ``depth`` is derived from the parent edges alone and
only when first asked for; the optional columns read as ``None`` when absent.
"""

import numpy as np
import pytest

from pedigree_graph import PedigreeGraph, _properties

ARRAY_PROPERTIES = [
    ("ids", np.int64),
    ("mother_ids", np.int64),
    ("father_ids", np.int64),
    ("twin_ids", np.int64),
    ("mother_rows", np.int32),
    ("father_rows", np.int32),
    ("twin_rows", np.int32),
    ("sex", np.int8),
    ("depth", np.int32),
    ("generation_labels", np.int32),
    ("birth_year", np.int32),
]
ARRAY_PROPERTY_NAMES = [name for name, _ in ARRAY_PROPERTIES]


def _full_data(**overrides):
    """Two founders and their MZ twin children, every optional column supplied."""
    base = {
        "id": np.array([10, 11, 12, 13], dtype=np.int64),
        "mother": np.array([-1, -1, 10, 10], dtype=np.int64),
        "father": np.array([-1, -1, 11, 11], dtype=np.int64),
        "twin": np.array([-1, -1, 13, 12], dtype=np.int64),
        "sex": np.array([0, 1, 1, 1], dtype=np.int8),
        "generation": np.array([0, 0, 1, 1], dtype=np.int32),
        "birth_year": np.array([1980, 1980, 2010, 2010], dtype=np.int32),
    }
    base.update(overrides)
    return base


def _minimal_data(**overrides):
    base = {
        "id": np.array([10, 11, 12], dtype=np.int64),
        "mother": np.array([-1, -1, 10], dtype=np.int64),
        "father": np.array([-1, -1, 11], dtype=np.int64),
    }
    base.update(overrides)
    return base


@pytest.fixture
def graph():
    return PedigreeGraph.from_frame(_full_data())


class TestArrayProperties:
    @pytest.mark.parametrize(("name", "dtype"), ARRAY_PROPERTIES)
    def test_dtype(self, graph, name, dtype):
        assert getattr(graph, name).dtype == dtype

    @pytest.mark.parametrize("name", ARRAY_PROPERTY_NAMES)
    def test_shape_is_one_entry_per_row(self, graph, name):
        assert getattr(graph, name).shape == (graph.n_individuals,)

    @pytest.mark.parametrize("name", ARRAY_PROPERTY_NAMES)
    def test_is_not_writeable(self, graph, name):
        assert getattr(graph, name).flags.writeable is False

    @pytest.mark.parametrize("name", ARRAY_PROPERTY_NAMES)
    def test_element_assignment_is_rejected(self, graph, name):
        values = getattr(graph, name)
        with pytest.raises(ValueError, match="read-only"):
            values[0] = 0

    @pytest.mark.parametrize("name", ARRAY_PROPERTY_NAMES)
    def test_every_access_returns_the_same_object(self, graph, name):
        assert getattr(graph, name) is getattr(graph, name)

    def test_stored_values(self, graph):
        assert graph.ids.tolist() == [10, 11, 12, 13]
        assert graph.mother_ids.tolist() == [-1, -1, 10, 10]
        assert graph.father_ids.tolist() == [-1, -1, 11, 11]
        assert graph.twin_ids.tolist() == [-1, -1, 13, 12]
        assert graph.mother_rows.tolist() == [-1, -1, 0, 0]
        assert graph.father_rows.tolist() == [-1, -1, 1, 1]
        assert graph.twin_rows.tolist() == [-1, -1, 3, 2]


class TestOptionalProperties:
    @pytest.mark.parametrize("name", ["sex", "generation_labels", "birth_year"])
    def test_absent_column_reads_as_none(self, name):
        pg = PedigreeGraph.from_frame(_minimal_data())
        assert getattr(pg, name) is None

    def test_depth_is_present_without_any_optional_column(self):
        pg = PedigreeGraph.from_frame(_minimal_data())
        assert pg.depth.dtype == np.int32
        assert pg.depth.tolist() == [0, 0, 1]


class TestDepth:
    def test_construction_does_not_compute_depth(self, monkeypatch):
        calls = 0
        real = _properties.structural_depth

        def counting(*args):
            nonlocal calls
            calls += 1
            return real(*args)

        monkeypatch.setattr(_properties, "structural_depth", counting)
        pg = PedigreeGraph.from_frame(_full_data())
        assert calls == 0
        assert pg.depth.tolist() == [0, 0, 1, 1]
        assert calls == 1
        assert pg.depth.tolist() == [0, 0, 1, 1]
        assert calls == 1

    def test_supplied_generation_labels_do_not_move_depth(self):
        pg = PedigreeGraph.from_frame(_full_data(generation=np.array([7, 7, 3, 3], dtype=np.int32)))
        assert pg.depth.tolist() == [0, 0, 1, 1]
        assert pg.generation_labels.tolist() == [7, 7, 3, 3]


class TestSize:
    def test_len_matches_n_individuals_and_ids(self, graph):
        assert len(graph) == 4
        assert graph.n_individuals == 4
        assert len(graph.ids) == 4

    def test_n_individuals_is_a_python_int(self, graph):
        assert isinstance(graph.n_individuals, int)

    def test_empty_pedigree_is_empty(self):
        pg = PedigreeGraph.from_frame({"id": [], "mother": [], "father": []})
        assert len(pg) == 0
        assert pg.n_individuals == 0
        assert pg.ids.shape == (0,)


class TestExternalReferences:
    def test_external_parent_keeps_its_id_and_loses_its_row(self):
        pg = PedigreeGraph.from_frame(_minimal_data(mother=np.array([-1, -1, 900], dtype=np.int64)))
        assert pg.mother_ids.tolist() == [-1, -1, 900]
        assert pg.mother_rows.tolist() == [-1, -1, -1]

    def test_external_co_twin_keeps_its_id_and_loses_its_row(self):
        pg = PedigreeGraph.from_frame(_minimal_data(twin=np.array([-1, -1, 900], dtype=np.int64)))
        assert pg.twin_ids.tolist() == [-1, -1, 900]
        assert pg.twin_rows.tolist() == [-1, -1, -1]
