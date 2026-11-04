"""The Python tables agree with the Rust core's (issues #54, #60).

``_errors.py`` is hand-written, because Python raises many codes of its own;
these tests read the core's codes through the binding, so a core code missing
from its Python table, or with other fields, fails here rather than at raise
time.  ``_registry.RELATIONSHIPS`` is built from the core's category table at
import, so only what the build adds (labels, the record type) is checked here.

No core variant is a metadata error yet, so ``METADATA_CODES`` has nothing
to agree with until one is; its row in ``_TABLES`` checks it from then on.
"""

import math
import pickle

import pytest

from pedigree_graph import _native
from pedigree_graph._errors import METADATA_CODES, RESOURCE_CODES, VALIDATION_CODES
from pedigree_graph._registry import RELATIONSHIPS

_TABLES = {
    "PedigreeValidationError": VALIDATION_CODES,
    "MissingMetadataError": METADATA_CODES,
    "ResourceError": RESOURCE_CODES,
}

_CORE_ERRORS = _native.error_codes()
_CORE_CATEGORIES = _native.relationship_categories()


def test_the_core_names_every_structured_code():
    # `error_codes` comes from one error of every core variant; the core's own
    # test holds that list complete.  An empty result would vacuously pass.
    assert {class_name for _, class_name, _ in _CORE_ERRORS} >= {"PedigreeValidationError", "ResourceError"}
    assert len({code for code, _, _ in _CORE_ERRORS}) == len(_CORE_ERRORS)


@pytest.mark.parametrize(("code", "class_name", "fields"), _CORE_ERRORS, ids=[c for c, _, _ in _CORE_ERRORS])
def test_every_core_code_is_in_its_python_table_with_the_same_fields(code, class_name, fields):
    table = _TABLES[class_name]
    assert code in table, f"{code!r} is raised by the core as {class_name} but missing from its Python table"
    assert table[code] == tuple(fields)


def test_the_registry_lists_the_core_categories_in_the_same_order():
    assert tuple(RELATIONSHIPS) == tuple(row[0] for row in _CORE_CATEGORIES)


@pytest.mark.parametrize("row", _CORE_CATEGORIES, ids=[row[0] for row in _CORE_CATEGORIES])
def test_every_registry_field_is_read_from_its_core_column(row):
    # Holds `_from_core`'s unpacking to the binding's column order.
    code, degree, first_role, second_role, nominal_kinship, up, down, ancestor_count = row
    category = RELATIONSHIPS[code]
    assert category.degree == degree
    assert (category.up, category.down, category.ancestor_count) == (up, down, ancestor_count)
    assert (category.first_role, category.second_role) == (first_role, second_role)
    assert math.isclose(category.nominal_kinship, nominal_kinship, rel_tol=1e-12)
    assert category.label


@pytest.mark.parametrize("code", list(RELATIONSHIPS))
def test_a_registry_record_survives_pickling(code):
    category = RELATIONSHIPS[code]
    assert pickle.loads(pickle.dumps(category)) == category
