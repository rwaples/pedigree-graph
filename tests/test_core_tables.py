"""The Python tables agree with the Rust core's (issues #54, #60).

``_errors.py`` is hand-written, because Python raises many codes of its own;
these tests read the core's codes through the binding, so a core code missing
from its Python table, or with other fields, fails here rather than at raise
time.  ``_registry.RELATIONSHIPS`` is built from the core's category table at
import, so only what the build adds (labels, the record type) is checked here.

No core variant is a metadata error yet, so ``METADATA_CODES`` has nothing
to agree with until one is; its row in ``_TABLES`` checks it from then on.
"""

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


def test_the_registry_is_the_core_table_with_labels():
    rows = tuple(
        (c.code, c.degree, c.first_role, c.second_role, c.nominal_kinship, c.up, c.down, c.ancestor_count)
        for c in RELATIONSHIPS.values()
    )
    assert rows == tuple(_CORE_CATEGORIES)
    assert all(c.label for c in RELATIONSHIPS.values())


@pytest.mark.parametrize("code", list(RELATIONSHIPS))
def test_a_registry_record_survives_pickling(code):
    category = RELATIONSHIPS[code]
    assert pickle.loads(pickle.dumps(category)) == category
