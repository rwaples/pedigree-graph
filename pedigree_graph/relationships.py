"""Public relationship vocabulary: the category records and their registry.

:data:`RELATIONSHIPS` maps each of the 23 relationship codes to an immutable
:class:`RelationshipCategory`.  Iteration order is the documented same-degree
precedence for closest-category classification.  See
:class:`RelationshipCategory` for the ``first`` / ``second`` orientation rule
and :data:`RelationshipRole` for the closed set of role names.
"""

from __future__ import annotations

from pedigree_graph._registry import RELATIONSHIPS, RelationshipCategory, RelationshipRole

__all__ = [
    "RELATIONSHIPS",
    "RelationshipCategory",
    "RelationshipRole",
]
