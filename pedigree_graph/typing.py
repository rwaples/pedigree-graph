"""Public typing protocols.

:class:`FrameLike` is the structural protocol any column table satisfies:
pandas and polars frames both do, and neither library is imported here.
"""

from __future__ import annotations

from pedigree_graph._frames import FrameLike

__all__ = ["FrameLike"]
