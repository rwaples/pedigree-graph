"""The close-relative counter returns results; its public wrapper owns the cache (ADR 0002)."""

from pedigree_graph import PedigreeGraph
from pedigree_graph._streaming_counter import _count_close_relatives


class TestEngineReadOnlyContract:
    """Engines compute results but never persist them (ADR 0002)."""

    def test_scalar_counter_does_not_write_the_cache(self, small_pedigree):
        pg = PedigreeGraph.from_frame(small_pedigree)
        counts = _count_close_relatives(pg)
        assert pg._close_relative_counts_cache is None
        assert isinstance(counts, dict)
        assert counts["MZ"] >= 0

        result = pg.close_relative_counts()
        assert pg._close_relative_counts_cache is result
        assert set(counts) == result.requested
        for code in result.requested:
            assert result[code] == counts[code], code
