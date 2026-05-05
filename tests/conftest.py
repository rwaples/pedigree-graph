"""Shared fixtures for pedigree_graph tests."""

from pathlib import Path

import pandas as pd
import pytest

_DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture
def small_pedigree() -> pd.DataFrame:
    """Snapshot of simace.run_simulation(seed=42, N=1000, G_ped=3, G_sim=3, ...).

    Captured once and shipped as a parquet so tests don't need a runtime
    dependency on simace.  Byte-identical to the upstream fixture.
    """
    return pd.read_parquet(_DATA_DIR / "small_pedigree.parquet")
