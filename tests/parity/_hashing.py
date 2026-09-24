"""Hash layout and constants the structural golden shares with the retired 0.7.1 baseline.

Copied from ``generate_baseline.py`` (removed with the 0.7.1 baseline) so the
digests in ``tests/data/structure_v0.10`` stay comparable, byte for byte, with
the ``v0_7_1_sha256`` each carried key records.
"""

from __future__ import annotations

import hashlib

import numpy as np

APPROX_THRESHOLD = 0.001
SUBSAMPLE_SEED = 7


def _sha(*arrays: np.ndarray) -> str:
    h = hashlib.sha256()
    for arr in arrays:
        c = np.ascontiguousarray(arr)
        h.update(str(c.dtype).encode())
        h.update(str(c.shape).encode())
        h.update(c.tobytes())
    return h.hexdigest()


def _sorted_pairs(i: np.ndarray, j: np.ndarray, *cols: np.ndarray):
    order = np.lexsort((j, i))
    return (i[order], j[order], *[c[order] for c in cols])


def _upper_coo(K):
    coo = K.tocoo()
    keep = coo.row <= coo.col
    r, c, v = coo.row[keep].astype(np.int32), coo.col[keep].astype(np.int32), coo.data[keep].astype(np.float32)
    return _sorted_pairs(r, c, v)
