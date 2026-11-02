"""The quiet path of the sex-aware Ne estimators on a ``from_arrays`` graph.

With both sexes present neither one warns.
"""

import warnings

import numpy as np

from pedigree_graph import PedigreeGraph
from pedigree_graph.effective_size import ne_sex_ratio, ne_variance_family_size


def test_no_warning_when_both_sexes_are_present():
    pg = PedigreeGraph.from_arrays(
        ids=np.array([0, 1, 2, 3]),
        mother_ids=np.array([-1, -1, 0, 0]),
        father_ids=np.array([-1, -1, 1, 1]),
        sex=np.array([0, 1, 0, 1], dtype=np.int8),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        ne_sex_ratio(pg)
        ne_variance_family_size(pg)
