"""The 0.9.1 numba kinship DP, kept verbatim as the differential oracle for the Rust kernel.

Test code only (ADR 0007): ``pedigree_graph`` never imports this and it is
not a fallback.  These modules are ``_kinship_dp``, ``_kinship_dp_depth``,
``_kinship_allocator``, ``_kinship_csc`` and the retirement-depth kernel of
``_kinship_depth`` as slice 14 found them, with only their imports rewritten,
so ``tests/test_native_kinship_matrix.py`` can hold
``_native.kinship_csc``, ``_native.approximate_kinship_csc`` and
``_native.generation_kinship_sums`` to the bytes the Python DP produced.
"""
