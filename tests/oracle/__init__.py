"""Readable pure-Python oracles for kernels that have migrated to the Rust core.

These are test code: they are never imported by ``pedigree_graph`` and are not
a fallback (ADR 0007).  Each one is the simplest correct statement of the
contract, kept slow and obvious so a differential test against the native
kernel means something.

Some are instead the retired kernel itself, kept verbatim: ``kinship_dp``
(the 0.9.1 Numba DP), and from 0.9.4 ``inbreeding``, ``lineage`` and ``eqg``
(the last three Numba modules) and ``founder_means`` (the NumPy adjoint
sweep).  ``remap`` is the 0.9.3 host-side depth-major remap the Numba
oracles sweep in.  The Numba oracles need the ``test`` extra, which carries
``numba`` since it left the runtime dependencies.
"""
