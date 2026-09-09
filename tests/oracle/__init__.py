"""Readable pure-Python oracles for kernels that have migrated to the Rust core.

These are test code: they are never imported by ``pedigree_graph`` and are not
a fallback (ADR 0007).  Each one is the simplest correct statement of the
contract, kept slow and obvious so a differential test against the native
kernel means something.
"""
