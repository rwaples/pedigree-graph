# Pedigree Graph Code Quality Review

This review has been split into separate issue documents under [`docs/code-quality-review/`](code-quality-review/).

## Issues

- [PGQ-001 — Fix pair coordinate-space mismatch after `from_subsample`](pgq-001-fix-pair-coordinate-space-mismatch-after-from-subsample.md)
- [PGQ-002 — Replace loose constructor inputs and dense ID remapping with a validated input model](pgq-002-replace-loose-constructor-inputs-and-dense-id-remapping-with-a-validated-input-model.md)
- [PGQ-003 — Decompose `PedigreeGraph`; move relationship engines out of `_core.py`](pgq-003-decompose-pedigreegraph-move-relationship-engines-out-of-core-py.md)
- [PGQ-004 — Make relationship extraction semantics a single explicit plan instead of duplicated degree branches](pgq-004-make-relationship-extraction-semantics-a-single-explicit-plan-instead-of-duplicated-degree-branches.md)
- [PGQ-005 — Replace stringly typed effective-size payload dicts with explicit typed models](pgq-005-replace-stringly-typed-effective-size-payload-dicts-with-explicit-typed-models.md)
- [PGQ-006 — Split `_effective_size.py` into focused estimator modules](pgq-006-split-effective-size-py-into-focused-estimator-modules.md)
- [PGQ-007 — Simplify or fully support `_assemble_csc()`'s extra contract](pgq-007-simplify-or-fully-support-assemble-csc-s-extra-contract.md)
- [PGQ-008 — Decompose `_kinship_kernel.py` and shrink `_dp_kinship()`'s state machine](pgq-008-decompose-kinship-kernel-py-and-shrink-dp-kinship-s-state-machine.md)
- [PGQ-009 — Reassess the experimental BFS engine after core relationship semantics are centralized](pgq-009-reassess-the-experimental-bfs-engine-after-core-relationship-semantics-are-centralized.md)
- [PGQ-010 — Add architecture guardrails for large files and hidden coordinate/type contracts](pgq-010-add-architecture-guardrails-for-large-files-and-hidden-coordinate-type-contracts.md)
