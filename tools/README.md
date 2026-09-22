# Consumer release gate

These scripts prove a pedigree-graph build against its consumers (simACE,
fitACE and its method packages, fitACE_epimight, pedsum). They run from a
simACE umbrella checkout with this repo at `external/pedigree-graph/`, and
borrow the family membership list from the umbrella's `tools/family_repos.py`;
outside that layout they exit at the first path assertion.

- `pg08_release_gate.py` — one check unit per family member, routed to a
  pedigree-graph build by `--routing`; writes one JSON record per unit under
  `docs/pedigree-graph-0.8-migration/gate/<stage>/`.
  `tests/test_consumer_gate_covers_family.py` holds it to the family list.
- `pg08_wheel_gate.sh` — clean-checkout build, isolated-env install, the
  package's own suite against the installed wheel, and a `--target` site the
  consumer gate can route to.
- `pg09_byte_parity.sh` — rebuilds the smoke scenario and hashes the two
  consumer artifacts that carry the pair contract furthest.
- `pg08_migration_diff.py` — snapshot and compare consumer-visible outputs
  across two pedigree-graph versions (the 0.7.1 to 0.8.0 record lives beside
  the gate evidence).
- `pg08_write_pedigree_100k.py` — times fitACE's sparse-GRM write at scale.

The evidence directory keeps the per-unit JSON records, slice notes, and
benchmark scripts; raw command logs were dropped in the move from simACE
(2026-09-22) and remain in simACE's git history.
