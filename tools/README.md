# Tools

Release and maintenance scripts. The gate tools prove a pedigree-graph build
against its consumers (simACE, fitACE and its method packages,
fitACE_epimight, pedsum). They run from a simACE umbrella checkout with this
repo at `external/pedigree-graph/` and borrow the family membership list from
the umbrella's `tools/family_repos.py`; outside that layout they exit at the
first path assertion.

## Release gate

- `consumer_gate.py` — one check unit per family member, routed to a
  pedigree-graph build. `run --wheel-ref <ref>` first builds the wheel and
  sdist from a clean worktree at the ref, checks both install into isolated
  venvs, runs the package's fast suite against the installed wheel, and stages
  a `--target` site the consumer units import from; `run --routing locked`
  checks the relocked consumers after publishing. Records go to
  `docs/release-gates/<stage>/`. `tests/test_consumer_gate_covers_family.py`
  holds its units to the family list.
- `byte_parity.sh` — rebuilds the smoke scenario and hashes the consumer
  products that carry pedigree-graph's results (report, pair list, sparse GRM,
  inbreeding, effective sizes, generation kinship summary). Run it before and
  after a relock and diff the manifests.
- `compare_floats.py` — where two `byte_parity.sh` manifests differ, parses
  both sides and reports the largest float difference against
  `rtol 1e-9, atol 1e-12`.

## R package

- `r_build_tarball.sh` — builds the R source tarball, staging the core and
  vendoring every dependency so it compiles offline. Refuses to run with
  `PG_CARGO_FEATURES` set: the tarball never carries the test hooks.
- `r_golden.py` — writes the R package's golden files from the Python
  package; `tests/test_r_goldens.py` checks they are current.
- `r_parity.py` with `r_parity.R` — runs the larger fixtures and the simACE
  study pedigrees through both hosts and compares every product byte for byte.

## Profiling

- `pg_tskit_profile.py` — fresh-process time and RSS for compact views and
  relationship-burden output.

Retired migration tools (`pg08_*`, `pg13/14/15_study_*`) are listed in
`docs/pedigree-graph-0.8-migration/README.md` with the commit they were last
present at.
