# AGENTS.md

pedigree-graph: a pedigree graph library with a host-neutral Rust core (`crates/core`), a pyo3 Python package (`crates/python`, `pedigree_graph/`), and an extendr R package (`r/`). [README.md](README.md) covers usage and layout; [docs/architecture.md](docs/architecture.md) and `docs/adr/` the design; [tools/README.md](tools/README.md) the release and consumer gates.

- Everything runs through pixi; bare `cargo` and `R` are not on PATH.
- Before pushing, run `pixi run ci-local`: the CI push jobs' Rust fmt, clippy, docs and tests, ruff, ty and pytest, and the R binding lint and testthat suite. The publish workflow also builds macOS, Windows and ARM wheels, so a float assertion in a test needs a tolerance, never `==`.
- The R binding crate `r/src/rust` builds only in the `r` env: `pixi run -e r lint-r-rust` (and `doc-r-rust`, `test-r-rust`). A cargo run outside it caches extendr's failed build script; `cargo clean -p extendr-api -p extendr-ffi` clears that.
- `crates/core` forbids unsafe code and host bindings (ADR 0007, asserted in CI). Check a design against both before proposing it.
- `pedigree_graph.__version__` is the version the native core was built from. The editable install's distribution metadata stays stale after a version bump (`test_core_version_is_the_distribution_version` fails) until `pixi reinstall pedigree-graph`.
- simACE, fitACE, fitACE_epimight, pedsum and pg-phenotype import this package; `tools/consumer_gate.py` runs their suites against a build. Before deleting or renaming public API, grep each consumer tree by explicit path.
