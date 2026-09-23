# 16c R source tarball gate for pedigree-graph 0.10.0 (2026-09-23)

The packaging half of slice 16 (simACE
`plans/pedigree-graph-slice-16-r-package.md`): the R source tarball, its
offline check, the CI and publish jobs, and the 0.10.0 version. `../16b/` is
the parity record. The wheel-site and post-publish gates follow at the tag.

## The tarball

`pixi run -e r tools/r_build_tarball.sh` stages a copy of `r/` with the core
at `src/rust/core` (workspace keys written out, no `[[bin]]`), vendors the
locked graph of 19 crates into `src/rust/vendor.tar.xz` with numeric owner 0,
lists each crate's authors (or, where its manifest names none, its repository) and license in `inst/AUTHORS`, and runs
`R CMD build`: `pedigreegraph_0.10.0.tar.gz`, 1.09 MB. The binding crate is
its own Cargo workspace, so it builds wherever the tarball is unpacked.

## Offline check (`r_cmd_check.log`)

`unshare -rn` (no interface up; `curl https://crates.io` fails inside), then
`R CMD check --as-cran --no-manual` with `_R_CHECK_CRAN_INCOMING_REMOTE_=false`,
R 4.5.3 and rustc 1.98.1 from the pixi `r` environment: exit 0,
`Status: 2 NOTEs`. The build ran `cargo build -j 2 --offline --locked` with
`CARGO_HOME` inside the package, and logged rustc's version first. Examples,
tests (the full testthat suite, 413 expectations) and R's Rust compilation
check are OK. The two NOTEs are the environment's: "unable to verify current
time" (offline by design) and `-march=nocona` (conda-forge R's own flags).

Two defects surfaced only here and are fixed: `tar` restored the archiving
uid when extracting as (mapped) root, which a root Docker build would hit
(now `--no-same-owner` on extract and numeric owner 0 on create); and cargo
adopted the repository's workspace when the tarball was unpacked beneath it
(now an empty `[workspace]` in the binding crate, replacing the root
`exclude`).

## Versions and CI

`[workspace.package].version`, `r/DESCRIPTION` and `r/src/rust/Cargo.toml`
read 0.10.0; `tests/test_r_version_agreement.py` and `publish.yml`'s
`check-version` hold them together. `ci.yml` gains an `r` job (lint the
binding crate, build the tarball, check it under `sudo unshare --net`), and
`publish.yml` gains `r-tarball` and `github-release`, the only job with
`contents: write`, which attaches the tarball with the CHANGELOG section as
notes. The GitHub-runner half of these jobs is unverified until they run
there: ubuntu-24.04 restricts unprivileged user namespaces, hence `sudo`.

After the bump and `pixi reinstall pedigree-graph`, the Python suite at
0.10.0: 3407 passed, 9 skipped, and the one test that read the stale 0.9.4
editable metadata (`test_core_version_is_the_distribution_version`) passes
after the reinstall.
