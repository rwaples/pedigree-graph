# 0.12.3 consumer gate (2026-10-09)

Release gate for 0.12.3: R's `relatives_per_person()`; R relationship calls
that cancel cleanly on an early interrupt; `relationship_moments()` in R
refusing classed `values` columns and malformed side codes; the
`allocation_failed` dtype naming the element being reserved; the burden
width read from the core's `relationships::DEGREES`; and the core-only API
changes (`Receiver`, `Operand`, removed constructors) listed in the
CHANGELOG.

## Build stage

`pixi run --frozen python external/pedigree-graph/tools/consumer_gate.py run --stage 0.12.3 --wheel-ref 7271591`
from the umbrella root of the main simACE checkout (`build.json`, from the
rerun below):

| artifact | sha256 |
|---|---|
| `pedigree_graph-0.12.3-cp313-abi3-linux_x86_64.whl` | `d4924072790ae358ef5c570db95151f6976ef0a603fdc9ab5486b6c37bf7c779` |
| `pedigree_graph-0.12.3.tar.gz` | `87194dabe3d0da65a765078a3864a3a580b87c1cfd1bdae486e0b6b2bb80e9ed` |

Both installs pass the import check. The package's own fast suite against
the installed wheel: `3469 passed, 3 skipped, 16 deselected`, exit 0, in
both builds.

## Consumer units

The first run passed 12 of 14 units. Two failed on the gate's tooling, not
on 0.12.3:

- **pedsum `cli-smoke`**: pedsum `0b49755` reads sex in PLINK coding only
  (1=male, 2=female, 0=unknown), and the `tsv` step wrote simACE's 0=female,
  1=male, so `validate` refused 991 rows as `unknown_sex`. The `tsv` step now
  recodes sex to PLINK (`tools/consumer_gate.py`).
- **pg-phenotype `test-against-pg`**: cargo applies a `[patch]` only when its
  version matches the lock file's, so the 0.12.3 candidate was ignored
  against a lock pinning 0.12.2 and the build used the pin; the script's
  patch check failed the run. pg-phenotype `9feb65a` re-resolves both lock
  files after writing the patch.

A rerun of the build stage and those two units with both fixes
(`--wheel-ref 7271591 --unit pedsum pg-phenotype`) passed: pedsum `537
passed` and its CLI smoke run exit 0; pg-phenotype's cargo leg compiled
`pedigree-graph-core v0.12.3` from the clean worktree, and its Python suite
gave `281 passed`. The records for those two units are from the rerun, the
other twelve from the first run.

All 14 units ok, every routed unit resolving `pedigree_graph` under
`target/consumer-gate/0.12.3/wheel-site`. Consumers: simACE `dev` at
`37e6b20`, fitACE `6cbe0d3`, fitACE_epimight `6df2e09` (an untracked
`docs/ci/deck/` from another session), pedsum `1f49295` (a one-line
uncommitted change to `pedsum/cli.py` from another session, reading
`ped_depth` through `to_numpy()`), pg-phenotype `471f4df` (first run) and
`9feb65a` (rerun). pedigree-graph `3517 passed`, simACE `2969 passed, 2
skipped`, fitACE `393 passed`, fitACE_epimight `393 passed, 25 deselected`,
pedsum `537 passed`, fitACE_iter_reml `110 passed, 4 skipped`, fitACE_pcgc
`220 passed, 4 skipped`, fitACE_pafgrs `122 passed`, fitACE_tetraher `34
passed`, fitACE_frailty `7 passed`; ace_iter_reml's fp32 and fp64
harnesses, fitACE_stan's import and tetraher_simace's LDAK checks all pass.

No Python consumer needs a change: 0.12.3 changes no public Python
signature, so the consumers' `<0.13` pins take it at the relock. The tag
`v0.12.3` is on the bump commit `7271591`, the gated wheel's ref.
