# 0.12.4 consumer gate (2026-10-16)

Release gate for 0.12.4: construction through a dense-id lookup table
(#43); R error messages that count rows and positions from 1 (#44); Python's
non-finite moments wording; and the core's `Error::fields` and
`Error::message`, which now take an `IndexBase`.

## Build stage

`pixi run --frozen python external/pedigree-graph/tools/consumer_gate.py run --stage 0.12.4 --wheel-ref 5be610f`
from the umbrella root of the main simACE checkout (`build.json`):

| artifact | sha256 |
|---|---|
| `pedigree_graph-0.12.4-cp313-abi3-linux_x86_64.whl` | `0eeba720473060587b73c019dfd97ad7d49d0fbeb9c32ec95d0ba7e8e7061f5f` |
| `pedigree_graph-0.12.4.tar.gz` | `c249a5c93c439e392bb5c11fca2c0c6963e31b3b990d2178222dbf2fb2601422` |

Both installs pass the import check. The package's own fast suite against
the installed wheel: `3469 passed, 3 skipped, 16 deselected`, exit 0.

## Consumer units

13 of 14 units ok, every routed unit resolving `pedigree_graph` under
`target/consumer-gate/0.12.4/wheel-site`. Consumers: simACE `dev` at
`1c81f16`, fitACE `781667d`, fitACE_epimight `cf497ce` (an untracked
`docs/ci/deck/` from another session), pedsum `33405c3`, pg-phenotype
`4090a70`. pedigree-graph `3517 passed`, simACE `3156 passed, 2 skipped`,
fitACE `404 passed`, fitACE_epimight `393 passed, 25 deselected`, pedsum
`708 passed` and its CLI smoke run exit 0, fitACE_iter_reml `110 passed, 4
skipped`, fitACE_pcgc `220 passed, 4 skipped`, fitACE_pafgrs `47 passed`,
fitACE_tetraher `34 passed`, fitACE_frailty `7 passed`; ace_iter_reml's
fp32 and fp64 harnesses, fitACE_stan's import and tetraher_simace's LDAK
checks all pass.

**pg-phenotype `test-against-pg` failed (exit 101), as the core API change
predicts.** `pg-phenotype-core` calls `PgError::fields()` with no argument
(`crates/core/src/error.rs:446`) and shifts index fields by name itself;
0.12.4's core shifts them and takes the base. pg-phenotype pins the core by
git rev (v0.12.3, `7271591`), so 0.12.4 breaks nothing until pg-phenotype
moves its pin.

The adaptation is `pg-phenotype/adaptation.diff`, on `4090a70`: pass the
base through to `fields` and `message` (converting pg-phenotype's
`IndexBase` to the core's), build the test samples' rows as `Position`s, and
hold pedigree errors' messages to the host base like the rest. Against the
0.12.4 core through a cargo `[patch]`, `cargo test -p pg-phenotype-core
--release` gives `150 passed`, and clippy `-D warnings` and `cargo fmt
--check` pass. The Python and R legs were not run: a fresh worktree's pixi
env builds pg-phenotype against the pin before the script can patch it.
Python output is unchanged by construction (base 0 is the old rendering);
in R, pedigree errors' messages then count from 1, which reverses the
"0-based in R as well" line of pg-phenotype's CHANGELOG.

No Python consumer needs a change: 0.12.4 changes no public Python
signature, so the consumers' `<0.13` pins take it at the relock. The tag
`v0.12.4` goes on the bump commit `5be610f`, the gated wheel's ref.
