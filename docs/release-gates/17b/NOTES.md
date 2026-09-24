# 17b consumer gate (2026-09-24): the new gate tools on 0.10.0

No release. This stage proves `tools/consumer_gate.py`, which absorbed
`pg08_wheel_gate.sh` as its `--wheel-ref` build stage, and the renamed
`tools/byte_parity.sh`, against the family. Plan: simACE
`plans/pedigree-graph-1.0-stabilization.md`, slice 17b.

## Build stage

`pixi run --frozen python external/pedigree-graph/tools/consumer_gate.py run --stage 17b --wheel-ref 7b81ac4`
from the umbrella root (`build.json`):

| artifact | sha256 |
|---|---|
| `pedigree_graph-0.10.0-cp313-abi3-linux_x86_64.whl` | `3886d771c16e5495479b6f57f2be004ebdfd3a9db4a15f3debb09ff5bb41fbc1` |
| `pedigree_graph-0.10.0.tar.gz` | `a7fc9c452706a20770f1c48538129a5fed1987743179065446e1326a2c1ef90d` |

Both installs pass the import check (13 root names, `core_version()` equals
the distribution version). The package's own fast suite against the
installed wheel: `3395 passed, 3 skipped, 10 deselected in 125.45s`, exit 0.
The skips include the panic-boundary probe, which installed artifacts skip
by design.

## Same-ref comparison with the old wheel gate

Both tools built `3adf97d` (the commit before the tools changed), run from
the umbrella root. The old `pg08_wheel_gate.sh` and the new build stage
(`build-3adf97d.json`) agree on everything that is deterministic:

| check | `pg08_wheel_gate.sh` | `consumer_gate.py` build stage |
|---|---|---|
| sdist sha256 | `a7fc9c45…c1ef90d` | `a7fc9c45…c1ef90d` |
| wheel and sdist import checks | pass, 13 root names | pass, 13 root names |
| installed fast suite | `3395 passed, 3 skipped, 10 deselected`, exit 0 | `3395 passed, 3 skipped, 10 deselected`, exit 0 |

The wheel is not byte-reproducible: two runs of the old script on the same
ref gave `5147c2ee…` and `8f1993db…`, and the new stage gave `70990325…`, so
wheel sha256 is not a comparison criterion. The sdist carries neither
`tests/` nor `tools/`, which is why `7b81ac4` above builds the same sdist.
`pg08_wheel_gate.sh` was deleted after this comparison.

The old script has one trap the new stage fixes: launched from the
repository root, its `python -` import check puts the cwd on `sys.path` and
imports the source package, so it failed there; the new stage runs its
checks with the work dir as cwd.

## Consumer units

`failed units: none`, exit 0: all 13 units ok, every routed unit resolving
`pedigree_graph` under `target/consumer-gate/17b/wheel-site`. The unit and
step set equals 16d's except the pedigree-graph unit, whose `pytest` step is
now `test-all` (3432 tests, the slow tier included) plus `test-rust`.

## Byte parity

`tools/byte_parity.sh target/byte-parity-17b`, run under the locked 0.10.0
consumers: the manifest is identical to
`../../pedigree-graph-0.8-migration/gate/16e/byte-parity/relocked-0.10.0/manifest.txt`
(`diff` exit 0), and `tools/compare_floats.py` reports every product
`identical`.
