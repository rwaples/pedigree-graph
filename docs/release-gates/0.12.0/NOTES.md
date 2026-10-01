# 0.12.0 consumer gate (2026-10-01)

Release gate for 0.12.0: the kinship walk that skips provable zeros (#34,
ADR 0016), the R moments, counts and burden bindings (#30, ADR 0015) and
the moments arithmetic in the Rust core, which changes
`RelationshipMoments`' constructor.

## Build stage

`pixi run --frozen python external/pedigree-graph/tools/consumer_gate.py run --stage 0.12.0 --wheel-ref 0974cdb`
from the umbrella root of the main simACE checkout (`build.json`):

| artifact | sha256 |
|---|---|
| `pedigree_graph-0.12.0-cp313-abi3-linux_x86_64.whl` | `12415ac8e4f942be2730d44a194914087332a47ce6c9ede8bd3269cf887e1c28` |
| `pedigree_graph-0.12.0.tar.gz` | `14be966418984f709d99f630382ae64d6fc200b964048a090af021573b53d689` |

Both installs pass the import check (15 root names). The package's own fast
suite against the installed wheel: `3389 passed, 2 skipped, 16 deselected
in 223.76s`, exit 0.

## Consumer units

All 13 units ok, every routed unit resolving `pedigree_graph` under
`target/consumer-gate/0.12.0/wheel-site`. fitACE (`f259467`) and pedsum
(`1fabb8e`) were clean. simACE `dev` at `b8f21ca` carried the consumer side
of the constructor change, uncommitted: `tests/analysis/moments_oracle.py`
builds its oracle table with `RelationshipMoments.from_exact`. simACE's
locked env still has 0.11.0, which has no `from_exact`, so the change is
committed with the pin bump and relock. simACE `2843 passed, 2 skipped`,
fitACE `387 passed`, fitACE_epimight `388 passed, 25 deselected`, pedsum
`323 passed, 3 skipped`.
