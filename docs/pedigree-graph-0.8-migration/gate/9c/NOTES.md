# 9c post-publish lock gate (2026-09-09)

`v0.8.0` (`05f06cb`) was tagged and pushed; `publish.yml` run 34328975589
succeeded, including its "Verify the built version matches the tag" step, which
is the only local proof that setuptools-scm derives `0.8.0` from the tag (the
9b wheel was built with `SETUPTOOLS_SCM_PRETEND_VERSION`).  PyPI serves
`pedigree_graph-0.8.0-py3-none-any.whl`, sha256
`fe8ba7931ccf7f4e6402f436ab105fa44d23f9dd665533f88ed1c55eaf664e53`.  That is
not the 9b local wheel (`42cf4d53…`): CI built from the tag, and the plan
never expected the two to be byte-identical.

## Locks

Deliberate `pixi lock` in simACE, fitACE, and pedsum.  Each reported exactly
one change, `~ (pypi) pedigree-graph 0.7.1 -> 0.8.0`, and no other package URL
moved in any of the three diffs.  The recorded `requires_dist` lines refreshed
from `pedigree-graph>=0.7.1,<0.8` to `>=0.8,<0.9` (once in simACE, three times
in fitACE, once in pedsum); they were stale because slice 8 raised the pins in
`pyproject.toml` without relocking.  pedsum's diff also moves an `iniconfig`
block, same URL, version, and sha256, reordered by the new entry.

`pixi install --locked` then succeeded in all three.

## Gate

`EPIMIGHT_CONDA_ENV=epimight-master pixi run python tools/pg08_release_gate.py
run --stage 9c --routing locked --unit <the nine consumer units>`.

All nine green, no failed steps.  Every `routing` step resolved
`pedigree_graph` under the unit's own `.pixi/envs/default/…/site-packages/`
with no `PYTHONPATH`, which is what this stage exists to prove.

| unit | suite | wall |
|---|---|---:|
| simACE | 1470 passed, 3 skipped (+ smoke `--forceall`, atlas) | 167.2 s |
| fitACE | 383 passed | 77.1 s |
| fitACE_pcgc | 211 passed, 4 skipped | 21.3 s |
| fitACE_iter_reml | 110 passed, 4 skipped | 97.7 s |
| fitACE_tetraher | 33 passed | 16.3 s |
| fitACE_pafgrs | 119 passed | 39.2 s |
| fitACE_frailty | 7 passed | 11.0 s |
| fitACE_epimight | 256 passed, 19 deselected | 168.7 s |
| pedsum | 325 passed (+ TSV export, CLI smoke) | 572.1 s |

Not run in 9c: the slow suites (9a only, per the plan), so the five
fitACE_epimight R-driver failures 9a proved pre-existing were not re-exercised
here; the `pedigree-graph` and `ace_iter_reml` units, which run from their own
manifests and never import a routed pedigree-graph.
