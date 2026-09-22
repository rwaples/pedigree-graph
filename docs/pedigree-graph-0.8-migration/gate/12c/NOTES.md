# 12c post-publish gate for pedigree-graph 0.9.0 (2026-09-20)

The second half of slice 12's commit 8. The pre-publish half is `../12b/`.
0.9.0 is on PyPI, the five caps have moved, the three locks that pin it from
PyPI are relocked, the byte-parity probe has been re-run, and the thirteen-unit
gate is green at `--routing locked`.

## Published

`v0.9.0` is pushed and PyPI serves six files. The tag moved before the push:
the two test-only commits closing issues #21 and #26 landed on `main` after the
changelog commit, and rather than release behind them the tag and the `v0.9`
branch were moved to `a4bacb7`. `../12b/NOTES.md` records the evidence that
nothing shipped changed. The published sdist hashes to
`45013ae13b1285e08223f53a7e17fd53d076455002c556046a10df9bf9550684`, equal to
the sdist the 12b wheel gate built locally, so what PyPI serves is the artifact
12b tested rather than a rebuild sharing a commit.

## Five caps, not four, and three locks, not four

The plan said three caps and this file previously said four. There are five:

| file | line |
|---|---|
| `pyproject.toml` | 13 |
| `fitACE/pyproject.toml` | 12 |
| `fitACE/fitACE_epimight/pyproject.toml` | 13 |
| `external/pedsum/pyproject.toml` | 20 |
| `external/pedsum/environment.yml` | 25 |

The fifth is the conda recipe's `pip:` block. Its own comment says it matches
the range `pyproject.toml` declares, but nothing tests that and `pedsum`'s
`pixi.toml` notes the recipe is neither locked nor exercised by CI, so it would
have drifted silently. A recursive `grep` from the umbrella root finds only the
first of the five: `fitACE/` and `external/` are gitignored there and the
default grep honours it. Sweep the family with `find`, not a recursive grep.

Relocking is three trees, not four. Only simACE, fitACE and pedsum pin
pedigree-graph from PyPI; `external/pedigree-graph/pixi.lock` resolves it as an
editable path and has nothing to bump. Each reported exactly one change,
`pedigree-graph 0.8.3 -> 0.9.0`, and scanning every changed `name:` and
`- pypi:` entry confirms it is the only package that moved in all three locks.
No reorder recurred of the kind 11b saw in `execnet`. The only other content
change is `pytest-xdist>=3.8,<4` appearing in 0.9.0's declared `test` extra.
All three pin wheel sha256
`5fe22fcea51ac4f50483cb411dc358dbd1ffe234f7c332b07cb471388480e946`.

11b's stale-index trap recurred with a new path. `pixi clean cache --pypi
--yes` freed 980.7 MiB and took the `uv-cache/simple-v21` rkyv with it, but a
newer `~/.cache/uv/simple-v24/pypi/pedigree-graph.rkyv` survives it and had to
be deleted by hand, or the updates report "already up-to-date".

After `pixi install --locked`, each environment imports 0.9.0 from its own
site-packages with `_native.core_version() == "0.9.0"`.

## Byte parity

`byte-parity/relocked-0.9.0/`, both envs confirmed at `pedigree-graph 0.9.0`.

| artifact | baseline 0.8.3 | relocked 0.9.0 | |
|---|---|---|---|
| `pairwise_relatedness.tsv` | `505a8fd3…` | `505a8fd3…` | identical |
| `report.yaml` | `6da909f7…` | `5a7b5d54…` | differs |

The pair table is byte-identical. That is the artifact the probe calls the
canonical pair list, and it is where 0.9.0's element-for-element promise is
tested; the promise holds.

`report.yaml` differs in four content lines out of 1889, all inside
`parameters:`, and none a computed value. No relationship correlation or count
moved.

```
+    max_degree: 3
-    simace_version: 2026.10.dev15+gb063438a0.d20260908
+    simace_version: 2026.10.dev50+g0dea8c9b4.d20260919
+    skip_ne_coancestry: true
```

The version line is the cause: the baseline was cut at simACE `b063438a0` on
2026-09-08 and this run is `0dea8c9b4`, 35 commits later. Both new keys are
newly recorded rather than newly set. `2a4d20d Record analysis provenance in
run artifacts` added them, and it postdates the baseline.
`DEFAULT_MAX_DEGREE: int = 3` is identical at both commits and
`config/_default.yaml:116` also says 3, so both runs extracted at the same
degree. `skip_ne_coancestry: true` reflects `35e7d65 Make the Ne_C coancestry
estimator opt-in`, which `git merge-base --is-ancestor` places before the
baseline, so the baseline run skipped it too and merely did not say so.

The baseline cannot be re-cut to remove this: 12b already records that simACE's
`dev` is red against the 0.8.3 its old lock pinned, since `ne_group_coancestry`
does not exist there. The pair table, which those 35 simACE commits do not
touch, is the half of the probe that still isolates pedigree-graph cleanly.

## Result

`failed units: none`, thirteen units at `--routing locked`. Every `routing`
step passed, so each consumer resolved `pedigree_graph` from its own locked
environment's site-packages.

| unit | suite | wall | peak RSS |
|---|---|---:|---:|
| pedigree-graph | 2954 passed, 9 skipped, 9 deselected | 530.1 s | 983.9 MiB |
| simACE | 1499 passed, 3 skipped (+ smoke, atlas) | 357.7 s | 552.4 MiB |
| fitACE | 383 passed | 5716.1 s | 490.8 MiB |
| fitACE_pcgc | 211 passed, 4 skipped | 27.2 s | 352.5 MiB |
| fitACE_iter_reml | 110 passed, 4 skipped | 64.1 s | 348.7 MiB |
| ace_iter_reml | six binary test units, all exit 0 | 18.4 s | 108.2 MiB |
| fitACE_tetraher | 34 passed | 6.0 s | 281.7 MiB |
| tetraher_simace | ruff, format, ldak runs, ldak is fork | 0.8 s | 107.4 MiB |
| fitACE_pafgrs | 121 passed | 13.6 s | 717.1 MiB |
| fitACE_stan | ruff, format, import beside fitace and simace | 1.6 s | 108.9 MiB |
| fitACE_frailty | 7 passed | 4.7 s | 211.9 MiB |
| fitACE_epimight | 257 passed, 19 deselected | 52.3 s | 752.0 MiB |
| pedsum | 324 passed | 180.3 s | 337.7 MiB |

**No wall here is a measurement.** The box ran this gate at 800 MHz under the
external PROCHOT clamp, roughly a 3.25x penalty; fitACE's suite self-reports
95.38 s against 27.6 s for the same unit at the same routing in 11b.

Two entries carry a caveat beyond the clock. `fitACE`'s 5716.1 s is dominated
by a pytest step the gate timed at 5711.9 s while pytest itself reported `383
passed in 95.38s`; the log-file mtimes bracket the gap exactly, 12:20:01 to
13:55:12, and `pytest.log` shows `bringing up nodes...` twice, which is not
normal xdist output. The unit passed and `pg08_release_gate.py` has no retry
logic, so the 93 minutes are unexplained and the number should not be quoted.
`fitACE_pafgrs` is a re-run of that unit alone after the fix below; every other
row is from the single full-gate invocation.

## What the gate caught

`fitACE_pafgrs` failed the first run, and not because of pedigree-graph.

```
E  TypeError: too many arguments: expected 4, got 8
   fitACE_pafgrs/fitace_pafgrs/pafgrs_bivariate.py:1061
7 failed, 114 passed in 11.81s
```

simACE `57d5c9a Stop _tetrachoric_core taking thresholds it discards`, at
2026-09-19 14:38:06, narrowed `_tetrachoric_core` from eight parameters to
four. It updated simACE's own caller at `simace/analysis/stats/tetrachoric.py`
but not pafgrs, which still passed `t1, t2, phi_t1, phi_t2`. 12b's
`fitACE_pafgrs` step ran at 14:20, eighteen minutes before that commit, which
is why no earlier gate saw it. This is CLAUDE.md's cross-package coupling
gotcha, caught where it should be.

The fix drops the four arguments and the lines computing them, which is what
`57d5c9a` intended: the kernel derives its own thresholds from the
canonicalized table, because transforming the caller's would reintroduce the
relabeling differences canonicalization exists to remove. The two imports those
lines alone used, `ndtri` and `norm_cdf`, went with them; `pafgrs.py` still
uses both. The degenerate-input guards stayed, so pafgrs keeps returning `0.0`
for an empty table or a saturated marginal rather than adopting simACE's `nan`,
which `test_rg_no_pairs_falls_back` depends on. `121 passed`, ruff and format
clean, and `test_asymmetric_thresholds` passes, which is the test that would
have caught a numerical shift. A sweep of the family found no other stale
caller.
