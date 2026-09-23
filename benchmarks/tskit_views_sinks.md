# Compact view and relationship burden benchmark

The measurements below compare native operations in fresh processes. Graph
construction and Python import occur before the timed region. Peak RSS includes
the graph, the operation, and its output. Six Rayon threads were configured.
The script is [`tools/pg_tskit_profile.py`](../tools/pg_tskit_profile.py);
run it with the pedigree-graph pixi interpreter after building the worktree's
native extension. The JSON from the three-repeat 30k run is not committed, but
the script records every run, closure size, output size, and parity digest.

## View extraction

The `random_30k` fixture contains 30,300 rows. Each cell below is the median
of three fresh-process runs. A seeded permutation chooses view rows. "Closure"
includes selected rows, represented ancestors, and MZ partners. Full and
compact outputs had identical digests at every fraction.

| View | Selected | Closure | Full wall | Compact wall | Full peak RSS | Compact peak RSS |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.1% | 30 | 1,447 | 13.4 ms | 1.2 ms | 67.9 MiB | 64.5 MiB |
| 1% | 303 | 7,692 | 15.2 ms | 6.7 ms | 67.8 MiB | 65.6 MiB |
| 10% | 3,030 | 17,423 | 39.8 ms | 28.8 ms | 68.5 MiB | 67.8 MiB |
| 50% | 15,150 | 25,894 | 165.4 ms | 143.9 ms | 82.3 MiB | 82.6 MiB |

On `random_300k` (303,000 rows), a single 1% view run selected 3,030 rows
and retained 74,859 rows. Full extraction took 180.9 ms at 119.6 MiB peak
RSS; compact extraction took 52.3 ms at 97.2 MiB. Counts on the same view
took 179.1 ms at 117.6 MiB for the full graph and 46.1 ms at 98.5 MiB for
the compact graph. The result digests matched in both comparisons. These large
fixture numbers are single runs, so they show scale rather than a stable
distribution.

The automatic dispatch threshold is at most 10% of a graph with at least
20,000 rows. Closure can be much larger than the view, so the threshold is a
conservative heuristic from these fixtures, not a guarantee of speedup on
every pedigree.

## Full relationship burden

Both operations classify all categories through degree five. `pairs` returns
materialized pair arrays; `burden` accumulates category, per-person degree,
and same-depth counts. On `random_30k`, three-run medians were 279.8 ms and
116.0 MiB peak RSS for pairs, versus 282.2 ms and 68.2 MiB for burden. Pair
output was 23.4 MiB; burden output was 0.58 MiB.

On `random_300k`, one run produced 235.1 MiB of pair output in 3.45 s at
620 MiB peak RSS. The burden arrays occupied 5.78 MiB; the traversal took
3.95 s at 126.2 MiB peak RSS. The native burden path therefore trades some
wall time in this large run for much lower peak memory. Pedsum additionally
avoids its former Python pair-key construction and sorting; its end-to-end
report time was not measured here.

Parity tests compare every category's compact pair blocks and view counts
against the full engine, and compare burden arrays against aggregation of
materialized pairs. Pedsum tests compare the old and new report schemas and
values on ordinary, MZ, and external-parent fixtures.
