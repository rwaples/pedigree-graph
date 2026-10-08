"""Write the R package's golden files from the Python package.

The R binding and the Python one call the same Rust core, so these goldens
catch binding bugs: a lost 1-based offset, a wrong order, a lossy promotion,
a dropped category or id-type drift.  For every hand-built core fixture
(``crates/core/tests/fixtures/*.tsv`` of at most ``MAX_ROWS`` rows: MZ
twins, external parents, mating loops, cousins) plus ``inbred0``, an inbred
pedigree at scale, the generator writes, under
``r/tests/testthat/golden/<fixture>/``:

* ``pedigree.tsv``: the ``id``/``mother``/``father``/``twin`` frame both hosts
  build from (``-1`` for none).  The fixtures are engine columns, rows plus
  original parent ids, so ids are reconstructed here once and R reads them
  rather than re-deriving them.
* ``pairs.tsv``: every pair up to degree 5, registry order then pair order,
  1-based rows, with ids.
* ``kinship.tsv``: the upper triangle (``i <= j``, 1-based) of the complete
  kinship matrix, values as hex floats so R compares them exactly.
* ``inbreeding.tsv``: ``F`` per row, as hex floats.
* ``counts.tsv``: the pairs of every category up to degree 5, by code.
* ``moments.tsv``, ``moments_exact.tsv``, ``moments_folded.tsv`` and
  ``moments_merged.tsv``: ``relationship_moments`` of nine categories under
  one fixed spec (:func:`_moments_spec`), in both ``symmetric`` modes, for
  the cells that hold pairs: every statistic R's ``as.data.frame`` gives,
  as hex floats; every exact accumulator, in decimal; the ``MO`` and ``FO``
  cells folded into one; and the merge of two calls whose second has every
  value scaled by 2^-600.
* ``burden.tsv``: each row's relatives at degrees 1 to 5
  (``relationship_burden``), and ``burden_depth.tsv``, its related pairs
  per structural depth.
* ``relatives.tsv``: ``relatives_per_person`` at degree 5 under one fixed
  pair of threshold columns (:func:`_relatives_spec`), one line per row and
  category where the row has relatives: the pair count and each column's.

and ``categories.tsv``, the Python registry, so the R registry (read from the
core) is held to it.

Run from the repository root::

    pixi run python tools/r_golden.py          # rewrite the goldens
    pixi run python tools/r_golden.py --check  # exit 1 if they are stale
"""

from __future__ import annotations

import argparse
import filecmp
import sys
import tempfile
from pathlib import Path

import numpy as np
import polars as pl
import scipy.sparse as sp

from pedigree_graph import PedigreeGraph
from pedigree_graph.relationships import RELATIONSHIPS

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "crates" / "core" / "tests" / "fixtures"
GOLDEN = REPO / "r" / "tests" / "testthat" / "golden"
# The goldens ship in the R source tarball, so they cover the hand-built
# edge cases and one inbred pedigree (about 1 MB in all); every larger fixture
# is compared by tools/r_parity.sh, which never ships.  Degree-5 pairs and the
# kinship matrix of deep_inbred_60g or random_1k alone are 8 to 11 MB.
MAX_ROWS = 20
EXTRA = ("inbred0",)


def _frame(fixture: pl.DataFrame) -> pl.DataFrame:
    """The id frame of an engine-column fixture.

    An internal parent's id is the ``orig_*`` its children name it by; a row
    no child names gets a fresh id above every id in the fixture.  External
    parents keep their ids and have no row.
    """
    n = fixture.height
    ids = np.full(n, -1, dtype=np.int64)
    for rows, orig in (("mother", "orig_mother"), ("father", "orig_father")):
        known = fixture.filter(pl.col(rows) >= 0)
        ids[known[rows].to_numpy()] = known[orig].to_numpy()
    named = np.concatenate([ids, fixture["orig_mother"].to_numpy(), fixture["orig_father"].to_numpy()])
    top = int(named.max(initial=-1))
    unnamed = np.flatnonzero(ids < 0)
    ids[unnamed] = top + 1 + np.arange(len(unnamed))
    twin = fixture["twin"].to_numpy()
    return pl.DataFrame(
        {
            "id": ids,
            "mother": fixture["orig_mother"].to_numpy(),
            "father": fixture["orig_father"].to_numpy(),
            "twin": np.where(twin >= 0, ids[np.maximum(twin, 0)], -1),
        }
    )


def _hex(values: np.ndarray) -> list[str]:
    return [float(v).hex() for v in values]


def _write(frame: pl.DataFrame, path: Path) -> None:
    frame.write_csv(path, separator="\t", line_terminator="\n")


def _golden(fixture: Path, out: Path) -> None:
    frame = _frame(pl.read_csv(fixture, separator="\t"))
    columns = {name: frame[name].to_numpy() for name in frame.columns}
    graph = PedigreeGraph.from_frame(columns)
    ids = columns["id"]
    out.mkdir(parents=True)
    _write(frame, out / "pedigree.tsv")

    pairs = graph.relationship_pairs(max_degree=5)
    code, first, second = [], [], []
    for block in pairs.values():
        code += [block.code] * len(block)
        first.append(block.first_rows)
        second.append(block.second_rows)
    first_rows = np.concatenate(first).astype(np.int64)
    second_rows = np.concatenate(second).astype(np.int64)
    _write(
        pl.DataFrame(
            {
                "code": code,
                "first": first_rows + 1,
                "second": second_rows + 1,
                "first_id": ids[first_rows],
                "second_id": ids[second_rows],
            },
            schema_overrides={"code": pl.String},
        ),
        out / "pairs.tsv",
    )

    upper = sp.triu(graph.kinship_matrix(), format="coo")
    order = np.lexsort((upper.row, upper.col))
    _write(
        pl.DataFrame(
            {
                "i": upper.row[order].astype(np.int64) + 1,
                "j": upper.col[order].astype(np.int64) + 1,
                "x": _hex(upper.data[order]),
            }
        ),
        out / "kinship.tsv",
    )
    _write(pl.DataFrame({"F": _hex(graph.inbreeding())}), out / "inbreeding.tsv")

    counts = graph.relationship_counts(max_degree=5)
    _write(pl.DataFrame({"code": list(counts), "count": [counts[code] for code in counts]}), out / "counts.tsv")
    burden = graph.relationship_burden()
    _write(
        pl.DataFrame({f"degree_{d}": burden.per_person[:, d - 1].astype(np.int64) for d in range(1, 6)}),
        out / "burden.tsv",
    )
    same_depth = burden.same_depth_pairs.astype(np.int64)
    _write(pl.DataFrame({"depth": np.arange(len(same_depth)), "pairs": same_depth}), out / "burden_depth.tsv")
    _relatives(graph, out)
    _moments(graph, columns["mother"], out)


def _relatives_spec(n: int) -> dict[str, tuple[np.ndarray, np.ndarray | float]]:
    """The threshold columns both hosts pass: a per-row threshold, and a scalar one.

    ``rowwise`` has NaN relatives every third row and ties between relative
    and threshold; ``scalar`` ties at its threshold.  Integer arithmetic only,
    so R builds the same doubles; ``test-golden.R`` and ``tools/r_parity.R``
    repeat it.
    """
    rows = np.arange(1, n + 1, dtype=np.float64)
    return {
        "rowwise": (np.where(rows % 3 == 0, np.nan, rows * 37 % 11), rows * 5 % 11),
        "scalar": (rows % 4, 2.0),
    }


def _relatives(graph: PedigreeGraph, out: Path) -> None:
    r = graph.relatives_per_person(max_degree=5, thresholds=_relatives_spec(graph.n_individuals))
    row, slot = np.nonzero(r.counts[:, :, 0])
    frame = {"row": row.astype(np.int64) + 1, "code": [r.categories[s] for s in slot]}
    for k, column in enumerate(r.columns):
        frame[column] = r.counts[row, slot, k].astype(np.int64)
    _write(pl.DataFrame(frame, schema_overrides={"code": pl.String}), out / "relatives.tsv")


MOMENT_CATEGORIES = ("MZ", "FS", "MO", "FO", "MHS", "PHS", "GP", "Av", "1C")
MOMENT_STATS = (
    "sum_first",
    "sum_second",
    "sumsq_first",
    "sumsq_second",
    "mean_first",
    "mean_second",
    "m2_first",
    "m2_second",
)


def _moments_spec(graph: PedigreeGraph) -> dict:
    """The moments spec both hosts run: factors, values, products and categories.

    Depth parity and a two-level row code over each member; ordinary,
    constant, half-ulp-tie and large values; the default products and one
    first x first; the mother as the equality key (passed separately).  R
    builds the same spec in ``test-golden.R`` and ``tools/r_parity.R``.
    """
    n = graph.n_individuals
    rows = np.arange(1, n + 1, dtype=np.float64)
    # max|x| = 2^43 gives e = 0, so every half-integer quantizes on a tie.
    tie = (rows % 7) - 2.5
    tie[0] = 2.0**43
    return {
        "categories": list(MOMENT_CATEGORIES),
        "first": {"parity": np.asarray(graph.depth) % 2, "code": (np.arange(n) % 2) + 1},
        # Integer arithmetic and one correctly rounded division or product,
        # so R computes every value bit for bit (no libm functions).
        "values": {
            "ordinary": (rows * 37 % 101) / 7 - 5,
            "constant": np.full(n, 0.3),
            "tie": tie,
            "large": 1e150 * ((rows * 13 % 17) - 8),
        },
        "products": [
            ("first.ordinary", "second.ordinary"),
            ("first.constant", "second.constant"),
            ("first.tie", "second.tie"),
            ("first.large", "second.large"),
            ("first.ordinary", "first.tie"),
        ],
    }


def _float_text(values: np.ndarray) -> list[str]:
    return ["NaN" if np.isnan(v) else float(v).hex() for v in values]


def _moment_frame(m, symmetric: str) -> pl.DataFrame:
    """R's ``as.data.frame`` of a result: axes as text (a code factor's labels), then every statistic."""
    levels = m.cell_levels()
    keep = m.counts.reshape(-1) > 0
    frame = {"symmetric": [symmetric] * int(keep.sum())}
    for axis in m.axes:
        values = levels[axis.name].reshape(-1)[keep]
        text = [f"r{int(v) - 1}" for v in values] if axis.name.endswith("_code") else [str(v) for v in values]
        frame[axis.name] = text
    frame["n"] = [str(int(v)) for v in m.counts.reshape(-1)[keep]]
    for stat in MOMENT_STATS:
        for j, column in enumerate(m.columns):
            if stat.startswith("mean_"):
                values = m.mean(f"{stat.removeprefix('mean_')}.{column}")
            else:
                values = getattr(m, stat)[..., j]
            frame[f"{stat}.{column}"] = _float_text(values.reshape(-1)[keep])
    names = [f"{a}:{b}" for a, b in m.products]
    for stat in ("cross", "comoment"):
        for j, name in enumerate(names):
            frame[f"{stat}.{name}"] = _float_text(getattr(m, stat)[..., j].reshape(-1)[keep])
    for (a, b), name in zip(m.products, names, strict=True):
        frame[f"pearson.{name}"] = _float_text(m.pearson(a, b).reshape(-1)[keep])
    return pl.DataFrame(frame)


def _exact_frame(m, symmetric: str) -> pl.DataFrame:
    """Every accumulator of the cells that hold pairs, cell by cell, in decimal."""
    stride = 1 + 4 * len(m.columns) + len(m.products)
    k = len(m.columns)
    slabs = [m.counts[..., np.newaxis], m.q_sum_first, m.q_sum_second, m.q_sumsq_first, m.q_sumsq_second, m.q_cross]
    stacked = np.concatenate([np.asarray(x, dtype=object) for x in slabs], axis=-1).reshape(-1, stride)
    assert stacked.shape[1] == 1 + 4 * k + len(m.products)
    values = [str(int(v)) for v in stacked[m.counts.reshape(-1) > 0].reshape(-1)]
    return pl.DataFrame({"symmetric": [symmetric] * len(values), "value": values})


def _moments(graph: PedigreeGraph, mother: np.ndarray, out: Path) -> None:
    spec = _moments_spec(graph)
    frames: dict[str, list[pl.DataFrame]] = {
        "moments": [],
        "moments_exact": [],
        "moments_folded": [],
        "moments_merged": [],
    }
    for symmetric in ("canonical", "both"):
        m = graph.relationship_moments(**spec, same={"mother": mother}, symmetric=symmetric)
        frames["moments"].append(_moment_frame(m, symmetric))
        frames["moments_exact"].append(_exact_frame(m, symmetric))
        folded = m.select(category=["MO", "FO"]).sum("category")
        frames["moments_folded"].append(_moment_frame(folded, symmetric))
        scaled = {**spec, "values": {name: column * 2.0**-600 for name, column in spec["values"].items()}}
        merged = m.merge(graph.relationship_moments(**scaled, same={"mother": mother}, symmetric=symmetric))
        frames["moments_merged"].append(_moment_frame(merged, symmetric))
    for name, parts in frames.items():
        _write(pl.concat(parts), out / f"{name}.tsv")


def _categories(out: Path) -> None:
    rows = list(RELATIONSHIPS.values())
    _write(
        pl.DataFrame(
            {
                "code": [c.code for c in rows],
                "degree": [c.degree for c in rows],
                "first_role": [c.first_role or "NA" for c in rows],
                "second_role": [c.second_role or "NA" for c in rows],
                "nominal_kinship": [float(c.nominal_kinship).hex() for c in rows],
            }
        ),
        out / "categories.tsv",
    )


def generate(out: Path) -> list[str]:
    """Write every golden under *out*; return the fixture names used."""
    names = []
    out.mkdir(parents=True, exist_ok=True)
    for fixture in sorted(FIXTURES.glob("*.tsv")):
        if fixture.stem not in EXTRA and pl.read_csv(fixture, separator="\t").height > MAX_ROWS:
            continue
        _golden(fixture, out / fixture.stem)
        names.append(fixture.stem)
    _categories(out)
    return names


def stale(committed: Path) -> list[str]:
    """Paths whose committed golden differs from a fresh one, or is missing or extra."""
    with tempfile.TemporaryDirectory() as tmp:
        fresh = Path(tmp)
        generate(fresh)
        want = {p.relative_to(fresh) for p in fresh.rglob("*") if p.is_file()}
        have = {p.relative_to(committed) for p in committed.rglob("*") if p.is_file()} if committed.exists() else set()
        diff = sorted(str(p) for p in want ^ have)
        diff += sorted(str(p) for p in want & have if not filecmp.cmp(fresh / p, committed / p, shallow=False))
    return diff


def main() -> int:
    """Rewrite the goldens, or with ``--check`` report stale ones and exit 1."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true", help="exit 1 if the committed goldens are stale")
    args = parser.parse_args()
    if args.check:
        diff = stale(GOLDEN)
        for path in diff:
            print(f"stale: {path}")
        return 1 if diff else 0
    if GOLDEN.exists():
        for path in sorted(GOLDEN.rglob("*"), reverse=True):
            path.unlink() if path.is_file() else path.rmdir()
    names = generate(GOLDEN)
    print(f"wrote goldens for {len(names)} fixtures under {GOLDEN.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
