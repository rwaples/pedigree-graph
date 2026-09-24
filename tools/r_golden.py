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
