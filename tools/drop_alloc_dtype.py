"""Delete the dtype argument from every ``alloc::`` call in ``crates/core/src``.

    python tools/drop_alloc_dtype.py [--check]

The element dtype of an ``Error::AllocationFailed`` comes from the element
type (``alloc::Dtype``), so the trailing string literal each call used to
pass is deleted.  A call whose last argument is not a string literal is
already migrated and left alone, so a rerun changes nothing.  ``--check``
lists what would change and exits non-zero when anything would, without
writing.  ``alloc.rs`` itself is
skipped: its internal calls are unqualified.  Run `cargo fmt` afterwards.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

CALL = re.compile(r"\balloc::(reserve_exact|reserve|with_capacity|filled|push|extend|collect|cloned)\(")
LITERAL = re.compile(r'"[a-z0-9]+"$')
SRC = Path(__file__).resolve().parent.parent / "crates" / "core" / "src"


def closing_paren(text: str, open_at: int) -> int:
    """The index of the ``)`` matching the ``(`` at ``open_at``."""
    depth = 0
    i = open_at
    while i < len(text):
        c = text[i]
        if c == '"':
            i += 1
            while text[i] != '"':
                i += 2 if text[i] == "\\" else 1
        elif c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    raise ValueError(f"unbalanced call at offset {open_at}")


def rewrite(text: str) -> tuple[str, int]:
    """``text`` with the dtype argument dropped, and the edit count."""
    out: list[str] = []
    done = 0
    pos = 0
    for m in CALL.finditer(text):
        if m.start() < pos:
            continue
        close = closing_paren(text, m.end() - 1)
        inner = text[m.end() : close]
        body = inner.rstrip()
        body = body.removesuffix(",").rstrip()
        lit = LITERAL.search(body)
        head = body[: lit.start()].rstrip() if lit else ""
        if not head.endswith(","):
            continue
        out.append(text[pos : m.end()])
        out.append(head[:-1])
        pos = close
        done += 1
    out.append(text[pos:])
    return "".join(out), done


def main() -> int:
    """Rewrite (or with ``--check``, report) every call still passing a dtype."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="report only; exit 1 when anything would change")
    args = ap.parse_args()
    total = 0
    for path in sorted(SRC.rglob("*.rs")):
        if path.name == "alloc.rs" and path.parent == SRC:
            continue
        text = path.read_text()
        new, n = rewrite(text)
        if n:
            total += n
            print(f"{path.relative_to(SRC)}: {n}")
            if not args.check:
                path.write_text(new)
    print(f"{total} calls")
    return 1 if args.check and total else 0


if __name__ == "__main__":
    sys.exit(main())
