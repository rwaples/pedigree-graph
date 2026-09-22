#!/usr/bin/env bash
# pedigree-graph release gate (slice 9b, kept for 0.8.1+): build from a clean
# checkout, install the artifacts into isolated environments, prove the import
# resolves there, run the package's own suite against the installed wheel, and
# stage a --target site the consumer gate (external/pedigree-graph/tools/pg08_release_gate.py run
# --routing <site>) imports from.
#
#   external/pedigree-graph/tools/pg08_wheel_gate.sh <work-dir> [ref]      # ref defaults to v0.8
#
# Writes <work-dir>/{clean,dist,build-venv,wheel-venv,sdist-venv,wheel-site} and
# <work-dir>/wheel-gate.json (sha256s, versions, import paths, pytest exit code).
# Never touches a pixi manifest or lock: the build venv is throwaway.  Since
# 0.8.1 the package is maturin-built (ADR 0007): the version is
# [workspace.package].version in Cargo.toml, the build needs the pixi env's
# cargo on PATH, and the wheel must carry the cp313-abi3 tag.
set -euo pipefail

WORK="$(realpath -m "${1:?work dir}")"
REF="${2:-v0.8}"
PG="$(cd "$(dirname "$0")/.." && pwd)"
ROOT="$(cd "$PG/../.." && pwd)"   # the simACE umbrella checkout this repo sits inside
PY="$PG/.pixi/envs/default/bin/python"
[ -x "$PY" ] || { echo "no interpreter at $PY (pixi install in $PG first)" >&2; exit 2; }
export PATH="$(dirname "$PY"):$PATH"
command -v cargo >/dev/null || { echo "no cargo on PATH after prepending $(dirname "$PY")" >&2; exit 2; }

mkdir -p "$WORK"
CLEAN="$WORK/clean"
if [ -e "$CLEAN" ]; then git -C "$PG" worktree remove --force "$CLEAN"; fi
git -C "$PG" worktree add --detach "$CLEAN" "$REF"
HEAD_SHA="$(git -C "$CLEAN" rev-parse HEAD)"
[ -z "$(git -C "$CLEAN" status --porcelain)" ] || { echo "clean worktree is dirty" >&2; exit 2; }
VERSION="$(sed -n '/^\[workspace\.package\]/,/^\[/{s/^version = "\(.*\)"$/\1/p}' "$CLEAN/Cargo.toml")"
[ -n "$VERSION" ] || { echo "no [workspace.package].version in $CLEAN/Cargo.toml" >&2; exit 2; }
echo "building version $VERSION from $HEAD_SHA"

rm -rf "$WORK/dist" "$WORK/build-venv"
"$PY" -m venv "$WORK/build-venv"
"$WORK/build-venv/bin/pip" install --quiet build
( cd "$CLEAN" && "$WORK/build-venv/bin/python" -m build --outdir "$WORK/dist" )
WHEEL="$(ls "$WORK"/dist/*.whl)"
SDIST="$(ls "$WORK"/dist/*.tar.gz)"
sha256sum "$WHEEL" "$SDIST"
case "$(basename "$WHEEL")" in
  pedigree_graph-"$VERSION"-cp313-abi3-*) ;;
  *) echo "wheel $(basename "$WHEEL") is not pedigree_graph-$VERSION-cp313-abi3-*" >&2; exit 2 ;;
esac
case "$(basename "$SDIST")" in
  pedigree_graph-"$VERSION".tar.gz) ;;
  *) echo "sdist $(basename "$SDIST") does not carry version $VERSION" >&2; exit 2 ;;
esac

check_install() {  # <venv> <artifact>
  local venv="$1" artifact="$2"
  rm -rf "$venv"
  "$PY" -m venv "$venv"
  "$venv/bin/pip" install --quiet "$artifact"
  "$venv/bin/python" - "$venv" <<'EOF'
import importlib.metadata as m, pathlib, sys
import pedigree_graph as p
venv = sys.argv[1]
file = pathlib.Path(p.__file__).resolve()
assert str(file).startswith(str(pathlib.Path(venv).resolve()) + "/"), file
assert "site-packages" in file.parts, file
assert (file.parent / "py.typed").is_file(), "py.typed missing"
assert (file.parent / "_native.pyi").is_file(), "_native.pyi missing"
import pedigree_graph._native as native
assert str(pathlib.Path(native.__file__).resolve()).startswith(str(file.parent) + "/"), native.__file__
version = m.version("pedigree-graph")
assert native.core_version() == version, (native.core_version(), version)
names = [getattr(p, n) for n in p.__all__]
import pedigree_graph.relationships, pedigree_graph.summaries, pedigree_graph.effective_size, pedigree_graph.typing
print("installed", version, file, native.__file__, len(names), "root names")
EOF
}

check_install "$WORK/wheel-venv" "$WHEEL"
check_install "$WORK/sdist-venv" "$SDIST"

# The package's own suite against the installed wheel.  cwd is the work dir,
# outside the clean tree, so neither pytest nor the fresh child processes the
# thread tests spawn (plain sys.executable, no -P) can pick up the source
# package over site-packages; the fixtures resolve relative to the test files.
"$WORK/wheel-venv/bin/pip" install --quiet "pedigree-graph[test]@file://$WHEEL"
( cd "$WORK" && "$WORK/wheel-venv/bin/python" -c "import pedigree_graph as p; assert '$WORK/wheel-venv/' in p.__file__, p.__file__" )
set +e
( cd "$WORK" && "$WORK/wheel-venv/bin/python" -m pytest -q -p no:cacheprovider -m "not slow" --rootdir "$CLEAN" -c "$CLEAN/pyproject.toml" "$CLEAN/tests" > "$WORK/wheel-pytest.log" 2>&1 )
PYTEST_RC=$?
set -e
tail -n 3 "$WORK/wheel-pytest.log"

rm -rf "$WORK/wheel-site"
"$PY" -m pip install --quiet --no-deps --target "$WORK/wheel-site" "$WHEEL"

"$PY" - "$WORK" "$HEAD_SHA" "$REF" "$VERSION" "$WHEEL" "$SDIST" "$PYTEST_RC" <<'EOF'
import hashlib, json, pathlib, sys
work, sha, ref, version, wheel, sdist, rc = sys.argv[1:]
digest = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
record = {
    "ref": ref, "head": sha, "version": version,
    "wheel": {"file": pathlib.Path(wheel).name, "sha256": digest(wheel)},
    "sdist": {"file": pathlib.Path(sdist).name, "sha256": digest(sdist)},
    "wheel_pytest_exit": int(rc),
    "wheel_site": str(pathlib.Path(work) / "wheel-site"),
}
out = pathlib.Path(work) / "wheel-gate.json"
out.write_text(json.dumps(record, indent=2) + "\n")
print(out.read_text())
EOF
exit "$PYTEST_RC"
