#!/usr/bin/env bash
# Build the R source tarball of pedigreegraph (slice 16c), the only producer
# of one: the checked-in r/ reaches the core through ../../../crates/core,
# which a tarball cannot, so this stages a copy.
#
#   tools/r_build_tarball.sh [out-dir]       # default: target/r
#
# In a temporary copy of r/ it stages crates/core at src/rust/core with the
# workspace-inherited keys written out, points the binding crate at it,
# vendors the complete locked dependency graph into src/rust/vendor.tar.xz
# (with the source-replacement config Makevars installs), lists every
# vendored crate's authors and license in inst/AUTHORS, and runs R CMD build.
# Needs cargo and R on PATH (pixi run -e r tools/r_build_tarball.sh).
set -euo pipefail

# The published tarball never carries test-only entry points (ADR 0007).  Its
# vendored build ignores PG_CARGO_FEATURES anyway; refusing here keeps a dev
# shell's setting from being mistaken for a supported build.
if [ -n "${PG_CARGO_FEATURES:-}" ]; then
  echo "r_build_tarball.sh: unset PG_CARGO_FEATURES (=$PG_CARGO_FEATURES); the tarball builds without features" >&2
  exit 1
fi

REPO="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$(realpath -m "${1:-$REPO/target/r}")"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
PKG="$STAGE/pedigreegraph"

workspace_key() {
  sed -n '/^\[workspace\.package\]/,/^\[/{s/^'"$1"' = "\(.*\)"$/\1/p}' "$REPO/Cargo.toml"
}
VERSION="$(workspace_key version)"
EDITION="$(workspace_key edition)"
LICENSE="$(workspace_key license)"
REPOSITORY="$(workspace_key repository)"

# The package, without build products.
mkdir -p "$PKG"
tar -C "$REPO/r" --exclude=src/rust/target --exclude='src/*.o' --exclude='src/*.so' \
  --exclude=src/.cargo -cf - . | tar -C "$PKG" -xf -

# The core: its library sources only, keys the workspace supplied written out.
CORE="$PKG/src/rust/core"
mkdir -p "$CORE"
cp -R "$REPO/crates/core/src" "$CORE/src"
rm -rf "$CORE/src/bin"
sed -e "s|^version.workspace = true|version = \"$VERSION\"|" \
    -e "s|^edition.workspace = true|edition = \"$EDITION\"|" \
    -e "s|^license.workspace = true|license = \"$LICENSE\"|" \
    -e "s|^repository.workspace = true|repository = \"$REPOSITORY\"|" \
    -e '/^\[\[bin\]\]/,/^$/d' \
    "$REPO/crates/core/Cargo.toml" > "$CORE/Cargo.toml"
if grep -q 'workspace = true' "$CORE/Cargo.toml"; then
  echo "error: an inherited key is left in the staged core Cargo.toml" >&2
  exit 1
fi
sed -i 's|path = "../../../crates/core"|path = "core"|' "$PKG/src/rust/Cargo.toml"

# Vendor exactly the locked graph, then compress it.
(
  cd "$PKG/src/rust"
  cargo vendor --locked --versioned-dirs vendor > vendor-config.toml
  tar --owner=0 --group=0 --numeric-owner -cJf vendor.tar.xz vendor
  rm -rf vendor
)

# Authors and licenses of every vendored crate, for DESCRIPTION's Copyright.
mkdir -p "$PKG/inst"
{
  echo "The pedigreegraph source tarball bundles these Rust crates (src/rust/vendor.tar.xz)."
  echo "Each is listed with its version, license and authors as its Cargo manifest states"
  echo "(or, where the manifest names none, its repository)."
  echo
  cargo metadata --locked --format-version 1 --manifest-path "$PKG/src/rust/Cargo.toml" |
    python3 -c '
import json, sys
meta = json.load(sys.stdin)
for p in sorted(meta["packages"], key=lambda p: (p["name"], p["version"])):
    if p["source"] is None:
        continue
    name, version = p["name"], p["version"]
    license = p["license"] or "see the crate"
    # A manifest may omit authors; its repository then names the holders.
    authors = ", ".join(p["authors"]) or "the authors of " + (p["repository"] or name)
    print(f"{name} {version} ({license}): {authors}")
'
} > "$PKG/inst/AUTHORS"

mkdir -p "$OUT"
( cd "$OUT" && R CMD build --no-manual "$PKG" )
ls -l "$OUT"/pedigreegraph_*.tar.gz
