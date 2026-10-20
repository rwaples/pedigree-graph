#!/usr/bin/env bash
# ADR 0007: the host-neutral core declares no host bindings and forbids unsafe code.
# Run by CI's rust job and by `pixi run check-core` (part of `ci-local`).
set -euo pipefail
cd "$(dirname "$0")/.."
test -f crates/core/Cargo.toml || { echo "::error::crates/core/Cargo.toml is missing"; exit 1; }
if grep -Eqi 'pyo3|extendr' crates/core/Cargo.toml; then
  echo "::error file=crates/core/Cargo.toml::the host-neutral core must not depend on pyo3 or extendr (ADR 0007)"
  exit 1
fi
echo "crates/core/Cargo.toml declares no host bindings"
test -f crates/core/src/lib.rs || { echo "::error::crates/core/src/lib.rs is missing"; exit 1; }
if ! grep -qF '#![forbid(unsafe_code)]' crates/core/src/lib.rs; then
  echo "::error file=crates/core/src/lib.rs::missing #![forbid(unsafe_code)] required by ADR 0007"
  exit 1
fi
echo "crates/core/src/lib.rs forbids unsafe code"
