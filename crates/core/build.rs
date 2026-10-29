//! Embeds the core's git revision as the compile-time `PG_CORE_REV`, which
//! both bindings report as `pg_core_rev()` (issue #46).  The first of:
//!
//! 1. the `PG_CORE_REV` environment variable, which the manylinux wheel
//!    builds set because git there cannot read the mounted checkout;
//! 2. a `REVISION` file beside this script, which `tools/r_build_tarball.sh`
//!    writes because the R source tarball carries no `.git`;
//! 3. `git rev-parse HEAD`, suffixed `-dirty` when anything under this crate
//!    differs from it, provided this crate's manifest is tracked by that
//!    repository (so a copy unpacked inside another one reads no revision);
//! 4. `unknown`.

use std::path::{Path, PathBuf};
use std::process::Command;

/// `git -C dir args`, its trimmed stdout on success.  The repository is
/// found from `dir` alone: a build started by a git hook inherits variables
/// that would point it at another repository or index.
fn git(dir: &Path, args: &[&str]) -> Option<String> {
    let out = Command::new("git")
        .env_remove("GIT_DIR")
        .env_remove("GIT_WORK_TREE")
        .env_remove("GIT_INDEX_FILE")
        .env_remove("GIT_COMMON_DIR")
        .arg("--no-optional-locks")
        .arg("-C")
        .arg(dir)
        .args(args)
        .output()
        .ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().to_owned())
}

/// Rerun when `path` changes.  A missing path is skipped: cargo would rerun
/// on every build.
fn watch(dir: &Path, path: &str) {
    let path = dir.join(path);
    if path.exists() {
        println!("cargo:rerun-if-changed={}", path.display());
    }
}

fn from_git(dir: &Path) -> Option<String> {
    git(dir, &["ls-files", "--error-unmatch", "Cargo.toml"])?;
    let rev = git(dir, &["rev-parse", "HEAD"])?;
    // A checkout rewrites HEAD, and every commit, checkout or reset appends
    // to HEAD's reflog, wherever the branch ref itself is stored.
    for path in ["HEAD", "logs/HEAD"] {
        watch(dir, &git(dir, &["rev-parse", "--git-path", path])?);
    }
    let dirty = git(dir, &["status", "--porcelain", "--", "."]).is_some_and(|s| !s.is_empty());
    Some(if dirty { format!("{rev}-dirty") } else { rev })
}

fn main() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    // The crate's own files, for the dirty check and for a REVISION file.
    println!("cargo:rerun-if-changed=.");
    println!("cargo:rerun-if-env-changed=PG_CORE_REV");
    let rev = std::env::var("PG_CORE_REV")
        .ok()
        .map(|rev| rev.trim().to_owned())
        .filter(|rev| !rev.is_empty())
        .or_else(|| {
            std::fs::read_to_string(dir.join("REVISION"))
                .ok()
                .map(|rev| rev.trim().to_owned())
        })
        .or_else(|| from_git(&dir))
        .unwrap_or_else(|| "unknown".to_owned());
    println!("cargo:rustc-env=PG_CORE_REV={rev}");
}
