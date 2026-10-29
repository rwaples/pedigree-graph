#' Revision of the Rust core
#'
#' The git revision of the Rust core this package was built from: a full
#' commit hash, suffixed `-dirty` when `crates/core` had uncommitted changes,
#' or `"unknown"` for a build that saw neither a git checkout nor a recorded
#' revision.  The Python package's `pg_core_rev()` reports the same value
#' for a build of the same tree.
#'
#' @return A character string.
#' @examples
#' pg_core_rev()
#' @export
pg_core_rev <- function() .native_pg_core_rev()
