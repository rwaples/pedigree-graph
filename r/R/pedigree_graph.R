#' Build a pedigree graph
#'
#' Validates a pedigree and returns a graph the other functions take.
#'
#' @param data A data frame (including tibbles and data.tables) or a named
#'   list with columns `id`, `mother` and `father`, and optionally `twin`,
#'   `sex`, `generation` and `birth_year`.  Other columns are ignored.
#'   Columns may be integer, whole-number double, or `bit64::integer64`; `NA`
#'   marks a missing parent or unknown value, and `id` may not be `NA`.
#' @param sex_encoding `"simace"` (0 female, 1 male, -1 unknown) or
#'   `"plink"` (2 female, 1 male, 0 unknown).
#' @return An object of class `pedigree_graph`.  `$id` holds the ids in the
#'   type they were given; `$native` and `$seal` are internal and must not be
#'   edited (a modified graph is refused with code `graph_modified`).
#' @export
pedigree_graph <- function(data, sex_encoding = "simace") {
  if (!is.list(data)) {
    .pg_usage("`data` must be a data frame or a named list of columns", call = sys.call())
  }
  if (!is.character(sex_encoding) || length(sex_encoding) != 1L || is.na(sex_encoding)) {
    sex_encoding <- ""
  }
  column <- function(name) if (name %in% names(data)) data[[name]] else NULL
  built <- .pg_call(.native_build_pedigree(
    column("id"), column("mother"), column("father"), column("twin"),
    column("sex"), column("generation"), column("birth_year"), sex_encoding
  ))
  structure(
    list(n = length(built$id), id = built$id, native = built$native, seal = built$seal),
    class = "pedigree_graph"
  )
}

# The graph's native fields after their seal checks out; every kernel starts here.
.pg_native <- function(pg, call = sys.call(-1L)) {
  if (!inherits(pg, "pedigree_graph")) {
    .pg_usage("`pg` must be a graph built by pedigree_graph()", call = call)
  }
  .pg_call(.native_check_graph(pg$native, pg$seal), call = call)
  pg$native
}

#' @export
print.pedigree_graph <- function(x, ...) {
  native <- .pg_native(x)
  founders <- sum(native$mother_rows < 0L & native$father_rows < 0L)
  optional <- c("sex", "generation", "birth_year")
  present <- optional[!vapply(native[optional], is.null, logical(1L))]
  if (any(native$twin_rows >= 0L)) present <- c("twin", present)
  cat(sprintf(
    "<pedigree_graph> %d individuals, %d founders, max depth %d\n",
    x$n, founders, if (x$n) max(native$depth) else 0L
  ))
  cat("  ids:", native$id_type, "\n")
  cat("  optional columns:", if (length(present)) paste(present, collapse = ", ") else "none", "\n")
  invisible(x)
}
