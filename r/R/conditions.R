# Failures come back from Rust as data (src/rust/src/errors.rs); this is the
# one place they become R conditions.  Rust never raises into R.

.pg_call <- function(value, call = sys.call(-1L)) {
  if (!inherits(value, "pedigree_graph_native_error")) {
    return(value)
  }
  code <- value$code
  condition <- structure(
    class = c(
      paste0("pedigree_graph_", value$class, "_error"),
      "pedigree_graph_error", "error", "condition"
    ),
    list(
      message = if (is.na(code)) value$message else paste0(value$message, " [", code, "]"),
      call = call,
      code = code,
      fields = value$fields
    )
  )
  stop(condition)
}
