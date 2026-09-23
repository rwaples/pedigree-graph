# A nuclear family: founders 1 and 2, children 3 and 4, and unrelated 5.
nuclear <- function() {
  data.frame(
    id = 1:5,
    mother = c(NA, NA, 1L, 1L, NA),
    father = c(NA, NA, 2L, 2L, NA)
  )
}

# Non-negative whole numbers below 2^31 as bit64::integer64, without bit64:
# each value's little-endian int64 bits as a low and a high int32 word.
# bit64's NA is INT64_MIN: low word 0, high word 0x80000000 (NA_integer_).
as_int64 <- function(x) {
  low <- ifelse(is.na(x), 0L, as.integer(x))
  high <- ifelse(is.na(x), NA_integer_, 0L)
  words <- as.vector(rbind(low, high))
  structure(readBin(writeBin(words, raw(), endian = "little"), "double", length(x), endian = "little"),
            class = "integer64")
}

expect_pg_error <- function(expr, class, code = NULL) {
  err <- expect_error(expr, class = paste0("pedigree_graph_", class, "_error"))
  if (!is.null(code)) expect_identical(err$code, code)
  invisible(err)
}

# Run R code in a fresh process, for state committed once per process.
run_rscript <- function(code, env = character(), timeout = 60) {
  script <- tempfile(fileext = ".R")
  on.exit(unlink(script))
  writeLines(c("library(pedigreegraph)", code), script)
  system2(file.path(R.home("bin"), "Rscript"), script, stdout = TRUE, stderr = TRUE, env = env,
          timeout = timeout)
}
