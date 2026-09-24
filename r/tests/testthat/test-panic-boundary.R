# A Rust panic inside a native call reaches R as an error, not an abort
# (ADR 0007).  The probe is a test-only entry point compiled with the
# `test-hooks` feature (`pixi run -e r r-install`).  It runs in a child
# Rscript, so a panic that did abort would fail the child rather than kill the
# test run.  PEDIGREE_GRAPH_REQUIRE_TEST_HOOKS=1 (set by `r-test`) turns a
# missing hook into a failure; only installed-tarball checks may skip.

hook_present <- function() {
  !is.null(get0("wrap__panic_for_test", envir = asNamespace("pedigreegraph"), inherits = FALSE))
}

test_that("a panic signals an R error and the session stays usable", {
  if (!hook_present()) {
    if (identical(Sys.getenv("PEDIGREE_GRAPH_REQUIRE_TEST_HOOKS"), "1")) {
      fail("the package was built without the test-hooks feature; run `pixi run -e r r-install`")
    }
    skip("installed without the test-hooks feature")
  }
  out <- run_rscript(c(
    "hook <- get('wrap__panic_for_test', envir = asNamespace('pedigreegraph'))",
    "e <- tryCatch(.Call(hook), error = function(e) e)",
    "cat('RAISED', inherits(e, 'error'), '\\n')",
    "g <- pedigree_graph(data.frame(id = 1:3, mother = c(NA, NA, 1L), father = c(NA, NA, 2L)))",
    "cat('ALIVE', nrow(relationship_pairs(g, max_degree = 1)), '\\n')"
  ))
  expect_null(attr(out, "status"))
  expect_true(any(grepl("^RAISED TRUE", out)), info = paste(out, collapse = "\n"))
  expect_true(any(grepl("^ALIVE [1-9]", out)), info = paste(out, collapse = "\n"))
})
