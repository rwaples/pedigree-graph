test_that("the budget defaults to 1", {
  out <- run_rscript(c("Sys.unsetenv('PEDIGREE_GRAPH_THREADS')", "cat(thread_budget())"))
  expect_identical(out, "1")
})

test_that("an empty PEDIGREE_GRAPH_THREADS is a usage error, as in Python", {
  out <- run_rscript(
    "e <- tryCatch(thread_budget(), pedigree_graph_usage_error = function(e) 'usage'); cat(e)",
    env = "PEDIGREE_GRAPH_THREADS="
  )
  expect_identical(out, "usage")
})

test_that("PEDIGREE_GRAPH_THREADS sets the budget", {
  expect_identical(run_rscript("cat(thread_budget())", env = "PEDIGREE_GRAPH_THREADS=3"), "3")
})

test_that("configure_threads wins over the environment and commits on first use", {
  out <- run_rscript(c(
    "configure_threads(2)",
    "cat(thread_budget(), '')",
    "configure_threads(2)",
    "e <- tryCatch(configure_threads(4), pedigree_graph_thread_conflict_error = function(e) e)",
    "cat(class(e)[1])"
  ), env = "PEDIGREE_GRAPH_THREADS=3")
  expect_identical(out, "2 pedigree_graph_thread_conflict_error")
})

test_that("configuring again before first use replaces the value", {
  out <- run_rscript(c("configure_threads(2)", "configure_threads(5)", "cat(thread_budget())"))
  expect_identical(out, "5")
})

test_that("a bad budget is a usage error", {
  for (n in list(0, 1.5, -1, NA, "2", c(1, 2), TRUE)) {
    expect_pg_error(configure_threads(n), "usage")
  }
  out <- run_rscript(
    "e <- tryCatch(thread_budget(), pedigree_graph_usage_error = function(e) 'usage'); cat(e)",
    env = "PEDIGREE_GRAPH_THREADS=two"
  )
  expect_identical(out, "usage")
})

test_that("a budget beyond an R integer is a usage error, not a wrapped value", {
  expect_pg_error(configure_threads(2^31), "usage")
  out <- run_rscript(
    "e <- tryCatch(thread_budget(), pedigree_graph_usage_error = function(e) 'usage'); cat(e)",
    env = "PEDIGREE_GRAPH_THREADS=3000000000"
  )
  expect_identical(out, "usage")
})
