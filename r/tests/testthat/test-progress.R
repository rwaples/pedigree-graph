# Progress reporting and interrupts for the long relationship calls
# (ADR 0017).  The engine runs as a job on the package pool while R's loop,
# .pg_watch(), waits on it a tick at a time.  Interrupts and time limits need
# a real signal, so those tests run in a child Rscript.

# `generations` of `per_generation` each, parents drawn from the generation
# before: mothers from its first half, fathers from its second.
long_graph <- function(per_generation, generations, seed = 39L) {
  n <- per_generation * generations
  mother <- father <- rep(NA_integer_, n)
  withr::with_seed(seed, {
    for (g in seq_len(generations - 1L)) {
      parents <- (g - 1L) * per_generation + seq_len(per_generation)
      children <- g * per_generation + seq_len(per_generation)
      half <- per_generation %/% 2L
      mother[children] <- sample(parents[seq_len(half)], per_generation, replace = TRUE)
      father[children] <- sample(parents[-seq_len(half)], per_generation, replace = TRUE)
    }
  })
  pedigree_graph(data.frame(id = seq_len(n), mother = mother, father = father))
}

every_call <- function(pg, progress) {
  list(
    pairs = relationship_pairs(pg, max_degree = 3, progress = progress),
    counts = relationship_counts(pg, max_degree = 5, progress = progress),
    burden = relationship_burden(pg, progress = progress),
    relatives = relatives_per_person(pg, max_degree = 5, progress = progress,
                                     thresholds = list(x = list(seq_len(pg$n) %% 7, 3))),
    moments = relationship_moments(
      pg, max_degree = 2, first = list(p = seq_len(pg$n) %% 2L),
      values = list(x = seq_len(pg$n) %% 7 / 7), progress = progress
    )
  )
}

# The lanes a moments pass ran on follow the thread budget; its table does not.
without_lanes <- function(results) {
  results$moments[c("lanes", "lane_pairs", "estimated_peak_bytes")] <- NULL
  results
}

# Definitions a child Rscript needs, as code.
as_code <- function(...) {
  fns <- list(...)
  vapply(names(fns), function(name) paste(name, "<-", paste(deparse(fns[[name]]), collapse = "\n")), "")
}

child_code <- c(
  as_code(long_graph = long_graph, every_call = every_call),
  "assignInNamespace('.pg_tick', function() 0.05, 'pedigreegraph')",
  "kill_soon <- function(after) system(sprintf(\"sh -c 'sleep %s; kill -INT %d'\", after, Sys.getpid()), wait = FALSE)",
  "clock <- function() proc.time()[['elapsed']]"
)

hook <- function(name) {
  found <- get0(paste0("wrap__", name), envir = asNamespace("pedigreegraph"), inherits = FALSE)
  if (is.null(found)) {
    if (identical(Sys.getenv("PEDIGREE_GRAPH_REQUIRE_TEST_HOOKS"), "1")) {
      fail("the package was built without the test-hooks feature; run `pixi run -e r r-install`")
    }
    skip("installed without the test-hooks feature")
  }
  found
}

phase_rank <- c(preparing = 1L, walking = 2L, finishing = 3L)

test_that("results are identical under every progress argument", {
  pg <- long_graph(150, 4)
  quiet <- every_call(pg, FALSE)
  expect_identical(every_call(pg, TRUE), quiet)
  expect_identical(every_call(pg, function(p) NULL), quiet)
  expect_identical(relationship_counts(pg, categories = character(), progress = FALSE),
                   relationship_counts(pg, categories = character(), progress = TRUE))
})

test_that("results at a budget of 4 threads equal the session's", {
  pg <- long_graph(150, 4)
  here <- without_lanes(every_call(pg, FALSE))
  saved <- tempfile(fileext = ".rds")
  on.exit(unlink(saved))
  out <- run_rscript(c(
    child_code,
    "configure_threads(4)",
    sprintf("saveRDS(every_call(long_graph(150, 4), TRUE), %s)", deparse(saved))
  ))
  expect_null(attr(out, "status"))
  expect_identical(without_lanes(readRDS(saved)), here)
})

test_that("a scripted job reports every phase, normalized, and its result", {
  scripted <- hook("scripted_job_for_test")
  local_mocked_bindings(.pg_tick = function() 0.02, .package = "pedigreegraph")
  handle <- .Call(scripted)
  total <- 3e9
  script <- list(
    list("walking", 0, total), list("walking", total / 2, total), list("walking", total, total),
    list("finishing", 0, total)
  )
  seen <- list()
  report <- .pg_progress(function(p) {
    seen[[length(seen) + 1L]] <<- p
    step <- script[length(seen)][[1]]
    if (is.null(step)) {
      .Call(hook("release_for_test"))
    } else {
      .pg_call(.Call(hook("script_for_test"), handle, step[[1]], step[[2]], step[[3]]))
    }
  }, "test")
  expect_identical(.pg_watch(handle, report), 42L)
  expect_gte(length(seen), 5L)
  observed <- lapply(seen, function(p) p[c("phase", "rows_done", "rows_total")])
  expected <- list(
    list(phase = "preparing", rows_done = NA_real_, rows_total = NA_real_),
    list(phase = "walking", rows_done = 0, rows_total = total),
    list(phase = "walking", rows_done = total / 2, rows_total = total),
    list(phase = "walking", rows_done = total, rows_total = total),
    list(phase = "finishing", rows_done = NA_real_, rows_total = total)
  )
  expect_identical(observed[1:5], expected)
  for (late in observed[-(1:5)]) expect_identical(late, expected[[5]])
  elapsed <- vapply(seen, `[[`, 0, "elapsed")
  expect_true(all(elapsed >= 0) && !is.unsorted(elapsed))
})

test_that("real calls report ordered, well-typed progress", {
  local_mocked_bindings(.pg_tick = function() 0.02, .package = "pedigreegraph")
  pg <- long_graph(2000, 6)
  calls <- list(
    function(progress) relationship_counts(pg, max_degree = 5, progress = progress),
    function(progress) relationship_pairs(pg, max_degree = 4, execution = "memory", ids = FALSE,
                                          progress = progress),
    function(progress) relationship_burden(pg, progress = progress),
    function(progress) relatives_per_person(pg, max_degree = 5, progress = progress),
    function(progress) relationship_moments(pg, max_degree = 5, progress = progress)
  )
  for (call in calls) {
    seen <- list()
    call(function(p) seen[[length(seen) + 1L]] <<- p)
    expect_gt(length(seen), 0L)
    expect_true(all(vapply(seen, function(p) {
      identical(names(p), c("phase", "rows_done", "rows_total", "elapsed")) &&
        is.character(p$phase) && is.double(p$rows_done) && is.double(p$rows_total) &&
        is.double(p$elapsed)
    }, NA)))
    phase <- vapply(seen, `[[`, "", "phase")
    done <- vapply(seen, `[[`, 0, "rows_done")
    total <- vapply(seen, `[[`, 0, "rows_total")
    expect_true(all(phase %in% names(phase_rank)))
    expect_identical(is.na(done), phase != "walking")
    expect_identical(is.na(total), phase == "preparing")
    expect_false(is.unsorted(phase_rank[phase]))
    walking <- phase == "walking"
    expect_true(all(done[walking] <= total[walking]))
    expect_false(is.unsorted(done[walking]))
  }
})

test_that("an engine error from the worker is raised, and the next call works", {
  pg <- pedigree_graph(nuclear())
  expected <- relationship_counts(pg, max_degree = 2)
  m <- relationship_moments(pg, max_degree = 2, first = list(p = c(0L, 1L, 0L, 1L, 0L)),
                            values = list(x = c(0.1, -0.4, 0.7, 0.3, 1.2)))
  one_lane <- 48 * prod(m$shape) * (1 + 4 + 1)
  err <- expect_pg_error(
    relationship_moments(pg, max_degree = 2, first = list(p = c(0L, 1L, 0L, 1L, 0L)),
                         values = list(x = c(0.1, -0.4, 0.7, 0.3, 1.2)),
                         memory_budget_bytes = one_lane - 1),
    "resource", "memory_budget_exceeded"
  )
  expect_identical(err$call[[1]], quote(relationship_moments))
  expect_identical(relationship_counts(pg, max_degree = 2), expected)
})

test_that("an error from the callback stops the call, and the next call works", {
  local_mocked_bindings(.pg_tick = function() 0.02, .package = "pedigreegraph")
  pg <- long_graph(2000, 6)
  small <- pedigree_graph(nuclear())
  expected <- relationship_counts(small, max_degree = 2)
  stop_here <- function(p) stop(structure(class = c("stop_here", "error", "condition"),
                                          list(message = "stop here", call = NULL)))
  expect_error(relationship_counts(pg, max_degree = 5, progress = stop_here), class = "stop_here")
  expect_identical(relationship_counts(small, max_degree = 2), expected)
})

test_that("a callback can start another call at one thread", {
  out <- run_rscript(c(
    child_code,
    "configure_threads(1)",
    "pg <- long_graph(2000, 6)",
    "small <- pedigree_graph(data.frame(id = 1:5, mother = c(NA, NA, 1L, 1L, NA), father = c(NA, NA, 2L, 2L, NA)))",
    "outer <- relationship_counts(pg, max_degree = 5)",
    "inner <- relationship_counts(small, max_degree = 2)",
    "nested <- list()",
    "got <- relationship_counts(pg, max_degree = 5, progress = function(p) nested[[length(nested) + 1L]] <<- relationship_counts(small, max_degree = 2))",
    "cat('NESTED', length(nested) > 0, identical(got, outer), all(vapply(nested, identical, NA, inner)), '\\n')"
  ))
  expect_null(attr(out, "status"))
  expect_true(any(out == "NESTED TRUE TRUE TRUE "), info = paste(out, collapse = "\n"))
})

test_that("the three line forms read as Python's", {
  expect_identical(.pg_format_progress("relationship_counts", "walking", 195758, 783029, 192.4),
                   "relationship_counts: 195,758/783,029 rows (25%) after 3m12s")
  expect_identical(.pg_format_progress("relationship_counts", "preparing", NA_real_, NA_real_, 45.9),
                   "relationship_counts: preparing after 45s")
  expect_identical(.pg_format_progress("relationship_counts", "finishing", NA_real_, 783029, 580),
                   "relationship_counts: all 783,029 rows walked, assembling after 9m40s")
  expect_identical(.pg_format_progress("x", "walking", 199, 1000, 7509),
                   "x: 199/1,000 rows (19%) after 2h05m09s")
  expect_identical(.pg_format_progress("x", "walking", 0, 0, 0), "x: 0/0 rows (100%) after 0s")
  expect_identical(.pg_format_progress("x", "walking", 5, NA_real_, 60), "x: preparing after 1m00s")
})

test_that("TRUE prints a line every 30 seconds, which suppressMessages() silences", {
  now <- 100
  local_mocked_bindings(.pg_clock = function() now, .package = "pedigreegraph")
  report <- .pg_progress(TRUE, "relationship_counts")
  state <- list(finished = FALSE, phase = "walking", rows_done = 10, rows_total = 100)
  now <- 129.9
  expect_no_message(report(state))
  now <- 130
  expect_message(report(state), "^relationship_counts: 10/100 rows \\(10%\\) after 30s\n$")
  now <- 159
  expect_no_message(report(state))
  now <- 160
  expect_message(report(state), "after 1m00s")
  now <- 190
  expect_no_message(suppressMessages(report(state)))
  expect_null(.pg_progress(FALSE, "relationship_counts"))
})

test_that("progress must be TRUE, FALSE or a function, and its default is the option", {
  pg <- pedigree_graph(nuclear())
  for (bad in list(NA, "yes", 1, c(TRUE, TRUE), NULL)) {
    err <- expect_pg_error(relationship_counts(pg, max_degree = 1, progress = bad), "usage")
    expect_identical(err$call[[1]], quote(relationship_counts))
  }
  withr::local_options(pedigreegraph.progress = "yes")
  expect_pg_error(relationship_burden(pg), "usage")
  withr::local_options(pedigreegraph.progress = FALSE)
  expect_identical(relationship_burden(pg), relationship_burden(pg, progress = TRUE))
})

test_that("Ctrl-C interrupts a long call, and the next call works", {
  out <- run_rscript(c(
    child_code,
    "configure_threads(1)",
    "pg <- long_graph(8000, 6)",
    "small <- long_graph(100, 3)",
    "expected <- relationship_counts(small, max_degree = 5)",
    "start <- clock(); invisible(relationship_counts(pg, max_degree = 5)); full <- clock() - start",
    "start <- clock()",
    "got <- tryCatch({ kill_soon(full / 4); relationship_counts(pg, max_degree = 5); 'finished' }, interrupt = function(e) 'interrupted')",
    "took <- clock() - start",
    "cat('FULL', full, '\\n')",
    "cat('INTERRUPT', got, took, took < full * 3 / 4, '\\n')",
    "cat('AFTER', identical(relationship_counts(small, max_degree = 5), expected), '\\n')"
  ))
  info <- paste(out, collapse = "\n")
  expect_null(attr(out, "status"))
  expect_true(any(grepl("^INTERRUPT interrupted [0-9.]+ TRUE", out)), info = info)
  expect_true(any(out == "AFTER TRUE "), info = info)
})

test_that("an uncaught interrupt halts Rscript", {
  out <- suppressWarnings(run_rscript(c(
    child_code,
    "configure_threads(1)",
    "pg <- long_graph(8000, 6)",
    "kill_soon(0.5)",
    "relationship_counts(pg, max_degree = 5)",
    "cat('NOT REACHED\\n')"
  )))
  expect_identical(attr(out, "status"), 1L)
  expect_true(any(out == "Execution halted"), info = paste(out, collapse = "\n"))
  expect_false(any(grepl("NOT REACHED", out)))
})

test_that("a time limit stops a long call as R's error, and the next call works", {
  out <- run_rscript(c(
    child_code,
    "configure_threads(1)",
    "pg <- long_graph(8000, 6)",
    "small <- long_graph(100, 3)",
    "expected <- relationship_counts(small, max_degree = 5)",
    "start <- clock(); invisible(relationship_counts(pg, max_degree = 5)); full <- clock() - start",
    "got <- local({ setTimeLimit(elapsed = full / 4, transient = TRUE); tryCatch(relationship_counts(pg, max_degree = 5), error = function(e) conditionMessage(e), interrupt = function(e) 'interrupt') })",
    "cat('LIMIT', got, '\\n')",
    "cat('AFTER', identical(relationship_counts(small, max_degree = 5), expected), '\\n')"
  ))
  info <- paste(out, collapse = "\n")
  expect_null(attr(out, "status"))
  expect_true(any(out == "LIMIT reached elapsed time limit "), info = info)
  expect_true(any(out == "AFTER TRUE "), info = info)
})

test_that("a panic in a job is an R error and the session stays usable", {
  hook("panicking_job_for_test")
  out <- run_rscript(c(
    "ns <- asNamespace('pedigreegraph')",
    "handle <- .Call(get('wrap__panicking_job_for_test', envir = ns))",
    "e <- tryCatch(ns$.pg_watch(handle, NULL), error = function(e) e)",
    "cat('RAISED', inherits(e, 'error'), '\\n')",
    "cat('CANCEL', is.null(ns$.native_cancel(handle)), '\\n')",
    "g <- pedigree_graph(data.frame(id = 1:3, mother = c(NA, NA, 1L), father = c(NA, NA, 2L)))",
    "cat('ALIVE', relationship_counts(g, max_degree = 1)[['MO']], '\\n')"
  ))
  info <- paste(out, collapse = "\n")
  expect_null(attr(out, "status"))
  expect_true(any(out == "RAISED TRUE "), info = info)
  expect_true(any(out == "CANCEL TRUE "), info = info)
  expect_true(any(out == "ALIVE 1 "), info = info)
})

test_that("garbage collecting an uncollected handle cancels its job", {
  start <- hook("cancellable_job_for_test")
  cancelled <- hook("was_cancelled_for_test")
  local(.Call(start))
  gc()
  waited <- 0
  while (!.Call(cancelled) && waited < 5) {
    Sys.sleep(0.01)
    waited <- waited + 0.01
  }
  expect_true(.Call(cancelled))
})

test_that("a call is cancelled when anything after its start fails", {
  start <- hook("cancellable_job_for_test")
  cancelled <- hook("was_cancelled_for_test")
  local_mocked_bindings(.pg_watch = function(...) stop("after the start"), .package = "pedigreegraph")
  expect_error(.pg_run(FALSE, "test", function() .Call(start)), "after the start")
  expect_true(.Call(cancelled))
})

test_that("cancel is a no-op twice and after collect; a second collect is refused", {
  scripted <- hook("scripted_job_for_test")
  local_mocked_bindings(.pg_tick = function() 0.02, .package = "pedigreegraph")
  handle <- .Call(scripted)
  .Call(hook("release_for_test"))
  expect_identical(.pg_watch(handle, NULL), 42L)
  expect_null(.pg_call(.native_cancel(handle)))
  expect_null(.pg_call(.native_cancel(handle)))
  expect_pg_error(.pg_call(.native_collect(handle)), "usage")

  running <- .Call(scripted)
  expect_null(.pg_call(.native_cancel(running)))
  expect_null(.pg_call(.native_cancel(running)))
  expect_true(.native_wait(running, 0.01)$finished)
  expect_pg_error(.pg_call(.native_collect(running)), "usage")
})

test_that("a bad tick or handle is a usage error", {
  scripted <- hook("scripted_job_for_test")
  handle <- .Call(scripted)
  on.exit(.native_cancel(handle))
  for (tick in c(0, -1, NaN, Inf)) expect_pg_error(.pg_call(.native_wait(handle, tick)), "usage")
  expect_pg_error(.pg_call(.native_wait(list(), 1)), "usage")
  expect_pg_error(.pg_call(.native_collect(NULL)), "usage")
})
