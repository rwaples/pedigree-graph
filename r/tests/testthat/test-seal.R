test_that("an unmodified graph passes its seal", {
  pg <- pedigree_graph(transform(nuclear(), sex = c(0L, 1L, 1L, 0L, -1L)))
  expect_identical(pedigreegraph:::.pg_native(pg), pg$native)
})

test_that("an in-range edit to any sealed field is graph_modified", {
  pg <- pedigree_graph(transform(nuclear(), sex = c(0L, 1L, 1L, 0L, -1L), generation = c(0L, 0L, 1L, 1L, 0L)))
  edits <- list(
    id_type = function(x) "double",
    ids = function(x) rev(x),
    mother_ids = function(x) rev(x),
    father_ids = function(x) rev(x),
    twin_ids = function(x) replace(x, 1, 0),
    mother_rows = function(x) replace(x, 5, 0L),
    father_rows = function(x) replace(x, 5, 1L),
    twin_rows = function(x) replace(x, 3:4, c(3L, 2L)),
    depth = function(x) replace(x, 3, 0L),
    sex = function(x) replace(x, 5, 1L),
    generation = function(x) replace(x, 5, 2L),
    birth_year = function(x) 1:5,
    rows_topological = function(x) !x
  )
  expect_setequal(names(edits), names(pg$native))
  for (field in names(edits)) {
    modified <- pg
    modified$native[[field]] <- edits[[field]](pg$native[[field]])
    expect_false(identical(modified$native[[field]], pg$native[[field]]), label = field)
    err <- expect_pg_error(pedigreegraph:::.pg_native(modified), "usage", "graph_modified")
  }
})

test_that("a changed type or a dropped field is graph_modified", {
  pg <- pedigree_graph(nuclear())
  retyped <- pg
  retyped$native$depth <- as.double(pg$native$depth)
  expect_pg_error(pedigreegraph:::.pg_native(retyped), "usage", "graph_modified")
  dropped <- pg
  dropped$native$depth <- NULL
  expect_pg_error(pedigreegraph:::.pg_native(dropped), "usage", "graph_modified")
  resealed <- pg
  resealed$seal <- "pg1:0000000000000000"
  expect_pg_error(pedigreegraph:::.pg_native(resealed), "usage", "graph_modified")
})

test_that("a graph survives saveRDS and readRDS", {
  pg <- pedigree_graph(transform(nuclear(), sex = c(0L, 1L, 1L, 0L, -1L)))
  path <- tempfile(fileext = ".rds")
  on.exit(unlink(path))
  saveRDS(pg, path)
  restored <- readRDS(path)
  expect_identical(restored, pg)
  expect_identical(pedigreegraph:::.pg_native(restored), pg$native)
})

test_that("an integer64 graph survives saveRDS too", {
  df <- nuclear()
  df[] <- lapply(df, as_int64)
  pg <- pedigree_graph(df)
  path <- tempfile(fileext = ".rds")
  on.exit(unlink(path))
  saveRDS(pg, path)
  expect_identical(pedigreegraph:::.pg_native(readRDS(path)), pg$native)
})

test_that("something that is not a graph is a usage error", {
  expect_pg_error(pedigreegraph:::.pg_native(list(native = list())), "usage")
})
