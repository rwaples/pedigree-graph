test_that("integer, double and integer64 ids build the same graph and keep their type", {
  base <- pedigree_graph(nuclear())
  as_double <- nuclear()
  as_double[] <- lapply(as_double, as.double)
  as_i64 <- nuclear()
  as_i64[] <- lapply(as_i64, as_int64)

  from_double <- pedigree_graph(as_double)
  from_i64 <- pedigree_graph(as_i64)

  expect_identical(base$id, 1:5)
  expect_identical(from_double$id, as.double(1:5))
  expect_s3_class(from_i64$id, "integer64")
  expect_identical(unclass(from_i64$id), unclass(as_int64(1:5)))
  for (field in c("ids", "mother_ids", "father_ids", "mother_rows", "father_rows", "depth")) {
    expect_identical(from_double$native[[field]], base$native[[field]], label = field)
    expect_identical(from_i64$native[[field]], base$native[[field]], label = field)
  }
})

test_that("NA is a missing parent, including an all-NA logical column", {
  pg <- pedigree_graph(data.frame(id = 1:3, mother = NA, father = NA))
  expect_identical(pg$native$mother_rows, rep(-1L, 3))
  expect_identical(pg$native$father_rows, rep(-1L, 3))
})

test_that("ids above 2^31 build from doubles", {
  big <- 2^40 + 0:2
  pg <- pedigree_graph(data.frame(id = big, mother = c(NA, NA, big[1]), father = c(NA, NA, big[2])))
  expect_identical(pg$id, big)
  expect_identical(pg$native$mother_rows, c(-1L, -1L, 0L))
})

test_that("a named list works as well as a data frame, and extra columns are ignored", {
  pg <- pedigree_graph(list(id = 1:2, mother = c(NA, 1L), father = c(NA, NA), note = c("a", "b")))
  expect_identical(pg$n, 2L)
})

test_that("values with no lossless integer form are invalid_integer_value at a 1-based position", {
  cases <- list(
    list(mother = c(NA, NA, 1.5, 1, NA), value = 1.5, position = 3),
    list(mother = c(NA, NA, Inf, 1, NA), value = Inf, position = 3),
    list(mother = c(NA, TRUE, NA, NA, NA), value = TRUE, position = 2),
    list(mother = c("a", "b", "c", "d", "e"), value = "character", position = 1),
    list(mother = factor(1:5), value = "factor", position = 1)
  )
  for (case in cases) {
    df <- nuclear()
    df$mother <- case$mother
    err <- expect_pg_error(pedigree_graph(df), "validation", "invalid_integer_value")
    expect_identical(err$fields$field, "mother")
    expect_identical(err$fields$position, case$position)
    expect_identical(err$fields$value, case$value)
  }
})

test_that("an NA id is rejected", {
  df <- nuclear()
  df$id[4] <- NA
  err <- expect_pg_error(pedigree_graph(df), "validation", "invalid_integer_value")
  expect_identical(err$fields[c("field", "position")], list(field = "id", position = 4))
})

test_that("a missing required column is missing_field", {
  err <- expect_pg_error(pedigree_graph(data.frame(id = 1:2, mother = NA)), "validation", "missing_field")
  expect_identical(err$fields$field, "father")
})

test_that("core validation errors report 1-based rows", {
  err <- expect_pg_error(
    pedigree_graph(data.frame(id = c(7, 8, 7), mother = NA, father = NA)),
    "validation", "duplicate_id"
  )
  expect_identical(err$fields$id, 7)
  expect_identical(err$fields$rows, c(1, 3))

  err <- expect_pg_error(
    pedigree_graph(data.frame(id = c(1, -2), mother = NA, father = NA)),
    "validation", "value_out_of_range"
  )
  expect_identical(err$fields$position, 2)

  expect_pg_error(
    pedigree_graph(data.frame(id = 1:2, mother = c(2L, 1L), father = NA)),
    "validation", "cycle"
  )
})

test_that("sex_encoding is simace or plink, anything else is a usage error", {
  df <- transform(nuclear(), sex = c(0L, 1L, 1L, 0L, -1L))
  expect_identical(pedigree_graph(df)$native$sex, c(0L, 1L, 1L, 0L, -1L))
  plink <- transform(nuclear(), sex = c(2L, 1L, 1L, 2L, 0L))
  expect_identical(pedigree_graph(plink, sex_encoding = "plink")$native$sex, c(0L, 1L, 1L, 0L, -1L))
  expect_pg_error(pedigree_graph(df, sex_encoding = "other"), "usage")
  expect_pg_error(pedigree_graph(df, sex_encoding = NA), "usage")
})

test_that("non-list input is a usage error", {
  expect_pg_error(pedigree_graph(1:3), "usage")
})

test_that("depth and print describe the graph", {
  pg <- pedigree_graph(nuclear())
  expect_identical(pg$native$depth, c(0L, 0L, 1L, 1L, 0L))
  expect_output(print(pg), "5 individuals, 3 founders, max depth 1")
})

test_that("an empty pedigree builds", {
  pg <- pedigree_graph(data.frame(id = integer(), mother = integer(), father = integer()))
  expect_identical(pg$n, 0L)
  expect_output(print(pg), "0 individuals")
})

test_that("an id above 2^53 in an error field comes back exact, as integer64", {
  # 2^60 and 2^60 + 1 differ only below a double's precision.
  big <- structure(readBin(writeBin(as.integer(c(0, 2^28, 1, 2^28, 0, 2^28)), raw(), endian = "little"),
                           "double", 3L, endian = "little"), class = "integer64")
  err <- expect_pg_error(
    pedigree_graph(list(id = big, mother = rep(NA, 3), father = rep(NA, 3))),
    "validation", "duplicate_id"
  )
  expect_s3_class(err$fields$id, "integer64")
  expect_identical(unclass(err$fields$id), unclass(big)[1])
  expect_identical(err$fields$rows, c(1, 3))
})
