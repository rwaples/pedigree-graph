# Three generations: founders 1, 2, 5; children 3, 4 of (1, 2); 6 of (3, 5).
three_gen <- function() {
  data.frame(
    id = 1:6,
    mother = c(NA, NA, 1L, 1L, NA, 3L),
    father = c(NA, NA, 2L, 2L, NA, 5L)
  )
}

test_that("exactly one selector is required", {
  pg <- pedigree_graph(three_gen())
  expect_pg_error(relationship_pairs(pg), "usage")
  expect_pg_error(relationship_pairs(pg, max_degree = 2, categories = "FS"), "usage")
})

test_that("max_degree outside 0-5 is max_degree_out_of_range", {
  pg <- pedigree_graph(three_gen())
  err <- expect_pg_error(relationship_pairs(pg, max_degree = 6), "validation", "max_degree_out_of_range")
  expect_identical(err$fields, list(value = 6, minimum = 0, maximum = 5))
  expect_pg_error(relationship_pairs(pg, max_degree = -1), "validation", "max_degree_out_of_range")
  expect_pg_error(relationship_pairs(pg, max_degree = 1.5), "usage")
  expect_pg_error(relationship_pairs(pg, max_degree = "2"), "usage")
})

test_that("unknown codes are reported sorted, NA included", {
  pg <- pedigree_graph(three_gen())
  err <- expect_pg_error(
    relationship_pairs(pg, categories = c("FS", "ZZ", NA, "AA", "ZZ")),
    "validation", "unknown_relationship_category"
  )
  expect_identical(err$fields$codes, c("AA", "NA", "ZZ"))
})

test_that("categories select codes; empty categories select nothing", {
  pg <- pedigree_graph(three_gen())
  fs <- relationship_pairs(pg, categories = "FS")
  expect_identical(as.character(fs$code), "FS")
  expect_identical(sum(attr(fs, "requested")), 1L)
  none <- relationship_pairs(pg, categories = character())
  expect_identical(nrow(none), 0L)
  expect_identical(levels(none$code), relationship_categories()$code)
  expect_false(any(attr(none, "requested")))
})

test_that("max_degree 0 requests MZ only", {
  pg <- pedigree_graph(three_gen())
  expect_identical(names(which(attr(relationship_pairs(pg, max_degree = 0), "requested"))), "MZ")
})

test_that("execution memory gives the same frame as speed; other values are usage errors", {
  pg <- pedigree_graph(three_gen())
  expect_identical(
    relationship_pairs(pg, max_degree = 5, execution = "memory"),
    relationship_pairs(pg, max_degree = 5)
  )
  expect_pg_error(relationship_pairs(pg, max_degree = 5, execution = "fast"), "usage")
  expect_pg_error(relationship_pairs(pg, max_degree = 5, ids = NA), "usage")
})

test_that("id columns follow the graph's id type, and ids = FALSE drops them", {
  df <- three_gen()
  expect_type(relationship_pairs(pedigree_graph(df), max_degree = 1)$first_id, "integer")
  dbl <- df
  dbl[] <- lapply(dbl, as.double)
  expect_type(relationship_pairs(pedigree_graph(dbl), max_degree = 1)$first_id, "double")
  i64 <- df
  i64[] <- lapply(i64, as_int64)
  pairs <- relationship_pairs(pedigree_graph(i64), max_degree = 1)
  expect_s3_class(pairs$first_id, "integer64")
  expect_identical(unclass(pairs$first_id), unclass(as_int64(df$id[pairs$first])))
  expect_named(relationship_pairs(pedigree_graph(df), max_degree = 1, ids = FALSE),
               c("code", "first", "second"))
})

test_that("asymmetric pairs keep their roles and symmetric ones are ordered", {
  pairs <- relationship_pairs(pedigree_graph(three_gen()), max_degree = 2)
  mo <- pairs[pairs$code == "MO", ]
  expect_identical(mo$first, c(3L, 4L, 6L))
  expect_identical(mo$second, c(1L, 1L, 3L))
  fs <- pairs[pairs$code == "FS", ]
  expect_true(all(fs$first < fs$second))
})

test_that("graphs with fewer than two individuals have no pairs", {
  for (n in 0:1) {
    pg <- pedigree_graph(data.frame(id = seq_len(n), mother = rep(NA, n), father = rep(NA, n)))
    pairs <- relationship_pairs(pg, max_degree = 5)
    expect_identical(nrow(pairs), 0L)
    expect_true(all(attr(pairs, "requested")))
  }
})

test_that("pair_kinship takes 1-based integer or double rows and validates them", {
  pg <- pedigree_graph(three_gen())
  expect_identical(pair_kinship(pg, c(3L, 1L, 6L), c(4L, 1L, 1L)), c(0.25, 0.5, 0.125))
  expect_identical(pair_kinship(pg, c(3, 1), c(4, 1)), c(0.25, 0.5))
  expect_identical(pair_kinship(pg, integer(), integer()), numeric())
  err <- expect_pg_error(pair_kinship(pg, c(1L, 7L), c(1L, 1L)), "validation", "value_out_of_range")
  expect_identical(err$fields[c("field", "position", "value")], list(field = "first", position = 2, value = 7L))
  expect_pg_error(pair_kinship(pg, c(1L, NA), c(1L, 1L)), "validation", "value_out_of_range")
  expect_pg_error(pair_kinship(pg, 0.5, 1), "validation", "value_out_of_range")
  expect_pg_error(pair_kinship(pg, 1:2, 1L), "usage")
  expect_pg_error(pair_kinship(pg, "1", "1"), "usage")
})

test_that("inbreeding is 2 * self-kinship - 1", {
  df <- data.frame(id = 1:5, mother = c(NA, NA, 1L, 1L, 3L), father = c(NA, NA, 2L, 2L, 4L))
  pg <- pedigree_graph(df)
  expect_identical(inbreeding(pg), 2 * pair_kinship(pg, 1:5, 1:5) - 1)
  expect_identical(inbreeding(pg)[5], 0.25)
  expect_identical(inbreeding(pedigree_graph(data.frame(id = integer(), mother = integer(), father = integer()))),
                   numeric())
})

test_that("the kinship matrix is a named dsCMatrix whose diagonal is 1 + F over 2", {
  pg <- pedigree_graph(three_gen())
  K <- kinship_matrix(pg)
  expect_s4_class(K, "dsCMatrix")
  expect_identical(dimnames(K), list(as.character(1:6), as.character(1:6)))
  expect_identical(unname(Matrix::diag(K)), (1 + inbreeding(pg)) / 2)
  expect_identical(K["6", "1"], 0.125)
  expect_identical(dim(kinship_matrix(pedigree_graph(data.frame(id = integer(), mother = integer(), father = integer())))),
                   c(0L, 0L))
})

test_that("integer64 ids name the matrix exactly", {
  big <- structure(readBin(writeBin(as.integer(c(1, 2^28, 2, 2^28)), raw(), endian = "little"),
                           "double", 2L, endian = "little"), class = "integer64")
  pg <- pedigree_graph(list(id = big, mother = c(NA, NA), father = c(NA, NA)))
  expect_identical(rownames(kinship_matrix(pg)), c("1152921504606846977", "1152921504606846978"))
})

test_that("a matrix past its entry cap is a resource error, never truncated", {
  pg <- pedigree_graph(three_gen())
  nnz <- as.double(length(kinship_matrix(pg)@x))
  expect_s4_class(pedigreegraph:::.pg_kinship_matrix(pg, nnz), "dsCMatrix")
  err <- expect_pg_error(pedigreegraph:::.pg_kinship_matrix(pg, nnz - 1), "resource", "csc_index_overflow")
  expect_identical(err$fields, list(nnz = nnz, maximum = nnz - 1))
})

test_that("every kernel refuses a modified graph", {
  pg <- pedigree_graph(three_gen())
  pg$native$depth[6] <- 1L
  expect_pg_error(relationship_pairs(pg, max_degree = 2), "usage", "graph_modified")
  expect_pg_error(pair_kinship(pg, 1L, 1L), "usage", "graph_modified")
  expect_pg_error(inbreeding(pg), "usage", "graph_modified")
  expect_pg_error(kinship_matrix(pg), "usage", "graph_modified")
  expect_pg_error(relationship_pairs(unclass(pg), max_degree = 2), "usage")
})

test_that("forked workers run pair queries after the parent used its pool", {
  skip_on_os("windows")
  out <- run_rscript(c(
    "configure_threads(2)",
    "df <- data.frame(id = 1:6, mother = c(NA, NA, 1L, 1L, NA, 3L), father = c(NA, NA, 2L, 2L, NA, 5L))",
    "pg <- pedigree_graph(df)",
    "parent <- nrow(relationship_pairs(pg, max_degree = 3))",
    "kids <- parallel::mclapply(1:4, function(i) nrow(relationship_pairs(pg, max_degree = 3)), mc.cores = 2)",
    "cat(parent, unlist(kids))"
  ), timeout = 120)
  expect_identical(out, "10 10 10 10 10")
})
