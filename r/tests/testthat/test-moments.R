# relationship_moments() and its algebra (ADR 0013, ADR 0015).  Bit-for-bit
# parity with Python is test-golden.R; these hold the R surface.

three_gen <- function() {
  data.frame(id = 1:6, mother = c(NA, NA, 1L, 1L, NA, 3L), father = c(NA, NA, 2L, 2L, NA, 5L))
}
x <- c(0.1, -0.4, 0.7, 0.3, 1.2, -0.2)

axis_of <- function(m, name) m$axes[[match(name, vapply(m$axes, `[[`, "", "name"))]]

test_that("a factor enters as its codes and keeps the labels that occur", {
  pg <- pedigree_graph(three_gen())
  f <- factor(c("b", "b", "c", "c", "b", "c"), levels = c("a", "b", "c", "d"))
  m <- relationship_moments(pg, categories = c("MO", "FS"), first = list(f = f), values = list(x = x))
  axis <- axis_of(m, "first_f")
  expect_identical(axis$levels, 2:3)
  expect_identical(axis$labels, c("b", "c"))
  expect_identical(axis_of(m, "second_f")$labels, c("b", "c"))
  df <- as.data.frame(m, stats = "n")
  expect_identical(levels(df$first_f), c("b", "c"))
  expect_identical(sum(df$n), 4)
})

test_that("integer, logical and whole doubles are factors; character and NA are refused", {
  pg <- pedigree_graph(three_gen())
  for (g in list(c(1L, 1L, 2L, 2L, 1L, 3L), c(1, 1, 2, 2, 1, 3), c(TRUE, TRUE, FALSE, FALSE, TRUE, FALSE))) {
    m <- relationship_moments(pg, max_degree = 1, first = list(g = g))
    expect_identical(axis_of(m, "first_g")$levels, sort(unique(g)))
  }
  expect_pg_error(relationship_moments(pg, max_degree = 1, first = list(g = letters[1:6])), "usage")
  expect_pg_error(relationship_moments(pg, max_degree = 1, first = list(g = c(1, 2, NA, 1, 1, 1))), "usage")
  expect_pg_error(relationship_moments(pg, max_degree = 1, first = list(g = c(1.5, 1, 1, 1, 1, 1))), "usage")
  expect_pg_error(relationship_moments(pg, max_degree = 1, first = list(g = factor(c("a", NA, "a", "a", "a", "a")))),
                  "usage")
  err <- expect_pg_error(relationship_moments(pg, max_degree = 1, first = list(g = 1:5)), "validation", "length_mismatch")
  expect_identical(err$fields$field, "first['g']")
})

test_that("NA in an equality key is unknown and never equal", {
  pg <- pedigree_graph(three_gen())
  m <- relationship_moments(pg, categories = "FS", same = list(k = c(NA, NA, NA, NA, 1L, 1L)))
  df <- as.data.frame(m, stats = "n")
  expect_identical(df$n[df$same_k == 1L], 0)
  same_key <- relationship_moments(pg, categories = "FS", same = list(k = c(NA, NA, 7L, 7L, 1L, 1L)))
  expect_identical(as.data.frame(same_key, stats = "n")$n, c(0, 1))
})

test_that("values must be finite, counted from row 1", {
  pg <- pedigree_graph(three_gen())
  expect_error(
    relationship_moments(pg, max_degree = 1, values = list(x = c(1, 2, NaN, 4, 5, 6))),
    "values\\['x'\\] is not finite at row 3", class = "pedigree_graph_usage_error"
  )
  expect_pg_error(relationship_moments(pg, max_degree = 1, values = list(x = c(1, 2, NA, 4, 5, 6))), "usage")
  expect_pg_error(relationship_moments(pg, max_degree = 1, values = list(x = x),
                                       products = list(c("first.x", "third.x"))), "usage")
  expect_pg_error(relationship_moments(pg, max_degree = 1, symmetric = "neither"), "usage")
})

test_that("a classed values column is refused, not read as its codes, bits or offsets", {
  pg <- pedigree_graph(three_gen())
  run <- function(column) relationship_moments(pg, max_degree = 1, values = list(x = column))
  expect_error(run(as_int64(1:6)), "values\\['x'\\] must be a numeric vector, not class integer64",
               class = "pedigree_graph_usage_error")
  expect_pg_error(run(factor(1:6)), "usage")
  expect_pg_error(run(as.Date(x)), "usage")
  expect_pg_error(run(as.difftime(x, units = "days")), "usage")
  expect_no_error(run(as.double(as.Date(x))))
})

test_that("select takes labels on a factor axis and values elsewhere, in the order given", {
  pg <- pedigree_graph(three_gen())
  f <- factor(c("u", "v", "u", "v", "u", "v"))
  m <- relationship_moments(pg, max_degree = 2, first = list(f = f, p = c(0L, 1L, 0L, 1L, 0L, 1L)),
                            values = list(x = x))
  s <- moments_select(m, category = c("GP", "MO"), first_f = "v", first_p = 1L)
  expect_identical(s$shape[1:3], c(2L, 1L, 1L))
  expect_identical(axis_of(s, "category")$levels, c("GP", "MO"))
  expect_identical(axis_of(s, "first_f")$labels, "v")
  expect_identical(s$lanes, m$lanes)
  expect_pg_error(moments_select(m, first_f = 2L), "usage")
  expect_pg_error(moments_select(m, first_f = "w"), "usage")
  expect_pg_error(moments_select(m, category = c("MO", "MO")), "usage")
  expect_pg_error(moments_select(m, nothing = 1), "usage")
})

test_that("a fold equals a direct call over the pooled cells, bit for bit", {
  pg <- pedigree_graph(three_gen())
  m <- relationship_moments(pg, categories = c("MO", "FO"), first = list(p = c(0L, 1L, 0L, 1L, 0L, 1L)),
                            values = list(x = x))
  folded <- moments_sum(m, "first_p", "second_p")
  direct <- relationship_moments(pg, categories = c("MO", "FO"), values = list(x = x))
  expect_identical(as.data.frame(folded), as.data.frame(direct))
  expect_identical(folded$lanes, m$lanes)
  expect_identical(moments_merge(m, m)$lanes, 0)
  all_folded <- moments_sum(folded, "category")
  expect_identical(nrow(as.data.frame(all_folded)), 1L)
})

test_that("merge needs one factor mapping, one layout and one rule", {
  pg <- pedigree_graph(three_gen())
  f <- factor(c("u", "v", "u", "v", "u", "v"))
  g <- factor(c("v", "u", "v", "u", "v", "u"), levels = c("v", "u"))
  a <- relationship_moments(pg, max_degree = 1, first = list(f = f), values = list(x = x))
  b <- relationship_moments(pg, max_degree = 1, first = list(f = g), values = list(x = x))
  err <- expect_pg_error(moments_merge(a, b), "usage")
  expect_match(conditionMessage(err), "axis 'first_f'")
  expect_pg_error(moments_merge(a, relationship_moments(pg, max_degree = 1, first = list(f = f))), "usage")
  expect_pg_error(moments_merge(a, relationship_moments(pg, max_degree = 1, first = list(f = f), values = list(x = x),
                                                         symmetric = "both")), "usage")
  doubled <- moments_merge(a, a)
  expect_identical(as.data.frame(doubled, stats = "n")$n, 2 * as.data.frame(a, stats = "n")$n)
  expect_identical(as.data.frame(doubled, stats = "mean_first"), as.data.frame(a, stats = "mean_first"))
})

test_that("a result survives saveRDS and readRDS", {
  pg <- pedigree_graph(three_gen())
  m <- relationship_moments(pg, max_degree = 2, values = list(x = x))
  path <- tempfile(fileext = ".rds")
  saveRDS(m, path)
  expect_identical(as.data.frame(readRDS(path)), as.data.frame(m))
})

test_that("a product side code other than 0 or 1 is a malformed object, not the second member", {
  pg <- pedigree_graph(three_gen())
  m <- relationship_moments(pg, max_degree = 2, values = list(x = x))
  expect_identical(m$operands, c(0L, 0L, 1L, 0L))
  for (code in c(2L, 256L, -1L)) {
    bad <- m
    bad$operands[3] <- code
    err <- expect_pg_error(as.data.frame(bad), "usage")
    expect_match(conditionMessage(err), "malformed relationship_moments object")
  }
})

test_that("constant large values: sums and means succeed, squares raise, centered moments are zero", {
  pg <- pedigree_graph(three_gen())
  m <- relationship_moments(pg, categories = c("MO", "FS"), values = list(big = rep(1e200, 6)))
  for (stat in c("sumsq_first", "sumsq_second", "cross")) {
    err <- expect_pg_error(as.data.frame(m, stats = stat), "usage")
    expect_match(conditionMessage(err), paste0("^", stat, " of '.*big.*' is not representable"))
    expect_match(conditionMessage(err), sprintf("stats = setdiff\\(stats, \"%s\"\\)", stat))
  }
  df <- as.data.frame(m, stats = c("n", "sum_first", "mean_first", "m2_first", "m2_second", "comoment", "pearson"))
  expect_true(all(df$sum_first.big[df$n > 0] > 0))
  expect_true(all(df$m2_first.big == 0) && all(df$m2_second.big == 0))
  expect_true(all(df$`comoment.first.big:second.big` == 0))
  expect_true(all(is.nan(df$`pearson.first.big:second.big`)))
})

test_that("varying large values: centered moments raise too, a pearson still succeeds", {
  pg <- pedigree_graph(three_gen())
  # Both members vary within MO (offspring 3, 4, 6; mothers 1, 1, 3) and FO (fathers 2, 2, 5).
  big <- c(1e200, 2e200, 2e200, 1e200, 1e200, 2e200)
  m <- relationship_moments(pg, categories = c("MO", "FO"), values = list(big = big))
  for (stat in c("sumsq_first", "sumsq_second", "cross", "m2_first", "m2_second", "comoment")) {
    expect_pg_error(as.data.frame(m, stats = stat), "usage")
  }
  df <- as.data.frame(m, stats = c("n", "sum_first", "mean_second", "pearson"))
  expect_true(all(is.finite(df$sum_first.big)))
  r <- df$`pearson.first.big:second.big`
  expect_true(any(is.finite(r)) && all(abs(r[is.finite(r)]) <= 1))
})

# A one-column table over `n` category cells with the given counts and zero sums.
counts_table <- function(counts) {
  codes <- c("FS", "MO", "FO", "MZ")[seq_along(counts)]
  exact <- as.vector(rbind(counts, matrix("0", nrow = 5, ncol = length(counts))))
  pedigreegraph:::.new_relationship_moments(
    axes = list(list(name = "category", levels = codes)), columns = "x",
    products = list(c("first.x", "second.x")), exponents = 0L, exact = exact
  )
}

test_that("n is exported to 2^53 and refused past it, the rest still derived", {
  ok <- counts_table("9007199254740992")
  expect_identical(as.data.frame(ok, stats = "n")$n, 2^53)
  past <- counts_table("9007199254740993")
  expect_pg_error(as.data.frame(past, stats = "n"), "resource", "count_exceeds_double")
  expect_identical(as.data.frame(past, stats = "sum_first")$sum_first.x, 0)
})

test_that("counts are checked per cell and totals are exact", {
  full <- counts_table("9223372036854775807")
  expect_pg_error(moments_merge(full, counts_table("1")), "resource", "arithmetic_overflow")
  expect_pg_error(moments_sum(counts_table(c("4611686018427387904", "4611686018427387904")), "category"),
                  "resource", "arithmetic_overflow")
  three <- counts_table(rep("9223372036854775807", 3))
  expect_output(print(three), "pairs: +27670116110564327421")
  expect_pg_error(counts_table("-1"), "usage")
})

test_that("both hosts plan the same budget: lanes of 16 bytes plus 32 per accumulator", {
  pg <- pedigree_graph(three_gen())
  m <- relationship_moments(pg, max_degree = 2, first = list(p = c(0L, 1L, 0L, 1L, 0L, 1L)), values = list(x = x))
  accumulators <- prod(m$shape) * (1 + 4 + 1)
  expect_identical(m$estimated_peak_bytes, m$lanes * 16 * accumulators + 32 * accumulators)
  one_lane <- 16 * accumulators + 32 * accumulators
  err <- expect_pg_error(
    relationship_moments(pg, max_degree = 2, first = list(p = c(0L, 1L, 0L, 1L, 0L, 1L)), values = list(x = x),
                         memory_budget_bytes = one_lane - 1),
    "resource", "memory_budget_exceeded"
  )
  expect_identical(err$fields$estimated_bytes, one_lane)
})

test_that("print shows the shape, columns, products and exact pairs", {
  pg <- pedigree_graph(three_gen())
  m <- relationship_moments(pg, categories = c("MO", "FS"), values = list(x = x))
  expect_output(print(m), "category=2")
  expect_output(print(m), "first.x x second.x")
  expect_output(print(m), "pairs: +4")
})
