# Parity with the Python package on the goldens tools/r_golden.py writes.
golden_dir <- test_path("golden")
hex <- function(x) as.numeric(x)
read_golden <- function(..., classes) read.delim(file.path(golden_dir, ...), colClasses = classes)

# The threshold columns of tools/r_golden.py::_relatives_spec.
relatives_spec <- function(n) {
  rows <- as.double(seq_len(n))
  list(
    rowwise = list(ifelse(rows %% 3 == 0, NaN, (rows * 37) %% 11), (rows * 5) %% 11),
    scalar = list(rows %% 4, 2)
  )
}

test_that("the registry matches the Python registry", {
  want <- read.delim(file.path(golden_dir, "categories.tsv"),
                     colClasses = c(nominal_kinship = "character"))
  want$nominal_kinship <- hex(want$nominal_kinship)
  expect_identical(relationship_categories(), want)
})

fixtures <- list.dirs(golden_dir, full.names = FALSE, recursive = FALSE)

test_that("every golden fixture is present", {
  expect_true(length(fixtures) >= 15)
})

for (fixture in fixtures) {
  test_that(paste("pairs, kinship and inbreeding match Python on", fixture), {
    pg <- pedigree_graph(read.delim(file.path(golden_dir, fixture, "pedigree.tsv")))

    want <- read.delim(
      file.path(golden_dir, fixture, "pairs.tsv"),
      colClasses = c("character", "integer", "integer", "integer", "integer")
    )
    got <- relationship_pairs(pg, max_degree = 5)
    expect_identical(as.character(got$code), want$code)
    for (column in c("first", "second", "first_id", "second_id")) {
      expect_identical(got[[column]], want[[column]], label = column)
    }
    expect_true(all(attr(got, "requested")))

    want <- read_golden(fixture, "kinship.tsv", classes = c("integer", "integer", "character"))
    K <- kinship_matrix(pg)
    expect_s4_class(K, "dsCMatrix")
    expect_identical(K@uplo, "U")
    stored <- Matrix::summary(K)
    stored <- stored[order(stored$j, stored$i), ]
    expect_identical(as.integer(stored$i), want$i)
    expect_identical(as.integer(stored$j), want$j)
    expect_identical(stored$x, hex(want$x))
    expect_identical(pair_kinship(pg, want$i, want$j), hex(want$x))

    want <- read_golden(fixture, "inbreeding.tsv", classes = "character")
    expect_identical(inbreeding(pg), hex(want$F))
  })

  test_that(paste("counts and burden match Python on", fixture), {
    pg <- pedigree_graph(read.delim(file.path(golden_dir, fixture, "pedigree.tsv")))

    want <- read_golden(fixture, "counts.tsv", classes = c("character", "numeric"))
    counts <- relationship_counts(pg, max_degree = 5)
    expect_identical(names(counts), want$code)
    expect_identical(as.vector(counts), want$count)
    expect_true(all(attr(counts, "requested")))

    burden <- relationship_burden(pg)
    want_rows <- read_golden(fixture, "burden.tsv", classes = rep("integer", 5))
    expect_identical(burden$per_person, `dimnames<-`(as.matrix(want_rows), list(NULL, names(want_rows))))
    expect_identical(burden$category_counts, stats::setNames(want$count, want$code))
    want <- read_golden(fixture, "burden_depth.tsv", classes = c("integer", "numeric"))
    expect_identical(burden$same_depth_pairs, want$pairs)
  })

  test_that(paste("relatives per person match Python on", fixture), {
    ped <- read.delim(file.path(golden_dir, fixture, "pedigree.tsv"))
    r <- relatives_per_person(pedigree_graph(ped), max_degree = 5, thresholds = relatives_spec(nrow(ped)))
    expect_identical(dimnames(r)[[2]], relationship_categories()$code)
    want <- read_golden(fixture, "relatives.tsv", classes = c("integer", "character", rep("integer", 3)))
    at <- which(matrix(r[, , "relatives"], nrow = dim(r)[1]) > 0, arr.ind = TRUE)
    at <- at[order(at[, 1], at[, 2]), , drop = FALSE]
    expect_identical(unname(at[, 1]), want$row)
    expect_identical(dimnames(r)[[2]][at[, 2]], want$code)
    for (column in c("relatives", "rowwise", "scalar")) {
      k <- rep(match(column, dimnames(r)[[3]]), nrow(at))
      expect_identical(r[cbind(at, k)], want[[column]], label = column)
    }
  })
}

# The spec of tools/r_golden.py::_moments_spec, built the same way here.
moments_of <- function(pg, ped, symmetric, scale = 1) {
  n <- nrow(ped)
  rows <- as.double(seq_len(n))
  tie <- (rows %% 7) - 2.5
  tie[1] <- 2^43
  values <- list(
    ordinary = ((rows * 37) %% 101) / 7 - 5, constant = rep(0.3, n), tie = tie,
    large = 1e150 * (((rows * 13) %% 17) - 8)
  )
  relationship_moments(
    pg, categories = c("MZ", "FS", "MO", "FO", "MHS", "PHS", "GP", "Av", "1C"),
    first = list(parity = pg$native$depth %% 2L,
                 code = factor(c("r0", "r1")[(seq_len(n) - 1L) %% 2L + 1L], levels = c("r0", "r1"))),
    values = lapply(values, function(v) v * scale),
    products = list(c("first.ordinary", "second.ordinary"), c("first.constant", "second.constant"),
                    c("first.tie", "second.tie"), c("first.large", "second.large"),
                    c("first.ordinary", "first.tie")),
    same = list(mother = ped$mother), symmetric = symmetric
  )
}

# The cells holding pairs, every column as the golden's text.
moment_rows <- function(m, symmetric) {
  df <- as.data.frame(m)
  df <- df[df$n > 0, , drop = FALSE]
  axes <- vapply(m$axes, `[[`, character(1), "name")
  for (column in names(df)) {
    df[[column]] <- if (column %in% c(axes, "n")) as.character(df[[column]]) else df[[column]]
  }
  cbind(symmetric = rep(symmetric, nrow(df)), df, stringsAsFactors = FALSE)
}

expect_moments_frame <- function(got, fixture, file) {
  want <- read.delim(file.path(golden_dir, fixture, file), colClasses = "character", check.names = FALSE)
  expect_identical(names(got), names(want))
  for (column in names(want)) {
    expected <- if (is.character(got[[column]])) want[[column]] else as.numeric(want[[column]])
    expect_identical(unname(got[[column]]), expected, label = paste(file, column))
  }
}

for (fixture in fixtures) {
  test_that(paste("moments match Python bit for bit on", fixture), {
    ped <- read.delim(file.path(golden_dir, fixture, "pedigree.tsv"))
    pg <- pedigree_graph(ped)
    frames <- list(moments = NULL, exact = NULL, folded = NULL, merged = NULL)
    for (symmetric in c("canonical", "both")) {
      m <- moments_of(pg, ped, symmetric)
      frames$moments <- rbind(frames$moments, moment_rows(m, symmetric))
      stride <- 1 + 4 * length(m$columns) + length(m$products)
      exact <- matrix(pedigreegraph:::.moments_exact(m), nrow = stride)
      counts <- as.numeric(exact[1, ])
      kept <- as.vector(exact[, counts > 0])
      frames$exact <- rbind(frames$exact, data.frame(symmetric = rep(symmetric, length(kept)), value = kept))
      folded <- moments_sum(moments_select(m, category = c("MO", "FO")), "category")
      frames$folded <- rbind(frames$folded, moment_rows(folded, symmetric))
      merged <- moments_merge(m, moments_of(pg, ped, symmetric, scale = 2^-600))
      frames$merged <- rbind(frames$merged, moment_rows(merged, symmetric))
    }
    expect_moments_frame(frames$moments, fixture, "moments.tsv")
    expect_moments_frame(frames$exact, fixture, "moments_exact.tsv")
    expect_moments_frame(frames$folded, fixture, "moments_folded.tsv")
    expect_moments_frame(frames$merged, fixture, "moments_merged.tsv")
  })
}
