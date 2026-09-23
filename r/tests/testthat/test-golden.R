# Parity with the Python package on the goldens tools/r_golden.py writes.
golden_dir <- test_path("golden")
hex <- function(x) as.numeric(x)
read_golden <- function(..., classes) read.delim(file.path(golden_dir, ...), colClasses = classes)

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
}
