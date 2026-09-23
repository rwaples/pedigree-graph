test_that("the registry has 23 categories in registry order", {
  cats <- relationship_categories()
  expect_identical(nrow(cats), 23L)
  expect_identical(cats$code, c(
    "MZ", "MO", "FO", "FS", "MHS", "PHS", "GP", "Av", "GGP", "HAv", "GAv", "1C",
    "GGGP", "HGAv", "GGAv", "H1C", "1C1R", "G3GP", "HGGAv", "G3Av", "H1C1R", "1C2R", "2C"
  ))
  expect_identical(cats$degree, c(0L, 1L, 1L, 1L, rep(2L, 4), rep(3L, 4), rep(4L, 5), rep(5L, 6)))
  expect_identical(cats$nominal_kinship, 0.5^(cats$degree + 1))
})

test_that("roles exist exactly for the asymmetric categories", {
  cats <- relationship_categories()
  symmetric <- c("MZ", "FS", "MHS", "PHS", "1C", "H1C", "2C")
  expect_identical(is.na(cats$first_role), cats$code %in% symmetric)
  expect_identical(is.na(cats$second_role), cats$code %in% symmetric)
  expect_identical(cats[cats$code == "MO", c("first_role", "second_role")],
                   data.frame(first_role = "offspring", second_role = "mother", row.names = 2L))
})
