test_that("pg_core_rev is a git revision", {
  expect_match(pg_core_rev(), "^[0-9a-f]{40}(-dirty)?$")
})
