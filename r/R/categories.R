#' The relationship categories
#'
#' One row per category in registry order: its `code`, kinship `degree`,
#' the semantic roles of the `first` and `second` pair members (`NA` for a
#' symmetric category, whose pairs have `first < second`), and its
#' `nominal_kinship`.
#'
#' @return A data frame with 23 rows.
#' @examples
#' relationship_categories()
#' @export
relationship_categories <- function() {
  as.data.frame(.native_relationship_categories(), stringsAsFactors = FALSE)
}

# The 23 codes in registry order, the levels of every pair frame's `code`.
.pg_codes <- function() .native_relationship_categories()$code
