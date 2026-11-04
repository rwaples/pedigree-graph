#' The relationship categories
#'
#' One row per category in registry order: its `code`, kinship `degree`,
#' the semantic roles of the `first` and `second` pair members (`NA` for a
#' symmetric category, whose pairs have `first < second`), its
#' `nominal_kinship`, and its path shape: `up`, the meioses from `first` up
#' to the shared ancestor(s), `down`, the meioses from them down to
#' `second`, and `ancestor_count`, the paths of that shape joining the pair
#' (2 full, 1 half or lineal, 0 for MZ).
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
