#' Relationship pairs
#'
#' Every pair of individuals in the selected relationship categories, each
#' pair under its closest category.
#'
#' @param pg A graph from [pedigree_graph()].
#' @param max_degree Select every category at or below this degree (0-5).
#' @param categories Or select these category codes (see
#'   [relationship_categories()]).  Give exactly one of the two.
#' @param execution `"speed"` (default) or `"memory"`, which uses about half
#'   the peak memory at roughly twice the time.  The result is identical.
#' @param ids Include `first_id` and `second_id` columns (default `TRUE`).
#' @return A data frame with one row per pair: `code`, a factor whose levels
#'   are all 23 category codes in registry order (so `table(pairs$code)`
#'   shows empty categories too); `first` and `second`, 1-based rows of the
#'   input; and, with `ids = TRUE`, `first_id` and `second_id` in the id
#'   type the graph was built from.  Rows are in registry order, then by pair.
#'   For an asymmetric category `first` holds the first role (for `MO`, the
#'   offspring; see [relationship_categories()]); for a symmetric one
#'   `first < second`.  `attr(pairs, "requested")` is a named logical over
#'   the 23 codes; a code that was not requested has no rows.
#' @examples
#' pg <- pedigree_graph(data.frame(
#'   id = 1:6, mother = c(NA, NA, 1, 1, NA, 3), father = c(NA, NA, 2, 2, NA, 5)
#' ))
#' pairs <- relationship_pairs(pg, max_degree = 3)
#' pairs
#' table(pairs$code)[1:8]
#' relationship_pairs(pg, categories = c("FS", "GP"), ids = FALSE)
#' @export
relationship_pairs <- function(pg, max_degree = NULL, categories = NULL,
                               execution = "speed", ids = TRUE) {
  native <- .pg_native(pg)
  if (!is.character(execution) || length(execution) != 1L || is.na(execution)) execution <- ""
  if (!is.logical(ids) || length(ids) != 1L || is.na(ids)) {
    .pg_usage("`ids` must be TRUE or FALSE")
  }
  if (!is.null(max_degree) && !is.numeric(max_degree)) max_degree <- NaN
  found <- .pg_call(.native_relationship_pairs(
    native, pg$seal, max_degree, categories, execution, ids
  ))
  codes <- .pg_codes()
  columns <- found[setdiff(names(found), "requested")]
  columns$code <- structure(columns$code, levels = codes, class = "factor")
  n <- length(columns$code)
  structure(
    columns,
    class = "data.frame",
    row.names = if (n) c(NA_integer_, -n) else integer(),
    requested = stats::setNames(found$requested, codes)
  )
}

#' Relationship counts
#'
#' The number of pairs in each selected relationship category, each pair
#' under its closest category, without building the pairs.
#'
#' @inheritParams relationship_pairs
#' @return A named double vector over all 23 category codes in registry
#'   order: the pair count of each selected category (equal to its number of
#'   rows in [relationship_pairs()]) and `NA` for every other.
#'   `attr(counts, "requested")` is a named logical over the 23 codes.  A
#'   count above 2^53, past which a double does not hold every integer, is
#'   refused with a `pedigree_graph_resource_error` (code
#'   `count_exceeds_double`).
#' @examples
#' pg <- pedigree_graph(data.frame(
#'   id = 1:6, mother = c(NA, NA, 1, 1, NA, 3), father = c(NA, NA, 2, 2, NA, 5)
#' ))
#' relationship_counts(pg, categories = c("FS", "MO", "FO", "GP"))
#' @export
relationship_counts <- function(pg, max_degree = NULL, categories = NULL) {
  native <- .pg_native(pg)
  if (!is.null(max_degree) && !is.numeric(max_degree)) max_degree <- NaN
  found <- .pg_call(.native_relationship_counts(native, pg$seal, max_degree, categories))
  codes <- .pg_codes()
  structure(
    stats::setNames(found$counts, codes),
    requested = stats::setNames(found$requested, codes)
  )
}

#' Relationship burden
#'
#' Every individual's relatives at degrees 1 to 5 and the pairs of every
#' category, from one pass that never builds the pairs.
#'
#' @param pg A graph from [pedigree_graph()].
#' @return A list:
#'   * `per_person`: an integer matrix with one row per input row, in input
#'     order, and columns `degree_1` to `degree_5`, the distinct relatives
#'     of that degree.  MZ co-twins are not counted here.
#'   * `category_counts`: a named double vector of the pairs of each of the
#'     23 categories (MZ included), as [relationship_counts()] with
#'     `max_degree = 5`.
#'   * `same_depth_pairs`: a double vector of the related pairs whose two
#'     members share a structural depth; element `d + 1` is depth `d`.
#'
#'   A total above 2^53 is refused as in [relationship_counts()].
#' @examples
#' pg <- pedigree_graph(data.frame(
#'   id = 1:6, mother = c(NA, NA, 1, 1, NA, 3), father = c(NA, NA, 2, 2, NA, 5)
#' ))
#' burden <- relationship_burden(pg)
#' burden$per_person
#' burden$category_counts[c("FS", "MO", "FO")]
#' @export
relationship_burden <- function(pg) {
  native <- .pg_native(pg)
  found <- .pg_call(.native_relationship_burden(native, pg$seal))
  dimnames(found$per_person) <- list(NULL, paste0("degree_", 1:5))
  names(found$category_counts) <- .pg_codes()
  found
}

#' Pairwise kinship
#'
#' The pedigree-expected kinship of each pair `(first[k], second[k])`.
#'
#' @param pg A graph from [pedigree_graph()].
#' @param first,second 1-based rows of the input, of equal length (for
#'   example the `first` and `second` columns of [relationship_pairs()]).
#' @return A double vector, one kinship per pair; values are the core's
#'   float32 recurrence, exactly representable as doubles.
#' @examples
#' pg <- pedigree_graph(data.frame(
#'   id = 1:6, mother = c(NA, NA, 1, 1, NA, 3), father = c(NA, NA, 2, 2, NA, 5)
#' ))
#' pairs <- relationship_pairs(pg, max_degree = 2)
#' pairs$kinship <- pair_kinship(pg, pairs$first, pairs$second)
#' pairs
#' @export
pair_kinship <- function(pg, first, second) {
  native <- .pg_native(pg)
  .pg_call(.native_pair_kinship(native, pg$seal, first, second))
}

#' Inbreeding coefficients
#'
#' @param pg A graph from [pedigree_graph()].
#' @return A double vector of `F`, one per input row, in input order.
#' @examples
#' # 5 is the child of full sibs 3 and 4.
#' pg <- pedigree_graph(data.frame(
#'   id = 1:5, mother = c(NA, NA, 1, 1, 3), father = c(NA, NA, 2, 2, 4)
#' ))
#' inbreeding(pg)
#' @export
inbreeding <- function(pg) {
  native <- .pg_native(pg)
  .pg_call(.native_inbreeding(native, pg$seal))
}

#' Kinship matrix
#'
#' Every nonzero pedigree-expected kinship, plus the diagonal.
#'
#' @param pg A graph from [pedigree_graph()].
#' @return A symmetric sparse `Matrix::dsCMatrix` (upper triangle stored),
#'   rows and columns in input order and named by id.  A matrix with more
#'   than 2^31 - 1 stored entries is refused with a
#'   `pedigree_graph_resource_error` (code `csc_index_overflow`).
#' @examples
#' pg <- pedigree_graph(data.frame(
#'   id = 1:6, mother = c(NA, NA, 1, 1, NA, 3), father = c(NA, NA, 2, 2, NA, 5)
#' ))
#' K <- kinship_matrix(pg)
#' K
#' K["6", "1"]
#' @export
kinship_matrix <- function(pg) .pg_kinship_matrix(pg, NULL)

# `max_nnz` lowers the entry cap for tests.
.pg_kinship_matrix <- function(pg, max_nnz, call = sys.call(-1L)) {
  native <- .pg_native(pg, call = call)
  slots <- .pg_call(.native_kinship_matrix(native, pg$seal, max_nnz), call = call)
  n <- length(slots$names)
  methods::new(
    "dsCMatrix",
    i = slots$i, p = slots$p, x = slots$x, Dim = c(n, n), uplo = "U",
    Dimnames = list(slots$names, slots$names)
  )
}
