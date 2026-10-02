#' Relationship moments
#'
#' Pair counts and value moments per relationship category and pair-label
#' cell, from one pass over the pedigree that never builds the pairs.  Every
#' integer is exact and every statistic is derived from them by the Rust
#' core the Python package shares, so both give the same floats bit for bit.
#'
#' Each pair of a category is counted once, in the category's orientation
#' (`symmetric = "canonical"`: for an asymmetric category the first role,
#' for a symmetric one the lower row first), or, for symmetric categories
#' under `symmetric = "both"`, once per orientation.
#'
#' @inheritParams relationship_pairs
#' @param first A named list of factors over the first member of a pair:
#'   factors, integer, logical or whole-number double vectors, one value per
#'   input row, no `NA`.  Character vectors are refused; convert them with
#'   [factor()].  Each becomes an axis `first_<name>` over the values that
#'   occur (a factor's unused levels are dropped).
#' @param second The same over the second member; `NULL` (default) reuses
#'   `first`.
#' @param values A named list of numeric columns, finite, one value per
#'   input row.  Each value is held to 43 significant bits of its column's
#'   largest magnitude.
#' @param products The cross products to accumulate, a list of length-2
#'   character vectors such as `c("first.x", "second.x")`; `NULL` (default)
#'   is `first.<c>` by `second.<c>` for every value column.
#' @param same A named list of equality keys: each adds an axis
#'   `same_<name>` that is 1 where both members have the same key and 0
#'   otherwise.  `NA` and negative keys are unknown and never equal.
#' @param symmetric `"canonical"` (default) or `"both"`.
#' @param memory_budget_bytes The accumulator memory the pass may use;
#'   fewer lanes run when a lane per thread would not fit, and a call that
#'   cannot fit one lane is refused with a `pedigree_graph_resource_error`
#'   (code `memory_budget_exceeded`).
#' @return A `relationship_moments` object.  Narrow it with
#'   [moments_select()], fold axes away with [moments_sum()], combine two
#'   with [moments_merge()], and read its statistics with
#'   `as.data.frame()`.  It holds the exact accumulators as a raw vector, so
#'   it saves and loads with [saveRDS()].
#' @examples
#' pg <- pedigree_graph(data.frame(
#'   id = 1:6, mother = c(NA, NA, 1, 1, NA, 3), father = c(NA, NA, 2, 2, NA, 5)
#' ))
#' x <- c(0.1, -0.4, 0.7, 0.3, 1.2, -0.2)
#' m <- relationship_moments(pg, categories = c("FS", "MO", "FO"), values = list(x = x))
#' m
#' as.data.frame(m, stats = c("n", "mean_first", "pearson"))
#' @export
relationship_moments <- function(pg, max_degree = NULL, categories = NULL,
                                 first = list(), second = NULL, values = list(),
                                 products = NULL, same = list(), symmetric = "canonical",
                                 memory_budget_bytes = 2^30,
                                 progress = getOption("pedigreegraph.progress", TRUE)) {
  native <- .pg_native(pg)
  if (!is.null(max_degree) && !is.numeric(max_degree)) max_degree <- NaN
  if (!is.character(symmetric) || length(symmetric) != 1L || is.na(symmetric)) symmetric <- ""
  if (!is.numeric(memory_budget_bytes) || length(memory_budget_bytes) != 1L) {
    .pg_usage("`memory_budget_bytes` must be one number")
  }
  first <- .pg_factor_list("first", first)
  if (!is.null(second)) second <- .pg_factor_list("second", second)
  report <- .pg_progress(progress, "relationship_moments")
  handle <- .pg_call(.native_start_moments(
    native, pg$seal, max_degree, categories, first, second, values, products, same,
    symmetric, as.double(memory_budget_bytes)
  ))
  found <- .pg_watch(handle, report)
  axes <- unname(c(
    list(list(name = "category", levels = found$categories)),
    .pg_factor_axes("first", first, found$first_levels),
    .pg_factor_axes("second", if (is.null(second)) first else second,
                    if (is.null(second)) found$first_levels else found$second_levels),
    lapply(names(same), function(key) list(name = paste0("same_", key), levels = 0:1))
  ))
  .pg_moments(
    axes = axes, columns = found$columns, products = found$products, operands = found$operands,
    exponents = found$exponents, width = found$width, table = found$table, symmetric = symmetric,
    lanes = found$lanes, lane_pairs = found$lane_pairs,
    estimated_peak_bytes = found$estimated_peak_bytes
  )
}

# A named list as given, each column checked for a type the core takes.
.pg_factor_list <- function(kind, columns, call = sys.call(-1L)) {
  if (!is.list(columns)) .pg_usage(sprintf("`%s` must be a named list of columns", kind), call)
  for (name in names(columns)) {
    if (is.character(columns[[name]])) {
      .pg_usage(sprintf("%s['%s'] is character; convert it with factor()", kind, name), call)
    }
  }
  columns
}

# One axis per factor: its occurring values, in the input's type, and a
# factor's labels for them.
.pg_factor_axes <- function(kind, columns, levels) {
  Map(function(name, column, occurring) {
    axis <- list(name = paste0(kind, "_", name))
    if (is.factor(column)) {
      axis$levels <- as.integer(occurring)
      axis$labels <- levels(column)[axis$levels]
    } else if (is.logical(column)) {
      axis$levels <- occurring == 1
    } else if (is.integer(column)) {
      axis$levels <- as.integer(occurring)
    } else {
      axis$levels <- occurring
    }
    axis
  }, names(columns), columns, levels)
}

# The object, its accumulators a raw matrix of one accumulator per column.
.pg_moments <- function(axes, columns, products, operands, exponents, width, table, symmetric,
                        lanes = 0, lane_pairs = numeric(), estimated_peak_bytes = 0) {
  shape <- vapply(axes, function(a) length(a$levels), integer(1))
  dim(table) <- c(width, length(table) / width)
  structure(
    list(
      axes = axes, columns = columns, products = products, shape = shape,
      n_columns = length(columns), operands = as.integer(operands), exponents = as.integer(exponents),
      width = as.integer(width), table = table, symmetric = symmetric, lanes = lanes,
      lane_pairs = lane_pairs, estimated_peak_bytes = estimated_peak_bytes
    ),
    class = "relationship_moments"
  )
}

# A table with every accumulator given as a decimal string, cell by cell in
# row-major order over the axes: count, sums of the first member's columns,
# of the second's, their sums of squares, then the cross sums.  For tests.
.new_relationship_moments <- function(axes, columns, products, exponents, exact,
                                      symmetric = "canonical") {
  side <- function(name) if (startsWith(name, "first.")) 0L else 1L
  column <- function(name) match(sub("^(first|second)\\.", "", name), columns) - 1L
  operands <- unlist(lapply(products, function(p) c(side(p[1]), column(p[1]), side(p[2]), column(p[2]))))
  encoded <- .pg_call(.native_moments_encode(as.character(exact)))
  m <- .pg_moments(axes, columns, products, if (is.null(operands)) integer() else operands, exponents,
                   encoded$width, encoded$table, symmetric)
  .pg_call(.native_moments_total(m))
  m
}

# The same object over new axes and a table core returned.
.pg_moments_like <- function(m, axes, out, keep_pass = FALSE) {
  .pg_moments(
    axes = axes, columns = m$columns, products = m$products, operands = m$operands,
    exponents = out$exponents, width = out$width, table = out$table, symmetric = m$symmetric,
    lanes = if (keep_pass) m$lanes else 0, lane_pairs = if (keep_pass) m$lane_pairs else numeric(),
    estimated_peak_bytes = if (keep_pass) m$estimated_peak_bytes else 0
  )
}

.pg_check_moments <- function(m, call = sys.call(-1L)) {
  if (!inherits(m, "relationship_moments")) {
    .pg_usage("expected a relationship_moments object from relationship_moments()", call)
  }
}

.pg_axis_index <- function(m, name, call = sys.call(-1L)) {
  names <- vapply(m$axes, `[[`, character(1), "name")
  at <- match(name, names)
  if (is.na(at)) {
    .pg_usage(sprintf("no axis '%s'; the axes are %s", name, paste(names, collapse = ", ")), call)
  }
  at
}

#' Narrow, fold and combine relationship moments
#'
#' `moments_select()` keeps the given levels of named axes, every axis
#' kept; `moments_sum()` folds named axes away by adding their cells;
#' `moments_merge()` adds two results over the same axes cell by cell.  All
#' three are exact integer operations, so the statistics of a fold equal
#' those of a direct call over the pooled cells bit for bit.
#'
#' @param m,a,b `relationship_moments` objects.
#' @param ... For `moments_select()`, `axis = levels` pairs: a factor axis
#'   takes labels, any other axis its values, kept in the order given.  For
#'   `moments_sum()`, axis names.
#' @return A `relationship_moments` object.  A cell count that would pass
#'   2^63 - 1 is refused with a `pedigree_graph_resource_error` (code
#'   `arithmetic_overflow`).  `moments_merge()` refuses results whose axes,
#'   factor labels, columns, products or `symmetric` rule differ; columns
#'   quantized at different scales are aligned exactly.
#' @examples
#' pg <- pedigree_graph(data.frame(
#'   id = 1:6, mother = c(NA, NA, 1, 1, NA, 3), father = c(NA, NA, 2, 2, NA, 5)
#' ))
#' x <- c(0.1, -0.4, 0.7, 0.3, 1.2, -0.2)
#' m <- relationship_moments(pg, categories = c("MO", "FO"), values = list(x = x))
#' po <- moments_sum(moments_select(m, category = c("MO", "FO")), "category")
#' as.data.frame(po, stats = c("n", "pearson"))
#' @name moments_algebra
NULL

#' @rdname moments_algebra
#' @export
moments_select <- function(m, ...) {
  .pg_check_moments(m)
  levels <- list(...)
  if (length(levels) && (is.null(names(levels)) || any(names(levels) == ""))) {
    .pg_usage("moments_select() takes axis = levels arguments")
  }
  for (name in names(levels)) {
    at <- .pg_axis_index(m, name)
    axis <- m$axes[[at]]
    wanted <- levels[[name]]
    known <- if (is.null(axis$labels)) axis$levels else axis$labels
    if (!is.null(axis$labels) && !is.character(wanted)) {
      .pg_usage(sprintf("axis '%s' is a factor; select its levels by label", name))
    }
    positions <- match(wanted, known)
    if (anyNA(positions)) {
      .pg_usage(sprintf("%s is not a level of axis '%s'", format(wanted[is.na(positions)][1]), name))
    }
    if (anyDuplicated(positions)) .pg_usage(sprintf("moments_select(%s = ...) names a level more than once", name))
    out <- .pg_call(.native_moments_select(m, at, positions))
    axis$levels <- axis$levels[positions]
    if (!is.null(axis$labels)) axis$labels <- axis$labels[positions]
    axes <- m$axes
    axes[[at]] <- axis
    m <- .pg_moments_like(m, axes, out, keep_pass = TRUE)
  }
  m
}

#' @rdname moments_algebra
#' @export
moments_sum <- function(m, ...) {
  .pg_check_moments(m)
  names <- c(...)
  if (!is.null(names) && !is.character(names)) .pg_usage("moments_sum() takes axis names")
  for (name in names) {
    at <- .pg_axis_index(m, name)
    out <- .pg_call(.native_moments_sum(m, at))
    m <- .pg_moments_like(m, m$axes[-at], out, keep_pass = TRUE)
  }
  m
}

#' @rdname moments_algebra
#' @export
moments_merge <- function(a, b) {
  .pg_check_moments(a)
  .pg_check_moments(b)
  same_axes <- length(a$axes) == length(b$axes) && all(mapply(function(x, y) {
    identical(x$name, y$name) && identical(x$levels, y$levels)
  }, a$axes, b$axes))
  if (!same_axes || !identical(a$columns, b$columns) || !identical(a$products, b$products)) {
    .pg_usage("moments_merge() needs two results with the same axes, columns and products")
  }
  for (i in seq_along(a$axes)) {
    if (!identical(a$axes[[i]]$labels, b$axes[[i]]$labels)) {
      .pg_usage(sprintf("moments_merge() needs one factor mapping on axis '%s'; the labels differ",
                        a$axes[[i]]$name))
    }
  }
  if (!identical(a$symmetric, b$symmetric)) {
    .pg_usage(sprintf("moments_merge() needs one symmetric rule, got '%s' and '%s'", a$symmetric, b$symmetric))
  }
  .pg_moments_like(a, a$axes, .pg_call(.native_moments_merge(a, b)))
}

.pg_stats <- c("n", "sum_first", "sum_second", "sumsq_first", "sumsq_second", "mean_first",
               "mean_second", "m2_first", "m2_second", "cross", "comoment", "pearson")

#' Relationship moments as a data frame
#'
#' One row per cell, with a column per axis and then the chosen statistics.
#'
#' @param x A `relationship_moments` object.
#' @param row.names,optional Ignored.
#' @param stats The statistics, any of `"n"` (pairs per cell), per value
#'   column `"sum_first"`, `"sum_second"`, `"sumsq_first"`,
#'   `"sumsq_second"`, `"mean_first"`, `"mean_second"`, `"m2_first"` and
#'   `"m2_second"` (centered second moments), and per product `"cross"`,
#'   `"comoment"` (centered co-moment) and `"pearson"`.  Columns are named
#'   `<stat>.<column>` and `<stat>.<a>:<b>`.
#' @param ... Ignored.
#' @return A data frame.  The `category` column and factor axes are
#'   factors over the axis levels; other axes keep their values; `same_*`
#'   columns are 0 or 1.  Every float is the exact rational rounded once.  A
#'   statistic past the double range is refused with a
#'   `pedigree_graph_usage_error` naming it and the column; `stats` without
#'   it gives the rest.  A cell count above 2^53 is refused for `"n"` only.
#' @export
as.data.frame.relationship_moments <- function(x, row.names = NULL, optional = FALSE, ...,
                                               stats = .pg_stats) {
  m <- x
  bad <- setdiff(stats, .pg_stats)
  if (length(bad)) .pg_usage(sprintf("unknown statistics: %s", paste(bad, collapse = ", ")))
  cells <- prod(m$shape)
  out <- list()
  stride <- 1
  for (i in rev(seq_along(m$axes))) {
    axis <- m$axes[[i]]
    index <- ((seq_len(cells) - 1) %/% stride) %% length(axis$levels) + 1
    stride <- stride * length(axis$levels)
    out[[axis$name]] <- if (axis$name == "category") {
      factor(axis$levels[index], levels = axis$levels)
    } else if (!is.null(axis$labels)) {
      factor(axis$labels[index], levels = axis$labels)
    } else {
      axis$levels[index]
    }
  }
  out <- out[rev(names(out))]
  derive <- function(stat, index, name) {
    tryCatch(
      .pg_call(.native_moments_derive(m, stat, index, name)),
      pedigree_graph_usage_error = function(e) {
        e$message <- sprintf("%s; leave it out with stats = setdiff(stats, \"%s\")",
                             conditionMessage(e), stat)
        stop(e)
      }
    )
  }
  products <- vapply(m$products, paste, character(1), collapse = ":")
  for (stat in intersect(.pg_stats, stats)) {
    if (stat == "n") {
      out$n <- .pg_call(.native_moments_counts(m))
    } else if (stat %in% c("cross", "comoment", "pearson")) {
      for (j in seq_along(products)) {
        out[[paste0(stat, ".", products[j])]] <- derive(stat, j, products[j])
      }
    } else {
      for (j in seq_along(m$columns)) {
        name <- if (startsWith(stat, "mean_")) paste0(sub("mean_", "", stat), ".", m$columns[j]) else m$columns[j]
        out[[paste0(stat, ".", m$columns[j])]] <- derive(stat, j, name)
      }
    }
  }
  structure(out, class = "data.frame", row.names = if (cells) c(NA_integer_, -cells) else integer())
}

#' @export
print.relationship_moments <- function(x, ...) {
  axes <- vapply(x$axes, function(a) sprintf("%s=%d", a$name, length(a$levels)), character(1))
  products <- vapply(x$products, paste, character(1), collapse = " x ")
  cat("<relationship_moments> ", paste(axes, collapse = ", "), "\n", sep = "")
  cat("  columns:  ", if (length(x$columns)) paste(x$columns, collapse = ", ") else "(none)", "\n", sep = "")
  cat("  products: ", if (length(products)) paste(products, collapse = ", ") else "(none)", "\n", sep = "")
  cat("  symmetric: ", x$symmetric, "; lanes: ", x$lanes, "\n", sep = "")
  cat("  pairs:    ", .pg_call(.native_moments_total(x)), "\n", sep = "")
  invisible(x)
}

# Every accumulator as a decimal string, cell by cell (the goldens' exact view).
.moments_exact <- function(m) .pg_call(.native_moments_exact(m))
