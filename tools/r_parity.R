# The R half of tools/r_parity.py: run one pedigree through pedigreegraph and
# write each product as raw little-endian arrays for the Python half to
# compare.  Usage: Rscript tools/r_parity.R <pedigree.tsv> <outdir> <max_degree> <matrix 0|1> <pair_kinship 0|1>
library(pedigreegraph)
args <- commandArgs(TRUE)
out <- args[2]
max_degree <- as.integer(args[3])
put <- function(x, name, size) writeBin(x, file.path(out, name), size = size, endian = "little")
timed <- function(name, expr) {
  t <- system.time(value <- expr)[["elapsed"]]
  cat(sprintf("%s\t%.3f\n", name, t), file = file.path(out, "r_seconds.tsv"), append = TRUE)
  value
}

pg <- pedigree_graph(read.delim(args[1]))
pairs <- timed("pairs", relationship_pairs(pg, max_degree = max_degree, ids = FALSE))
put(as.integer(pairs$code), "pairs_code.bin", 4L)
put(pairs$first, "pairs_first.bin", 4L)
put(pairs$second, "pairs_second.bin", 4L)
put(timed("inbreeding", inbreeding(pg)), "inbreeding.bin", 8L)
if (args[5] == "1") {
  close <- as.integer(pairs$code) %in% which(relationship_categories()$degree <= 3L)
  put(timed("pair_kinship", pair_kinship(pg, pairs$first[close], pairs$second[close])),
      "pair_kinship.bin", 8L)
}
counts <- timed("counts", relationship_counts(pg, max_degree = max_degree))
counts[is.na(counts)] <- -1
put(as.double(counts), "counts.bin", 8L)
burden <- timed("burden", relationship_burden(pg))
put(as.vector(burden$per_person), "burden_rows.bin", 4L)
put(burden$category_counts, "burden_categories.bin", 8L)
put(burden$same_depth_pairs, "burden_depth.bin", 8L)

# tools/r_golden.py::_moments_spec, as test-golden.R builds it.
ped <- read.delim(args[1])
n <- nrow(ped)
rows <- as.double(seq_len(n))
tie <- (rows %% 7) - 2.5
tie[1] <- 2^43
m <- timed("moments", relationship_moments(
  pg, categories = c("MZ", "FS", "MO", "FO", "MHS", "PHS", "GP", "Av", "1C"),
  first = list(parity = pg$native$depth %% 2L,
               code = factor(c("r0", "r1")[(seq_len(n) - 1L) %% 2L + 1L], levels = c("r0", "r1"))),
  values = list(ordinary = ((rows * 37) %% 101) / 7 - 5, constant = rep(0.3, n), tie = tie,
                large = 1e150 * (((rows * 13) %% 17) - 8)),
  products = list(c("first.ordinary", "second.ordinary"), c("first.constant", "second.constant"),
                  c("first.tie", "second.tie"), c("first.large", "second.large"),
                  c("first.ordinary", "first.tie")),
  same = list(mother = ped$mother)
))
df <- timed("moments_frame", as.data.frame(m))
axes <- vapply(m$axes, `[[`, character(1), "name")
put(unlist(df[setdiff(names(df), axes)], use.names = FALSE), "moments_stats.bin", 8L)
writeBin(charToRaw(paste(pedigreegraph:::.moments_exact(m), collapse = "\n")), file.path(out, "moments_exact.txt"))

if (args[4] == "1") {
  K <- timed("kinship_matrix", kinship_matrix(pg))
  stored <- Matrix::summary(K)
  stored <- stored[order(stored$j, stored$i), ]
  put(as.integer(stored$i), "kinship_i.bin", 4L)
  put(as.integer(stored$j), "kinship_j.bin", 4L)
  put(stored$x, "kinship_x.bin", 8L)
}
