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
if (args[4] == "1") {
  K <- timed("kinship_matrix", kinship_matrix(pg))
  stored <- Matrix::summary(K)
  stored <- stored[order(stored$j, stored$i), ]
  put(as.integer(stored$i), "kinship_i.bin", 4L)
  put(as.integer(stored$j), "kinship_j.bin", 4L)
  put(stored$x, "kinship_x.bin", 8L)
}
