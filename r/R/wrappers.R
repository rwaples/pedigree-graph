# The native routines, registered by useDynLib(.registration = TRUE).

.native_build_pedigree <- function(id, mother, father, twin, sex, generation, birth_year, sex_encoding) {
  .Call(wrap__build_pedigree, id, mother, father, twin, sex, generation, birth_year, sex_encoding)
}
.native_check_graph <- function(native, seal) .Call(wrap__check_graph, native, seal)
.native_configure_threads <- function(n) .Call(wrap__configure_threads, n)
.native_thread_budget <- function() .Call(wrap__thread_budget)
.native_relationship_categories <- function() .Call(wrap__relationship_categories)
.native_relationship_pairs <- function(native, seal, max_degree, categories, execution, ids) {
  .Call(wrap__relationship_pairs, native, seal, max_degree, categories, execution, ids)
}
.native_pair_kinship <- function(native, seal, first, second) .Call(wrap__pair_kinship, native, seal, first, second)
.native_inbreeding <- function(native, seal) .Call(wrap__inbreeding, native, seal)
.native_kinship_matrix <- function(native, seal, max_nnz) .Call(wrap__kinship_matrix, native, seal, max_nnz)
