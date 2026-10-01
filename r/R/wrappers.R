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
.native_relationship_counts <- function(native, seal, max_degree, categories) {
  .Call(wrap__relationship_counts, native, seal, max_degree, categories)
}
.native_relationship_burden <- function(native, seal) .Call(wrap__relationship_burden, native, seal)
.native_pair_kinship <- function(native, seal, first, second) .Call(wrap__pair_kinship, native, seal, first, second)
.native_inbreeding <- function(native, seal) .Call(wrap__inbreeding, native, seal)
.native_kinship_matrix <- function(native, seal, max_nnz) .Call(wrap__kinship_matrix, native, seal, max_nnz)
.native_relationship_moments <- function(native, seal, max_degree, categories, first, second, values,
                                         products, same, symmetric, memory_budget_bytes) {
  .Call(wrap__relationship_moments, native, seal, max_degree, categories, first, second, values,
        products, same, symmetric, memory_budget_bytes)
}
.native_moments_select <- function(m, axis, positions) .Call(wrap__moments_select, m, axis, positions)
.native_moments_sum <- function(m, axis) .Call(wrap__moments_sum, m, axis)
.native_moments_merge <- function(a, b) .Call(wrap__moments_merge, a, b)
.native_moments_derive <- function(m, statistic, index, name) .Call(wrap__moments_derive, m, statistic, index, name)
.native_moments_counts <- function(m) .Call(wrap__moments_counts, m)
.native_moments_total <- function(m) .Call(wrap__moments_total, m)
.native_moments_exact <- function(m) .Call(wrap__moments_exact, m)
.native_moments_encode <- function(values) .Call(wrap__moments_encode, values)
