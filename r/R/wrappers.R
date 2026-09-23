# The native routines, registered by useDynLib(.registration = TRUE).

.native_build_pedigree <- function(id, mother, father, twin, sex, generation, birth_year, sex_encoding) {
  .Call(wrap__build_pedigree, id, mother, father, twin, sex, generation, birth_year, sex_encoding)
}
.native_check_graph <- function(native, seal) .Call(wrap__check_graph, native, seal)
.native_configure_threads <- function(n) .Call(wrap__configure_threads, n)
.native_thread_budget <- function() .Call(wrap__thread_budget)
.native_relationship_categories <- function() .Call(wrap__relationship_categories)
