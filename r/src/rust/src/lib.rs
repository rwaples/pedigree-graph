//! extendr host binding for pedigree-graph-core (ADR 0007; slice 16).
//!
//! Every exported function returns its value or, on failure, the classed
//! list the R wrapper `.pg_call()` signals (`errors.rs`).  Nothing here
//! raises into R.

mod errors;
mod graph;
mod input;
mod kernels;
mod moments;
mod test_hooks;
mod threads;

use errors::{finish, HostError};
use extendr_api::prelude::*;
use pedigree_graph_core::relationships::Category;

/// Validate and build a pedigree; see `graph::build`.
#[extendr]
#[allow(clippy::too_many_arguments)]
fn build_pedigree(
    id: Robj,
    mother: Robj,
    father: Robj,
    twin: Robj,
    sex: Robj,
    generation: Robj,
    birth_year: Robj,
    sex_encoding: &str,
) -> Robj {
    finish(graph::build(
        id,
        mother,
        father,
        twin,
        sex,
        generation,
        birth_year,
        sex_encoding,
    ))
}

/// `TRUE` when a graph's `native` fields still match its seal.
#[extendr]
fn check_graph(native: Robj, seal: Robj) -> Robj {
    finish(graph::verified(&native, &seal).map(|_| true.into()))
}

/// Pairs of the selected categories; see `kernels::relationship_pairs`.
#[extendr]
fn relationship_pairs(
    native: Robj,
    seal: Robj,
    max_degree: Robj,
    categories: Robj,
    execution: &str,
    ids: bool,
) -> Robj {
    finish(kernels::relationship_pairs(
        &native,
        &seal,
        &max_degree,
        &categories,
        execution,
        ids,
    ))
}

/// Pairs per category; see `kernels::relationship_counts`.
#[extendr]
fn relationship_counts(native: Robj, seal: Robj, max_degree: Robj, categories: Robj) -> Robj {
    finish(kernels::relationship_counts(
        &native,
        &seal,
        &max_degree,
        &categories,
    ))
}

/// Per-person relatives and pair totals; see `kernels::relationship_burden`.
#[extendr]
fn relationship_burden(native: Robj, seal: Robj) -> Robj {
    finish(kernels::relationship_burden(&native, &seal))
}

/// Relationship moments; see `moments::relationship_moments`.
#[extendr]
#[allow(clippy::too_many_arguments)]
fn relationship_moments(
    native: Robj,
    seal: Robj,
    max_degree: Robj,
    categories: Robj,
    first: Robj,
    second: Robj,
    values: Robj,
    products: Robj,
    same: Robj,
    symmetric: &str,
    memory_budget_bytes: f64,
) -> Robj {
    finish(moments::relationship_moments(
        &native,
        &seal,
        &max_degree,
        &categories,
        &first,
        &second,
        &values,
        &products,
        &same,
        symmetric,
        memory_budget_bytes,
    ))
}

/// Keep 1-based `positions` of 1-based `axis` of a moments table.
#[extendr]
fn moments_select(m: Robj, axis: i32, positions: Vec<i32>) -> Robj {
    finish(moments::select(&m, axis, &positions))
}

/// Fold 1-based `axis` of a moments table away.
#[extendr]
fn moments_sum(m: Robj, axis: i32) -> Robj {
    finish(moments::sum(&m, axis))
}

/// Add two moments tables cell by cell.
#[extendr]
fn moments_merge(a: Robj, b: Robj) -> Robj {
    finish(moments::merge(&a, &b))
}

/// One statistic per cell for 1-based column or product `index`.
#[extendr]
fn moments_derive(m: Robj, statistic: &str, index: i32, name: &str) -> Robj {
    finish(moments::derive(&m, statistic, index, name))
}

/// The pair count of every cell, as doubles.
#[extendr]
fn moments_counts(m: Robj) -> Robj {
    finish(moments::counts(&m))
}

/// The table's pairs, exactly, as a decimal string.
#[extendr]
fn moments_total(m: Robj) -> Robj {
    finish(moments::total(&m))
}

/// Every accumulator as a decimal string.
#[extendr]
fn moments_exact(m: Robj) -> Robj {
    finish(moments::exact(&m))
}

/// Decimal integers as an encoded table.
#[extendr]
fn moments_encode(values: Robj) -> Robj {
    finish(moments::encode(&values))
}

/// Kinship per 1-based row pair.
#[extendr]
fn pair_kinship(native: Robj, seal: Robj, first: Robj, second: Robj) -> Robj {
    finish(kernels::pair_kinship(&native, &seal, &first, &second))
}

/// Inbreeding per row.
#[extendr]
fn inbreeding(native: Robj, seal: Robj) -> Robj {
    finish(kernels::inbreeding(&native, &seal))
}

/// The upper-triangle kinship matrix slots; `max_nnz` (`NULL` normally)
/// lowers the entry cap for tests.
#[extendr]
fn kinship_matrix(native: Robj, seal: Robj, max_nnz: Robj) -> Robj {
    let max_nnz = max_nnz.as_real().map(|v| v as usize);
    finish(kernels::kinship_matrix(&native, &seal, max_nnz))
}

/// Record a thread budget; R passes `n` as a double (or `NaN` for a non-number).
#[extendr]
fn configure_threads(n: f64) -> Robj {
    // Whole and within usize; `threads::configure` applies the range.
    if !(n.is_finite() && n == n.trunc() && n >= 0.0 && n <= usize::MAX as f64) {
        return HostError::usage(format!(
            "configure_threads(n) requires a whole number from 1 to {}, got {n}",
            threads::MAX_THREADS
        ))
        .into_robj();
    }
    finish(threads::configure(n as usize).map(|()| ().into()))
}

/// The committed thread budget, committing it on first call.
#[extendr]
fn thread_budget() -> Robj {
    finish(threads::budget().map(|n| (n as i32).into()))
}

/// The relationship registry as columns, in registry order.
#[extendr]
fn relationship_categories() -> Robj {
    let role = |pick: fn((&'static str, &'static str)) -> &'static str| {
        Strings::from_values(Category::ALL.iter().map(|c| match c.roles() {
            Some(roles) => Rstr::from(pick(roles)),
            None => Rstr::na(),
        }))
        .into_robj()
    };
    List::from_names_and_values(
        [
            "code",
            "degree",
            "first_role",
            "second_role",
            "nominal_kinship",
        ],
        [
            Strings::from_values(Category::ALL.iter().map(|c| c.code())).into_robj(),
            Integers::from_values(Category::ALL.iter().map(|c| i32::from(c.degree()))).into_robj(),
            role(|r| r.0),
            role(|r| r.1),
            // Every registry row's nominal kinship is 2^-(degree + 1).
            Doubles::from_values(
                Category::ALL
                    .iter()
                    .map(|c| 0.5f64.powi(i32::from(c.degree()) + 1)),
            )
            .into_robj(),
        ],
    )
    .expect("five names and five values")
    .into_robj()
}

extendr_module! {
    mod pedigreegraph;
    use test_hooks;
    fn build_pedigree;
    fn relationship_pairs;
    fn relationship_counts;
    fn relationship_burden;
    fn relationship_moments;
    fn moments_select;
    fn moments_sum;
    fn moments_merge;
    fn moments_derive;
    fn moments_counts;
    fn moments_total;
    fn moments_exact;
    fn moments_encode;
    fn pair_kinship;
    fn inbreeding;
    fn kinship_matrix;
    fn check_graph;
    fn configure_threads;
    fn thread_budget;
    fn relationship_categories;
}
