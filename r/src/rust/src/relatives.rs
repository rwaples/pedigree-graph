//! Per-person relative counts for R (ADR 0014).
//!
//! Core credits the pairs; this module turns R's threshold columns into
//! owned float64 sides for the job and the core's row-major counts into an
//! R integer array `[row, category, column]`.

use crate::errors::{HostError, HostResult};
use crate::job;
use crate::kernels::{package_pool, selection, walk_of, Native};
use crate::moments::{check_length, named_columns};
use crate::threads;
use extendr_api::prelude::*;
use pedigree_graph_core::relationships::{
    relatives_per_person as relatives_of, Progress, Threshold, ThresholdColumn, MAX_VALUE_COLUMNS,
};
use std::num::NonZeroUsize;

/// The pair-count column, always first; no threshold column may take its name.
const RELATIVES: &str = "relatives";

/// The credited member's side of one column, owned for the job.
enum OwnedThreshold {
    Scalar(f64),
    Rows(Vec<f64>),
}

/// One threshold column, owned for the job.
struct OwnedColumn {
    relative: Vec<f64>,
    threshold: OwnedThreshold,
}

/// A double or integer vector as float64, `NA` as NaN (which never counts).
fn real_column(field: &str, column: &Robj) -> HostResult<Vec<f64>> {
    let refused = |what: &str| {
        Err(HostError::usage(format!(
            "{field} must be a numeric vector, not {what}"
        )))
    };
    // A factor's codes, integer64's bits and a Date's or POSIXct's epoch
    // offsets are not the values they print as; Python refuses datetime64.
    if let Some(class) = column.class().and_then(|mut c| c.next()) {
        return refused(&format!("class {class}; convert it with as.double()"));
    }
    if let Some(values) = column.as_real_slice() {
        return Ok(values.to_vec());
    }
    if let Some(values) = column.as_integer_slice() {
        return Ok(values
            .iter()
            .map(|&v| {
                if v == i32::MIN {
                    f64::NAN
                } else {
                    f64::from(v)
                }
            })
            .collect());
    }
    // Python refuses bool too; `NA` alone is logical, so name NA_real_.
    if column.is_logical() {
        return refused("logical (use NA_real_ for a missing value)");
    }
    refused("this type")
}

/// The `(relative, threshold)` of one entry: an unnamed length-2 list, or
/// one named `relative` and `threshold`.
fn sides(field: &str, entry: &Robj) -> HostResult<(Robj, Robj)> {
    let shape = || {
        HostError::usage(format!(
            "{field} must be list(relative, threshold): a per-row numeric vector and a per-row \
             numeric vector or one number"
        ))
    };
    let list = List::try_from(entry.clone())
        .ok()
        .filter(|l| l.len() == 2)
        .ok_or_else(shape)?;
    let values: Vec<Robj> = list.values().collect();
    let names: Vec<&str> = list.names().map(Iterator::collect).unwrap_or_default();
    match names.as_slice() {
        [] | ["", ""] | ["relative", "threshold"] => Ok((values[0].clone(), values[1].clone())),
        ["threshold", "relative"] => Ok((values[1].clone(), values[0].clone())),
        _ => Err(shape()),
    }
}

fn owned_column(name: &str, entry: &Robj, n: usize) -> HostResult<OwnedColumn> {
    let field = format!("thresholds['{name}']");
    let (relative, threshold) = sides(&field, entry)?;
    let relative_field = format!("{field}.relative");
    let relative_values = real_column(&relative_field, &relative)?;
    check_length(&relative_field, &relative, n)?;
    let threshold_field = format!("{field}.threshold");
    let threshold_values = real_column(&threshold_field, &threshold)?;
    let threshold = if threshold_values.len() == 1 {
        OwnedThreshold::Scalar(threshold_values[0])
    } else {
        check_length(&threshold_field, &threshold, n)?;
        OwnedThreshold::Rows(threshold_values)
    };
    Ok(OwnedColumn {
        relative: relative_values,
        threshold,
    })
}

/// Start `relatives_per_person()`; collecting the job gives
/// `list(counts, categories)`: an integer array with `dim = c(rows,
/// categories, 1 + columns)` in input rows, the requested categories in
/// registry order, and the pair count then each threshold column; and those
/// categories' codes.
pub fn start_relatives(
    native: &Robj,
    seal: &Robj,
    max_degree: &Robj,
    categories: &Robj,
    thresholds: &Robj,
) -> HostResult<Robj> {
    let graph = Native::verified(native, seal)?;
    let requested = selection(max_degree, categories)?;
    let n = graph.len();
    let named = named_columns("thresholds", thresholds)?;
    if named.iter().any(|(name, _)| name == RELATIVES) {
        return Err(HostError::usage(format!(
            "'{RELATIVES}' is the pair-count column and cannot name a threshold column"
        )));
    }
    if named.len() > MAX_VALUE_COLUMNS {
        return Err(HostError::usage(format!(
            "at most {MAX_VALUE_COLUMNS} threshold columns are supported, got {}",
            named.len()
        )));
    }
    let columns = named
        .iter()
        .map(|(name, entry)| owned_column(name, entry, n))
        .collect::<HostResult<Vec<_>>>()?;
    let n_categories = requested.iter().count();
    let stride = 1 + columns.len();

    let threads = NonZeroUsize::new(threads::budget()?).expect("a budget of at least 1");
    let pool = package_pool()?;
    let walk = walk_of(&graph, requested)?;
    let work = move |progress: &Progress| match walk {
        Some((max_degree, rows)) => {
            let borrowed: Vec<ThresholdColumn<'_>> = columns
                .iter()
                .map(|c| ThresholdColumn {
                    relative: &c.relative,
                    threshold: match &c.threshold {
                        OwnedThreshold::Scalar(t) => Threshold::Scalar(*t),
                        OwnedThreshold::Rows(t) => Threshold::Rows(t),
                    },
                })
                .collect();
            let relatives = relatives_of(
                &rows.pedigree()?,
                max_degree,
                requested,
                None,
                false,
                &borrowed,
                threads,
                progress,
            )?;
            Ok(relatives.counts)
        }
        None => Ok(vec![0u32; n * n_categories * stride]),
    };
    Ok(job::spawn(pool, work, move |counts| {
        let codes: Vec<&str> = requested.iter().map(|c| c.code()).collect();
        let codes = Strings::from_values(codes).into_robj();
        Ok(List::from_names_and_values(
            ["counts", "categories"],
            [column_major(&counts, n, n_categories, stride), codes],
        )
        .expect("two names and two values")
        .into_robj())
    }))
}

/// Row-major `[row][category][column]` counts as an R integer array of the
/// same dimensions, read once in order.
fn column_major(counts: &[u32], n: usize, n_categories: usize, stride: usize) -> Robj {
    let mut out = Integers::new(counts.len());
    let cells = n * n_categories;
    for (cell, columns) in counts.chunks_exact(stride).enumerate() {
        let (row, category) = (cell / n_categories, cell % n_categories);
        for (column, &count) in columns.iter().enumerate() {
            // A row's relatives in one category are other rows, so fewer
            // than the graph's int32 row count.
            out[row + n * category + cells * column] = Rint::from(count as i32);
        }
    }
    let mut out = out.into_robj();
    out.set_attrib(dim_symbol(), [n as i32, n_categories as i32, stride as i32])
        .expect("an integer dim");
    out
}
