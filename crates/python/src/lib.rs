//! `pedigree_graph._native`: the PyO3 host binding of pedigree-graph-core (ADR 0007).
//!
//! Every function takes and returns host arrays; nothing here retains core
//! state between calls.  Core errors cross the boundary as the structured
//! exception classes of `pedigree_graph._errors`, keyed by their `.code`.

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pedigree_graph_core::error::{Error, ErrorClass};
use pedigree_graph_core::topology::{self, Order};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// `(order, inverse)` intp arrays of a depth-major permutation.
type Permutation<'py> = (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>);

/// The Cargo workspace version, which is also the Python distribution version.
#[pyfunction]
fn core_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

fn check_parent_lengths(mother: &[i32], father: &[i32]) -> PyResult<()> {
    if mother.len() != father.len() {
        return Err(PyValueError::new_err(format!(
            "mother_rows and father_rows must have the same length, got {} and {}",
            mother.len(),
            father.len()
        )));
    }
    Ok(())
}

fn to_pyerr(py: Python<'_>, err: Error) -> PyErr {
    let class_name = match err.class() {
        ErrorClass::Validation => "PedigreeValidationError",
        ErrorClass::Metadata => "MissingMetadataError",
        ErrorClass::Resource => "ResourceError",
    };
    let fields = PyDict::new(py);
    let populate = match &err {
        Error::Cycle { ids } => fields.set_item("ids", ids.clone()),
        Error::PedigreeTooLarge {
            n_individuals,
            maximum,
        } => fields
            .set_item("n_individuals", *n_individuals)
            .and_then(|()| fields.set_item("maximum", *maximum)),
    };
    if let Err(e) = populate {
        return e;
    }
    let raise = || -> PyResult<PyErr> {
        let module = py.import("pedigree_graph._errors")?;
        let class = module.getattr(class_name)?;
        let instance = class.call((err.code(), err.to_string()), Some(&fields))?;
        Ok(PyErr::from_value(instance))
    };
    raise().unwrap_or_else(|e| e)
}

/// True iff every represented parent row strictly precedes its child row.
#[pyfunction]
fn is_topological(
    mother_rows: PyReadonlyArray1<'_, i32>,
    father_rows: PyReadonlyArray1<'_, i32>,
) -> PyResult<bool> {
    let mother = mother_rows.as_slice()?;
    let father = father_rows.as_slice()?;
    check_parent_lengths(mother, father)?;
    Ok(topology::is_topological(mother, father))
}

/// Structural depth per row: founders 0, otherwise `max(parent depths) + 1`.
///
/// Requires an acyclic pedigree; run `validate_acyclic` first.
#[pyfunction]
fn structural_depth<'py>(
    py: Python<'py>,
    mother_rows: PyReadonlyArray1<'py, i32>,
    father_rows: PyReadonlyArray1<'py, i32>,
) -> PyResult<Bound<'py, PyArray1<i32>>> {
    let mother = mother_rows.as_slice()?;
    let father = father_rows.as_slice()?;
    check_parent_lengths(mother, father)?;
    Ok(topology::structural_depth(mother, father).into_pyarray(py))
}

/// The stable depth-major permutation as `(order, inverse)` intp arrays, or
/// `None` when the rows are already depth-major.
#[pyfunction]
fn depth_major_order<'py>(
    py: Python<'py>,
    depth: PyReadonlyArray1<'py, i32>,
) -> PyResult<Option<Permutation<'py>>> {
    Ok(match topology::depth_major_order(depth.as_slice()?) {
        Order::Identity => None,
        Order::Permuted { order, inverse } => {
            Some((order.into_pyarray(py), inverse.into_pyarray(py)))
        }
    })
}

/// Raise `PedigreeValidationError("cycle")` with the deterministic id witness
/// when the parent edges are cyclic.
#[pyfunction]
fn validate_acyclic(
    py: Python<'_>,
    ids: PyReadonlyArray1<'_, i64>,
    mother_rows: PyReadonlyArray1<'_, i32>,
    father_rows: PyReadonlyArray1<'_, i32>,
) -> PyResult<()> {
    let ids = ids.as_slice()?;
    let mother = mother_rows.as_slice()?;
    let father = father_rows.as_slice()?;
    check_parent_lengths(mother, father)?;
    if ids.len() != mother.len() {
        return Err(PyValueError::new_err(format!(
            "ids and parent rows must have the same length, got {} and {}",
            ids.len(),
            mother.len()
        )));
    }
    topology::validate_acyclic(ids, mother, father).map_err(|e| to_pyerr(py, e))
}

#[pymodule(name = "_native")]
fn native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(core_version, m)?)?;
    m.add_function(wrap_pyfunction!(is_topological, m)?)?;
    m.add_function(wrap_pyfunction!(structural_depth, m)?)?;
    m.add_function(wrap_pyfunction!(depth_major_order, m)?)?;
    m.add_function(wrap_pyfunction!(validate_acyclic, m)?)?;
    Ok(())
}
