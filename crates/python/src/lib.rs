//! `pedigree_graph._native`: the PyO3 host binding of pedigree-graph-core (ADR 0007).
//!
//! Every function takes host arrays and returns host arrays or a small value
//! object built from them; nothing here retains core state between calls.
//! Core errors cross the boundary as the structured exception classes of
//! `pedigree_graph._errors`, keyed by their `.code`, with the keyword fields
//! rebuilt from `Error::fields`.  A usage error crosses as a plain `ValueError`.

use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1};
use pedigree_graph_core::alloc::{self, Family};
use pedigree_graph_core::error::{Error, ErrorClass, FieldValue, MAX_ROWS};
use pedigree_graph_core::graph::{self, Columns, IdIndex, Limits, SexEncoding};
use pedigree_graph_core::pool;
use pedigree_graph_core::relationships::{self, Category, CategorySet, Execution, Pedigree};
use pedigree_graph_core::topology::{self, Order};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use std::num::NonZeroUsize;

/// `(order, inverse)` intp arrays of a depth-major permutation.
type Permutation<'py> = (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>);

/// The Cargo workspace version, which is also the Python distribution version.
#[pyfunction]
fn core_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// The deepest `max_degree` this build counts, derived from the category registry.
#[pyfunction]
fn max_degree_max() -> u8 {
    relationships::MaxDegree::MAX.get()
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

/// Every parent row is `-1` or a row of the pedigree, so the core can index
/// by it without a bounds failure reaching the user.
fn check_parent_rows(mother: &[i32], father: &[i32]) -> PyResult<()> {
    check_parent_lengths(mother, father)?;
    check_rows("mother_rows", mother, mother.len())?;
    check_rows("father_rows", father, mother.len())
}

fn check_rows(name: &str, rows: &[i32], n: usize) -> PyResult<()> {
    let n = n as i64;
    if let Some(position) = rows
        .iter()
        .position(|&row| i64::from(row) < -1 || i64::from(row) >= n)
    {
        return Err(PyValueError::new_err(format!(
            "{name}[{position}] = {} is not -1 or a row below {n}",
            rows[position]
        )));
    }
    Ok(())
}

fn check_same_length(name: &str, len: usize, n: usize) -> PyResult<()> {
    if len != n {
        return Err(PyValueError::new_err(format!(
            "{name} must have the same length as mother_rows, got {len} and {n}"
        )));
    }
    Ok(())
}

fn field_object<'py>(py: Python<'py>, value: FieldValue) -> PyResult<Bound<'py, PyAny>> {
    Ok(match value {
        FieldValue::Int(n) => n.into_pyobject(py)?.into_any(),
        FieldValue::Str(s) => s.into_pyobject(py)?.into_any(),
        FieldValue::Ints(v) => v.into_pyobject(py)?.into_any(),
        FieldValue::Strs(v) => v.into_pyobject(py)?.into_any(),
    })
}

fn to_pyerr(py: Python<'_>, err: Error) -> PyErr {
    // A pool already built for a different budget is the native half of the
    // rule `_threads.configure_threads` states, so it raises what that
    // function raises rather than the `Usage` family's ValueError.
    if matches!(err, Error::ThreadPoolConflict { .. }) {
        return PyRuntimeError::new_err(err.to_string());
    }
    let class_name = match err.class() {
        ErrorClass::Validation => "PedigreeValidationError",
        ErrorClass::Metadata => "MissingMetadataError",
        ErrorClass::Resource => "ResourceError",
        ErrorClass::Usage => return PyValueError::new_err(err.to_string()),
    };
    let raise = || -> PyResult<PyErr> {
        let fields = PyDict::new(py);
        for (name, value) in err.fields() {
            fields.set_item(name, field_object(py, value)?)?;
        }
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
    check_parent_rows(mother, father)?;
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
    check_parent_rows(mother, father)?;
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
    check_parent_rows(mother, father)?;
    if ids.len() != mother.len() {
        return Err(PyValueError::new_err(format!(
            "ids and parent rows must have the same length, got {} and {}",
            ids.len(),
            mother.len()
        )));
    }
    topology::validate_acyclic(ids, mother, father).map_err(|e| to_pyerr(py, e))
}

/// The validated columns of one pedigree, each an owned numpy array.
///
/// `sex`, `generation` and `birth_year` are `None` when the host had no such
/// column or every entry was unknown.
#[pyclass(frozen, get_all, module = "pedigree_graph._native")]
struct BuiltPedigree {
    ids: Py<PyArray1<i64>>,
    mother_ids: Py<PyArray1<i64>>,
    father_ids: Py<PyArray1<i64>>,
    twin_ids: Py<PyArray1<i64>>,
    mother_rows: Py<PyArray1<i32>>,
    father_rows: Py<PyArray1<i32>>,
    twin_rows: Py<PyArray1<i32>>,
    sex: Option<Py<PyArray1<i8>>>,
    generation: Option<Py<PyArray1<i32>>>,
    birth_year: Option<Py<PyArray1<i32>>>,
    rows_topological: bool,
}

fn optional_slice<'a>(
    column: &'a Option<PyReadonlyArray1<'_, i64>>,
) -> PyResult<Option<&'a [i64]>> {
    match column {
        Some(array) => Ok(Some(array.as_slice()?)),
        None => Ok(None),
    }
}

/// Validate host-coerced int64 columns and build the pedigree (ADR 0006).
///
/// Raises the structured construction errors in the core's order of
/// precedence, or `ValueError` for an unknown `sex_encoding`.  `max_rows`
/// lowers the row capacity for tests; the default is the int32 row limit.
#[pyfunction]
#[pyo3(signature = (ids, mother, father, twin=None, sex=None, generation=None, birth_year=None, *, sex_encoding, max_rows=None))]
#[allow(clippy::too_many_arguments)]
fn build_pedigree<'py>(
    py: Python<'py>,
    ids: PyReadonlyArray1<'py, i64>,
    mother: PyReadonlyArray1<'py, i64>,
    father: PyReadonlyArray1<'py, i64>,
    twin: Option<PyReadonlyArray1<'py, i64>>,
    sex: Option<PyReadonlyArray1<'py, i64>>,
    generation: Option<PyReadonlyArray1<'py, i64>>,
    birth_year: Option<PyReadonlyArray1<'py, i64>>,
    sex_encoding: &str,
    max_rows: Option<usize>,
) -> PyResult<BuiltPedigree> {
    let encoding = SexEncoding::parse(sex_encoding).map_err(|e| to_pyerr(py, e))?;
    let columns = Columns {
        ids: ids.as_slice()?,
        mother: mother.as_slice()?,
        father: father.as_slice()?,
        twin: optional_slice(&twin)?,
        sex: optional_slice(&sex)?,
        generation: optional_slice(&generation)?,
        birth_year: optional_slice(&birth_year)?,
    };
    let limits = Limits {
        max_rows: max_rows.unwrap_or(MAX_ROWS),
    };
    let graph = graph::build(columns, encoding, limits).map_err(|e| to_pyerr(py, e))?;
    Ok(BuiltPedigree {
        ids: graph.ids.into_pyarray(py).unbind(),
        mother_ids: graph.mother_ids.into_pyarray(py).unbind(),
        father_ids: graph.father_ids.into_pyarray(py).unbind(),
        twin_ids: graph.twin_ids.into_pyarray(py).unbind(),
        mother_rows: graph.mother_rows.into_pyarray(py).unbind(),
        father_rows: graph.father_rows.into_pyarray(py).unbind(),
        twin_rows: graph.twin_rows.into_pyarray(py).unbind(),
        sex: graph.sex.map(|v| v.into_pyarray(py).unbind()),
        generation: graph.generation.map(|v| v.into_pyarray(py).unbind()),
        birth_year: graph.birth_year.map(|v| v.into_pyarray(py).unbind()),
        rows_topological: graph.rows_topological,
    })
}

/// The package Rayon pool, built on first use with `threads` workers (ADR 0007).
///
/// Calling again with the same value is a no-op; a different value raises
/// `ValueError`, as does `threads == 0`.  Returns the pool's size.
#[pyfunction]
fn configure_pool(py: Python<'_>, threads: usize) -> PyResult<usize> {
    Ok(checked_pool(py, threads)?.current_num_threads())
}

fn checked_pool(py: Python<'_>, threads: usize) -> PyResult<&'static rayon::ThreadPool> {
    let threads = NonZeroUsize::new(threads)
        .ok_or_else(|| PyValueError::new_err("threads must be at least 1"))?;
    pool::configure(threads).map_err(|e| to_pyerr(py, e))
}

/// A graph's engine columns, borrowed from its [`BuiltPedigree`] for one call.
struct EngineColumns<'py> {
    mother_rows: PyReadonlyArray1<'py, i32>,
    father_rows: PyReadonlyArray1<'py, i32>,
    twin_rows: PyReadonlyArray1<'py, i32>,
    mother_ids: PyReadonlyArray1<'py, i64>,
    father_ids: PyReadonlyArray1<'py, i64>,
}

impl<'py> EngineColumns<'py> {
    fn borrow(py: Python<'py>, pedigree: &BuiltPedigree) -> EngineColumns<'py> {
        EngineColumns {
            mother_rows: pedigree.mother_rows.bind(py).readonly(),
            father_rows: pedigree.father_rows.bind(py).readonly(),
            twin_rows: pedigree.twin_rows.bind(py).readonly(),
            mother_ids: pedigree.mother_ids.bind(py).readonly(),
            father_ids: pedigree.father_ids.bind(py).readonly(),
        }
    }

    /// The columns as checked engine input; `build_pedigree` validated them,
    /// and the core rechecks its own preconditions as it borrows.
    fn pedigree(&self, py: Python<'py>) -> PyResult<Pedigree<'_>> {
        Pedigree::try_new(
            self.mother_rows.as_slice()?,
            self.father_rows.as_slice()?,
            self.twin_rows.as_slice()?,
            self.mother_ids.as_slice()?,
            self.father_ids.as_slice()?,
        )
        .map_err(|e| to_pyerr(py, e))
    }
}

fn checked_max_degree(py: Python<'_>, max_degree: u8) -> PyResult<relationships::MaxDegree> {
    relationships::MaxDegree::try_new(max_degree).map_err(|e| to_pyerr(py, e))
}

/// Exact closest-category pair counts, keyed by registry code in registry order.
///
/// `pedigree` is the graph's own [`BuiltPedigree`].  With `selected`, a bool
/// per row, only pairs whose two rows are both selected are counted (the
/// view contract).  `threads` configures the package pool (see
/// `configure_pool`); the counts are the same for every value.  The GIL is
/// released while counting.
#[pyfunction]
#[pyo3(signature = (pedigree, *, max_degree, threads, selected=None))]
fn relationship_counts<'py>(
    py: Python<'py>,
    pedigree: &BuiltPedigree,
    max_degree: u8,
    threads: usize,
    selected: Option<PyReadonlyArray1<'py, bool>>,
) -> PyResult<Bound<'py, PyDict>> {
    let columns = EngineColumns::borrow(py, pedigree);
    let ped = columns.pedigree(py)?;
    let max_degree = checked_max_degree(py, max_degree)?;
    let mask = match &selected {
        Some(array) => {
            let mask = array.as_slice()?;
            check_same_length("selected", mask.len(), ped.len())?;
            Some(mask)
        }
        None => None,
    };
    let pool = checked_pool(py, threads)?;
    let counts = py
        .detach(|| pool.install(|| relationships::count_pairs(&ped, max_degree, mask)))
        .map_err(|e| to_pyerr(py, e))?;
    let values = PyDict::new(py);
    for &cat in Category::ALL.iter() {
        values.set_item(cat.code(), counts.get(cat) as i64)?;
    }
    Ok(values)
}

/// The oriented pairs of the `requested` categories, keyed by registry code
/// in registry order, each an `(first, second)` pair of owned int32 arrays.
///
/// Every category up to `max_degree` is classified; only the requested
/// blocks are filled, the rest are empty.  With `view_rows`, the int32 view
/// row of every graph row (`-1` unselected), blocks are in view rows and
/// sorted by the view-space key; without it they are graph rows in
/// canonical-key order.  `execution` is `"speed"` or `"memory"` (ADR 0006
/// as amended) and changes resource use only.  The arrays are moved out of
/// the core without a copy and retain nothing else.  The GIL is released
/// while classifying and assembling.
#[pyfunction]
#[pyo3(signature = (pedigree, *, max_degree, requested, threads, execution, view_rows=None))]
fn relationship_pairs<'py>(
    py: Python<'py>,
    pedigree: &BuiltPedigree,
    max_degree: u8,
    requested: Vec<String>,
    threads: usize,
    execution: &str,
    view_rows: Option<PyReadonlyArray1<'py, i32>>,
) -> PyResult<Bound<'py, PyDict>> {
    let columns = EngineColumns::borrow(py, pedigree);
    let ped = columns.pedigree(py)?;
    let max_degree = checked_max_degree(py, max_degree)?;
    let mut categories = CategorySet::EMPTY;
    for code in &requested {
        let cat = Category::parse(code)
            .ok_or_else(|| PyValueError::new_err(format!("unknown relationship code {code:?}")))?;
        categories.insert(cat);
    }
    let execution = Execution::parse(execution).ok_or_else(|| {
        PyValueError::new_err(format!(
            "execution must be \"speed\" or \"memory\", got {execution:?}"
        ))
    })?;
    let view = match &view_rows {
        Some(array) => {
            let map = array.as_slice()?;
            check_same_length("view_rows", map.len(), ped.len())?;
            Some(map)
        }
        None => None,
    };
    let pool = checked_pool(py, threads)?;
    let blocks = py
        .detach(|| {
            pool.install(|| {
                relationships::pair_blocks(&ped, max_degree, categories, view, execution)
            })
        })
        .map_err(|e| to_pyerr(py, e))?;
    let values = PyDict::new(py);
    for (cat, block) in Category::ALL.iter().zip(blocks.0) {
        let first = block.first.into_pyarray(py);
        let second = block.second.into_pyarray(py);
        values.set_item(cat.code(), PyTuple::new(py, [first, second])?)?;
    }
    Ok(values)
}

/// The allocation family names, in `Family::ALL` order.
///
/// The test seam's parametrisation reads this rather than keeping its own
/// copy of the list, which cannot then drift from the core's.
#[pyfunction]
fn allocation_families() -> Vec<&'static str> {
    Family::ALL.iter().map(|f| f.name()).collect()
}

/// The environment variable that unlocks [`fail_next_allocation`].
const SEAM_ENV: &str = "PEDIGREE_GRAPH_ALLOW_TEST_SEAM";

/// Test seam: make the next reservation of the named allocation family fail
/// with `ResourceError("allocation_failed")`, or clear the plant with `None`.
///
/// `min_elements` aims the plant at a reservation of at least that size, so
/// the error's `requested_elements` is the number the host would show.
///
/// The plant is process-global and is consumed by whichever thread reserves
/// that family next, so arming it from a released wheel would fail an
/// unrelated call.  It is refused unless the process was started with
/// `PEDIGREE_GRAPH_ALLOW_TEST_SEAM=1`, which the package's own child-process
/// tests set.
#[pyfunction]
#[pyo3(signature = (family, min_elements = 0))]
fn fail_next_allocation(family: Option<&str>, min_elements: usize) -> PyResult<()> {
    if std::env::var(SEAM_ENV).as_deref() != Ok("1") {
        return Err(PyRuntimeError::new_err(format!(
            "the allocation test seam is off; set {SEAM_ENV}=1 before starting the process"
        )));
    }
    let family =
        match family {
            None => None,
            Some(name) => Some(Family::parse(name).ok_or_else(|| {
                PyValueError::new_err(format!("unknown allocation family {name:?}"))
            })?),
        };
    alloc::fail_next_above(family, min_elements);
    Ok(())
}

/// Sorted-id lookup over a graph's unique ids, for repeated id selections.
#[pyclass(frozen, name = "IdIndex", module = "pedigree_graph._native")]
struct PyIdIndex {
    index: IdIndex,
}

#[pymethods]
impl PyIdIndex {
    #[new]
    fn new(ids: PyReadonlyArray1<'_, i64>) -> PyResult<PyIdIndex> {
        let ids = ids.as_slice()?;
        if ids.len() > MAX_ROWS {
            return Err(PyValueError::new_err(format!(
                "an id index holds at most {MAX_ROWS} ids, got {}",
                ids.len()
            )));
        }
        Ok(PyIdIndex {
            index: IdIndex::build(ids),
        })
    }

    /// The int32 row per query id, `-1` for a negative id or one not indexed.
    fn resolve<'py>(
        &self,
        py: Python<'py>,
        query: PyReadonlyArray1<'py, i64>,
    ) -> PyResult<Bound<'py, PyArray1<i32>>> {
        Ok(self.index.resolve(query.as_slice()?).into_pyarray(py))
    }
}

#[pymodule(name = "_native")]
fn native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(core_version, m)?)?;
    m.add_function(wrap_pyfunction!(max_degree_max, m)?)?;
    m.add_function(wrap_pyfunction!(is_topological, m)?)?;
    m.add_function(wrap_pyfunction!(structural_depth, m)?)?;
    m.add_function(wrap_pyfunction!(depth_major_order, m)?)?;
    m.add_function(wrap_pyfunction!(validate_acyclic, m)?)?;
    m.add_function(wrap_pyfunction!(build_pedigree, m)?)?;
    m.add_function(wrap_pyfunction!(configure_pool, m)?)?;
    m.add_function(wrap_pyfunction!(relationship_counts, m)?)?;
    m.add_function(wrap_pyfunction!(relationship_pairs, m)?)?;
    m.add_function(wrap_pyfunction!(allocation_families, m)?)?;
    m.add_function(wrap_pyfunction!(fail_next_allocation, m)?)?;
    m.add_class::<BuiltPedigree>()?;
    m.add_class::<PyIdIndex>()?;
    Ok(())
}
