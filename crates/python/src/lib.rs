//! `pedigree_graph._native`: the PyO3 host binding of pedigree-graph-core (ADR 0007).
//!
//! Every function takes host arrays and returns host arrays or a small value
//! object built from them; nothing here retains core state between calls.
//! Core errors cross the boundary as the structured exception classes of
//! `pedigree_graph._errors`, keyed by their `.code`, with the keyword fields
//! rebuilt from `Error::fields`.  A usage error crosses as a plain `ValueError`.

use num_bigint::{BigInt, BigUint};
use numpy::{
    IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyReadwriteArray2,
    PyUntypedArrayMethods,
};
use pedigree_graph_core::alloc::{self, Family};
use pedigree_graph_core::error::{Error, ErrorClass, FieldValue, MAX_ROWS};
use pedigree_graph_core::graph::{self, Columns, IdIndex, Limits, SexEncoding};
use pedigree_graph_core::kinship::{self, Csc, KinshipPedigree};
use pedigree_graph_core::lineage::{self, ParentColumns};
use pedigree_graph_core::pool;
use pedigree_graph_core::relationships::{
    self, Category, CategorySet, Execution, MomentsInput, MomentsPlan, MomentsShape, MomentsTable,
    Operand, Pedigree, Product, Progress, Side, Snapshot, Statistic, Symmetric, Threshold,
    ThresholdColumn,
};
use pedigree_graph_core::topology::{self, Order};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use std::borrow::Cow;
use std::num::NonZeroUsize;
use std::str::FromStr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::Thread;
use std::time::{Duration, Instant};

/// `(order, inverse)` intp arrays of a depth-major permutation.
type Permutation<'py> = (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>);

/// `F` and the distinct ancestor counts, one of each per graph row.
type InbreedingArrays<'py> = (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<i32>>);

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

/// Wakes the watching thread when the engine job ends, by return or by panic.
struct Done<'a> {
    finished: &'a AtomicBool,
    caller: Thread,
}

impl Drop for Done<'_> {
    fn drop(&mut self) {
        self.finished.store(true, Ordering::Release);
        self.caller.unpark();
    }
}

/// Run `work` as one job in `pool` while this thread watches it (ADR 0017).
///
/// The GIL is released for the call.  Every `tick` the watcher attaches,
/// checks for signals, and passes `(phase, rows_done, rows_total)` to
/// `callback` if there is one.  The first exception from either cancels the
/// call; once every worker has left its current 64 rows or step, that exception
/// is raised in place of the result.  The core never calls the host: the
/// job only updates the [`Progress`] the watcher reads.
fn run_watched<T: Send>(
    py: Python<'_>,
    pool: &rayon::ThreadPool,
    tick: f64,
    callback: Option<Py<PyAny>>,
    work: impl FnOnce(&Progress) -> Result<T, Error> + Send,
) -> PyResult<T> {
    let tick = Duration::try_from_secs_f64(tick)
        .ok()
        .filter(|t| !t.is_zero())
        .ok_or_else(|| {
            PyValueError::new_err(format!(
                "tick must be a positive number of seconds, got {tick}"
            ))
        })?;
    let progress = Progress::default();
    let finished = AtomicBool::new(false);
    let mut slot = None;
    let mut kept: Option<PyErr> = None;
    py.detach(|| {
        let caller = std::thread::current();
        pool.in_place_scope(|scope| {
            scope.spawn(|_| {
                let _done = Done {
                    finished: &finished,
                    caller,
                };
                slot = Some(work(&progress));
            });
            let mut next = Instant::now() + tick;
            while !finished.load(Ordering::Acquire) {
                let now = Instant::now();
                if now < next {
                    std::thread::park_timeout(next - now);
                    continue;
                }
                next = now + tick;
                if kept.is_some() {
                    continue;
                }
                let observed = Python::attach(|py| {
                    py.check_signals()?;
                    let Some(callback) = &callback else {
                        return Ok(());
                    };
                    let (phase, done, total) = match progress.snapshot() {
                        Snapshot::Preparing => ("preparing", 0, None),
                        Snapshot::Walking { done, total } => ("walking", done, Some(total)),
                        Snapshot::Finishing { total } => ("finishing", total, Some(total)),
                    };
                    callback.bind(py).call1((phase, done, total)).map(drop)
                });
                if let Err(err) = observed {
                    progress.cancel();
                    kept = Some(err);
                }
            }
        });
    });
    if let Some(err) = kept {
        return Err(err);
    }
    // The scope re-raises a panicking job, so a job that got here filled the slot.
    slot.expect("the watched job ran")
        .map_err(|e| to_pyerr(py, e))
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
    let order = topology::depth_major_order(depth.as_slice()?).map_err(|e| to_pyerr(py, e))?;
    Ok(match order {
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
#[pyo3(signature = (pedigree, *, max_degree, threads, selected=None, progress=None, tick=1.0))]
fn relationship_counts<'py>(
    py: Python<'py>,
    pedigree: &BuiltPedigree,
    max_degree: u8,
    threads: usize,
    selected: Option<PyReadonlyArray1<'py, bool>>,
    progress: Option<Py<PyAny>>,
    tick: f64,
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
    let counts = run_watched(py, pool, tick, progress, |progress| {
        relationships::count_pairs(&ped, max_degree, mask, progress)
    })?;
    let values = PyDict::new(py);
    for &cat in Category::ALL.iter() {
        values.set_item(cat.code(), counts.get(cat) as i64)?;
    }
    Ok(values)
}

/// Exact view counts after compacting to the view's represented ancestry.
#[pyfunction]
#[pyo3(signature = (pedigree, view_rows, *, max_degree, threads, progress=None, tick=1.0))]
fn compact_view_counts<'py>(
    py: Python<'py>,
    pedigree: &BuiltPedigree,
    view_rows: PyReadonlyArray1<'py, i32>,
    max_degree: u8,
    threads: usize,
    progress: Option<Py<PyAny>>,
    tick: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let columns = EngineColumns::borrow(py, pedigree);
    let ped = columns.pedigree(py)?;
    let view = view_rows.as_slice()?;
    check_same_length("view_rows", view.len(), ped.len())?;
    let max_degree = checked_max_degree(py, max_degree)?;
    let pool = checked_pool(py, threads)?;
    let counts = run_watched(py, pool, tick, progress, |progress| {
        relationships::count_view_pairs_compact(&ped, max_degree, view, progress)
    })?;
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
#[pyo3(signature = (pedigree, *, max_degree, requested, threads, execution, view_rows=None, compact=false, progress=None, tick=1.0))]
#[allow(clippy::too_many_arguments)]
fn relationship_pairs<'py>(
    py: Python<'py>,
    pedigree: &BuiltPedigree,
    max_degree: u8,
    requested: Vec<String>,
    threads: usize,
    execution: &str,
    view_rows: Option<PyReadonlyArray1<'py, i32>>,
    compact: bool,
    progress: Option<Py<PyAny>>,
    tick: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let columns = EngineColumns::borrow(py, pedigree);
    let ped = columns.pedigree(py)?;
    let max_degree = checked_max_degree(py, max_degree)?;
    let categories = requested_categories(&requested)?;
    let execution = Execution::parse(execution).ok_or_else(|| {
        PyValueError::new_err(format!(
            "execution must be \"speed\" or \"memory\", got {execution:?}"
        ))
    })?;
    let view = view_map(&view_rows, ped.len())?;
    if compact && view.is_none() {
        return Err(PyValueError::new_err("compact requires view_rows"));
    }
    let pool = checked_pool(py, threads)?;
    let blocks = run_watched(py, pool, tick, progress, |progress| {
        if compact {
            relationships::pair_blocks_compact(
                &ped,
                max_degree,
                categories,
                view.unwrap(),
                execution,
                progress,
            )
        } else {
            relationships::pair_blocks(&ped, max_degree, categories, view, execution, progress)
        }
    })?;
    let values = PyDict::new(py);
    for (cat, block) in Category::ALL.iter().zip(blocks.0) {
        let first = block.first.into_pyarray(py);
        let second = block.second.into_pyarray(py);
        values.set_item(cat.code(), PyTuple::new(py, [first, second])?)?;
    }
    Ok(values)
}

/// The requested categories of a pair query, from their registry codes.
fn requested_categories(requested: &[String]) -> PyResult<CategorySet> {
    let mut categories = CategorySet::EMPTY;
    for code in requested {
        let cat = Category::parse(code)
            .ok_or_else(|| PyValueError::new_err(format!("unknown relationship code {code:?}")))?;
        categories.insert(cat);
    }
    Ok(categories)
}

/// The view map of a query, checked to have one entry per graph row.
fn view_map<'a>(
    view_rows: &'a Option<PyReadonlyArray1<'_, i32>>,
    n: usize,
) -> PyResult<Option<&'a [i32]>> {
    match view_rows {
        Some(array) => {
            let map = array.as_slice()?;
            check_same_length("view_rows", map.len(), n)?;
            Ok(Some(map))
        }
        None => Ok(None),
    }
}

/// What [`relationship_moments`] hands back: the bytes per accumulator and
/// the encoded accumulators (adopted, not copied), the cells per category,
/// the integers per cell, the lanes the pass ran on, the pairs each lane
/// reduced, and the planned accumulator peak in bytes.
type MomentsArrays<'py> = (
    usize,
    Bound<'py, PyArray1<u8>>,
    usize,
    usize,
    usize,
    Vec<u64>,
    u64,
);

/// The lane count and planned accumulator peak of a call of the given
/// sizes, as `(lanes, estimated_peak_bytes)`, without running it; the same
/// plan the pass makes, so a degenerate call can be refused the same way.
#[pyfunction]
#[pyo3(signature = (*, n_categories, n_labels_first, n_labels_second, n_columns, n_products, n_same, threads, memory_budget_bytes))]
#[allow(clippy::too_many_arguments)]
fn moments_plan(
    py: Python<'_>,
    n_categories: usize,
    n_labels_first: usize,
    n_labels_second: usize,
    n_columns: usize,
    n_products: usize,
    n_same: usize,
    threads: usize,
    memory_budget_bytes: u64,
) -> PyResult<(usize, u64)> {
    let threads = NonZeroUsize::new(threads)
        .ok_or_else(|| PyValueError::new_err("threads must be at least 1"))?;
    let shape = MomentsShape {
        n_categories,
        n_labels_first,
        n_labels_second,
        n_columns,
        n_products,
        n_same,
    };
    let plan =
        MomentsPlan::new(shape, threads, memory_budget_bytes).map_err(|e| to_pyerr(py, e))?;
    Ok((plan.lanes.get(), plan.estimated_peak_bytes))
}

/// Relationship moments per requested category and pair-label cell (ADR
/// 0013, ADR 0015): the exact accumulators `CellReducer::finish` describes,
/// encoded at their minimal width, in a `(width, table, cells, stride,
/// lanes, lane_pairs, estimated_peak_bytes)` tuple.
///
/// Inputs are in receiver rows: `labels_first` / `labels_second` are one
/// int32 label per row below their label counts, `values` is a C-contiguous
/// int64 `[n, k]` array of quantized values, `products` names each cross
/// product as `(side_a, column_a, side_b, column_b)` with side `0` for the
/// first member and `1` for the second, and `same` is an int64 `[n, s]`
/// array of equality keys.  `symmetric` is `"canonical"` or `"both"`.  The
/// lane count is at most `threads` and is cut to fit `memory_budget_bytes`;
/// a budget one lane cannot fit raises `ResourceError` before anything is
/// allocated.  The GIL is released for the pass.
#[pyfunction]
#[pyo3(signature = (pedigree, *, max_degree, requested, threads, labels_first, n_labels_first, labels_second, n_labels_second, values, products, same, symmetric, memory_budget_bytes, view_rows=None, compact=false, progress=None, tick=1.0))]
#[allow(clippy::too_many_arguments)]
fn relationship_moments<'py>(
    py: Python<'py>,
    pedigree: &BuiltPedigree,
    max_degree: u8,
    requested: Vec<String>,
    threads: usize,
    labels_first: PyReadonlyArray1<'py, i32>,
    n_labels_first: usize,
    labels_second: PyReadonlyArray1<'py, i32>,
    n_labels_second: usize,
    values: PyReadonlyArray2<'py, i64>,
    products: Vec<(u8, usize, u8, usize)>,
    same: PyReadonlyArray2<'py, i64>,
    symmetric: &str,
    memory_budget_bytes: u64,
    view_rows: Option<PyReadonlyArray1<'py, i32>>,
    compact: bool,
    progress: Option<Py<PyAny>>,
    tick: f64,
) -> PyResult<MomentsArrays<'py>> {
    let columns = EngineColumns::borrow(py, pedigree);
    let ped = columns.pedigree(py)?;
    let max_degree = checked_max_degree(py, max_degree)?;
    let categories = requested_categories(&requested)?;
    let symmetric = Symmetric::parse(symmetric).ok_or_else(|| {
        PyValueError::new_err(format!(
            "symmetric must be \"canonical\" or \"both\", got {symmetric:?}"
        ))
    })?;
    let view = view_map(&view_rows, ped.len())?;
    if compact && view.is_none() {
        return Err(PyValueError::new_err("compact requires view_rows"));
    }
    let n_columns = values.shape()[1];
    let n_same = same.shape()[1];
    let resolved = resolve_products(products)?;
    let input = MomentsInput {
        labels_first: labels_first.as_slice()?,
        n_labels_first,
        labels_second: labels_second.as_slice()?,
        n_labels_second,
        values: values.as_slice()?,
        n_columns,
        products: &resolved,
        same: same.as_slice()?,
        n_same,
    };
    let threads = NonZeroUsize::new(threads)
        .ok_or_else(|| PyValueError::new_err("threads must be at least 1"))?;
    let pool = pool::configure(threads).map_err(|e| to_pyerr(py, e))?;
    let moments = run_watched(py, pool, tick, progress, |progress| {
        relationships::relationship_moments(
            &ped,
            max_degree,
            categories,
            view,
            compact,
            &input,
            symmetric,
            threads,
            memory_budget_bytes,
            progress,
        )
    })?;
    Ok((
        moments.width,
        moments.table.into_pyarray(py),
        moments.cells,
        moments.stride,
        moments.lanes,
        moments.lane_pairs,
        moments.estimated_peak_bytes,
    ))
}

/// Product operands as the host spells them, `(side_a, column_a, side_b,
/// column_b)` with side `0` for the first member and `1` for the second.
type ProductArg = (u8, usize, u8, usize);

fn resolve_products(products: Vec<ProductArg>) -> PyResult<Vec<Product>> {
    let side = |code: u8| {
        Side::from_code(code).ok_or_else(|| {
            PyValueError::new_err(format!(
                "product sides are 0 (first) or 1 (second), got {code}"
            ))
        })
    };
    products
        .into_iter()
        .map(|(side_a, a, side_b, b)| {
            Ok(Product {
                a: Operand {
                    side: side(side_a)?,
                    column: a,
                },
                b: Operand {
                    side: side(side_b)?,
                    column: b,
                },
            })
        })
        .collect()
}

/// What [`moments_pack`] hands back: the labels, the label count and each
/// factor's levels.
type Packed<'py> = (
    Bound<'py, PyArray1<i32>>,
    usize,
    Vec<Bound<'py, PyArray1<i64>>>,
);

/// Named factors packed by mixed radix into int32 labels (ADR 0015), as
/// `(labels, n_labels, levels)`: `levels` holds each factor's distinct
/// values ascending.  `kind` is `"first"` or `"second"`, which names the
/// label range a refusal reports.
#[pyfunction]
fn moments_pack<'py>(
    py: Python<'py>,
    factors: Vec<PyReadonlyArray1<'py, i64>>,
    n: usize,
    kind: &str,
) -> PyResult<Packed<'py>> {
    let field = match kind {
        "first" => "first label",
        "second" => "second label",
        _ => {
            return Err(PyValueError::new_err(format!(
                "kind must be 'first' or 'second', got {kind:?}"
            )))
        }
    };
    let columns = factors
        .iter()
        .map(|f| f.as_slice())
        .collect::<Result<Vec<_>, _>>()?;
    let packed = relationships::pack_labels(&columns, n, field).map_err(|e| to_pyerr(py, e))?;
    let levels = packed
        .levels
        .into_iter()
        .map(|l| l.into_pyarray(py))
        .collect();
    Ok((packed.labels.into_pyarray(py), packed.n_labels, levels))
}

/// Quantize one float64 value column into column `j` of the C-contiguous
/// int64 `[n, k]` array `out` and return its exponent (ADR 0013, ADR 0015).
/// `field` names the column in a non-finite refusal.
#[pyfunction]
fn moments_quantize(
    py: Python<'_>,
    column: PyReadonlyArray1<'_, f64>,
    mut out: PyReadwriteArray2<'_, i64>,
    j: usize,
    field: &str,
) -> PyResult<i64> {
    let k = out.shape()[1];
    let column = column.as_slice()?;
    if out.shape()[0] != column.len() || j >= k {
        return Err(PyValueError::new_err(format!(
            "out must be [{}, k] with j < k, got {:?} and j={j}",
            column.len(),
            out.shape()
        )));
    }
    let out = out.as_slice_mut()?;
    relationships::quantize_column(column, out, k, j, field).map_err(|e| to_pyerr(py, e))
}

/// `numerator / (denominator · 2^shift)` rounded once to float64 (ADR
/// 0015), the integers as decimal strings; `None` past the float64 range.
#[pyfunction]
fn moments_ratio(numerator: &str, denominator: &str, shift: i64) -> PyResult<Option<f64>> {
    let parse_error =
        |what: &str| PyValueError::new_err(format!("{what} is not a decimal integer"));
    let numerator = BigInt::from_str(numerator).map_err(|_| parse_error("numerator"))?;
    let denominator = BigUint::from_str(denominator).map_err(|_| parse_error("denominator"))?;
    if denominator == BigUint::ZERO {
        return Err(PyValueError::new_err("denominator must be positive"));
    }
    Ok(relationships::ratio(&numerator, &denominator, shift))
}

/// A moments table as the host hands it over: `(shape, n_columns, products,
/// exponents, width, data)`, `data` a uint8 array of little-endian two's
/// complement accumulators, `width` bytes each.  Core borrows `data`.
type TableArg<'py> = (
    Vec<usize>,
    usize,
    Vec<ProductArg>,
    Vec<i64>,
    usize,
    PyReadonlyArray1<'py, u8>,
);

/// A table core returns: `(shape, exponents, width, data)`, at its minimal
/// width.
type TableOut<'py> = (Vec<usize>, Vec<i64>, usize, Bound<'py, PyArray1<u8>>);

fn table<'a>(py: Python<'_>, arg: &'a TableArg<'_>) -> PyResult<MomentsTable<'a>> {
    let (shape, n_columns, products, exponents, width, data) = arg;
    MomentsTable::from_bytes(
        shape.clone(),
        *n_columns,
        resolve_products(products.clone())?,
        exponents.clone(),
        *width,
        Cow::Borrowed(data.as_slice()?),
    )
    .map_err(|e| to_pyerr(py, e))
}

fn table_out<'py>(py: Python<'py>, table: MomentsTable<'_>) -> TableOut<'py> {
    let (shape, exponents, width) = (
        table.shape().to_vec(),
        table.exponents().to_vec(),
        table.width(),
    );
    (shape, exponents, width, table.into_bytes().into_pyarray(py))
}

/// The int64 pair count of every cell of a moments table, after core has
/// checked the table: its layout, and every count in `[0, 2^63 - 1]`.
#[pyfunction]
fn moments_table_counts<'py>(
    py: Python<'py>,
    table_arg: TableArg<'py>,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    Ok(table(py, &table_arg)?.counts().into_pyarray(py))
}

/// Fold `axis` of a moments table away, exactly (ADR 0015); refuses a cell
/// whose pair count would pass 2^63 - 1.
#[pyfunction]
fn moments_table_sum<'py>(
    py: Python<'py>,
    table_arg: TableArg<'py>,
    axis: usize,
) -> PyResult<TableOut<'py>> {
    let input = table(py, &table_arg)?;
    let out = py.detach(|| input.sum(axis)).map_err(|e| to_pyerr(py, e))?;
    Ok(table_out(py, out))
}

/// Keep `positions` of `axis` of a moments table, in that order (ADR 0015).
#[pyfunction]
fn moments_table_select<'py>(
    py: Python<'py>,
    table_arg: TableArg<'py>,
    axis: usize,
    positions: Vec<usize>,
) -> PyResult<TableOut<'py>> {
    let input = table(py, &table_arg)?;
    let out = py
        .detach(|| input.select(axis, &positions))
        .map_err(|e| to_pyerr(py, e))?;
    Ok(table_out(py, out))
}

/// Add two moments tables of one layout cell by cell, aligning differing
/// exponents exactly (ADR 0015); refuses a cell whose pair count would
/// pass 2^63 - 1.
#[pyfunction]
fn moments_table_merge<'py>(
    py: Python<'py>,
    a: TableArg<'py>,
    b: TableArg<'py>,
) -> PyResult<TableOut<'py>> {
    let (left, right) = (table(py, &a)?, table(py, &b)?);
    let out = py
        .detach(|| left.merge(&right))
        .map_err(|e| to_pyerr(py, e))?;
    Ok(table_out(py, out))
}

/// One float64 per cell of `statistic` (`"sum_first"`, ..., `"pearson"`)
/// for column or product `index` of a moments table (ADR 0015); `name`
/// spells that column or product in an unrepresentable-output refusal.
#[pyfunction]
fn moments_table_derive<'py>(
    py: Python<'py>,
    table_arg: TableArg<'py>,
    statistic: &str,
    index: usize,
    name: &str,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let statistic = Statistic::parse(statistic)
        .ok_or_else(|| PyValueError::new_err(format!("unknown statistic {statistic:?}")))?;
    let input = table(py, &table_arg)?;
    let out = py
        .detach(|| input.derive(statistic, index, name))
        .map_err(|e| to_pyerr(py, e))?;
    Ok(out.into_pyarray(py))
}

/// What [`relationship_burden`] hands back: category counts keyed by code,
/// the per-person degree-1..5 counts (row-major, five per graph row) and the
/// related-pair count per structural depth.
type BurdenArrays<'py> = (
    Bound<'py, PyDict>,
    Bound<'py, PyArray1<u32>>,
    Bound<'py, PyArray1<u64>>,
);

/// Counts and per-person degree burden from one relationship traversal.
/// `depth` is structural depth in graph rows; no pair blocks are returned.
#[pyfunction]
#[pyo3(signature = (pedigree, depth, *, threads, progress=None, tick=1.0))]
fn relationship_burden<'py>(
    py: Python<'py>,
    pedigree: &BuiltPedigree,
    depth: PyReadonlyArray1<'py, i32>,
    threads: usize,
    progress: Option<Py<PyAny>>,
    tick: f64,
) -> PyResult<BurdenArrays<'py>> {
    let columns = EngineColumns::borrow(py, pedigree);
    let ped = columns.pedigree(py)?;
    let depth = depth.as_slice()?;
    check_same_length("depth", depth.len(), ped.len())?;
    if depth.iter().any(|&d| d < 0 || d as usize >= ped.len()) {
        return Err(PyValueError::new_err(
            "depth must be in the graph row range",
        ));
    }
    let pool = checked_pool(py, threads)?;
    let burden = run_watched(py, pool, tick, progress, |progress| {
        relationships::relationship_burden(&ped, depth, progress)
    })?;
    let categories = PyDict::new(py);
    for (cat, count) in Category::ALL.iter().zip(burden.categories) {
        categories.set_item(cat.code(), count)?;
    }
    Ok((
        categories,
        burden.per_person.into_pyarray(py),
        burden.same_depth.into_pyarray(py),
    ))
}

/// The credited member's side of one [`relatives_per_person`] column: one
/// float64 per receiver row, or a float that is never broadcast.
#[derive(FromPyObject)]
enum ThresholdArg<'py> {
    Rows(PyReadonlyArray1<'py, f64>),
    Scalar(f64),
}

/// Per receiver row and requested category, the relative count and the
/// count passing each threshold column, as a `(counts, rows, n_categories,
/// stride, lanes, lane_pairs)` tuple: `counts` is the flat row-major uint32
/// `[rows, n_categories, stride]` array, moved out of the core without a
/// copy, `stride` is one plus the column count, and the pass ran on `lanes`
/// lanes that credited `lane_pairs` pairs each.
///
/// `columns` is a list of `(relative, threshold)` pairs in receiver rows:
/// `relative` a contiguous float64 array, `threshold` one or a float.  A
/// relative counts for the credited row when `relative[relative row] <=
/// threshold[credited row]`; NaN never counts.  Symmetric categories credit
/// both members and directional ones the junior only.  The inputs are
/// borrowed, and the GIL is released for the pass.
#[pyfunction]
#[pyo3(signature = (pedigree, *, max_degree, requested, threads, columns, view_rows=None, compact=false, progress=None, tick=1.0))]
#[allow(clippy::too_many_arguments)]
fn relatives_per_person<'py>(
    py: Python<'py>,
    pedigree: &BuiltPedigree,
    max_degree: u8,
    requested: Vec<String>,
    threads: usize,
    columns: Vec<(PyReadonlyArray1<'py, f64>, ThresholdArg<'py>)>,
    view_rows: Option<PyReadonlyArray1<'py, i32>>,
    compact: bool,
    progress: Option<Py<PyAny>>,
    tick: f64,
) -> PyResult<RelativesArrays<'py>> {
    let engine = EngineColumns::borrow(py, pedigree);
    let ped = engine.pedigree(py)?;
    let max_degree = checked_max_degree(py, max_degree)?;
    let categories = requested_categories(&requested)?;
    let view = view_map(&view_rows, ped.len())?;
    if compact && view.is_none() {
        return Err(PyValueError::new_err("compact requires view_rows"));
    }
    let mut borrowed = Vec::with_capacity(columns.len());
    for (relative, threshold) in &columns {
        borrowed.push(ThresholdColumn {
            relative: relative.as_slice()?,
            threshold: match threshold {
                ThresholdArg::Rows(rows) => Threshold::Rows(rows.as_slice()?),
                ThresholdArg::Scalar(value) => Threshold::Scalar(*value),
            },
        });
    }
    let threads = NonZeroUsize::new(threads)
        .ok_or_else(|| PyValueError::new_err("threads must be at least 1"))?;
    let pool = pool::configure(threads).map_err(|e| to_pyerr(py, e))?;
    let relatives = run_watched(py, pool, tick, progress, |progress| {
        relationships::relatives_per_person(
            &ped, max_degree, categories, view, compact, &borrowed, threads, progress,
        )
    })?;
    Ok((
        relatives.counts.into_pyarray(py),
        relatives.rows,
        relatives.n_categories,
        relatives.stride,
        relatives.lanes,
        relatives.lane_pairs,
    ))
}

/// What [`relatives_per_person`] hands back: the flat counts, rows,
/// categories, stride, lanes and the pairs each lane credited.
type RelativesArrays<'py> = (
    Bound<'py, PyArray1<u32>>,
    usize,
    usize,
    usize,
    usize,
    Vec<u64>,
);

/// The recurrence's columns, borrowed from a graph's [`BuiltPedigree`] and
/// its cached depth for one call.
struct KinshipColumns<'py> {
    mother_rows: PyReadonlyArray1<'py, i32>,
    father_rows: PyReadonlyArray1<'py, i32>,
    twin_rows: PyReadonlyArray1<'py, i32>,
    depth: PyReadonlyArray1<'py, i32>,
}

impl<'py> KinshipColumns<'py> {
    fn borrow(
        py: Python<'py>,
        pedigree: &BuiltPedigree,
        depth: PyReadonlyArray1<'py, i32>,
    ) -> KinshipColumns<'py> {
        KinshipColumns {
            mother_rows: pedigree.mother_rows.bind(py).readonly(),
            father_rows: pedigree.father_rows.bind(py).readonly(),
            twin_rows: pedigree.twin_rows.bind(py).readonly(),
            depth,
        }
    }

    fn pedigree(&self, py: Python<'py>) -> PyResult<KinshipPedigree<'_>> {
        KinshipPedigree::try_new(
            self.mother_rows.as_slice()?,
            self.father_rows.as_slice()?,
            self.twin_rows.as_slice()?,
            self.depth.as_slice()?,
        )
        .map_err(|e| to_pyerr(py, e))
    }
}

/// Pedigree-expected kinship per requested pair (ADR 0009), in graph rows.
///
/// `pedigree` is the graph's own [`BuiltPedigree`] and `depth` its structural
/// depth.  `first` and `second` are validated graph rows of one length.  The
/// walk runs in the package pool at `threads` with the GIL released, one
/// memo per worker, each freed before the call returns.
#[pyfunction]
#[pyo3(signature = (pedigree, depth, first, second, /, *, threads))]
fn pair_kinship<'py>(
    py: Python<'py>,
    pedigree: &BuiltPedigree,
    depth: PyReadonlyArray1<'py, i32>,
    first: PyReadonlyArray1<'py, i32>,
    second: PyReadonlyArray1<'py, i32>,
    threads: usize,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let columns = KinshipColumns::borrow(py, pedigree, depth);
    let ped = columns.pedigree(py)?;
    let first = first.as_slice()?;
    let second = second.as_slice()?;
    let pool = checked_pool(py, threads)?;
    let values = py
        .detach(|| pool.install(|| kinship::pair_kinship(ped, first, second)))
        .map_err(|e| to_pyerr(py, e))?;
    Ok(values.into_pyarray(py))
}

/// The kinship of every entry of a symmetric CSC support, as its `data`.
///
/// `indptr` (int64, `n + 1` entries) and `indices` (int32, sorted within
/// each column) describe the support; the result is float32 of length
/// `nnz`, with each upper entry evaluated once and its mirror written from
/// it.  A missing mirror or an unsorted column is a validation error.  The
/// columns are walked in the package pool at `threads`.
#[pyfunction]
#[pyo3(signature = (pedigree, depth, indptr, indices, /, *, threads))]
fn kinship_support_values<'py>(
    py: Python<'py>,
    pedigree: &BuiltPedigree,
    depth: PyReadonlyArray1<'py, i32>,
    indptr: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i32>,
    threads: usize,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let columns = KinshipColumns::borrow(py, pedigree, depth);
    let ped = columns.pedigree(py)?;
    let indptr = indptr.as_slice()?;
    let indices = indices.as_slice()?;
    let pool = checked_pool(py, threads)?;
    let values = py
        .detach(|| pool.install(|| kinship::support_values(ped, indptr, indices)))
        .map_err(|e| to_pyerr(py, e))?;
    Ok(values.into_pyarray(py))
}

/// `(indptr, indices, data)` of a symmetric CSC in graph rows.
type CscArrays<'py> = (
    Bound<'py, PyArray1<i32>>,
    Bound<'py, PyArray1<i32>>,
    Bound<'py, PyArray1<f32>>,
);

fn csc_arrays(py: Python<'_>, csc: Csc) -> CscArrays<'_> {
    (
        csc.indptr.into_pyarray(py),
        csc.indices.into_pyarray(py),
        csc.data.into_pyarray(py),
    )
}

/// The complete kinship matrix as `(indptr, indices, data)`: int32 `indptr`
/// of `n + 1`, int32 `indices` and float32 `data` of `nnz`, rows ascending
/// within each column, the diagonal always present (ADR 0006, 0009).
///
/// `pedigree` is the graph's own [`BuiltPedigree`] and `depth` its structural
/// depth, which must be non-negative and strictly above both parents' for
/// every row.  The DP runs in depth-major order on the calling thread with
/// the GIL released; the arrays are moved out of the core without a copy.
#[pyfunction]
#[pyo3(signature = (pedigree, depth, /))]
fn kinship_csc<'py>(
    py: Python<'py>,
    pedigree: &BuiltPedigree,
    depth: PyReadonlyArray1<'py, i32>,
) -> PyResult<CscArrays<'py>> {
    let columns = KinshipColumns::borrow(py, pedigree, depth);
    let ped = columns.pedigree(py)?;
    let csc = py
        .detach(|| kinship::kinship_csc(ped))
        .map_err(|e| to_pyerr(py, e))?;
    Ok(csc_arrays(py, csc))
}

/// Exact values on the propagation-pruned support, as `kinship_csc` lays
/// them out.  `threshold` must be finite and in `[0, 1]`; every intermediate
/// value at or below it is dropped while the support propagates, and every
/// retained entry is then recomputed by the complete recurrence.
#[pyfunction]
#[pyo3(signature = (pedigree, depth, threshold, /))]
fn approximate_kinship_csc<'py>(
    py: Python<'py>,
    pedigree: &BuiltPedigree,
    depth: PyReadonlyArray1<'py, i32>,
    threshold: f64,
) -> PyResult<CscArrays<'py>> {
    let columns = KinshipColumns::borrow(py, pedigree, depth);
    let ped = columns.pedigree(py)?;
    let csc = py
        .detach(|| kinship::approximate_kinship_csc(ped, threshold))
        .map_err(|e| to_pyerr(py, e))?;
    Ok(csc_arrays(py, csc))
}

/// Per bucket, the float64 kinship summed over unordered same-bucket pairs
/// of distinct rows that are not MZ co-twins.  `inbreeding` is the graph's
/// float64 `F` from [`inbreeding`]; `labels` is one int32 bucket per graph
/// row, in `0..n_buckets` or `n_buckets` for none.  One backward sweep per
/// bucket; no kinship is stored and no walk is run.
#[pyfunction]
#[pyo3(signature = (pedigree, depth, inbreeding, labels, n_buckets, /))]
fn generation_kinship_sums<'py>(
    py: Python<'py>,
    pedigree: &BuiltPedigree,
    depth: PyReadonlyArray1<'py, i32>,
    inbreeding: PyReadonlyArray1<'py, f64>,
    labels: PyReadonlyArray1<'py, i32>,
    n_buckets: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let columns = KinshipColumns::borrow(py, pedigree, depth);
    let ped = columns.pedigree(py)?;
    let inbreeding = inbreeding.as_slice()?;
    let labels = labels.as_slice()?;
    let sums = py
        .detach(|| kinship::generation_kinship_sums(ped, inbreeding, labels, n_buckets))
        .map_err(|e| to_pyerr(py, e))?;
    Ok(sums.into_pyarray(py))
}

/// Inbreeding `F` per graph row, float64, and distinct strict ancestors per
/// graph row, int32: one Meuwissen-Luo walk over the genome-node pedigree
/// (ADR 0008), serial, with the GIL released.
#[pyfunction]
#[pyo3(signature = (pedigree, depth, /))]
fn inbreeding<'py>(
    py: Python<'py>,
    pedigree: &BuiltPedigree,
    depth: PyReadonlyArray1<'py, i32>,
) -> PyResult<InbreedingArrays<'py>> {
    let columns = KinshipColumns::borrow(py, pedigree, depth);
    let ped = columns.pedigree(py)?;
    let (f, ancestors) = py
        .detach(|| kinship::inbreeding_and_ancestor_counts(ped))
        .map_err(|e| to_pyerr(py, e))?;
    Ok((f.into_pyarray(py), ancestors.into_pyarray(py)))
}

/// The parent columns of a graph's [`BuiltPedigree`] and, when the host has
/// it, its structural depth, borrowed for one parents-first sweep.  The
/// columns and the parents-first flag are construction's, so they are not
/// re-validated.
struct ParentColumnsRef<'py> {
    mother_rows: PyReadonlyArray1<'py, i32>,
    father_rows: PyReadonlyArray1<'py, i32>,
    depth: Option<PyReadonlyArray1<'py, i32>>,
    rows_topological: bool,
}

impl<'py> ParentColumnsRef<'py> {
    fn borrow(
        py: Python<'py>,
        pedigree: &BuiltPedigree,
        depth: Option<PyReadonlyArray1<'py, i32>>,
    ) -> ParentColumnsRef<'py> {
        ParentColumnsRef {
            mother_rows: pedigree.mother_rows.bind(py).readonly(),
            father_rows: pedigree.father_rows.bind(py).readonly(),
            depth,
            rows_topological: pedigree.rows_topological,
        }
    }

    fn columns(&self, py: Python<'py>) -> PyResult<ParentColumns<'_>> {
        let depth = match &self.depth {
            Some(depth) => Some(depth.as_slice()?),
            None => None,
        };
        ParentColumns::validated(
            self.mother_rows.as_slice()?,
            self.father_rows.as_slice()?,
            depth,
            self.rows_topological,
        )
        .map_err(|e| to_pyerr(py, e))
    }
}

/// Distinct strict ancestors per graph row, int32.
///
/// `depth` may be `None` when the graph rows are parents-first (every parent
/// row before its children), which the sweep then walks as they are; it is
/// required, and checked structural, only when the rows need sorting.
#[pyfunction]
#[pyo3(signature = (pedigree, depth, /))]
fn distinct_ancestor_counts<'py>(
    py: Python<'py>,
    pedigree: &BuiltPedigree,
    depth: Option<PyReadonlyArray1<'py, i32>>,
) -> PyResult<Bound<'py, PyArray1<i32>>> {
    let columns = ParentColumnsRef::borrow(py, pedigree, depth);
    let ped = columns.columns(py)?;
    let counts = py
        .detach(|| lineage::distinct_ancestor_counts(ped))
        .map_err(|e| to_pyerr(py, e))?;
    Ok(counts.into_pyarray(py))
}

/// Descendant paths per graph row, int64; a count past int64 raises
/// `ResourceError("arithmetic_overflow")` rather than wrapping.
#[pyfunction]
#[pyo3(signature = (pedigree, depth, /))]
fn descendant_path_counts<'py>(
    py: Python<'py>,
    pedigree: &BuiltPedigree,
    depth: Option<PyReadonlyArray1<'py, i32>>,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let columns = ParentColumnsRef::borrow(py, pedigree, depth);
    let ped = columns.columns(py)?;
    let counts = py
        .detach(|| lineage::descendant_path_counts(ped))
        .map_err(|e| to_pyerr(py, e))?;
    Ok(counts.into_pyarray(py))
}

/// Equivalent complete generations per graph row, float64 (Maignel 1996).
#[pyfunction]
#[pyo3(signature = (pedigree, depth, /))]
fn equivalent_generations<'py>(
    py: Python<'py>,
    pedigree: &BuiltPedigree,
    depth: Option<PyReadonlyArray1<'py, i32>>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let columns = ParentColumnsRef::borrow(py, pedigree, depth);
    let ped = columns.columns(py)?;
    let values = py
        .detach(|| kinship::equivalent_generations(ped))
        .map_err(|e| to_pyerr(py, e))?;
    Ok(values.into_pyarray(py))
}

/// Per-cohort mean founder-genome contributions, float64, row-major
/// `(cohort, genome)` of `n_cohorts * n_genomes`.  `cohort` is one int32
/// cohort per graph row in `0..=n_cohorts` (`n_cohorts` for none);
/// `founder_column` is the int64 genome column a founder row seeds, or `-1`.
#[pyfunction]
#[pyo3(signature = (pedigree, depth, cohort, n_cohorts, founder_column, n_genomes, /))]
fn founder_contribution_means<'py>(
    py: Python<'py>,
    pedigree: &BuiltPedigree,
    depth: PyReadonlyArray1<'py, i32>,
    cohort: PyReadonlyArray1<'py, i32>,
    n_cohorts: usize,
    founder_column: PyReadonlyArray1<'py, i64>,
    n_genomes: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let columns = KinshipColumns::borrow(py, pedigree, depth);
    let ped = columns.pedigree(py)?;
    let cohort = cohort.as_slice()?;
    let founder_column = founder_column.as_slice()?;
    let means = py
        .detach(|| {
            kinship::founder_contribution_means(ped, cohort, n_cohorts, founder_column, n_genomes)
        })
        .map_err(|e| to_pyerr(py, e))?;
    Ok(means.into_pyarray(py))
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

/// Test hook: panic inside a native call, so the package tests can show a
/// Rust panic reaches Python as `PanicException` and leaves the process
/// usable.  Compiled only with the `test-hooks` feature.
#[cfg(feature = "test-hooks")]
#[pyfunction]
fn _panic_for_test() {
    panic!("pedigree-graph test hook: deliberate panic");
}

/// Test hook: panic inside a parallel iterator of a watched job, on a pool
/// worker, so the package tests can show the watcher wakes at once instead
/// of after a `tick`, and the pool stays usable.  Compiled only with the
/// `test-hooks` feature.
#[cfg(feature = "test-hooks")]
#[pyfunction]
#[pyo3(signature = (*, threads, tick))]
fn _panic_in_watched_worker_for_test(py: Python<'_>, threads: usize, tick: f64) -> PyResult<()> {
    use rayon::prelude::*;
    let pool = checked_pool(py, threads)?;
    run_watched(py, pool, tick, None, |_| {
        (0..64u32).into_par_iter().for_each(|i| {
            if i == 63 {
                panic!("pedigree-graph test hook: deliberate panic in a watched worker");
            }
        });
        Ok(())
    })
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
    m.add_function(wrap_pyfunction!(compact_view_counts, m)?)?;
    m.add_function(wrap_pyfunction!(relationship_pairs, m)?)?;
    m.add_function(wrap_pyfunction!(relationship_burden, m)?)?;
    m.add_function(wrap_pyfunction!(relationship_moments, m)?)?;
    m.add_function(wrap_pyfunction!(moments_plan, m)?)?;
    m.add(
        "MOMENTS_MAX_VALUE_COLUMNS",
        relationships::MAX_VALUE_COLUMNS,
    )?;
    m.add("MOMENTS_MAX_SAME_KEYS", relationships::MAX_SAME_KEYS)?;
    m.add_function(wrap_pyfunction!(moments_pack, m)?)?;
    m.add_function(wrap_pyfunction!(moments_quantize, m)?)?;
    m.add_function(wrap_pyfunction!(moments_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(moments_table_counts, m)?)?;
    m.add_function(wrap_pyfunction!(moments_table_sum, m)?)?;
    m.add_function(wrap_pyfunction!(moments_table_select, m)?)?;
    m.add_function(wrap_pyfunction!(moments_table_merge, m)?)?;
    m.add_function(wrap_pyfunction!(moments_table_derive, m)?)?;
    m.add_function(wrap_pyfunction!(relatives_per_person, m)?)?;
    m.add_function(wrap_pyfunction!(pair_kinship, m)?)?;
    m.add_function(wrap_pyfunction!(kinship_support_values, m)?)?;
    m.add_function(wrap_pyfunction!(kinship_csc, m)?)?;
    m.add_function(wrap_pyfunction!(approximate_kinship_csc, m)?)?;
    m.add_function(wrap_pyfunction!(generation_kinship_sums, m)?)?;
    m.add_function(wrap_pyfunction!(inbreeding, m)?)?;
    m.add_function(wrap_pyfunction!(distinct_ancestor_counts, m)?)?;
    m.add_function(wrap_pyfunction!(descendant_path_counts, m)?)?;
    m.add_function(wrap_pyfunction!(equivalent_generations, m)?)?;
    m.add_function(wrap_pyfunction!(founder_contribution_means, m)?)?;
    m.add_function(wrap_pyfunction!(allocation_families, m)?)?;
    m.add_function(wrap_pyfunction!(fail_next_allocation, m)?)?;
    #[cfg(feature = "test-hooks")]
    m.add_function(wrap_pyfunction!(_panic_for_test, m)?)?;
    #[cfg(feature = "test-hooks")]
    m.add_function(wrap_pyfunction!(_panic_in_watched_worker_for_test, m)?)?;
    m.add_class::<BuiltPedigree>()?;
    m.add_class::<PyIdIndex>()?;
    Ok(())
}
