//! Relationship moments for R (ADR 0013, ADR 0015).
//!
//! Core packs the factors, quantizes the values, runs the engine and owns
//! every operation on the result, exactly as for Python; this module only
//! turns R vectors into core's inputs and core's outputs into R vectors.  A
//! table crosses as the list the R object keeps (`shape`, `n_columns`,
//! `operands`, `exponents`, `width`, `table`), its accumulators a raw
//! vector that core borrows.

use crate::errors::{HostError, HostResult};
use crate::input::doubles;
use crate::job;
use crate::kernels::{exact_double, package_pool, selection, walk_of, Native};
use crate::threads;
use extendr_api::prelude::*;
use num_bigint::BigInt;
use pedigree_graph_core::error::Error;
use pedigree_graph_core::relationships::{
    encode_big, pack_labels, quantize_column, relationship_moments as moments_of, Moments,
    MomentsInput, MomentsPlan, MomentsTable, PackedLabels, Product, Progress, Side, Statistic,
    Symmetric, MAX_SAME_KEYS, MAX_VALUE_COLUMNS,
};
use std::borrow::Cow;
use std::num::NonZeroUsize;
use std::str::FromStr;

/// The named columns of a `first`, `second`, `values` or `same` list.
pub(crate) fn named_columns(kind: &str, list: &Robj) -> HostResult<Vec<(String, Robj)>> {
    if list.is_null() {
        return Ok(Vec::new());
    }
    let list = List::try_from(list.clone())
        .map_err(|_| HostError::usage(format!("`{kind}` must be a named list of columns")))?;
    if list.is_empty() {
        return Ok(Vec::new());
    }
    let names: Vec<String> = match list.names() {
        Some(names) => names.map(str::to_string).collect(),
        None => Vec::new(),
    };
    if names.len() != list.len() || names.iter().any(|n| n.is_empty() || n == "NA") {
        return Err(HostError::usage(format!(
            "every column of `{kind}` must be named"
        )));
    }
    let mut seen = names.clone();
    seen.sort();
    seen.dedup();
    if seen.len() != names.len() {
        return Err(HostError::usage(format!(
            "`{kind}` names a column more than once"
        )));
    }
    Ok(names.into_iter().zip(list.values()).collect())
}

pub(crate) fn check_length(field: &str, column: &Robj, n: usize) -> HostResult<()> {
    if column.len() != n {
        return Err(HostError::validation(
            "length_mismatch",
            format!(
                "{field} has length {}, expected {n} from the graph",
                column.len()
            ),
            vec![
                ("field", field.into()),
                ("expected_length", (n as f64).into()),
                ("actual_length", (column.len() as f64).into()),
            ],
        ));
    }
    Ok(())
}

/// A factor, integer, logical or whole-number double column as int64 codes.
/// `NA` is refused, or with `na` given (an equality key) it is that value.
fn factor_codes(field: &str, column: &Robj, n: usize, na: Option<i64>) -> HostResult<Vec<i64>> {
    check_length(field, column, n)?;
    let missing = |position: usize| match na {
        Some(value) => Ok(value),
        None => Err(HostError::usage(format!(
            "{field} is NA at row {}; a factor level cannot be missing",
            position + 1
        ))),
    };
    if let Some(values) = column.as_integer_slice() {
        // Factors land here too: their integer codes are the levels.
        return values
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                if v == i32::MIN {
                    missing(i)
                } else {
                    Ok(i64::from(v))
                }
            })
            .collect();
    }
    if let Some(values) = column.as_logical_slice() {
        return values
            .iter()
            .enumerate()
            .map(|(i, v)| {
                if v.is_na() {
                    missing(i)
                } else {
                    Ok(i64::from(v.is_true()))
                }
            })
            .collect();
    }
    if let Some(values) = column.as_real_slice() {
        return values
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                if v.is_nan() {
                    missing(i)
                } else if v == v.trunc() && v.abs() < 9.223_372_036_854_776e18 {
                    Ok(v as i64)
                } else {
                    Err(HostError::usage(format!(
                        "{field} must hold whole numbers; row {} is {v}",
                        i + 1
                    )))
                }
            })
            .collect();
    }
    let hint = if column.is_string() {
        "; convert it with factor()"
    } else {
        ""
    };
    Err(HostError::usage(format!(
        "{field} must be a factor, integer, logical or whole-number double vector{hint}"
    )))
}

/// A numeric value column as float64; `NA` becomes NaN, which core refuses.
fn value_column(field: &str, column: &Robj, n: usize) -> HostResult<Vec<f64>> {
    check_length(field, column, n)?;
    if let Some(values) = doubles(field, column)? {
        return Ok(values);
    }
    if let Some(values) = column.as_logical_slice() {
        return Ok(values
            .iter()
            .map(|v| {
                if v.is_na() {
                    f64::NAN
                } else {
                    f64::from(u8::from(v.is_true()))
                }
            })
            .collect());
    }
    Err(HostError::usage(format!(
        "{field} must be a numeric vector"
    )))
}

fn operand(name: &str, columns: &[String]) -> HostResult<(Side, usize)> {
    let (side, column) = name.split_once('.').unwrap_or(("", name));
    let side = match side {
        "first" => Some(Side::First),
        "second" => Some(Side::Second),
        _ => None,
    };
    match (side, columns.iter().position(|c| c == column)) {
        (Some(side), Some(at)) => Ok((side, at)),
        _ => Err(HostError::usage(format!(
            "product operand '{name}' must be 'first.<column>' or 'second.<column>' over the values ({})",
            columns.join(", ")
        ))),
    }
}

/// Each product's `(a, b)` operand names, as the R object keeps them.
type ProductNames = Vec<(String, String)>;

/// The products as `(a, b)` names and core operands; `NULL` is the
/// first x second product of every column.
fn products(products: &Robj, columns: &[String]) -> HostResult<(ProductNames, Vec<Product>)> {
    let named: Vec<(String, String)> = if products.is_null() {
        columns
            .iter()
            .map(|c| (format!("first.{c}"), format!("second.{c}")))
            .collect()
    } else {
        let list = List::try_from(products.clone()).map_err(|_| {
            HostError::usage("`products` must be a list of length-2 character vectors".to_string())
        })?;
        list.values()
            .map(|pair| match pair.as_str_vector() {
                Some(names) if names.len() == 2 => Ok((names[0].to_string(), names[1].to_string())),
                _ => Err(HostError::usage(
                    "each product must be a length-2 character vector of operand names".to_string(),
                )),
            })
            .collect::<HostResult<_>>()?
    };
    let resolved = named
        .iter()
        .map(|(a, b)| {
            Ok(Product {
                a: operand(a, columns)?,
                b: operand(b, columns)?,
            })
        })
        .collect::<HostResult<_>>()?;
    Ok((named, resolved))
}

/// A whole number of bytes from an R double, saturating at `u64::MAX`.
fn budget_bytes(budget: f64) -> HostResult<u64> {
    if !(budget.is_finite() && budget >= 0.0 && budget == budget.trunc()) {
        return Err(HostError::usage(format!(
            "memory_budget_bytes must be a whole number of bytes from 0, got {budget}"
        )));
    }
    Ok(if budget >= 18_446_744_073_709_551_615.0 {
        u64::MAX
    } else {
        budget as u64
    })
}

/// The levels of each packed factor, as doubles (every one came from an R
/// integer or a whole double, so each is exact).
fn levels_list(levels: &[Vec<i64>]) -> Robj {
    List::from_values(
        levels
            .iter()
            .map(|l| Doubles::from_values(l.iter().map(|&v| v as f64)).into_robj()),
    )
    .into_robj()
}

fn table_out(table: MomentsTable<'_>) -> Robj {
    let (shape, exponents, width) = (
        table.shape().iter().map(|&s| s as i32).collect::<Vec<_>>(),
        table
            .exponents()
            .iter()
            .map(|&e| e as i32)
            .collect::<Vec<_>>(),
        table.width() as i32,
    );
    let bytes = Raw::from_bytes(table.bytes()).into_robj();
    List::from_names_and_values(
        ["shape", "exponents", "width", "table"],
        [
            Integers::from_values(shape).into_robj(),
            Integers::from_values(exponents).into_robj(),
            width.into(),
            bytes,
        ],
    )
    .expect("four names and four values")
    .into_robj()
}

/// Start `relationship_moments()`; collecting the job gives the list the R
/// object is built from: the categories, each factor's levels, and the
/// encoded table with its exponents and the pass's lanes, lane pairs and
/// planned peak.
#[allow(clippy::too_many_arguments)]
pub fn start_moments(
    native: &Robj,
    seal: &Robj,
    max_degree: &Robj,
    categories: &Robj,
    first: &Robj,
    second: &Robj,
    values: &Robj,
    products_arg: &Robj,
    same: &Robj,
    symmetric: &str,
    memory_budget_bytes: f64,
) -> HostResult<Robj> {
    let graph = Native::verified(native, seal)?;
    let requested = selection(max_degree, categories)?;
    let symmetric = Symmetric::parse(symmetric).ok_or_else(|| {
        HostError::usage(format!(
            "symmetric must be \"canonical\" or \"both\", got {symmetric:?}"
        ))
    })?;
    let budget = budget_bytes(memory_budget_bytes)?;
    let n = graph.len();

    let pack = |kind: &str, field: &'static str, list: &Robj| -> HostResult<_> {
        let columns = named_columns(kind, list)?
            .iter()
            .map(|(name, column)| factor_codes(&format!("{kind}['{name}']"), column, n, None))
            .collect::<HostResult<Vec<_>>>()?;
        let slices: Vec<&[i64]> = columns.iter().map(Vec::as_slice).collect();
        Ok(pack_labels(&slices, n, field)?)
    };
    let first_packed = pack("first", "first label", first)?;
    let second_packed = if second.is_null() {
        None
    } else {
        Some(pack("second", "second label", second)?)
    };

    let value_columns = named_columns("values", values)?;
    if value_columns.len() > MAX_VALUE_COLUMNS {
        return Err(HostError::usage(format!(
            "at most {MAX_VALUE_COLUMNS} value columns are supported, got {}",
            value_columns.len()
        )));
    }
    let names: Vec<String> = value_columns.iter().map(|(name, _)| name.clone()).collect();
    let k = names.len();
    let mut quantized = vec![0i64; n * k];
    let mut exponents = Vec::with_capacity(k);
    for (j, (name, column)) in value_columns.iter().enumerate() {
        let field = format!("values['{name}']");
        let column = value_column(&field, column, n)?;
        let e =
            quantize_column(&column, &mut quantized, k, j, &field).map_err(|err| match err {
                // R counts rows from 1.
                Error::NonFiniteValue { field, position } => HostError::usage(format!(
                    "{field} is not finite at row {}; mask it with a factor level instead",
                    position + 1
                )),
                other => other.into(),
            })?;
        exponents.push(e);
    }
    let (product_names, resolved) = products(products_arg, &names)?;

    let keys = named_columns("same", same)?;
    if keys.len() > MAX_SAME_KEYS {
        return Err(HostError::usage(format!(
            "at most {MAX_SAME_KEYS} same keys are supported, got {}",
            keys.len()
        )));
    }
    let s = keys.len();
    let mut same_keys = vec![0i64; n * s];
    for (j, (name, column)) in keys.iter().enumerate() {
        // NA is an unknown key, never equal, as a negative one is.
        let codes = factor_codes(&format!("same['{name}']"), column, n, Some(-1))?;
        for (row, code) in codes.into_iter().enumerate() {
            same_keys[row * s + j] = code;
        }
    }

    let threads = NonZeroUsize::new(threads::budget()?).expect("a budget of at least 1");
    let pool = package_pool()?;
    let walk = walk_of(&graph, requested)?;
    let operands: Vec<i32> = resolved
        .iter()
        .flat_map(|p| [p.a, p.b])
        .flat_map(|(side, column)| [i32::from(side == Side::Second), column as i32])
        .collect();
    let PackedLabels {
        labels: first_labels,
        n_labels: n_first,
        levels: first_levels,
    } = first_packed;
    let (second_labels, second_levels) = match second_packed {
        Some(p) => (Some((p.labels, p.n_labels)), Some(p.levels)),
        None => (None, None),
    };
    let work = move |progress: &Progress| {
        let (labels_second, n_labels_second) = match &second_labels {
            Some((labels, n)) => (labels.as_slice(), *n),
            None => (first_labels.as_slice(), n_first),
        };
        let input = MomentsInput {
            labels_first: &first_labels,
            n_labels_first: n_first,
            labels_second,
            n_labels_second,
            values: &quantized,
            n_columns: k,
            products: &resolved,
            same: &same_keys,
            n_same: s,
        };
        match walk {
            Some((max_degree, rows)) => moments_of(
                &rows.pedigree()?,
                max_degree,
                requested,
                None,
                false,
                &input,
                symmetric,
                threads,
                budget,
                progress,
            ),
            None => {
                let plan = MomentsPlan::new(input.shape(requested), threads, budget)?;
                let accumulators = requested.iter().count() * plan.cells() * plan.stride();
                Ok(Moments {
                    width: 1,
                    table: vec![0u8; accumulators],
                    stride: plan.stride(),
                    cells: plan.cells(),
                    lanes: 0,
                    lane_pairs: Vec::new(),
                    estimated_peak_bytes: plan.estimated_peak_bytes,
                })
            }
        }
    };
    let finish = move |moments: Moments| {
        let codes: Vec<&str> = requested.iter().map(|c| c.code()).collect();
        Ok(List::from_names_and_values(
            [
                "categories",
                "first_levels",
                "second_levels",
                "columns",
                "products",
                "operands",
                "exponents",
                "width",
                "table",
                "lanes",
                "lane_pairs",
                "estimated_peak_bytes",
            ],
            [
                Strings::from_values(codes).into_robj(),
                levels_list(&first_levels),
                second_levels
                    .as_deref()
                    .map_or_else(|| Robj::from(()), levels_list),
                Strings::from_values(&names).into_robj(),
                List::from_values(
                    product_names
                        .iter()
                        .map(|(a, b)| Strings::from_values([a.as_str(), b.as_str()]).into_robj()),
                )
                .into_robj(),
                Integers::from_values(operands).into_robj(),
                Integers::from_values(exponents.iter().map(|&e| e as i32)).into_robj(),
                (moments.width as i32).into(),
                Raw::from_bytes(&moments.table).into_robj(),
                (moments.lanes as f64).into(),
                Doubles::from_values(moments.lane_pairs.iter().map(|&p| p as f64)).into_robj(),
                (moments.estimated_peak_bytes as f64).into(),
            ],
        )
        .expect("one value per name")
        .into_robj())
    };
    Ok(job::spawn(pool, work, finish))
}

fn int_field<'a>(m: &'a List, name: &str) -> HostResult<&'a [i32]> {
    m.dollar(name)
        .ok()
        .and_then(|v| v.as_integer_slice())
        .ok_or_else(|| {
            HostError::usage(format!(
                "a relationship_moments object needs an integer `{name}`"
            ))
        })
}

/// Run `f` on the core table an R moments object holds, borrowing its raw
/// accumulators; core checks the table's consistency.
fn with_table<T>(m: &Robj, f: impl FnOnce(&MomentsTable<'_>) -> HostResult<T>) -> HostResult<T> {
    let list = List::try_from(m.clone())
        .map_err(|_| HostError::usage("not a relationship_moments table".to_string()))?;
    let shape = int_field(&list, "shape")?;
    let n_columns = int_field(&list, "n_columns")?;
    let operands = int_field(&list, "operands")?;
    let exponents = int_field(&list, "exponents")?;
    let width = int_field(&list, "width")?;
    let raw = list.dollar("table").map_err(|_| {
        HostError::usage("a relationship_moments object needs a raw `table`".to_string())
    })?;
    let bytes = raw.as_raw_slice().ok_or_else(|| {
        HostError::usage("a relationship_moments object needs a raw `table`".to_string())
    })?;
    if shape
        .iter()
        .chain(n_columns)
        .chain(width)
        .chain(operands)
        .any(|&v| v < 0)
        || n_columns.len() != 1
        || width.len() != 1
        || operands.len() % 4 != 0
    {
        return Err(HostError::usage(
            "a malformed relationship_moments object".to_string(),
        ));
    }
    let side = |code: i32| if code == 0 { Side::First } else { Side::Second };
    let products = operands
        .chunks_exact(4)
        .map(|o| Product {
            a: (side(o[0]), o[1] as usize),
            b: (side(o[2]), o[3] as usize),
        })
        .collect();
    let table = MomentsTable::from_bytes(
        shape.iter().map(|&s| s as usize).collect(),
        n_columns[0] as usize,
        products,
        exponents.iter().map(|&e| i64::from(e)).collect(),
        width[0] as usize,
        Cow::Borrowed(bytes),
    )?;
    f(&table)
}

/// 1-based R positions to 0-based ones.
fn zero_based(what: &str, values: &[i32]) -> HostResult<Vec<usize>> {
    values
        .iter()
        .map(|&v| {
            if v >= 1 {
                Ok(v as usize - 1)
            } else {
                Err(HostError::usage(format!(
                    "{what} must be 1-based positions"
                )))
            }
        })
        .collect()
}

pub fn select(m: &Robj, axis: i32, positions: &[i32]) -> HostResult<Robj> {
    let axis = zero_based("axis", &[axis])?[0];
    let positions = zero_based("positions", positions)?;
    with_table(m, |t| Ok(table_out(t.select(axis, &positions)?)))
}

pub fn sum(m: &Robj, axis: i32) -> HostResult<Robj> {
    let axis = zero_based("axis", &[axis])?[0];
    with_table(m, |t| Ok(table_out(t.sum(axis)?)))
}

pub fn merge(a: &Robj, b: &Robj) -> HostResult<Robj> {
    with_table(a, |a| with_table(b, |b| Ok(table_out(a.merge(b)?))))
}

pub fn derive(m: &Robj, statistic: &str, index: i32, name: &str) -> HostResult<Robj> {
    let statistic = Statistic::parse(statistic)
        .ok_or_else(|| HostError::usage(format!("unknown statistic {statistic:?}")))?;
    let index = zero_based("index", &[index])?[0];
    with_table(m, |t| {
        Ok(Doubles::from_values(t.derive(statistic, index, name)?).into_robj())
    })
}

/// Every cell's pair count as a double, refused past 2^53.
pub fn counts(m: &Robj) -> HostResult<Robj> {
    with_table(m, |t| {
        let counts = t
            .counts()
            .into_iter()
            .map(|n| exact_double("a cell", n as u64))
            .collect::<HostResult<Vec<_>>>()?;
        Ok(Doubles::from_values(counts).into_robj())
    })
}

/// The pairs of every cell added exactly, as a decimal string.
pub fn total(m: &Robj) -> HostResult<Robj> {
    with_table(m, |t| Ok(t.total_pairs().to_string().into()))
}

/// Every accumulator, cell by cell, as decimal strings.
pub fn exact(m: &Robj) -> HostResult<Robj> {
    with_table(m, |t| {
        let values = (0..t.cells())
            .flat_map(|cell| (0..t.stride()).map(move |at| (cell, at)))
            .map(|(cell, at)| t.read(cell, at).to_string())
            .collect::<Vec<_>>();
        Ok(Strings::from_values(values).into_robj())
    })
}

/// Decimal integers at their minimal width, as `list(width, table)`.
pub fn encode(values: &Robj) -> HostResult<Robj> {
    let strings = values
        .as_str_vector()
        .ok_or_else(|| HostError::usage("values must be decimal strings".to_string()))?;
    let ints = strings
        .iter()
        .map(|s| {
            BigInt::from_str(s)
                .map_err(|_| HostError::usage(format!("{s:?} is not a decimal integer")))
        })
        .collect::<HostResult<Vec<_>>>()?;
    let (width, bytes) = encode_big(&ints);
    Ok(List::from_names_and_values(
        ["width", "table"],
        [(width as i32).into(), Raw::from_bytes(&bytes).into_robj()],
    )
    .expect("two names and two values")
    .into_robj())
}
