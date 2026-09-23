//! Host coercion of R columns to int64 (slice 16, decision 3).
//!
//! The rules are Python's (`pedigree_graph/_input.py`): integers pass, an
//! integral finite double passes, a null is `-1`, and anything with no
//! lossless integer form is `invalid_integer_value`.  `bit64::integer64` is a
//! double vector whose bits are an `i64`; it is read without depending on
//! bit64.  An all-`NA` logical column is all nulls.  Positions in errors
//! are 1-based.

use crate::errors::{HostError, HostResult};
use extendr_api::prelude::*;

/// The R storage an id column arrived in, so ids can go back out in it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IdType {
    Integer,
    Double,
    Integer64,
}

impl IdType {
    pub fn name(self) -> &'static str {
        match self {
            IdType::Integer => "integer",
            IdType::Double => "double",
            IdType::Integer64 => "integer64",
        }
    }

    pub fn parse(name: &str) -> Option<IdType> {
        [IdType::Integer, IdType::Double, IdType::Integer64]
            .into_iter()
            .find(|t| t.name() == name)
    }
}

/// One coerced column: its values with nulls as `-1`, and its storage.
pub struct Coerced {
    pub values: Vec<i64>,
    pub nulls: Vec<bool>,
    pub storage: IdType,
}

const TWO_POW_63: f64 = 9_223_372_036_854_775_808.0;

fn invalid(field: &'static str, position: usize, value: Robj) -> HostError {
    HostError::validation(
        "invalid_integer_value",
        format!(
            "'{field}' value at position {} is not a lossless integer",
            position + 1
        ),
        vec![
            ("field", field.into()),
            ("position", ((position + 1) as f64).into()),
            ("value", value),
        ],
    )
}

fn out_of_range(field: &'static str, position: usize, value: f64) -> HostError {
    HostError::validation(
        "value_out_of_range",
        format!(
            "'{field}' value at position {} is outside the int64 range",
            position + 1
        ),
        vec![
            ("field", field.into()),
            ("position", ((position + 1) as f64).into()),
            ("value", value.into()),
            ("minimum", (i64::MIN as f64).into()),
            ("maximum", (i64::MAX as f64).into()),
        ],
    )
}

/// The R class (or `typeof`) an unsupported column reports as its `value`.
fn describe(column: &Robj) -> Robj {
    if let Some(class) = column.class().and_then(|mut c| c.next()) {
        return class.into();
    }
    match column.rtype() {
        Rtype::Logicals => "logical",
        Rtype::Strings => "character",
        Rtype::Complexes => "complex",
        Rtype::List => "list",
        Rtype::Raw => "raw",
        _ => "unsupported",
    }
    .into()
}

/// Coerce one column to int64, nulls to `-1`.
pub fn coerce(field: &'static str, column: &Robj) -> HostResult<Coerced> {
    let n = column.len();
    // An empty column of any atomic type has a lossless (empty) form, as in Python.
    if n == 0 && column.rtype() == Rtype::Strings {
        return Ok(Coerced {
            values: Vec::new(),
            nulls: Vec::new(),
            storage: IdType::Integer,
        });
    }
    if column.inherits("factor") {
        return Err(invalid(field, 0, "factor".into()));
    }
    match column.rtype() {
        Rtype::Integers => {
            let slice = column.as_integer_slice().expect("an integer vector");
            let nulls: Vec<bool> = slice.iter().map(|&v| v == i32::MIN).collect();
            let values = slice
                .iter()
                .map(|&v| if v == i32::MIN { -1 } else { i64::from(v) })
                .collect();
            Ok(Coerced {
                values,
                nulls,
                storage: IdType::Integer,
            })
        }
        Rtype::Doubles if column.inherits("integer64") => {
            let slice = column.as_real_slice().expect("a double vector");
            let raw: Vec<i64> = slice.iter().map(|v| v.to_bits() as i64).collect();
            let nulls: Vec<bool> = raw.iter().map(|&v| v == i64::MIN).collect();
            let values = raw
                .into_iter()
                .map(|v| if v == i64::MIN { -1 } else { v })
                .collect();
            Ok(Coerced {
                values,
                nulls,
                storage: IdType::Integer64,
            })
        }
        Rtype::Doubles => {
            let slice = column.as_real_slice().expect("a double vector");
            let mut values = Vec::with_capacity(n);
            let mut nulls = Vec::with_capacity(n);
            for (position, &v) in slice.iter().enumerate() {
                if v.is_nan() {
                    values.push(-1);
                    nulls.push(true);
                    continue;
                }
                if !v.is_finite() || v != v.trunc() {
                    return Err(invalid(field, position, v.into()));
                }
                if !(-TWO_POW_63..TWO_POW_63).contains(&v) {
                    return Err(out_of_range(field, position, v));
                }
                values.push(v as i64);
                nulls.push(false);
            }
            Ok(Coerced {
                values,
                nulls,
                storage: IdType::Double,
            })
        }
        // `NA` alone is a logical in R, so an all-`NA` column (`mother = NA`)
        // is a column of nulls, as a column of `None` is in Python.  Any
        // TRUE/FALSE has no integer form.
        Rtype::Logicals => {
            let slice = column.as_logical_slice().expect("a logical vector");
            match slice.iter().position(|v| !v.is_na()) {
                Some(position) => Err(invalid(field, position, slice[position].to_bool().into())),
                None => Ok(Coerced {
                    values: vec![-1; n],
                    nulls: vec![true; n],
                    storage: IdType::Integer,
                }),
            }
        }
        _ => Err(invalid(field, 0, describe(column))),
    }
}

/// Coerce the required `id` column, which has no null form.
pub fn coerce_ids(column: &Robj) -> HostResult<Coerced> {
    let coerced = coerce("id", column)?;
    if let Some(position) = coerced.nulls.iter().position(|&null| null) {
        let na = Strings::from_values([Rstr::na()]).into_robj();
        return Err(invalid("id", position, na));
    }
    Ok(coerced)
}
