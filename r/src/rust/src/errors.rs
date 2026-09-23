//! Failures handed to R as data, never raised from Rust (slice 16, decision 6).
//!
//! A binding that fails returns a list classed `pedigree_graph_native_error`
//! carrying the condition class, code, message and fields; the R wrapper
//! `.pg_call()` turns it into a classed condition.  Raising from Rust would
//! give an untyped `simpleError` and unwind through R's longjmp.

use extendr_api::prelude::*;
use pedigree_graph_core::error::{Error, ErrorClass, FieldValue};

/// One failure on its way to R.
pub struct HostError {
    class: &'static str,
    code: Option<&'static str>,
    message: String,
    fields: Vec<(&'static str, Robj)>,
}

pub type HostResult<T> = std::result::Result<T, HostError>;

impl HostError {
    /// A pedigree-contract failure the binding detects itself (host coercion).
    pub fn validation(
        code: &'static str,
        message: String,
        fields: Vec<(&'static str, Robj)>,
    ) -> HostError {
        HostError {
            class: "validation",
            code: Some(code),
            message,
            fields,
        }
    }

    /// A capacity the binding itself enforces (an R representation limit).
    pub fn resource(
        code: &'static str,
        message: String,
        fields: Vec<(&'static str, Robj)>,
    ) -> HostError {
        HostError {
            class: "resource",
            code: Some(code),
            message,
            fields,
        }
    }

    /// API misuse: a bad argument, with no code and no fields.
    pub fn usage(message: String) -> HostError {
        HostError {
            class: "usage",
            code: None,
            message,
            fields: Vec::new(),
        }
    }

    /// A graph whose sealed fields no longer match their seal.
    pub fn graph_modified() -> HostError {
        HostError {
            class: "usage",
            code: Some("graph_modified"),
            message: "the graph's fields were modified after pedigree_graph() built it; \
                      rebuild it with pedigree_graph()"
                .to_string(),
            fields: Vec::new(),
        }
    }

    /// The R list `.pg_call()` signals.
    pub fn into_robj(self) -> Robj {
        let (names, values): (Vec<&str>, Vec<Robj>) = self.fields.into_iter().unzip();
        let fields = List::from_names_and_values(names, values)
            .expect("names and values have one entry each")
            .into_robj();
        let code: Robj = match self.code {
            Some(code) => code.into(),
            None => Strings::from_values([Rstr::na()]).into_robj(),
        };
        let mut out = List::from_names_and_values(
            ["class", "code", "message", "fields"],
            [self.class.into(), code, self.message.into(), fields],
        )
        .expect("four names and four values")
        .into_robj();
        out.set_class(["pedigree_graph_native_error"])
            .expect("a character class attribute");
        out
    }
}

impl From<Error> for HostError {
    fn from(err: Error) -> HostError {
        let class = match (&err, err.class()) {
            (Error::ThreadPoolConflict { .. }, _) => "thread_conflict",
            (_, ErrorClass::Validation) => "validation",
            (_, ErrorClass::Metadata) => "metadata",
            (_, ErrorClass::Resource) => "resource",
            (_, ErrorClass::Usage) => "usage",
        };
        let code = match err.code() {
            "" => None,
            code => Some(code),
        };
        let fields = err
            .fields()
            .into_iter()
            .map(|(name, value)| (name, field_robj(name, value)))
            .collect();
        HostError {
            class,
            code,
            message: err.to_string(),
            fields,
        }
    }
}

/// Whether a core field holds 0-based row or column indices, which R reports 1-based.
///
/// The rule is by name, so a new core field spelled like these converts
/// without a change here.
fn is_index_field(name: &str) -> bool {
    matches!(name, "row" | "rows" | "column" | "columns" | "position")
        || ["_row", "_rows", "_column", "_columns"]
            .iter()
            .any(|suffix| name.ends_with(suffix))
}

/// The largest magnitude a double holds exactly.
const EXACT_IN_DOUBLE: i64 = 1 << 53;

/// Integer field values: doubles when every value is exact in one, else
/// bit64 `integer64`, so an id above 2^53 is never reported as its neighbour.
fn int_field(values: Vec<i64>, shift: i64) -> Robj {
    let values: Vec<i64> = values.into_iter().map(|n| n + shift).collect();
    if values
        .iter()
        .all(|n| n.unsigned_abs() <= EXACT_IN_DOUBLE as u64)
    {
        return Doubles::from_values(values.iter().map(|&n| n as f64)).into_robj();
    }
    let mut out =
        Doubles::from_values(values.iter().map(|&n| f64::from_bits(n as u64))).into_robj();
    out.set_class(["integer64"])
        .expect("a character class attribute");
    out
}

/// A core field value as R holds it.
fn field_robj(name: &str, value: FieldValue) -> Robj {
    let shift = i64::from(is_index_field(name));
    match value {
        FieldValue::Int(n) => int_field(vec![n], shift),
        FieldValue::Str(s) => s.into(),
        FieldValue::Ints(v) => int_field(v, shift),
        FieldValue::Strs(v) => Strings::from_values(v).into_robj(),
    }
}

/// A binding's value, or its failure as the list `.pg_call()` signals.
pub fn finish(result: HostResult<Robj>) -> Robj {
    result.unwrap_or_else(HostError::into_robj)
}

#[cfg(test)]
mod tests {
    use super::is_index_field;

    #[test]
    fn index_fields_are_named_rows_or_columns() {
        for name in [
            "row",
            "rows",
            "column",
            "position",
            "child_row",
            "parent_row",
        ] {
            assert!(is_index_field(name), "{name}");
        }
        for name in [
            "id",
            "ids",
            "field",
            "maximum",
            "expected_length",
            "twin_id",
            "nnz",
        ] {
            assert!(!is_index_field(name), "{name}");
        }
    }
}
