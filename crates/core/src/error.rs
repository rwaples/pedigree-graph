//! Structured core errors, one variant per failure a host must distinguish.
//!
//! Hosts do not parse messages.  Every error carries a stable [`Error::code`]
//! and the operands that produced it, so the Python facade maps the code onto
//! its exception class (ADR 0006) and rebuilds the keyword fields from the
//! variant, and R raises a classed condition from the same pair.
//! [`Error::class`] names which of the three host exception families a variant
//! belongs to, which is the only routing decision a binding has to make.
//!
//! The `Display` prose is byte-identical to the Python raise site it replaces.
//! That is not part of the contract, but it keeps the migration invisible to
//! anyone reading a traceback across the boundary.

/// Row coordinates are int32, so the row count is the hard capacity limit.
pub const MAX_ROWS: usize = i32::MAX as usize;

/// The host exception family a variant maps onto.
///
/// `Metadata` carries no variant yet; it exists because the three families are
/// the boundary contract, not because every one is populated in this slice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorClass {
    /// The input violates the pedigree contract.
    Validation,
    /// A requested operation needs metadata the graph does not carry.
    Metadata,
    /// The input exceeds a representational or allocation capacity.
    Resource,
}

/// A structured core failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// Parent references form a cycle through `ids`, the deterministic witness.
    Cycle {
        /// The cycle's original ids, rotated so the smallest comes first.
        ids: Vec<i64>,
    },
    /// The pedigree has more rows than an int32 row coordinate can address.
    PedigreeTooLarge {
        /// The offending row count.
        n_individuals: usize,
        /// The capacity that was exceeded, always [`MAX_ROWS`].
        maximum: usize,
    },
}

impl Error {
    /// The host exception family this error routes to.
    pub fn class(&self) -> ErrorClass {
        match self {
            Error::Cycle { .. } => ErrorClass::Validation,
            Error::PedigreeTooLarge { .. } => ErrorClass::Resource,
        }
    }

    /// The stable code hosts branch on, matching the ADR 0006 code tables.
    pub fn code(&self) -> &'static str {
        match self {
            Error::Cycle { .. } => "cycle",
            Error::PedigreeTooLarge { .. } => "pedigree_too_large",
        }
    }
}

/// Render `ids` the way Python renders a tuple of ints, trailing comma and all.
fn python_tuple(ids: &[i64]) -> String {
    match ids {
        [] => "()".to_string(),
        [only] => format!("({only},)"),
        _ => {
            let joined: Vec<String> = ids.iter().map(|id| id.to_string()).collect();
            format!("({})", joined.join(", "))
        }
    }
}

/// Render `n` with comma thousands separators, Python's `{n:,}` format.
fn grouped(n: usize) -> String {
    let digits = n.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    let lead = digits.len() % 3;
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && i % 3 == lead {
            out.push(',');
        }
        out.push(c);
    }
    out
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Cycle { ids } => write!(
                f,
                "parent references form a cycle through ids {}",
                python_tuple(ids)
            ),
            Error::PedigreeTooLarge { n_individuals, .. } => write!(
                f,
                "pedigree has {} rows, exceeding the int32 row-coordinate capacity",
                grouped(*n_individuals)
            ),
        }
    }
}

impl std::error::Error for Error {}

#[cfg(test)]
mod tests {
    use super::{Error, ErrorClass, MAX_ROWS};

    #[test]
    fn max_rows_is_the_int32_capacity() {
        assert_eq!(MAX_ROWS, 2147483647);
    }

    #[test]
    fn cycle_is_a_validation_error() {
        let err = Error::Cycle { ids: vec![7] };
        assert_eq!(err.class(), ErrorClass::Validation);
        assert_eq!(err.code(), "cycle");
    }

    #[test]
    fn pedigree_too_large_is_a_resource_error() {
        let err = Error::PedigreeTooLarge {
            n_individuals: 2147483648,
            maximum: MAX_ROWS,
        };
        assert_eq!(err.class(), ErrorClass::Resource);
        assert_eq!(err.code(), "pedigree_too_large");
    }

    #[test]
    fn one_id_renders_with_a_trailing_comma() {
        assert_eq!(
            Error::Cycle { ids: vec![7] }.to_string(),
            "parent references form a cycle through ids (7,)"
        );
    }

    #[test]
    fn two_ids_render_comma_space_separated() {
        assert_eq!(
            Error::Cycle { ids: vec![3, 8] }.to_string(),
            "parent references form a cycle through ids (3, 8)"
        );
    }

    #[test]
    fn three_ids_render_in_witness_order() {
        assert_eq!(
            Error::Cycle {
                ids: vec![20, 40, 30]
            }
            .to_string(),
            "parent references form a cycle through ids (20, 40, 30)"
        );
    }

    #[test]
    fn no_ids_render_as_the_empty_tuple() {
        assert_eq!(
            Error::Cycle { ids: Vec::new() }.to_string(),
            "parent references form a cycle through ids ()"
        );
    }

    #[test]
    fn row_count_renders_thousands_grouped() {
        assert_eq!(
            Error::PedigreeTooLarge {
                n_individuals: 2147483648,
                maximum: MAX_ROWS,
            }
            .to_string(),
            "pedigree has 2,147,483,648 rows, exceeding the int32 row-coordinate capacity"
        );
    }

    #[test]
    fn grouping_handles_every_leading_group_width() {
        for (n, expected) in [
            (0usize, "0"),
            (7, "7"),
            (42, "42"),
            (999, "999"),
            (1000, "1,000"),
            (12345, "12,345"),
            (100000, "100,000"),
            (1000000, "1,000,000"),
        ] {
            assert_eq!(super::grouped(n), expected, "grouping {n}");
        }
    }
}
