//! Structured core errors, one variant per failure a host must distinguish.
//!
//! Hosts do not parse messages.  Every error carries a stable [`Error::code`]
//! and the operands that produced it, so the Python facade maps the code onto
//! its exception class (ADR 0006) and rebuilds the keyword fields from
//! [`Error::fields`], and R raises a classed condition from the same pair.
//! [`Error::class`] names which of the host exception families a variant
//! belongs to, which is the only routing decision a binding has to make.
//!
//! The `Display` prose is byte-identical to the Python raise site it replaces.
//! That is not part of the contract, but it keeps the migration invisible to
//! anyone reading a traceback across the boundary.

/// Row coordinates are int32, so the row count is the hard capacity limit.
pub const MAX_ROWS: usize = i32::MAX as usize;

/// The host exception family a variant maps onto.
///
/// `Metadata` carries no variant yet; it exists because the three structured
/// families are the boundary contract, not because every one is populated.
/// `Usage` is API misuse (a bad argument, not bad pedigree data) and maps onto
/// the host's plain argument error with no code and no fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorClass {
    /// The input violates the pedigree contract.
    Validation,
    /// A requested operation needs metadata the graph does not carry.
    Metadata,
    /// The input exceeds a representational or allocation capacity.
    Resource,
    /// The caller passed an argument the API does not define.
    Usage,
}

/// The parent role an error names, rendered as the host spells it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParentRole {
    /// The mother column.
    Mother,
    /// The father column.
    Father,
}

impl ParentRole {
    /// The host-facing name.
    pub fn name(self) -> &'static str {
        match self {
            ParentRole::Mother => "mother",
            ParentRole::Father => "father",
        }
    }
}

/// One keyword field of a structured error, in the value kinds hosts receive.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FieldValue {
    /// An integer scalar.
    Int(i64),
    /// A fixed name (a field or role).
    Str(&'static str),
    /// A tuple of integers.
    Ints(Vec<i64>),
    /// A tuple of fixed names.
    Strs(Vec<&'static str>),
}

/// A structured core failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// A column does not have one entry per id.
    LengthMismatch {
        /// The input field.
        field: &'static str,
        /// The id count.
        expected_length: usize,
        /// The column's length.
        actual_length: usize,
    },
    /// A value of `field` lies outside the range that field accepts.
    ValueOutOfRange {
        /// The input field.
        field: &'static str,
        /// The first offending position.
        position: usize,
        /// The offending value.
        value: i64,
        /// The reported lower bound.
        minimum: i64,
        /// The reported upper bound.
        maximum: i64,
    },
    /// An id appears on more than one row.
    DuplicateId {
        /// The smallest repeated id.
        id: i64,
        /// Every row carrying that id, in row order.
        rows: Vec<i64>,
        /// How many rows repeat an earlier id, over every repeated id.
        duplicate_count: usize,
    },
    /// A row names one id in both parent roles.
    SameParentId {
        /// The offending row.
        row: usize,
        /// That row's id.
        child_id: i64,
        /// The id named twice.
        parent_id: i64,
    },
    /// Parent references form a cycle through `ids`, the deterministic witness.
    Cycle {
        /// The cycle's original ids, rotated so the smallest comes first.
        ids: Vec<i64>,
    },
    /// A row names itself as its MZ co-twin.
    MzSelfReference {
        /// The offending row.
        row: usize,
        /// That row's id.
        id: i64,
    },
    /// A row's represented co-twin does not name it back.
    MzNonreciprocal {
        /// The referencing row.
        row: usize,
        /// That row's id.
        id: i64,
        /// The row that did not reciprocate.
        twin_row: usize,
        /// That row's id.
        twin_id: i64,
    },
    /// Reciprocal co-twins name different parents.
    MzParentMismatch {
        /// The lower row of the pair.
        row: usize,
        /// That row's id.
        id: i64,
        /// The co-twin's row.
        twin_row: usize,
        /// The co-twin's id.
        twin_id: i64,
        /// The roles whose ids differ, mother first.
        parent_roles: Vec<ParentRole>,
    },
    /// Reciprocal co-twins have different known sexes.
    MzSexMismatch {
        /// The lower row of the pair.
        row: usize,
        /// That row's id.
        id: i64,
        /// The co-twin's row.
        twin_row: usize,
        /// The co-twin's id.
        twin_id: i64,
        /// The lower row's stored sex.
        sex: i8,
        /// The co-twin's stored sex.
        twin_sex: i8,
    },
    /// A child's birth year precedes its parent's.
    BirthYearTopology {
        /// The role of the offending edge.
        parent_role: ParentRole,
        /// The child row.
        child_row: usize,
        /// The parent row.
        parent_row: usize,
        /// The child's id.
        child_id: i64,
        /// The parent's id.
        parent_id: i64,
        /// The child's birth year.
        child_birth_year: i32,
        /// The parent's birth year.
        parent_birth_year: i32,
        /// How many edges of this role violate the order.
        violation_count: usize,
    },
    /// The pedigree has more rows than an int32 row coordinate can address.
    PedigreeTooLarge {
        /// The offending row count.
        n_individuals: usize,
        /// The capacity that was exceeded, normally [`MAX_ROWS`].
        maximum: usize,
    },
    /// The caller asked for a relationship degree outside the supported range.
    MaxDegreeOutOfRange {
        /// The degree as given.
        value: i64,
        /// The lowest supported degree.
        minimum: i64,
        /// The highest supported degree.
        maximum: i64,
    },
    /// The caller named a sex encoding the API does not define.
    UnknownSexEncoding {
        /// The name as given.
        name: String,
    },
}

impl Error {
    /// The host exception family this error routes to.
    pub fn class(&self) -> ErrorClass {
        match self {
            Error::LengthMismatch { .. }
            | Error::ValueOutOfRange { .. }
            | Error::DuplicateId { .. }
            | Error::SameParentId { .. }
            | Error::Cycle { .. }
            | Error::MzSelfReference { .. }
            | Error::MzNonreciprocal { .. }
            | Error::MzParentMismatch { .. }
            | Error::MzSexMismatch { .. }
            | Error::BirthYearTopology { .. }
            | Error::MaxDegreeOutOfRange { .. } => ErrorClass::Validation,
            Error::PedigreeTooLarge { .. } => ErrorClass::Resource,
            Error::UnknownSexEncoding { .. } => ErrorClass::Usage,
        }
    }

    /// The stable code hosts branch on, matching the ADR 0006 code tables.
    ///
    /// A `Usage` error has no code; it renders as the empty string.
    pub fn code(&self) -> &'static str {
        match self {
            Error::LengthMismatch { .. } => "length_mismatch",
            Error::ValueOutOfRange { .. } => "value_out_of_range",
            Error::DuplicateId { .. } => "duplicate_id",
            Error::SameParentId { .. } => "same_parent_id",
            Error::Cycle { .. } => "cycle",
            Error::MzSelfReference { .. } => "mz_self_reference",
            Error::MzNonreciprocal { .. } => "mz_nonreciprocal",
            Error::MzParentMismatch { .. } => "mz_parent_mismatch",
            Error::MzSexMismatch { .. } => "mz_sex_mismatch",
            Error::BirthYearTopology { .. } => "birth_year_topology",
            Error::PedigreeTooLarge { .. } => "pedigree_too_large",
            Error::MaxDegreeOutOfRange { .. } => "max_degree_out_of_range",
            Error::UnknownSexEncoding { .. } => "",
        }
    }

    /// The keyword fields the code requires, in the order the code table lists them.
    pub fn fields(&self) -> Vec<(&'static str, FieldValue)> {
        use FieldValue::{Int, Ints, Str, Strs};
        let int = |n: usize| Int(n as i64);
        match self {
            Error::LengthMismatch {
                field,
                expected_length,
                actual_length,
            } => vec![
                ("field", Str(field)),
                ("expected_length", int(*expected_length)),
                ("actual_length", int(*actual_length)),
            ],
            Error::ValueOutOfRange {
                field,
                position,
                value,
                minimum,
                maximum,
            } => vec![
                ("field", Str(field)),
                ("position", int(*position)),
                ("value", Int(*value)),
                ("minimum", Int(*minimum)),
                ("maximum", Int(*maximum)),
            ],
            Error::DuplicateId {
                id,
                rows,
                duplicate_count,
            } => vec![
                ("id", Int(*id)),
                ("rows", Ints(rows.clone())),
                ("duplicate_count", int(*duplicate_count)),
            ],
            Error::SameParentId {
                row,
                child_id,
                parent_id,
            } => vec![
                ("row", int(*row)),
                ("child_id", Int(*child_id)),
                ("parent_id", Int(*parent_id)),
            ],
            Error::Cycle { ids } => vec![("ids", Ints(ids.clone()))],
            Error::MzSelfReference { row, id } => vec![("row", int(*row)), ("id", Int(*id))],
            Error::MzNonreciprocal {
                row, id, twin_id, ..
            } => vec![
                ("row", int(*row)),
                ("id", Int(*id)),
                ("twin_id", Int(*twin_id)),
            ],
            Error::MzParentMismatch {
                row,
                id,
                twin_id,
                parent_roles,
                ..
            } => vec![
                ("row", int(*row)),
                ("id", Int(*id)),
                ("twin_id", Int(*twin_id)),
                (
                    "parent_roles",
                    Strs(parent_roles.iter().map(|role| role.name()).collect()),
                ),
            ],
            Error::MzSexMismatch {
                row,
                id,
                twin_id,
                sex,
                twin_sex,
                ..
            } => vec![
                ("row", int(*row)),
                ("id", Int(*id)),
                ("twin_id", Int(*twin_id)),
                ("sex", Int(i64::from(*sex))),
                ("twin_sex", Int(i64::from(*twin_sex))),
            ],
            Error::BirthYearTopology {
                parent_role,
                child_row,
                parent_row,
                child_id,
                parent_id,
                child_birth_year,
                parent_birth_year,
                violation_count,
            } => vec![
                ("parent_role", Str(parent_role.name())),
                ("child_row", int(*child_row)),
                ("parent_row", int(*parent_row)),
                ("child_id", Int(*child_id)),
                ("parent_id", Int(*parent_id)),
                ("child_birth_year", Int(i64::from(*child_birth_year))),
                ("parent_birth_year", Int(i64::from(*parent_birth_year))),
                ("violation_count", int(*violation_count)),
            ],
            Error::PedigreeTooLarge {
                n_individuals,
                maximum,
            } => vec![
                ("n_individuals", int(*n_individuals)),
                ("maximum", int(*maximum)),
            ],
            Error::MaxDegreeOutOfRange {
                value,
                minimum,
                maximum,
            } => vec![
                ("value", Int(*value)),
                ("minimum", Int(*minimum)),
                ("maximum", Int(*maximum)),
            ],
            Error::UnknownSexEncoding { .. } => Vec::new(),
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

/// Render `name` the way Python's `repr` renders a str: single-quoted unless
/// the text holds a single quote and no double quote.
fn python_str(name: &str) -> String {
    let quote = if name.contains('\'') && !name.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut out = String::with_capacity(name.len() + 2);
    out.push(quote);
    for c in name.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            '\r' => out.push_str("\\r"),
            c => out.push(c),
        }
    }
    out.push(quote);
    out
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
            Error::LengthMismatch {
                field,
                expected_length,
                actual_length,
            } => write!(
                f,
                "{} has length {actual_length}, expected {expected_length} from the id field",
                python_str(field)
            ),
            Error::ValueOutOfRange {
                field,
                position,
                value: _,
                minimum,
                maximum,
            } => write!(
                f,
                "{} value at position {position} is outside [{minimum}, {maximum}]",
                python_str(field)
            ),
            Error::DuplicateId {
                id,
                rows,
                duplicate_count,
            } => write!(
                f,
                "id {id} appears at rows {}; {duplicate_count} row(s) repeat an earlier id",
                python_tuple(rows)
            ),
            Error::SameParentId {
                row,
                child_id: _,
                parent_id,
            } => write!(
                f,
                "row {row} names id {parent_id} as both mother and father"
            ),
            Error::Cycle { ids } => write!(
                f,
                "parent references form a cycle through ids {}",
                python_tuple(ids)
            ),
            Error::MzSelfReference { row, .. } => {
                write!(f, "row {row} names itself as its MZ co-twin")
            }
            Error::MzNonreciprocal { row, twin_row, .. } => write!(
                f,
                "the MZ reference at row {row} is not reciprocated by row {twin_row}"
            ),
            Error::MzParentMismatch { row, twin_row, .. } => write!(
                f,
                "MZ co-twins at rows {row} and {twin_row} do not name the same parents"
            ),
            Error::MzSexMismatch { row, twin_row, .. } => write!(
                f,
                "MZ co-twins at rows {row} and {twin_row} have different known sexes"
            ),
            Error::BirthYearTopology {
                parent_role,
                child_row,
                ..
            } => write!(
                f,
                "birth_year topology violation: {role}-child edge at row {child_row} \
                 has child.birth_year below {role}.birth_year",
                role = parent_role.name()
            ),
            Error::PedigreeTooLarge { n_individuals, .. } => write!(
                f,
                "pedigree has {} rows, exceeding the int32 row-coordinate capacity",
                grouped(*n_individuals)
            ),
            Error::MaxDegreeOutOfRange {
                value,
                minimum,
                maximum,
            } => write!(
                f,
                "max_degree must be in [{minimum}, {maximum}], got {value}"
            ),
            Error::UnknownSexEncoding { name } => write!(
                f,
                "sex_encoding must be one of ['plink', 'simace'], got {}",
                python_str(name)
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
