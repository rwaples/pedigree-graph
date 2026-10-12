//! Structured core errors, one variant per failure a host must distinguish.
//!
//! Hosts do not parse messages.  Every error carries a stable [`Error::code`]
//! and the operands that produced it, so the Python facade maps the code onto
//! its exception class (ADR 0006) and rebuilds the keyword fields from
//! [`Error::fields`], and R raises a classed condition from the same pair.
//! [`Error::class`] names which of the host exception families a variant
//! belongs to, which is the only routing decision a binding has to make.
//!
//! A host renders an error under its [`IndexBase`]: Python counts rows and
//! positions from 0, R from 1, and the message and the fields count alike.
//! Every row, column or position that locates the failure is a [`Position`],
//! so neither can show one raw.  An offending value and its bounds are
//! reported as given, even when the field itself holds rows
//! ([`Error::ValueOutOfRange`]).
//!
//! The prose is not part of the contract; most of it began as the Python
//! raise site it replaced.

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

/// A row, column or list position an error names, counted from 0.
///
/// It has no `Display` and no integer conversion: a message or field reads
/// it only through [`Position::at`], so it always appears in the host's base.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Position(usize);

impl From<usize> for Position {
    fn from(position: usize) -> Position {
        Position(position)
    }
}

impl Position {
    /// The position as a host counting from `base` writes it.
    pub fn at(self, base: IndexBase) -> i64 {
        self.0 as i64 + base.offset()
    }
}

/// Where a host starts counting rows and positions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexBase {
    /// Python and Rust.
    Zero,
    /// R.
    One,
}

impl IndexBase {
    fn offset(self) -> i64 {
        match self {
            IndexBase::Zero => 0,
            IndexBase::One => 1,
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
    ///
    /// `position` locates the value; `value`, `minimum` and `maximum` are in
    /// the field's own units, 0-based when the field holds rows.
    ValueOutOfRange {
        /// The input field.
        field: &'static str,
        /// The first offending position.
        position: Position,
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
        rows: Vec<Position>,
        /// How many rows repeat an earlier id, over every repeated id.
        duplicate_count: usize,
    },
    /// A row names one id in both parent roles.
    SameParentId {
        /// The offending row.
        row: Position,
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
        row: Position,
        /// That row's id.
        id: i64,
    },
    /// A row's represented co-twin does not name it back.
    MzNonreciprocal {
        /// The referencing row.
        row: Position,
        /// That row's id.
        id: i64,
        /// The row that did not reciprocate.
        twin_row: Position,
        /// That row's id.
        twin_id: i64,
    },
    /// Reciprocal co-twins name different parents.
    MzParentMismatch {
        /// The lower row of the pair.
        row: Position,
        /// That row's id.
        id: i64,
        /// The co-twin's row.
        twin_row: Position,
        /// The co-twin's id.
        twin_id: i64,
        /// The roles whose ids differ, mother first.
        parent_roles: Vec<ParentRole>,
    },
    /// Reciprocal co-twins have different known sexes.
    MzSexMismatch {
        /// The lower row of the pair.
        row: Position,
        /// That row's id.
        id: i64,
        /// The co-twin's row.
        twin_row: Position,
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
        child_row: Position,
        /// The parent row.
        parent_row: Position,
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
    /// A large allocation the input reaches was refused by the allocator.
    AllocationFailed {
        /// What the engine was building, e.g. `"pair_block"`.
        operation: &'static str,
        /// How many elements it asked for.
        requested_elements: usize,
        /// The element type, in NumPy spelling, e.g. `"int32"`.
        dtype: &'static str,
    },
    /// An integer result outgrew its dtype.
    ArithmeticOverflow {
        /// What the engine was computing, e.g. `"descendant_path_counts"`.
        operation: &'static str,
        /// The dtype that overflowed, in NumPy spelling.
        dtype: &'static str,
    },
    /// The package thread pool is already configured with another size.
    ThreadPoolConflict {
        /// The size the pool was built with.
        configured: usize,
        /// The size now requested.
        requested: usize,
    },
    /// The operating system refused to start the pool's worker threads.
    ThreadPoolUnavailable {
        /// The builder's message.
        reason: String,
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
    /// A kinship support column's row indices are not strictly increasing.
    KinshipSupportUnsorted {
        /// The offending column.
        column: Position,
    },
    /// An upper entry of a kinship support has no mirror in its row's column.
    KinshipSupportAsymmetric {
        /// The entry's row.
        row: Position,
        /// The entry's column.
        column: Position,
    },
    /// The symmetric CSC a kinship matrix would assemble has more entries
    /// than an int32 index can address.
    CscIndexOverflow {
        /// The entry count the DP produced.
        nnz: u64,
        /// The largest count an int32 `indptr` can end at.
        maximum: i64,
    },
    /// The caller asked for a propagation threshold that is not finite or
    /// not in `[0, 1]`.
    KinshipThresholdOutOfRange {
        /// The threshold as given, rendered.
        value: String,
    },
    /// The graph-to-view map is not a partial permutation of view rows.
    InvalidViewMap {
        /// The graph row carrying the offending entry.
        position: Position,
        /// The offending view row.
        value: Position,
        /// Why it is rejected.
        reason: &'static str,
    },
    /// The caller named a sex encoding the API does not define.
    UnknownSexEncoding {
        /// The name as given.
        name: String,
    },
    /// One lane of accumulators plus the output would not fit the caller's
    /// memory budget, so nothing was allocated.
    MemoryBudgetExceeded {
        /// The operation that planned the allocation, e.g. `"relationship_moments"`.
        operation: &'static str,
        /// The planned peak in bytes; `u64::MAX` when a size was not
        /// representable, which the message says instead of the number.
        estimated_bytes: u64,
        /// The budget the caller set, in bytes.
        budget_bytes: u64,
    },
    /// A relationship-moments value column holds a value that is not finite.
    NonFiniteValue {
        /// The column as the host names it, e.g. `values['x']`.
        field: String,
        /// The first offending row.
        position: Position,
    },
    /// A float a relationship-moments table derives is past the float64 range.
    NotRepresentable {
        /// The statistic, e.g. `"sumsq_first"`.
        statistic: &'static str,
        /// The column or product as the host names it.
        name: String,
    },
    /// A relationship-moments table handed in is inconsistent, or an
    /// operation names a table it does not fit.
    InvalidMomentsTable {
        /// What is wrong.
        reason: String,
    },
    /// The host cancelled the call through its `Progress` (ADR 0017).  The
    /// Python host raises the exception that caused the cancel instead, so
    /// this reaches a user only from a host that cancels without one.
    Cancelled,
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
            | Error::MaxDegreeOutOfRange { .. }
            | Error::KinshipSupportUnsorted { .. }
            | Error::KinshipSupportAsymmetric { .. } => ErrorClass::Validation,
            Error::PedigreeTooLarge { .. } => ErrorClass::Resource,
            Error::AllocationFailed { .. } => ErrorClass::Resource,
            Error::ArithmeticOverflow { .. } => ErrorClass::Resource,
            Error::CscIndexOverflow { .. } => ErrorClass::Resource,
            Error::MemoryBudgetExceeded { .. } => ErrorClass::Resource,
            Error::KinshipThresholdOutOfRange { .. } => ErrorClass::Usage,
            Error::ThreadPoolConflict { .. } | Error::ThreadPoolUnavailable { .. } => {
                ErrorClass::Usage
            }
            Error::InvalidViewMap { .. } | Error::UnknownSexEncoding { .. } => ErrorClass::Usage,
            Error::NonFiniteValue { .. }
            | Error::NotRepresentable { .. }
            | Error::InvalidMomentsTable { .. } => ErrorClass::Usage,
            Error::Cancelled => ErrorClass::Usage,
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
            Error::AllocationFailed { .. } => "allocation_failed",
            Error::ArithmeticOverflow { .. } => "arithmetic_overflow",
            Error::ThreadPoolConflict { .. } | Error::ThreadPoolUnavailable { .. } => "",
            Error::MaxDegreeOutOfRange { .. } => "max_degree_out_of_range",
            Error::KinshipSupportUnsorted { .. } => "kinship_support_unsorted",
            Error::KinshipSupportAsymmetric { .. } => "kinship_support_asymmetric",
            Error::CscIndexOverflow { .. } => "csc_index_overflow",
            Error::MemoryBudgetExceeded { .. } => "memory_budget_exceeded",
            Error::KinshipThresholdOutOfRange { .. }
            | Error::InvalidViewMap { .. }
            | Error::UnknownSexEncoding { .. }
            | Error::NonFiniteValue { .. }
            | Error::NotRepresentable { .. }
            | Error::InvalidMomentsTable { .. }
            | Error::Cancelled => "",
        }
    }

    /// The keyword fields the code requires, in the order the code table
    /// lists them, rows and positions counted from `base`.
    pub fn fields(&self, base: IndexBase) -> Vec<(&'static str, FieldValue)> {
        use FieldValue::{Int, Ints, Str, Strs};
        let int = |n: usize| Int(n as i64);
        let at = |p: &Position| Int(p.at(base));
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
                ("position", at(position)),
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
                ("rows", Ints(rows.iter().map(|p| p.at(base)).collect())),
                ("duplicate_count", int(*duplicate_count)),
            ],
            Error::SameParentId {
                row,
                child_id,
                parent_id,
            } => vec![
                ("row", at(row)),
                ("child_id", Int(*child_id)),
                ("parent_id", Int(*parent_id)),
            ],
            Error::Cycle { ids } => vec![("ids", Ints(ids.clone()))],
            Error::MzSelfReference { row, id } => vec![("row", at(row)), ("id", Int(*id))],
            Error::MzNonreciprocal {
                row, id, twin_id, ..
            } => vec![
                ("row", at(row)),
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
                ("row", at(row)),
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
                ("row", at(row)),
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
                ("child_row", at(child_row)),
                ("parent_row", at(parent_row)),
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
            Error::AllocationFailed {
                operation,
                requested_elements,
                dtype,
            } => vec![
                ("operation", Str(operation)),
                ("requested_elements", int(*requested_elements)),
                ("dtype", Str(dtype)),
            ],
            Error::ArithmeticOverflow { operation, dtype } => {
                vec![("operation", Str(operation)), ("dtype", Str(dtype))]
            }
            Error::ThreadPoolConflict { .. }
            | Error::ThreadPoolUnavailable { .. }
            | Error::InvalidViewMap { .. } => vec![],
            Error::MaxDegreeOutOfRange {
                value,
                minimum,
                maximum,
            } => vec![
                ("value", Int(*value)),
                ("minimum", Int(*minimum)),
                ("maximum", Int(*maximum)),
            ],
            Error::KinshipSupportUnsorted { column } => vec![("column", at(column))],
            Error::KinshipSupportAsymmetric { row, column } => {
                vec![("row", at(row)), ("column", at(column))]
            }
            Error::CscIndexOverflow { nnz, maximum } => {
                vec![("nnz", Int(*nnz as i64)), ("maximum", Int(*maximum))]
            }
            Error::MemoryBudgetExceeded {
                operation,
                estimated_bytes,
                budget_bytes,
            } => vec![
                ("operation", Str(operation)),
                (
                    "estimated_bytes",
                    Int(i64::try_from(*estimated_bytes).unwrap_or(i64::MAX)),
                ),
                (
                    "budget_bytes",
                    Int(i64::try_from(*budget_bytes).unwrap_or(i64::MAX)),
                ),
            ],
            Error::KinshipThresholdOutOfRange { .. }
            | Error::UnknownSexEncoding { .. }
            | Error::NonFiniteValue { .. }
            | Error::NotRepresentable { .. }
            | Error::InvalidMomentsTable { .. }
            | Error::Cancelled => Vec::new(),
        }
    }

    /// The message, rows and positions counted from `base`.
    pub fn message(&self, base: IndexBase) -> String {
        Message { err: self, base }.to_string()
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

/// An error's message with its rows and positions counted from `base`.
struct Message<'a> {
    err: &'a Error,
    base: IndexBase,
}

impl std::fmt::Display for Message<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let base = self.base;
        match self.err {
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
                "{} value at position {} is outside [{minimum}, {maximum}]",
                python_str(field),
                position.at(base)
            ),
            Error::DuplicateId {
                id,
                rows,
                duplicate_count,
            } => write!(
                f,
                "id {id} appears at rows {}; {duplicate_count} row(s) repeat an earlier id",
                python_tuple(&rows.iter().map(|p| p.at(base)).collect::<Vec<_>>())
            ),
            Error::SameParentId {
                row,
                child_id: _,
                parent_id,
            } => write!(
                f,
                "row {} names id {parent_id} as both mother and father",
                row.at(base)
            ),
            Error::Cycle { ids } => write!(
                f,
                "parent references form a cycle through ids {}",
                python_tuple(ids)
            ),
            Error::MzSelfReference { row, .. } => {
                write!(f, "row {} names itself as its MZ co-twin", row.at(base))
            }
            Error::MzNonreciprocal { row, twin_row, .. } => write!(
                f,
                "the MZ reference at row {} is not reciprocated by row {}",
                row.at(base),
                twin_row.at(base)
            ),
            Error::MzParentMismatch { row, twin_row, .. } => write!(
                f,
                "MZ co-twins at rows {} and {} do not name the same parents",
                row.at(base),
                twin_row.at(base)
            ),
            Error::MzSexMismatch { row, twin_row, .. } => write!(
                f,
                "MZ co-twins at rows {} and {} have different known sexes",
                row.at(base),
                twin_row.at(base)
            ),
            Error::BirthYearTopology {
                parent_role,
                child_row,
                ..
            } => write!(
                f,
                "birth_year topology violation: {role}-child edge at row {child_row} \
                 has child.birth_year below {role}.birth_year",
                role = parent_role.name(),
                child_row = child_row.at(base)
            ),
            Error::PedigreeTooLarge { n_individuals, .. } => write!(
                f,
                "pedigree has {} rows, exceeding the int32 row-coordinate capacity",
                grouped(*n_individuals)
            ),
            Error::AllocationFailed {
                operation,
                requested_elements,
                dtype,
            } => write!(
                f,
                "could not allocate {} {dtype} elements for {operation}",
                grouped(*requested_elements)
            ),
            Error::ArithmeticOverflow { operation, dtype } => {
                write!(f, "{operation} overflowed {dtype}")
            }
            Error::ThreadPoolConflict {
                configured,
                requested,
            } => write!(
                f,
                "the thread pool is already configured with {configured} threads and cannot be changed to {requested}"
            ),
            Error::ThreadPoolUnavailable { reason } => {
                write!(f, "could not start the thread pool: {reason}")
            }
            Error::InvalidViewMap {
                position,
                value,
                reason,
            } => write!(
                f,
                "view_rows[{}] is {}: {reason}",
                position.at(base),
                value.at(base)
            ),
            Error::MaxDegreeOutOfRange {
                value,
                minimum,
                maximum,
            } => write!(
                f,
                "max_degree must be in [{minimum}, {maximum}], got {value}"
            ),
            Error::KinshipSupportUnsorted { column } => write!(
                f,
                "kinship support column {} is not sorted by row",
                column.at(base)
            ),
            Error::KinshipSupportAsymmetric { row, column } => write!(
                f,
                "kinship support entry ({row}, {column}) has no mirror at ({column}, {row})",
                row = row.at(base),
                column = column.at(base)
            ),
            Error::CscIndexOverflow { nnz, maximum } => write!(
                f,
                "kinship matrix nnz exceeds the int32 CSC index range ({nnz} > {maximum})"
            ),
            Error::KinshipThresholdOutOfRange { value } => write!(
                f,
                "min_propagated_kinship must be finite and in [0, 1], got {value}"
            ),
            Error::UnknownSexEncoding { name } => write!(
                f,
                "sex_encoding must be one of ['plink', 'simace'], got {}",
                python_str(name)
            ),
            Error::MemoryBudgetExceeded {
                operation,
                estimated_bytes,
                budget_bytes,
            } => {
                if *estimated_bytes == u64::MAX {
                    write!(
                        f,
                        "{operation} accumulator sizes are not representable in memory"
                    )
                } else {
                    write!(
                        f,
                        "{operation} needs about {} bytes of accumulators, over the memory budget of {} bytes",
                        grouped(*estimated_bytes as usize),
                        grouped(*budget_bytes as usize)
                    )
                }
            }
            Error::NonFiniteValue { field, position } => write!(
                f,
                "{field} is not finite at row {}; mask it with a factor level instead",
                position.at(base)
            ),
            Error::NotRepresentable { statistic, name } => write!(
                f,
                "{statistic} of {} is not representable in float64 (an output overflowed)",
                python_str(name)
            ),
            Error::InvalidMomentsTable { reason } => {
                write!(f, "invalid relationship moments table: {reason}")
            }
            Error::Cancelled => write!(f, "the call was cancelled"),
        }
    }
}

impl std::fmt::Display for Error {
    /// The message as Python and Rust count, from 0.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let message = Message {
            err: self,
            base: IndexBase::Zero,
        };
        message.fmt(f)
    }
}

impl std::error::Error for Error {}

#[cfg(test)]
mod tests {
    use super::{Error, ErrorClass, FieldValue, IndexBase, ParentRole, Position, MAX_ROWS};

    /// How many variants `Error` has.
    const VARIANTS: usize = 27;

    /// Each variant's slot in [`samples`]; the exhaustive match makes a new
    /// variant fail to compile until it is counted here.
    fn variant_index(err: &Error) -> usize {
        match err {
            Error::LengthMismatch { .. } => 0,
            Error::ValueOutOfRange { .. } => 1,
            Error::DuplicateId { .. } => 2,
            Error::SameParentId { .. } => 3,
            Error::Cycle { .. } => 4,
            Error::MzSelfReference { .. } => 5,
            Error::MzNonreciprocal { .. } => 6,
            Error::MzParentMismatch { .. } => 7,
            Error::MzSexMismatch { .. } => 8,
            Error::BirthYearTopology { .. } => 9,
            Error::PedigreeTooLarge { .. } => 10,
            Error::AllocationFailed { .. } => 11,
            Error::ArithmeticOverflow { .. } => 12,
            Error::ThreadPoolConflict { .. } => 13,
            Error::ThreadPoolUnavailable { .. } => 14,
            Error::MaxDegreeOutOfRange { .. } => 15,
            Error::KinshipSupportUnsorted { .. } => 16,
            Error::KinshipSupportAsymmetric { .. } => 17,
            Error::CscIndexOverflow { .. } => 18,
            Error::KinshipThresholdOutOfRange { .. } => 19,
            Error::InvalidViewMap { .. } => 20,
            Error::UnknownSexEncoding { .. } => 21,
            Error::MemoryBudgetExceeded { .. } => 22,
            Error::NonFiniteValue { .. } => 23,
            Error::NotRepresentable { .. } => 24,
            Error::InvalidMomentsTable { .. } => 25,
            Error::Cancelled => 26,
        }
    }

    /// One error of every variant.  Positions are 61 and 93, numbers no
    /// other operand contains, so a field or message shows which base wrote
    /// them.
    fn samples() -> Vec<Error> {
        let (p, q) = (Position(61), Position(93));
        vec![
            Error::LengthMismatch {
                field: "sex",
                expected_length: 4,
                actual_length: 3,
            },
            Error::ValueOutOfRange {
                field: "mother",
                position: p,
                value: -7,
                minimum: -1,
                maximum: 5,
            },
            Error::DuplicateId {
                id: 5,
                rows: vec![p, q],
                duplicate_count: 1,
            },
            Error::SameParentId {
                row: p,
                child_id: 5,
                parent_id: 7,
            },
            Error::Cycle { ids: vec![5, 7] },
            Error::MzSelfReference { row: p, id: 5 },
            Error::MzNonreciprocal {
                row: p,
                id: 5,
                twin_row: q,
                twin_id: 7,
            },
            Error::MzParentMismatch {
                row: p,
                id: 5,
                twin_row: q,
                twin_id: 7,
                parent_roles: vec![ParentRole::Mother, ParentRole::Father],
            },
            Error::MzSexMismatch {
                row: p,
                id: 5,
                twin_row: q,
                twin_id: 7,
                sex: 0,
                twin_sex: 1,
            },
            Error::BirthYearTopology {
                parent_role: ParentRole::Father,
                child_row: p,
                parent_row: q,
                child_id: 5,
                parent_id: 7,
                child_birth_year: 1980,
                parent_birth_year: 1985,
                violation_count: 2,
            },
            Error::PedigreeTooLarge {
                n_individuals: MAX_ROWS + 1,
                maximum: MAX_ROWS,
            },
            Error::AllocationFailed {
                operation: "pair_block",
                requested_elements: 1000,
                dtype: "int32",
            },
            Error::ArithmeticOverflow {
                operation: "descendant_path_counts",
                dtype: "int64",
            },
            Error::ThreadPoolConflict {
                configured: 4,
                requested: 8,
            },
            Error::ThreadPoolUnavailable {
                reason: "refused".to_string(),
            },
            Error::MaxDegreeOutOfRange {
                value: 9,
                minimum: 1,
                maximum: 5,
            },
            Error::KinshipSupportUnsorted { column: p },
            Error::KinshipSupportAsymmetric { row: p, column: q },
            Error::CscIndexOverflow {
                nnz: 3_000_000_000,
                maximum: i64::from(i32::MAX),
            },
            Error::KinshipThresholdOutOfRange {
                value: "1.5".to_string(),
            },
            Error::InvalidViewMap {
                position: p,
                value: q,
                reason: "two graph rows map to this view row",
            },
            Error::UnknownSexEncoding {
                name: "x".to_string(),
            },
            Error::MemoryBudgetExceeded {
                operation: "relationship_moments",
                estimated_bytes: 2048,
                budget_bytes: 1024,
            },
            Error::NonFiniteValue {
                field: "values['x']".to_string(),
                position: p,
            },
            Error::NotRepresentable {
                statistic: "sumsq_first",
                name: "x".to_string(),
            },
            Error::InvalidMomentsTable {
                reason: "bad".to_string(),
            },
            Error::Cancelled,
        ]
    }

    #[test]
    fn samples_cover_every_variant() {
        let mut seen = [false; VARIANTS];
        for err in samples() {
            seen[variant_index(&err)] = true;
        }
        let missing: Vec<usize> = (0..VARIANTS).filter(|&i| !seen[i]).collect();
        assert!(missing.is_empty(), "add a sample of variant(s) {missing:?}");
    }

    /// An integer field's values, if it has any.
    fn ints(value: &FieldValue) -> Option<Vec<i64>> {
        match value {
            FieldValue::Int(n) => Some(vec![*n]),
            FieldValue::Ints(v) => Some(v.clone()),
            _ => None,
        }
    }

    #[test]
    fn positions_count_from_the_hosts_base() {
        for err in samples() {
            let (zero, one) = (err.fields(IndexBase::Zero), err.fields(IndexBase::One));
            assert_eq!(zero.len(), one.len(), "{err:?}");
            for ((name, v0), (name1, v1)) in zero.iter().zip(&one) {
                assert_eq!(name, name1, "{err:?}");
                match ints(v0).zip(ints(v1)) {
                    Some((p0, p1)) if p0.iter().any(|n| [61, 93].contains(n)) => {
                        let shifted: Vec<i64> = p0.iter().map(|n| n + 1).collect();
                        assert_eq!(p1, shifted, "{name} of {err:?}");
                    }
                    _ => assert_eq!(v0, v1, "{name} of {err:?} is not a position"),
                }
            }
            let message = err.message(IndexBase::Zero);
            assert_eq!(message, err.to_string(), "{err:?}");
            assert_eq!(
                err.message(IndexBase::One),
                message.replace("61", "62").replace("93", "94"),
                "{err:?}"
            );
        }
    }

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
