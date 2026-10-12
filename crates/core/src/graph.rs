//! Native pedigree construction: the one boundary every host constructor crosses.
//!
//! The host hands over integer columns it has already coerced from its own
//! representations (frames, nullable dtypes, host nulls as `-1`).  Everything
//! that is pedigree semantics rather than host representation happens here, in
//! the order the Python parser established (ADR 0006): per-field range, sex
//! encoding, duplicate ids, a shared parent id, id to row resolution, the
//! topological check and cycle witness, the MZ pair contract, collapse of a
//! wholly unknown optional column, and the birth-year order of every known
//! parent-child edge.  The first violation found wins, and which one that is
//! depends on this order, so the order is part of the contract the tests pin.
//!
//! Missing and unresolved references are distinct and need no third sentinel.
//! A relation is missing when its id is `-1`; it is unresolved (external to
//! the represented rows) when its id is `>= 0` and its row is `-1`.

use crate::error::{Error, ParentRole, MAX_ROWS};
use crate::topology;

/// The host's coerced columns, one entry per row.
///
/// `ids`, `mother` and `father` are required; the rest are `None` when the
/// host had no such column.  A host null is already `-1` everywhere but `ids`,
/// where the host has rejected it.
#[derive(Debug, Clone, Copy)]
pub struct Columns<'a> {
    /// Row ids.
    pub ids: &'a [i64],
    /// Mother id per row, `-1` when missing.
    pub mother: &'a [i64],
    /// Father id per row, `-1` when missing.
    pub father: &'a [i64],
    /// MZ co-twin id per row, `-1` when missing.
    pub twin: Option<&'a [i64]>,
    /// Sex per row in the caller's encoding.
    pub sex: Option<&'a [i64]>,
    /// Generation label per row, `-1` when unknown.
    pub generation: Option<&'a [i64]>,
    /// Birth year per row, `-1` when unknown.
    pub birth_year: Option<&'a [i64]>,
}

/// How the caller spells sex, never inferred from the values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SexEncoding {
    /// `0` female, `1` male, `-1` unknown: the stored encoding itself.
    Simace,
    /// `2` female, `1` male, `0` unknown; `-1` is also accepted as unknown.
    Plink,
}

impl SexEncoding {
    /// The encoding a host names, or [`Error::UnknownSexEncoding`].
    pub fn parse(name: &str) -> Result<SexEncoding, Error> {
        match name {
            "simace" => Ok(SexEncoding::Simace),
            "plink" => Ok(SexEncoding::Plink),
            _ => Err(Error::UnknownSexEncoding {
                name: name.to_string(),
            }),
        }
    }

    /// The raw range accepted, then the range an error reports.
    fn ranges(self) -> ((i64, i64), (i64, i64)) {
        match self {
            SexEncoding::Simace => ((-1, 1), (-1, 1)),
            SexEncoding::Plink => ((-1, 2), (0, 2)),
        }
    }

    /// A raw value inside the accepted range, in the stored `0/1/-1` encoding.
    fn store(self, raw: i64) -> i8 {
        match self {
            SexEncoding::Simace => raw as i8,
            SexEncoding::Plink => match raw {
                1 => 1,
                2 => 0,
                _ => -1,
            },
        }
    }
}

/// Capacities a build is held to.  The default is the real one; tests lower it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Limits {
    /// The most rows a pedigree may have.
    pub max_rows: usize,
}

impl Default for Limits {
    fn default() -> Limits {
        Limits { max_rows: MAX_ROWS }
    }
}

/// A validated pedigree in graph-space rows.
///
/// Every vector has one entry per row.  Ids are unique and nonnegative; row
/// references are `-1` when missing or external; an optional column is `None`
/// when the host had none or every entry was unknown.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PedigreeGraph {
    /// Row ids.
    pub ids: Vec<i64>,
    /// Mother id per row, `-1` when missing.
    pub mother_ids: Vec<i64>,
    /// Father id per row, `-1` when missing.
    pub father_ids: Vec<i64>,
    /// MZ co-twin id per row, `-1` when missing.
    pub twin_ids: Vec<i64>,
    /// Mother row per row, `-1` when missing or external.
    pub mother_rows: Vec<i32>,
    /// Father row per row, `-1` when missing or external.
    pub father_rows: Vec<i32>,
    /// MZ co-twin row per row, `-1` when missing or external.
    pub twin_rows: Vec<i32>,
    /// Sex per row as `0` female, `1` male, `-1` unknown.
    pub sex: Option<Vec<i8>>,
    /// Generation label per row, `-1` when unknown.
    pub generation: Option<Vec<i32>>,
    /// Birth year per row, `-1` when unknown.
    pub birth_year: Option<Vec<i32>>,
    /// Whether every represented parent row precedes its child row.
    pub rows_topological: bool,
}

impl PedigreeGraph {
    /// The number of rows.
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// Whether there are no rows.
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }
}

/// The widest id span, in slots per id, that [`IdIndex`] stores as a table.
///
/// Three `i32` slots per id is 12 bytes per row, what the sorted layout costs,
/// so no table grows with the largest id.
const DENSE_SLOTS_PER_ID: i128 = 3;

/// Ids mapped to rows, for repeated id lookups.
///
/// Rows are `i32` because a graph never has more than [`MAX_ROWS`] rows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IdIndex {
    layout: Layout,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Layout {
    /// `rows[id - min]` is the row of `id`, or `-1` where no row has it.
    Dense { min: i64, rows: Vec<i32> },
    /// Ids ascending, equal ids in row order, beside their rows.
    Sorted { ids: Vec<i64>, rows: Vec<i32> },
}

impl Layout {
    /// A table over `[min, min + span)`.
    fn dense(ids: &[i64], min: i64, span: usize) -> Layout {
        // Scatter without reading the slot first: a read before each write
        // made the shuffled 5.4M-row build 1.75x slower.
        let mut rows = vec![-1; span];
        for (row, &id) in ids.iter().enumerate() {
            rows[(id - min) as usize] = row as i32;
        }
        Layout::Dense { min, rows }
    }

    /// Sorted ids and rows, for ids too sparse for a table.
    fn sorted(ids: &[i64]) -> Layout {
        let mut pairs: Vec<(i64, i32)> = ids
            .iter()
            .enumerate()
            .map(|(row, &id)| (id, row as i32))
            .collect();
        pairs.sort_unstable();
        let rows = pairs.iter().map(|&(_, row)| row).collect();
        // std collects in place into the pairs' buffer, so building peaks at
        // 20 bytes per row and the index keeps 12 (pinned by a test).
        let mut ids: Vec<i64> = pairs.into_iter().map(|(id, _)| id).collect();
        ids.shrink_to_fit();
        Layout::Sorted { ids, rows }
    }
}

impl IdIndex {
    /// Index `ids`, which are at most [`MAX_ROWS`] long.
    ///
    /// Ids spanning at most [`DENSE_SLOTS_PER_ID`] slots per id get a table
    /// indexed by `id - min`; others are sorted.  Either way the index is a
    /// function of the ids alone.  Which row of a repeated id
    /// [`IdIndex::row`] returns is unspecified.
    pub fn build(ids: &[i64]) -> IdIndex {
        let (min, max) = ids.iter().fold((i64::MAX, i64::MIN), |(min, max), &id| {
            (min.min(id), max.max(id))
        });
        let span = i128::from(max) - i128::from(min) + 1;
        let layout = if ids.is_empty() || span > DENSE_SLOTS_PER_ID * ids.len() as i128 {
            Layout::sorted(ids)
        } else {
            Layout::dense(ids, min, span as usize)
        };
        IdIndex { layout }
    }

    /// The smallest repeated id and how many rows repeat an id an earlier row
    /// has, for the `ids` this index was built from.
    fn duplicate(&self, ids: &[i64]) -> Option<(i64, usize)> {
        match &self.layout {
            Layout::Dense { min, rows } => {
                // A repeated id overwrote its earlier rows, leaving fewer
                // slots filled than rows.
                let repeats = ids.len() - rows.iter().filter(|&&row| row >= 0).count();
                if repeats == 0 {
                    return None;
                }
                ids.iter()
                    .enumerate()
                    .filter(|&(row, &id)| rows[(id - min) as usize] != row as i32)
                    .map(|(_, &id)| id)
                    .min()
                    .map(|id| (id, repeats))
            }
            Layout::Sorted { ids: sorted, .. } => {
                let mut repeats = sorted
                    .windows(2)
                    .filter(|pair| pair[0] == pair[1])
                    .map(|pair| pair[0]);
                repeats.next().map(|id| (id, 1 + repeats.count()))
            }
        }
    }

    /// The row of `id`, or `-1` for a negative id or one not in the index.
    pub fn row(&self, id: i64) -> i32 {
        if id < 0 {
            return -1;
        }
        match &self.layout {
            Layout::Dense { min, rows } => id
                .checked_sub(*min)
                .and_then(|offset| usize::try_from(offset).ok())
                .and_then(|offset| rows.get(offset))
                .map_or(-1, |&row| row),
            Layout::Sorted { ids, rows } => match ids.binary_search(&id) {
                Ok(position) => rows[position],
                Err(_) => -1,
            },
        }
    }

    /// The row per query id, `-1` where unresolved.
    pub fn resolve(&self, query: &[i64]) -> Vec<i32> {
        query.iter().map(|&id| self.row(id)).collect()
    }
}

/// The first position whose value lies outside `[minimum, maximum]`.
fn check_range(
    field: &'static str,
    values: &[i64],
    (minimum, maximum): (i64, i64),
    reported: (i64, i64),
) -> Result<(), Error> {
    match values
        .iter()
        .position(|&value| value < minimum || value > maximum)
    {
        Some(position) => Err(Error::ValueOutOfRange {
            field,
            position: position.into(),
            value: values[position],
            minimum: reported.0,
            maximum: reported.1,
        }),
        None => Ok(()),
    }
}

/// `length_mismatch` unless `values` has one entry per id.
fn check_length(field: &'static str, values: &[i64], n: usize) -> Result<(), Error> {
    if values.len() == n {
        Ok(())
    } else {
        Err(Error::LengthMismatch {
            field,
            expected_length: n,
            actual_length: values.len(),
        })
    }
}

/// `duplicate_id` naming the smallest repeated id and all its rows.
fn check_duplicate_ids(ids: &[i64], index: &IdIndex) -> Result<(), Error> {
    let Some((duplicated, repeats)) = index.duplicate(ids) else {
        return Ok(());
    };
    let rows = ids
        .iter()
        .enumerate()
        .filter(|(_, &id)| id == duplicated)
        .map(|(row, _)| row.into())
        .collect();
    Err(Error::DuplicateId {
        id: duplicated,
        rows,
        duplicate_count: repeats,
    })
}

/// `same_parent_id` for the first row naming one id in both parent roles.
fn check_same_parent(ids: &[i64], mother: &[i64], father: &[i64]) -> Result<(), Error> {
    match mother
        .iter()
        .zip(father)
        .position(|(&m, &f)| m == f && m >= 0)
    {
        Some(row) => Err(Error::SameParentId {
            row: row.into(),
            child_id: ids[row],
            parent_id: mother[row],
        }),
        None => Ok(()),
    }
}

/// Reject represented MZ references that break the ADR 0006 pair contract.
///
/// Each kind of violation is swept over every row before the next kind runs,
/// so a third row pointing into an otherwise valid pair is reported as
/// non-reciprocal rather than compared for parents against a row that is not
/// its partner.  The symmetric checks read only the lower row of each pair, so
/// one violation is reported once.  Parents are compared by id, so co-twins
/// naming the same unrepresented parent agree.
fn check_mz_pairs(
    ids: &[i64],
    mother: &[i64],
    father: &[i64],
    twin_rows: &[i32],
    sex: Option<&[i8]>,
) -> Result<(), Error> {
    let referencing: Vec<(usize, usize)> = twin_rows
        .iter()
        .enumerate()
        .filter(|(_, &partner)| partner >= 0)
        .map(|(row, &partner)| (row, partner as usize))
        .collect();

    if let Some(&(row, _)) = referencing.iter().find(|&&(row, partner)| row == partner) {
        return Err(Error::MzSelfReference {
            row: row.into(),
            id: ids[row],
        });
    }
    if let Some(&(row, partner)) = referencing
        .iter()
        .find(|&&(row, partner)| twin_rows[partner] != row as i32)
    {
        return Err(Error::MzNonreciprocal {
            row: row.into(),
            id: ids[row],
            twin_row: partner.into(),
            twin_id: ids[partner],
        });
    }

    let pairs = referencing.iter().filter(|&&(row, partner)| row < partner);
    for &(row, partner) in pairs.clone() {
        let mut parent_roles = Vec::new();
        if mother[partner] != mother[row] {
            parent_roles.push(ParentRole::Mother);
        }
        if father[partner] != father[row] {
            parent_roles.push(ParentRole::Father);
        }
        if !parent_roles.is_empty() {
            return Err(Error::MzParentMismatch {
                row: row.into(),
                id: ids[row],
                twin_row: partner.into(),
                twin_id: ids[partner],
                parent_roles,
            });
        }
    }

    let Some(sex) = sex else {
        return Ok(());
    };
    for &(row, partner) in pairs {
        if sex[row] != -1 && sex[partner] != -1 && sex[row] != sex[partner] {
            return Err(Error::MzSexMismatch {
                row: row.into(),
                id: ids[row],
                twin_row: partner.into(),
                twin_id: ids[partner],
                sex: sex[row],
                twin_sex: sex[partner],
            });
        }
    }
    Ok(())
}

/// `birth_year_topology` for the first known edge, mother role first, whose
/// child was born before its parent.
fn check_birth_year_topology(
    ids: &[i64],
    mother_rows: &[i32],
    father_rows: &[i32],
    birth_year: &[i32],
) -> Result<(), Error> {
    for (role, parents) in [
        (ParentRole::Mother, mother_rows),
        (ParentRole::Father, father_rows),
    ] {
        let mut first: Option<(usize, usize)> = None;
        let mut violation_count = 0usize;
        for (child, &parent) in parents.iter().enumerate() {
            if parent < 0 {
                continue;
            }
            let parent = parent as usize;
            if birth_year[child] >= 0
                && birth_year[parent] >= 0
                && birth_year[child] < birth_year[parent]
            {
                violation_count += 1;
                first.get_or_insert((child, parent));
            }
        }
        if let Some((child_row, parent_row)) = first {
            return Err(Error::BirthYearTopology {
                parent_role: role,
                child_row: child_row.into(),
                parent_row: parent_row.into(),
                child_id: ids[child_row],
                parent_id: ids[parent_row],
                child_birth_year: birth_year[child_row],
                parent_birth_year: birth_year[parent_row],
                violation_count,
            });
        }
    }
    Ok(())
}

/// An optional column as stored, or `None` when absent, empty, or wholly unknown.
fn normalize_optional<T: Copy + PartialEq>(values: Option<Vec<T>>, unknown: T) -> Option<Vec<T>> {
    values.filter(|values| !values.is_empty() && values.iter().any(|&v| v != unknown))
}

/// Validate `columns` and build the graph, or report the first violation.
///
/// # Errors
///
/// `pedigree_too_large`, `length_mismatch`, `value_out_of_range`,
/// `duplicate_id`, `same_parent_id`, `cycle`, the four MZ codes, and
/// `birth_year_topology`, in that order of precedence.
pub fn build(
    columns: Columns<'_>,
    encoding: SexEncoding,
    limits: Limits,
) -> Result<PedigreeGraph, Error> {
    let n = columns.ids.len();
    if n > limits.max_rows {
        return Err(Error::PedigreeTooLarge {
            n_individuals: n,
            maximum: limits.max_rows,
        });
    }
    check_length("mother", columns.mother, n)?;
    check_length("father", columns.father, n)?;
    for (field, values) in [
        ("twin", columns.twin),
        ("sex", columns.sex),
        ("generation", columns.generation),
        ("birth_year", columns.birth_year),
    ] {
        if let Some(values) = values {
            check_length(field, values, n)?;
        }
    }

    const ID_RANGE: (i64, i64) = (0, i64::MAX);
    const REFERENCE_RANGE: (i64, i64) = (-1, i64::MAX);
    const LABEL_RANGE: (i64, i64) = (-1, i32::MAX as i64);
    check_range("id", columns.ids, ID_RANGE, ID_RANGE)?;
    check_range("mother", columns.mother, REFERENCE_RANGE, REFERENCE_RANGE)?;
    check_range("father", columns.father, REFERENCE_RANGE, REFERENCE_RANGE)?;
    if let Some(twin) = columns.twin {
        check_range("twin", twin, REFERENCE_RANGE, REFERENCE_RANGE)?;
    }
    let sex: Option<Vec<i8>> = match columns.sex {
        Some(raw) => {
            let (accepted, reported) = encoding.ranges();
            check_range("sex", raw, accepted, reported)?;
            Some(raw.iter().map(|&value| encoding.store(value)).collect())
        }
        None => None,
    };
    if let Some(generation) = columns.generation {
        check_range("generation", generation, LABEL_RANGE, LABEL_RANGE)?;
    }
    if let Some(birth_year) = columns.birth_year {
        check_range("birth_year", birth_year, LABEL_RANGE, LABEL_RANGE)?;
    }

    let index = IdIndex::build(columns.ids);
    check_duplicate_ids(columns.ids, &index)?;
    check_same_parent(columns.ids, columns.mother, columns.father)?;

    let mother_rows = index.resolve(columns.mother);
    let father_rows = index.resolve(columns.father);
    let (twin_ids, twin_rows) = match columns.twin {
        Some(twin) => (twin.to_vec(), index.resolve(twin)),
        None => (vec![-1; n], vec![-1; n]),
    };
    drop(index);

    let rows_topological = topology::is_topological(&mother_rows, &father_rows);
    if !rows_topological {
        topology::validate_acyclic(columns.ids, &mother_rows, &father_rows)?;
    }
    check_mz_pairs(
        columns.ids,
        columns.mother,
        columns.father,
        &twin_rows,
        sex.as_deref(),
    )?;

    let narrow = |values: Option<&[i64]>| -> Option<Vec<i32>> {
        values.map(|values| values.iter().map(|&value| value as i32).collect())
    };
    let sex = normalize_optional(sex, -1);
    let generation = normalize_optional(narrow(columns.generation), -1);
    let birth_year = normalize_optional(narrow(columns.birth_year), -1);
    if let Some(birth_year) = &birth_year {
        check_birth_year_topology(columns.ids, &mother_rows, &father_rows, birth_year)?;
    }

    Ok(PedigreeGraph {
        ids: columns.ids.to_vec(),
        mother_ids: columns.mother.to_vec(),
        father_ids: columns.father.to_vec(),
        twin_ids,
        mother_rows,
        father_rows,
        twin_rows,
        sex,
        generation,
        birth_year,
        rows_topological,
    })
}

#[cfg(test)]
mod tests {
    use super::{build, Columns, IdIndex, Layout, Limits, PedigreeGraph, SexEncoding};
    use crate::error::{Error, ParentRole};

    fn columns<'a>(ids: &'a [i64], mother: &'a [i64], father: &'a [i64]) -> Columns<'a> {
        Columns {
            ids,
            mother,
            father,
            twin: None,
            sex: None,
            generation: None,
            birth_year: None,
        }
    }

    fn trio() -> PedigreeGraph {
        build(
            columns(&[10, 20, 30], &[-1, -1, 10], &[-1, -1, 20]),
            SexEncoding::Simace,
            Limits::default(),
        )
        .expect("a valid trio")
    }

    #[test]
    fn a_trio_resolves_its_rows_and_is_topological() {
        let graph = trio();
        assert_eq!(graph.mother_rows, vec![-1, -1, 0]);
        assert_eq!(graph.father_rows, vec![-1, -1, 1]);
        assert_eq!(graph.twin_ids, vec![-1, -1, -1]);
        assert_eq!(graph.twin_rows, vec![-1, -1, -1]);
        assert!(graph.rows_topological);
        assert_eq!(graph.sex, None);
        assert_eq!(graph.len(), 3);
    }

    #[test]
    fn an_external_parent_keeps_its_id_and_has_no_row() {
        let graph = build(
            columns(&[1, 2], &[99, -1], &[-1, -1]),
            SexEncoding::Simace,
            Limits::default(),
        )
        .unwrap();
        assert_eq!(graph.mother_ids, vec![99, -1]);
        assert_eq!(graph.mother_rows, vec![-1, -1]);
    }

    #[test]
    fn children_first_input_is_not_topological_but_valid() {
        let graph = build(
            columns(&[30, 10, 20], &[10, -1, -1], &[20, -1, -1]),
            SexEncoding::Simace,
            Limits::default(),
        )
        .unwrap();
        assert!(!graph.rows_topological);
        assert_eq!(graph.mother_rows, vec![1, -1, -1]);
    }

    #[test]
    fn the_row_limit_is_checked_first() {
        let err = build(
            columns(&[0, 0, 0], &[-1, -1, -1], &[-1, -1, -1]),
            SexEncoding::Simace,
            Limits { max_rows: 2 },
        )
        .unwrap_err();
        assert_eq!(
            err,
            Error::PedigreeTooLarge {
                n_individuals: 3,
                maximum: 2
            }
        );
    }

    #[test]
    fn a_negative_id_is_out_of_range_with_the_id_bounds() {
        let err = build(
            columns(&[0, -1], &[-1, -1], &[-1, -1]),
            SexEncoding::Simace,
            Limits::default(),
        )
        .unwrap_err();
        assert_eq!(
            err,
            Error::ValueOutOfRange {
                field: "id",
                position: 1.into(),
                value: -1,
                minimum: 0,
                maximum: i64::MAX
            }
        );
        assert_eq!(
            err.to_string(),
            format!("'id' value at position 1 is outside [0, {}]", i64::MAX)
        );
    }

    #[test]
    fn plink_sex_reports_its_own_range_and_maps_onto_the_stored_encoding() {
        let err = build(
            Columns {
                sex: Some(&[3, 1, 2]),
                ..columns(&[0, 1, 2], &[-1, -1, -1], &[-1, -1, -1])
            },
            SexEncoding::Plink,
            Limits::default(),
        )
        .unwrap_err();
        assert_eq!(
            err,
            Error::ValueOutOfRange {
                field: "sex",
                position: 0.into(),
                value: 3,
                minimum: 0,
                maximum: 2
            }
        );
        let graph = build(
            Columns {
                sex: Some(&[-1, 0, 1, 2]),
                ..columns(&[0, 1, 2, 3], &[-1; 4], &[-1; 4])
            },
            SexEncoding::Plink,
            Limits::default(),
        )
        .unwrap();
        assert_eq!(graph.sex, Some(vec![-1, -1, 1, 0]));
    }

    #[test]
    fn a_wholly_unknown_optional_column_collapses_to_none() {
        let graph = build(
            Columns {
                sex: Some(&[-1, -1]),
                generation: Some(&[-1, -1]),
                birth_year: Some(&[1990, -1]),
                ..columns(&[0, 1], &[-1, -1], &[-1, -1])
            },
            SexEncoding::Simace,
            Limits::default(),
        )
        .unwrap();
        assert_eq!(graph.sex, None);
        assert_eq!(graph.generation, None);
        assert_eq!(graph.birth_year, Some(vec![1990, -1]));
    }

    #[test]
    fn duplicate_id_names_the_smallest_repeated_id_and_every_row() {
        let err = build(
            columns(&[5, 3, 5, 3, 3], &[-1; 5], &[-1; 5]),
            SexEncoding::Simace,
            Limits::default(),
        )
        .unwrap_err();
        assert_eq!(
            err,
            Error::DuplicateId {
                id: 3,
                rows: vec![1.into(), 3.into(), 4.into()],
                duplicate_count: 3
            }
        );
        assert_eq!(
            err.to_string(),
            "id 3 appears at rows (1, 3, 4); 3 row(s) repeat an earlier id"
        );
    }

    #[test]
    fn same_parent_id_includes_an_external_id() {
        let err = build(
            columns(&[0, 1], &[-1, 99], &[-1, 99]),
            SexEncoding::Simace,
            Limits::default(),
        )
        .unwrap_err();
        assert_eq!(
            err,
            Error::SameParentId {
                row: 1.into(),
                child_id: 1,
                parent_id: 99
            }
        );
        assert_eq!(
            err.to_string(),
            "row 1 names id 99 as both mother and father"
        );
    }

    #[test]
    fn a_cycle_is_reported_after_the_id_checks() {
        let err = build(
            columns(&[0, 1], &[1, 0], &[-1, -1]),
            SexEncoding::Simace,
            Limits::default(),
        )
        .unwrap_err();
        assert_eq!(err, Error::Cycle { ids: vec![0, 1] });
    }

    #[test]
    fn mz_checks_run_in_contract_order() {
        let base = columns(&[0, 1, 2], &[-1, -1, -1], &[-1, -1, -1]);
        let self_ref = build(
            Columns {
                twin: Some(&[0, -1, -1]),
                ..base
            },
            SexEncoding::Simace,
            Limits::default(),
        )
        .unwrap_err();
        assert_eq!(
            self_ref,
            Error::MzSelfReference {
                row: 0.into(),
                id: 0
            }
        );
        assert_eq!(self_ref.to_string(), "row 0 names itself as its MZ co-twin");

        let nonreciprocal = build(
            Columns {
                twin: Some(&[1, 2, 1]),
                ..base
            },
            SexEncoding::Simace,
            Limits::default(),
        )
        .unwrap_err();
        assert_eq!(
            nonreciprocal,
            Error::MzNonreciprocal {
                row: 0.into(),
                id: 0,
                twin_row: 1.into(),
                twin_id: 1
            }
        );
        assert_eq!(
            nonreciprocal.to_string(),
            "the MZ reference at row 0 is not reciprocated by row 1"
        );

        let parents = build(
            Columns {
                twin: Some(&[1, 0, -1]),
                ..columns(&[0, 1, 2], &[7, 8, -1], &[9, 9, -1])
            },
            SexEncoding::Simace,
            Limits::default(),
        )
        .unwrap_err();
        assert_eq!(
            parents,
            Error::MzParentMismatch {
                row: 0.into(),
                id: 0,
                twin_row: 1.into(),
                twin_id: 1,
                parent_roles: vec![ParentRole::Mother]
            }
        );
        assert_eq!(
            parents.to_string(),
            "MZ co-twins at rows 0 and 1 do not name the same parents"
        );

        let sexes = build(
            Columns {
                twin: Some(&[1, 0, -1]),
                sex: Some(&[0, 1, -1]),
                ..base
            },
            SexEncoding::Simace,
            Limits::default(),
        )
        .unwrap_err();
        assert_eq!(
            sexes,
            Error::MzSexMismatch {
                row: 0.into(),
                id: 0,
                twin_row: 1.into(),
                twin_id: 1,
                sex: 0,
                twin_sex: 1
            }
        );
        assert_eq!(
            sexes.to_string(),
            "MZ co-twins at rows 0 and 1 have different known sexes"
        );
    }

    #[test]
    fn an_external_co_twin_forms_no_pair() {
        let graph = build(
            Columns {
                twin: Some(&[77, -1]),
                ..columns(&[0, 1], &[-1, -1], &[-1, -1])
            },
            SexEncoding::Simace,
            Limits::default(),
        )
        .unwrap();
        assert_eq!(graph.twin_ids, vec![77, -1]);
        assert_eq!(graph.twin_rows, vec![-1, -1]);
    }

    #[test]
    fn birth_year_topology_reports_the_mother_role_first_with_its_count() {
        let err = build(
            Columns {
                birth_year: Some(&[2000, 2000, 1990, 1980]),
                ..columns(&[0, 1, 2, 3], &[-1, -1, 0, 0], &[-1, -1, 1, 1])
            },
            SexEncoding::Simace,
            Limits::default(),
        )
        .unwrap_err();
        assert_eq!(
            err,
            Error::BirthYearTopology {
                parent_role: ParentRole::Mother,
                child_row: 2.into(),
                parent_row: 0.into(),
                child_id: 2,
                parent_id: 0,
                child_birth_year: 1990,
                parent_birth_year: 2000,
                violation_count: 2
            }
        );
        assert_eq!(
            err.to_string(),
            "birth_year topology violation: mother-child edge at row 2 \
             has child.birth_year below mother.birth_year"
        );
    }

    #[test]
    fn unknown_birth_years_constrain_nothing() {
        let graph = build(
            Columns {
                birth_year: Some(&[-1, 1990]),
                ..columns(&[0, 1], &[-1, 0], &[-1, -1])
            },
            SexEncoding::Simace,
            Limits::default(),
        )
        .unwrap();
        assert_eq!(graph.birth_year, Some(vec![-1, 1990]));
    }

    #[test]
    fn an_unknown_encoding_is_usage_not_validation() {
        let err = SexEncoding::parse("foo").unwrap_err();
        assert_eq!(
            err.to_string(),
            "sex_encoding must be one of ['plink', 'simace'], got 'foo'"
        );
        assert_eq!(err.code(), "");
    }

    #[test]
    fn the_id_index_resolves_negative_and_unknown_ids_to_minus_one() {
        let index = IdIndex::build(&[30, 10, 20]);
        assert_eq!(index.resolve(&[10, 20, 30, -1, 40]), vec![1, 2, 0, -1, -1]);
        assert_eq!(IdIndex::build(&[]).resolve(&[1]), vec![-1]);
    }

    #[test]
    fn ids_spanning_three_slots_per_id_get_a_table_and_sparser_ids_are_sorted() {
        let dense = |ids: &[i64]| matches!(IdIndex::build(ids).layout, Layout::Dense { .. });
        assert!(dense(&[0, 5]));
        assert!(!dense(&[0, 6]));
        assert!(dense(&[1_000_000, 1_000_005]));
        assert!(!dense(&[i64::MIN, i64::MAX]));
    }

    #[test]
    fn both_layouts_resolve_the_same_rows() {
        let ids = [12, 4, 9, 5, 11, 7];
        let query = [4, 5, 6, 7, 9, 11, 12, 13, 3, -1, -4, i64::MAX];
        let expected = vec![1, 3, -1, 5, 2, 4, 0, -1, -1, -1, -1, -1];
        let index = IdIndex::build(&ids);
        assert!(matches!(index.layout, Layout::Dense { min: 4, .. }));
        assert_eq!(index.resolve(&query), expected);
        let sorted = IdIndex {
            layout: Layout::sorted(&ids),
        };
        assert!(matches!(sorted.layout, Layout::Sorted { .. }));
        assert_eq!(sorted.resolve(&query), expected);
    }

    #[test]
    fn a_table_starting_below_zero_resolves_without_overflow() {
        let index = IdIndex::build(&[-5, 0, 2]);
        assert!(matches!(index.layout, Layout::Dense { min: -5, .. }));
        assert_eq!(
            index.resolve(&[0, 2, -5, 3, i64::MAX]),
            vec![1, 2, -1, -1, -1]
        );
        let extremes = IdIndex::build(&[i64::MIN, i64::MAX]);
        assert_eq!(extremes.resolve(&[i64::MAX, 0]), vec![1, -1]);
    }

    #[test]
    fn both_layouts_report_the_smallest_repeated_id_and_the_repeat_count() {
        let sorted = |ids: &[i64]| IdIndex {
            layout: Layout::sorted(ids),
        };
        let ids = [7, 3, 7, 3, 7, 5];
        assert!(matches!(IdIndex::build(&ids).layout, Layout::Dense { .. }));
        assert_eq!(IdIndex::build(&ids).duplicate(&ids), Some((3, 3)));
        assert_eq!(sorted(&ids).duplicate(&ids), Some((3, 3)));
        assert_eq!(IdIndex::build(&[3, 7, 5]).duplicate(&[3, 7, 5]), None);
        assert_eq!(sorted(&[3, 7, 5]).duplicate(&[3, 7, 5]), None);
    }

    #[test]
    fn ids_collect_into_the_sorted_pairs_buffer() {
        // Layout::sorted's 20-byte-per-row peak rests on this std optimisation.
        let pairs: Vec<(i64, i32)> = vec![(1, 0), (2, 1), (3, 2)];
        let buffer = pairs.as_ptr() as usize;
        let ids: Vec<i64> = pairs.into_iter().map(|(id, _)| id).collect();
        assert_eq!(ids.as_ptr() as usize, buffer);
    }

    #[test]
    fn duplicate_id_is_the_same_error_for_sparse_ids() {
        let err = build(
            columns(&[5000, 3, 5000, 3, 3], &[-1; 5], &[-1; 5]),
            SexEncoding::Simace,
            Limits::default(),
        )
        .unwrap_err();
        assert_eq!(
            err,
            Error::DuplicateId {
                id: 3,
                rows: vec![1.into(), 3.into(), 4.into()],
                duplicate_count: 3
            }
        );
    }

    #[test]
    fn an_empty_pedigree_builds() {
        let graph = build(
            Columns {
                sex: Some(&[]),
                ..columns(&[], &[], &[])
            },
            SexEncoding::Simace,
            Limits::default(),
        )
        .unwrap();
        assert!(graph.is_empty());
        assert!(graph.rows_topological);
        assert_eq!(graph.sex, None);
    }
}
