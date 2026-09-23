//! The four kernels (slice 16b): pairs, pairwise kinship, the kinship
//! matrix and inbreeding.  Each starts from a graph whose seal checked out
//! (`graph::verified`), borrows its `native` columns, and runs the core.

use crate::errors::{HostError, HostResult};
use crate::graph::{self, NATIVE_FIELDS};
use crate::input::IdType;
use crate::threads;
use extendr_api::prelude::*;
use pedigree_graph_core::error::Error;
use pedigree_graph_core::kinship::{self, KinshipPedigree};
use pedigree_graph_core::pool;
use pedigree_graph_core::relationships::{
    pair_blocks, Category, CategorySet, Execution, MaxDegree, PairBlock, Pedigree, N_CATEGORIES,
};
use std::num::NonZeroUsize;

/// A verified graph's `native` fields, held so their slices stay valid.
pub struct Native {
    fields: Vec<Robj>,
}

impl Native {
    pub fn verified(native: &Robj, seal: &Robj) -> HostResult<Native> {
        let list = graph::verified(native, seal)?;
        Ok(Native {
            fields: list.values().collect(),
        })
    }

    fn field(&self, name: &str) -> &Robj {
        let at = NATIVE_FIELDS
            .iter()
            .position(|f| *f == name)
            .expect("a native field name");
        &self.fields[at]
    }

    /// An int32 field; the seal fixed its type when the graph was built.
    fn rows(&self, name: &str) -> &[i32] {
        self.field(name)
            .as_integer_slice()
            .expect("a sealed integer field")
    }

    /// An int64 field, stored as its bits in doubles.
    fn int64(&self, name: &str) -> Vec<i64> {
        self.field(name)
            .as_real_slice()
            .expect("a sealed double field")
            .iter()
            .map(|v| v.to_bits() as i64)
            .collect()
    }

    pub fn len(&self) -> usize {
        self.rows("mother_rows").len()
    }

    fn id_type(&self) -> IdType {
        IdType::parse(self.field("id_type").as_str().unwrap_or("")).expect("a sealed id type")
    }

    fn kinship(&self) -> HostResult<KinshipPedigree<'_>> {
        Ok(KinshipPedigree::try_new(
            self.rows("mother_rows"),
            self.rows("father_rows"),
            self.rows("twin_rows"),
            self.rows("depth"),
        )?)
    }

    /// The ids at `rows` in the graph's id type, as R vectors of that type.
    fn ids_at(&self, rows: impl ExactSizeIterator<Item = i32>) -> Robj {
        let ids = self.int64("ids");
        let at = |r: i32| ids[r as usize];
        match self.id_type() {
            IdType::Integer => Integers::from_values(rows.map(|r| at(r) as i32)).into_robj(),
            IdType::Double => Doubles::from_values(rows.map(|r| at(r) as f64)).into_robj(),
            IdType::Integer64 => {
                let mut out =
                    Doubles::from_values(rows.map(|r| f64::from_bits(at(r) as u64))).into_robj();
                out.set_class(["integer64"])
                    .expect("a character class attribute");
                out
            }
        }
    }

    /// Every id as a string, for matrix dimnames: exact for int64 ids, which
    /// `as.character()` cannot print without bit64.
    fn id_names(&self) -> Robj {
        Strings::from_values(self.int64("ids").iter().map(|id| id.to_string())).into_robj()
    }
}

/// An iterator whose length is known up front, for extendr's exact-size
/// vector constructors, so chained blocks fill an R vector without a copy.
struct Exact<I> {
    inner: I,
    left: usize,
}

impl<I: Iterator> Exact<I> {
    fn new(inner: I, len: usize) -> Exact<I> {
        Exact { inner, left: len }
    }
}

impl<I: Iterator> Iterator for Exact<I> {
    type Item = I::Item;

    fn next(&mut self) -> Option<I::Item> {
        let item = self.inner.next()?;
        self.left -= 1;
        Some(item)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.left, Some(self.left))
    }
}

impl<I: Iterator> ExactSizeIterator for Exact<I> {}

/// The categories one selector names, in registry order.
fn selection(max_degree: &Robj, categories: &Robj) -> HostResult<CategorySet> {
    match (max_degree.is_null(), categories.is_null()) {
        (false, true) => {
            let degree = match (
                max_degree.len(),
                max_degree.as_real(),
                max_degree.as_integer(),
            ) {
                (1, Some(v), _) if v.is_finite() && v == v.trunc() => v,
                (1, _, Some(v)) if v != i32::MIN => f64::from(v),
                _ => {
                    return Err(HostError::usage(
                        "max_degree must be one whole number".to_string(),
                    ))
                }
            };
            if !(0.0..=f64::from(MaxDegree::MAX.get())).contains(&degree) {
                return Err(Error::MaxDegreeOutOfRange {
                    value: degree as i64,
                    minimum: 0,
                    maximum: i64::from(MaxDegree::MAX.get()),
                }
                .into());
            }
            Ok(CategorySet::up_to_degree(degree as u8))
        }
        (true, false) => {
            let codes = Strings::try_from(categories.clone()).map_err(|_| {
                HostError::usage("categories must be a character vector of codes".to_string())
            })?;
            let mut unknown: Vec<String> = codes
                .iter()
                .filter(|code| code.is_na() || Category::parse(code.as_ref()).is_none())
                .map(|code| {
                    if code.is_na() {
                        "NA".to_string()
                    } else {
                        code.to_string()
                    }
                })
                .collect();
            if !unknown.is_empty() {
                unknown.sort();
                unknown.dedup();
                return Err(HostError::validation(
                    "unknown_relationship_category",
                    format!(
                        "unknown relationship category code(s): {}",
                        unknown.join(", ")
                    ),
                    vec![("codes", Strings::from_values(unknown).into_robj())],
                ));
            }
            Ok(codes
                .iter()
                .filter_map(|code| Category::parse(code.as_ref()))
                .collect())
        }
        _ => Err(HostError::usage(
            "exactly one of max_degree or categories is required".to_string(),
        )),
    }
}

/// The package pool at the committed budget.
fn package_pool() -> HostResult<&'static rayon::ThreadPool> {
    let threads = NonZeroUsize::new(threads::budget()?).expect("a budget of at least 1");
    Ok(pool::configure(threads)?)
}

/// The rows an R data frame can hold: compact row names are int32.
const MAX_FRAME_ROWS: usize = i32::MAX as usize;

/// `list(code, first, second[, first_id, second_id], requested)`.
///
/// `code` is the 1-based registry index per pair (the R wrapper makes it a
/// factor over all 23 codes); `first`/`second` are 1-based graph rows.
pub fn relationship_pairs(
    native: &Robj,
    seal: &Robj,
    max_degree: &Robj,
    categories: &Robj,
    execution: &str,
    ids: bool,
) -> HostResult<Robj> {
    let graph = Native::verified(native, seal)?;
    let requested = selection(max_degree, categories)?;
    let execution = Execution::parse(execution).ok_or_else(|| {
        HostError::usage(format!(
            "execution must be \"speed\" or \"memory\", got {execution:?}"
        ))
    })?;
    // Every public operation commits the budget, as in Python.
    let pool = package_pool()?;

    let mut blocks: Vec<PairBlock> = vec![PairBlock::default(); N_CATEGORIES];
    if let (Some(top), true) = (requested.top_degree(), graph.len() >= 2) {
        let mother_ids = graph.int64("mother_ids");
        let father_ids = graph.int64("father_ids");
        let ped = Pedigree::try_new(
            graph.rows("mother_rows"),
            graph.rows("father_rows"),
            graph.rows("twin_rows"),
            &mother_ids,
            &father_ids,
        )?;
        let max_degree = MaxDegree::try_new(top)?;
        let found = pool.install(|| pair_blocks(&ped, max_degree, requested, None, execution))?;
        blocks = found.0;
    }

    let total: usize = blocks.iter().map(PairBlock::len).sum();
    if total > MAX_FRAME_ROWS {
        return Err(HostError::resource(
            "pairs_exceed_frame_rows",
            format!(
                "{total} pairs exceed the {MAX_FRAME_ROWS} rows an R data frame holds; \
                 select fewer categories"
            ),
            vec![
                ("n_pairs", (total as f64).into()),
                ("maximum", (MAX_FRAME_ROWS as f64).into()),
            ],
        ));
    }
    // Built straight into R vectors from the blocks: no concatenated copy.
    let code = Integers::from_values(Exact::new(
        blocks
            .iter()
            .enumerate()
            .flat_map(|(i, block)| std::iter::repeat_n(i as i32 + 1, block.len())),
        total,
    ))
    .into_robj();
    let rows = |pick: fn(&PairBlock) -> &[i32]| {
        Integers::from_values(Exact::new(
            blocks.iter().flat_map(|b| pick(b).iter().map(|&r| r + 1)),
            total,
        ))
        .into_robj()
    };
    let first = rows(|b| &b.first);
    let second = rows(|b| &b.second);
    let requested_flags =
        Logicals::from_values(Category::ALL.iter().map(|c| requested.contains(*c))).into_robj();

    let mut names = vec!["code", "first", "second"];
    let mut values = vec![code, first, second];
    if ids {
        let all = |pick: fn(&PairBlock) -> &[i32]| {
            graph.ids_at(Exact::new(
                blocks.iter().flat_map(|b| pick(b).iter().copied()),
                total,
            ))
        };
        names.extend(["first_id", "second_id"]);
        values.push(all(|b| &b.first));
        values.push(all(|b| &b.second));
    }
    names.push("requested");
    values.push(requested_flags);
    Ok(List::from_names_and_values(names, values)
        .expect("one value per name")
        .into_robj())
}

/// 1-based R rows to 0-based graph rows, or the first one that is not a row.
fn graph_rows(name: &'static str, rows: &Robj, n: usize) -> HostResult<Vec<i32>> {
    let bad = |position: usize, value: Robj| {
        HostError::validation(
            "value_out_of_range",
            format!(
                "'{name}' value at position {} is not a row from 1 to {n}",
                position + 1
            ),
            vec![
                ("field", name.into()),
                ("position", ((position + 1) as f64).into()),
                ("value", value),
                ("minimum", 1.0.into()),
                ("maximum", (n as f64).into()),
            ],
        )
    };
    if let Some(slice) = rows.as_integer_slice() {
        return slice
            .iter()
            .enumerate()
            .map(|(p, &r)| {
                if r >= 1 && (r as usize) <= n {
                    Ok(r - 1)
                } else if r == i32::MIN {
                    Err(bad(p, Rint::na().into()))
                } else {
                    Err(bad(p, r.into()))
                }
            })
            .collect();
    }
    if let (Some(slice), false) = (rows.as_real_slice(), rows.inherits("integer64")) {
        return slice
            .iter()
            .enumerate()
            .map(|(p, &r)| {
                if r >= 1.0 && r <= n as f64 && r == r.trunc() {
                    Ok(r as i32 - 1)
                } else {
                    Err(bad(p, r.into()))
                }
            })
            .collect();
    }
    Err(HostError::usage(format!(
        "{name} must be integer or whole-number double rows"
    )))
}

/// Kinship per `(first[k], second[k])`, 1-based rows, as doubles.
pub fn pair_kinship(native: &Robj, seal: &Robj, first: &Robj, second: &Robj) -> HostResult<Robj> {
    let graph = Native::verified(native, seal)?;
    let n = graph.len();
    let first = graph_rows("first", first, n)?;
    let second = graph_rows("second", second, n)?;
    if first.len() != second.len() {
        return Err(HostError::usage(format!(
            "first and second must have the same length, got {} and {}",
            first.len(),
            second.len()
        )));
    }
    let values = kinship::pair_kinship(graph.kinship()?, &first, &second)?;
    Ok(Doubles::from_values(values.into_iter().map(f64::from)).into_robj())
}

/// `F` per row, as doubles.
pub fn inbreeding(native: &Robj, seal: &Robj) -> HostResult<Robj> {
    let graph = Native::verified(native, seal)?;
    Ok(Doubles::from_values(kinship::inbreeding(graph.kinship()?)?).into_robj())
}

/// `list(i, p, x, names)`: the upper triangle as `dsCMatrix` slots
/// (0-based `i`, int32 `p`, double `x`) and the ids as dimnames.
pub fn kinship_matrix(native: &Robj, seal: &Robj, max_nnz: Option<usize>) -> HostResult<Robj> {
    let graph = Native::verified(native, seal)?;
    let csc =
        kinship::kinship_csc_upper(graph.kinship()?, max_nnz.unwrap_or(kinship::MAX_CSC_NNZ))?;
    let i = Integers::from_values(csc.indices.iter().copied()).into_robj();
    drop(csc.indices);
    let p = Integers::from_values(csc.indptr.iter().copied()).into_robj();
    let x = Doubles::from_values(csc.data.iter().map(|&v| f64::from(v))).into_robj();
    Ok(
        List::from_names_and_values(["i", "p", "x", "names"], [i, p, x, graph.id_names()])
            .expect("four names and four values")
            .into_robj(),
    )
}
