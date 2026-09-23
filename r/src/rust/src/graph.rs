//! Construction and the seal over what kernels read (slice 16, decision 2).
//!
//! `pedigree_graph()` keeps the validated columns as plain R vectors in the
//! graph's `native` list, so a graph survives `saveRDS` and forked workers.
//! Because a list is user-editable, `seal` fingerprints every `native` field;
//! every kernel re-hashes them and refuses a mismatch as `graph_modified`.
//! The hash is xxh3-64 over a fixed layout, never `std`'s `DefaultHasher`,
//! whose output may change between Rust releases and would reject graphs
//! saved under an older build.

use crate::errors::{HostError, HostResult};
use crate::input::{coerce, coerce_ids, IdType};
use extendr_api::prelude::*;
use pedigree_graph_core::graph::{self, Columns, Limits, SexEncoding};
use pedigree_graph_core::topology;
use xxhash_rust::xxh3::Xxh3;

/// The seal's layout version; a change to the hashed layout bumps it.
const SEAL_VERSION: &str = "pg1";

/// The `native` fields in the order they are stored and hashed.
pub const NATIVE_FIELDS: [&str; 13] = [
    "id_type",
    "ids",
    "mother_ids",
    "father_ids",
    "twin_ids",
    "mother_rows",
    "father_rows",
    "twin_rows",
    "depth",
    "sex",
    "generation",
    "birth_year",
    "rows_topological",
];

/// `i64` values as a double vector holding their bits, bit64's layout.
fn int64_bits(values: &[i64]) -> Robj {
    Doubles::from_values(values.iter().map(|&v| f64::from_bits(v as u64))).into_robj()
}

fn integers(values: &[i32]) -> Robj {
    Integers::from_values(values.iter().copied()).into_robj()
}

fn optional_integers<T: Copy + Into<i32>>(values: Option<Vec<T>>) -> Robj {
    match values {
        Some(v) => Integers::from_values(v.into_iter().map(Into::into)).into_robj(),
        None => ().into(),
    }
}

/// The user-facing ids in the storage they arrived in.
fn ids_as(values: &[i64], storage: IdType) -> Robj {
    match storage {
        // Ids that arrived as R integers fit in one.
        IdType::Integer => Integers::from_values(values.iter().map(|&v| v as i32)).into_robj(),
        IdType::Double => Doubles::from_values(values.iter().map(|&v| v as f64)).into_robj(),
        IdType::Integer64 => {
            let mut out = int64_bits(values);
            out.set_class(["integer64"])
                .expect("a character class attribute");
            out
        }
    }
}

/// Build a graph's pieces: `list(native = <list>, id = <ids>, seal = <chr>)`.
#[allow(clippy::too_many_arguments)]
pub fn build(
    id: Robj,
    mother: Robj,
    father: Robj,
    twin: Robj,
    sex: Robj,
    generation: Robj,
    birth_year: Robj,
    sex_encoding: &str,
) -> HostResult<Robj> {
    for (name, column) in [("id", &id), ("mother", &mother), ("father", &father)] {
        if column.is_null() {
            return Err(HostError::validation(
                "missing_field",
                format!("input is missing the required '{name}' field"),
                vec![("field", name.into())],
            ));
        }
    }
    let encoding = SexEncoding::parse(sex_encoding)?;
    let ids = coerce_ids(&id)?;
    let mother = coerce("mother", &mother)?.values;
    let father = coerce("father", &father)?.values;
    let optional = |name: &'static str, column: &Robj| -> HostResult<Option<Vec<i64>>> {
        if column.is_null() {
            Ok(None)
        } else {
            Ok(Some(coerce(name, column)?.values))
        }
    };
    let twin = optional("twin", &twin)?;
    let sex = optional("sex", &sex)?;
    let generation = optional("generation", &generation)?;
    let birth_year = optional("birth_year", &birth_year)?;
    let built = graph::build(
        Columns {
            ids: &ids.values,
            mother: &mother,
            father: &father,
            twin: twin.as_deref(),
            sex: sex.as_deref(),
            generation: generation.as_deref(),
            birth_year: birth_year.as_deref(),
        },
        encoding,
        Limits::default(),
    )?;
    let depth = topology::structural_depth(&built.mother_rows, &built.father_rows);

    let native = List::from_names_and_values(
        NATIVE_FIELDS,
        [
            ids.storage.name().into(),
            int64_bits(&built.ids),
            int64_bits(&built.mother_ids),
            int64_bits(&built.father_ids),
            int64_bits(&built.twin_ids),
            integers(&built.mother_rows),
            integers(&built.father_rows),
            integers(&built.twin_rows),
            integers(&depth),
            optional_integers(built.sex),
            optional_integers(built.generation),
            optional_integers(built.birth_year),
            built.rows_topological.into(),
        ],
    )
    .expect("one value per native field")
    .into_robj();
    let seal = seal_of(&native)?;
    Ok(List::from_names_and_values(
        ["native", "id", "seal"],
        [native, ids_as(&built.ids, ids.storage), seal.into()],
    )
    .expect("three names and three values")
    .into_robj())
}

/// Feed bytes through a fixed buffer so large vectors hash without a copy of their size.
struct Feed {
    hasher: Xxh3,
    buffer: Vec<u8>,
}

impl Feed {
    fn new() -> Feed {
        Feed {
            hasher: Xxh3::new(),
            buffer: Vec::with_capacity(1 << 16),
        }
    }

    fn push(&mut self, bytes: &[u8]) {
        if self.buffer.len() + bytes.len() > self.buffer.capacity() {
            self.hasher.update(&self.buffer);
            self.buffer.clear();
        }
        self.buffer.extend_from_slice(bytes);
    }

    fn finish(mut self) -> u64 {
        self.hasher.update(&self.buffer);
        self.hasher.digest()
    }
}

/// R's own `SEXPTYPE` code for a field's type (Rinternals.h), stable across
/// R and extendr releases, unlike the position of extendr's `Rtype` variant.
fn sexptype(rtype: Rtype) -> u8 {
    match rtype {
        Rtype::Null => 0,
        Rtype::Logicals => 10,
        Rtype::Integers => 13,
        Rtype::Doubles => 14,
        Rtype::Strings => 16,
        _ => 255,
    }
}

/// The seal of a `native` list: its version and the xxh3-64 of its fields.
///
/// Each field contributes its name, R type tag, length and little-endian
/// values, so a changed type or length changes the seal as a value does.
fn seal_of(native: &Robj) -> HostResult<String> {
    let names: Vec<&str> = native.names().map(|n| n.collect()).unwrap_or_default();
    if native.rtype() != Rtype::List || names != NATIVE_FIELDS {
        return Err(HostError::graph_modified());
    }
    let list = List::try_from(native.clone()).map_err(|_| HostError::graph_modified())?;
    let mut feed = Feed::new();
    for (name, value) in NATIVE_FIELDS.iter().zip(list.values()) {
        feed.push(name.as_bytes());
        feed.push(&[0, sexptype(value.rtype())]);
        feed.push(&(value.len() as u64).to_le_bytes());
        match value.rtype() {
            Rtype::Integers => {
                for v in value.as_integer_slice().expect("an integer vector") {
                    feed.push(&v.to_le_bytes());
                }
            }
            Rtype::Doubles => {
                for v in value.as_real_slice().expect("a double vector") {
                    feed.push(&v.to_bits().to_le_bytes());
                }
            }
            Rtype::Logicals => {
                for v in value.as_logical_slice().expect("a logical vector") {
                    feed.push(&[if v.is_true() {
                        1
                    } else if v.is_false() {
                        0
                    } else {
                        2
                    }]);
                }
            }
            Rtype::Strings => {
                for s in value.as_str_vector().expect("a character vector") {
                    feed.push(&(s.len() as u64).to_le_bytes());
                    feed.push(s.as_bytes());
                }
            }
            Rtype::Null => {}
            _ => return Err(HostError::graph_modified()),
        }
    }
    Ok(format!("{SEAL_VERSION}:{:016x}", feed.finish()))
}

/// A graph's `native` list after its seal checked out.
pub fn verified(native: &Robj, seal: &Robj) -> HostResult<List> {
    let expected = seal.as_str().ok_or_else(HostError::graph_modified)?;
    if seal_of(native)? != expected {
        return Err(HostError::graph_modified());
    }
    let list = List::try_from(native.clone()).map_err(|_| HostError::graph_modified())?;
    if IdType::parse(list.values().next().and_then(|v| v.as_str()).unwrap_or("")).is_none() {
        return Err(HostError::graph_modified());
    }
    Ok(list)
}
