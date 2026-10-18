//! The memoised walk of the ADR 0009 recurrence.
//!
//! `phi(a, a) = (1 + phi(m_a, f_a)) / 2` with a missing parent contributing 0
//! and an MZ co-twin taking the self formula; otherwise the endpoint of
//! greater structural depth, ties to the greater row, is peeled to its
//! parents, `phi(a, b) = (phi(m_c, o) + phi(f_c, o)) / 2`.  Every step is the
//! correctly rounded float32 of a half-sum of two float32 operands, so each
//! key has one value whatever order it is reached in, and a memo hit returns
//! the bit a cold walk would store.
//!
//! The walk is an explicit-stack post-order: a key is pushed for *expand*
//! (discover unresolved dependencies) and re-pushed for *compute* (combine
//! them).  Each frame carries what was learned where the key was met, so
//! a key's rule is read once per meeting rather than at every step.  LIFO
//! order resolves a node's whole subtree before its compute marker, so
//! every distinct key is computed once per walker even when shared across
//! requested pairs, and endpoint order cannot change a bit.
//!
//! A key whose value needs no walk is never pushed or stored ([`Rule`]):
//! a self-like key with a missing parent, a key whose peeled endpoint is a
//! founder, and a key whose endpoints share no ancestor, which
//! [`AncestorSignatures`] proves.  A call splits its pairs or support
//! columns into chunks over the current Rayon pool, one walker per worker
//! ([`run_chunks`]); since a key has one value whoever computes it, the
//! output is the same bits for every thread count.

use super::ancestry::AncestorSignatures;
use super::memo::PairMemo;
use crate::alloc::{self, Family};
use crate::error::Error;
use crate::relationships::{check_column_length, check_row_range};
use rayon::prelude::*;
use std::ops::Range;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

/// The columns the recurrence reads, borrowed from the host for one call.
#[derive(Clone, Copy)]
pub struct KinshipPedigree<'a> {
    mother: &'a [i32],
    father: &'a [i32],
    twin: &'a [i32],
    depth: &'a [i32],
}

impl<'a> KinshipPedigree<'a> {
    /// Check the four columns have one entry per row and every parent or
    /// twin row is `-1` or a row of the pedigree.
    ///
    /// # Errors
    ///
    /// [`Error::LengthMismatch`] and [`Error::ValueOutOfRange`], as the
    /// relationship engine reports them.
    pub fn try_new(
        mother: &'a [i32],
        father: &'a [i32],
        twin: &'a [i32],
        depth: &'a [i32],
    ) -> Result<KinshipPedigree<'a>, Error> {
        let n = mother.len();
        check_column_length("father", father.len(), n)?;
        check_column_length("twin", twin.len(), n)?;
        check_column_length("depth", depth.len(), n)?;
        check_row_range("mother", mother, n)?;
        check_row_range("father", father, n)?;
        check_row_range("twin", twin, n)?;
        Ok(KinshipPedigree {
            mother,
            father,
            twin,
            depth,
        })
    }

    pub fn len(&self) -> usize {
        self.mother.len()
    }

    pub fn is_empty(&self) -> bool {
        self.mother.is_empty()
    }

    pub fn mother(&self) -> &'a [i32] {
        self.mother
    }

    pub fn father(&self) -> &'a [i32] {
        self.father
    }

    pub fn twin(&self) -> &'a [i32] {
        self.twin
    }

    pub fn depth(&self) -> &'a [i32] {
        self.depth
    }
}

#[inline]
fn canon(a: i32, b: i32) -> (u32, u32) {
    let (a, b) = (a as u32, b as u32);
    if a <= b {
        (a, b)
    } else {
        (b, a)
    }
}

/// How the recurrence evaluates one canonical key.
enum Rule {
    /// Fixed without a walk, so the key takes no memo slot: `0.5` for a
    /// self-like key with a missing parent, `0` for a key whose peeled
    /// endpoint is a founder or whose endpoints share no ancestor.
    Known(f32),
    /// Combined from its dependencies, once each is resolved.
    Walk(Walk),
}

/// The rule of a key that needs a walk.
#[derive(Clone, Copy)]
enum Walk {
    /// A self-like key: `(1 + phi(mother, father)) / 2`.
    SelfLike((u32, u32)),
    /// `(phi(mother, other) + phi(father, other)) / 2`, a missing parent
    /// contributing 0.
    Peel([Option<(u32, u32)>; 2]),
}

/// An operand of a compute frame: its value, when the rule fixes it or the
/// memo holds it at expand time, or else the key its own walk will store.
#[derive(Clone, Copy)]
enum Dep {
    Value(f32),
    Pending((u32, u32)),
}

/// One item of the walk stack.  An expand frame carries its key's rule, read
/// once where the key was met.  A compute frame carries the two operands of
/// its key's half-sum, `1` standing in for a self-like key's constant term
/// (`(1 + phi) / 2` is the same bits as `(phi + 1) / 2`), so neither frame
/// re-reads the rows of a key it has already classified.
#[derive(Clone, Copy)]
enum Frame {
    Expand((u32, u32), Walk),
    Compute((u32, u32), [Dep; 2]),
}

impl alloc::Dtype for Frame {
    const NAME: &'static str = "object";
}

/// One row's recurrence columns side by side, so a rule reads one cache
/// line per endpoint rather than one per column.
#[derive(Clone, Copy)]
#[repr(C, align(16))]
struct Row {
    depth: i32,
    mother: i32,
    father: i32,
    twin: i32,
}

impl alloc::Dtype for Row {
    const NAME: &'static str = "object";
}

/// A pedigree's recurrence columns packed one row to 16 bytes, beside the
/// signatures' 32.  Build it once per pedigree, next to its
/// [`AncestorSignatures`], and share it with every [`Walker`] over that
/// pedigree.
pub struct KinshipRows(Vec<Row>);

impl KinshipRows {
    /// Pack `ped`'s depth, parent and twin columns.
    ///
    /// # Errors
    ///
    /// [`Error::AllocationFailed`] with operation `kinship_columns`.
    pub fn new(ped: &KinshipPedigree<'_>) -> Result<Self, Error> {
        let mut rows = Vec::new();
        alloc::extend(
            &mut rows,
            (0..ped.len()).map(|i| Row {
                depth: ped.depth[i],
                mother: ped.mother[i],
                father: ped.father[i],
                twin: ped.twin[i],
            }),
            Family::KinshipColumns,
        )?;
        Ok(KinshipRows(rows))
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

/// One memo and one stack over a pedigree's packed rows and its ancestor
/// signatures: the state of one worker.
pub struct Walker<'a> {
    rows: &'a [Row],
    signatures: &'a AncestorSignatures,
    memo: PairMemo,
    stack: Vec<Frame>,
}

impl<'a> Walker<'a> {
    /// A walker with an empty memo.  `rows` and `signatures` must be built
    /// from the same pedigree.
    ///
    /// # Errors
    ///
    /// [`Error::AllocationFailed`] with operation `kinship_memo`.
    pub fn new(rows: &'a KinshipRows, signatures: &'a AncestorSignatures) -> Result<Self, Error> {
        Ok(Walker {
            rows: &rows.0,
            signatures,
            memo: PairMemo::new(rows.len())?,
            stack: Vec::new(),
        })
    }

    pub fn memo(&self) -> &PairMemo {
        &self.memo
    }

    #[inline]
    fn push(&mut self, frame: Frame) -> Result<(), Error> {
        alloc::push(&mut self.stack, frame, Family::KinshipStack)
    }

    /// The rule of a canonical key.  The peeled endpoint is the deeper one,
    /// ties to the greater row.
    ///
    /// `always`: under plain `#[inline]` rustc kept this out of line, and
    /// inlining it alone made `pair_kinship` 7 to 11% faster on the simACE
    /// `bench_scale` pedigrees of 1M to 10M rows.
    #[inline(always)]
    fn rule(&self, (lo, hi): (u32, u32)) -> Rule {
        let (l, h) = (lo as usize, hi as usize);
        let (rl, rh) = (self.rows[l], self.rows[h]);
        let (peeled, other, p, o) = if rl.depth > rh.depth {
            (l, h, rl, rh)
        } else {
            (h, l, rh, rl)
        };
        let (m, f) = (p.mother, p.father);
        if other == peeled || o.twin == peeled as i32 || p.twin == other as i32 {
            return if m < 0 || f < 0 {
                Rule::Known(0.5)
            } else {
                Rule::Walk(Walk::SelfLike(canon(m, f)))
            };
        }
        if (m < 0 && f < 0) || self.signatures.disjoint(lo, hi) {
            return Rule::Known(0.0);
        }
        let other = other as i32;
        Rule::Walk(Walk::Peel(
            [m, f].map(|parent| (parent >= 0).then(|| canon(parent, other))),
        ))
    }

    /// A dependency's value if its rule fixes it or the memo holds it, else
    /// its pending key and the expand frame that will resolve it.
    #[inline(always)]
    fn dep(&self, key: (u32, u32)) -> (Dep, Option<Frame>) {
        match self.rule(key) {
            Rule::Known(v) => (Dep::Value(v), None),
            Rule::Walk(walk) => match self.memo.get(key.0, key.1) {
                Some(v) => (Dep::Value(v), None),
                None => (Dep::Pending(key), Some(Frame::Expand(key, walk))),
            },
        }
    }

    #[expect(
        clippy::expect_used,
        reason = "a compute frame is pushed beneath the expand frames of its pending \
                  dependencies, so each is memoised before it is read, and the root, \
                  pushed first, resolves last"
    )]
    #[inline]
    fn stored(&self, (lo, hi): (u32, u32)) -> f32 {
        self.memo.get(lo, hi).expect("dependency resolved")
    }

    /// Kinship of rows `a` and `b`, walking whatever the memo lacks.  Only
    /// keys whose rule is [`Rule::Walk`] are pushed or stored.
    pub fn resolve(&mut self, a: i32, b: i32) -> Result<f32, Error> {
        let root = canon(a, b);
        let walk = match self.rule(root) {
            Rule::Known(v) => return Ok(v),
            Rule::Walk(walk) => walk,
        };
        if let Some(v) = self.memo.get(root.0, root.1) {
            return Ok(v);
        }
        self.push(Frame::Expand(root, walk))?;
        while let Some(frame) = self.stack.pop() {
            match frame {
                Frame::Expand(key, walk) => {
                    if self.memo.get(key.0, key.1).is_some() {
                        continue;
                    }
                    let [(d0, e0), (d1, e1)] = match walk {
                        Walk::SelfLike(parents) => [self.dep(parents), (Dep::Value(1.0), None)],
                        Walk::Peel(keys) => {
                            keys.map(|key| key.map_or((Dep::Value(0.0), None), |key| self.dep(key)))
                        }
                    };
                    self.push(Frame::Compute(key, [d0, d1]))?;
                    for expand in [e0, e1].into_iter().flatten() {
                        self.push(expand)?;
                    }
                }
                Frame::Compute(key, deps) => {
                    let [v0, v1] = deps.map(|dep| match dep {
                        Dep::Value(v) => v,
                        Dep::Pending(key) => self.stored(key),
                    });
                    self.memo.insert(key.0, key.1, 0.5f32 * (v0 + v1))?;
                }
            }
        }
        Ok(self.stored(root))
    }
}

/// An `f32` stored atomically as its bits.
#[repr(transparent)]
struct AtomicF32(AtomicU32);

impl alloc::Dtype for AtomicF32 {
    const NAME: &'static str = "float32";
}

/// A float32 output the workers of one call write at disjoint positions.
struct SharedOutput(Vec<AtomicF32>);

impl SharedOutput {
    fn new(len: usize) -> Result<Self, Error> {
        let mut values = alloc::with_capacity(len, Family::KinshipOutput)?;
        values.resize_with(len, || AtomicF32(AtomicU32::new(0)));
        Ok(SharedOutput(values))
    }

    #[inline]
    fn set(&self, position: usize, value: f32) {
        self.0[position].0.store(value.to_bits(), Ordering::Relaxed);
    }

    /// The values, in the same allocation: `AtomicF32` is transparent over
    /// `AtomicU32`, which shares size and alignment with `f32`, so the
    /// standard library collects in place (`shared_output_converts_in_place`
    /// holds it to that).
    fn into_values(self) -> Vec<f32> {
        self.0
            .into_iter()
            .map(|value| f32::from_bits(value.0.into_inner()))
            .collect()
    }
}

/// Run `fill` on the chunks of `0..len`, `chunk` items each, across the
/// current Rayon pool.
///
/// Each worker takes the next chunk from a shared counter and keeps one
/// walker, built on its first chunk, for every chunk it takes, so shared
/// ancestor keys are walked once per worker rather than once per chunk.  A
/// key's value does not depend on which walker reaches it, so the result is
/// the same for every thread count.  After a failure the remaining chunks
/// are abandoned and one error is returned.  `len` must be positive, since
/// the caller builds the rows and signatures only for a call with work.
fn run_chunks<F>(
    rows: &KinshipRows,
    signatures: &AncestorSignatures,
    len: usize,
    chunk: usize,
    fill: F,
) -> Result<(), Error>
where
    F: Fn(&mut Walker<'_>, Range<usize>) -> Result<(), Error> + Sync,
{
    let chunks = len.div_ceil(chunk);
    let next = AtomicUsize::new(0);
    let workers = rayon::current_num_threads().min(chunks);
    (0..workers).into_par_iter().try_for_each(|_| {
        let mut walker = None;
        loop {
            let index = next.fetch_add(1, Ordering::Relaxed);
            if index >= chunks {
                return Ok(());
            }
            let items = index * chunk..((index + 1) * chunk).min(len);
            let result = match &mut walker {
                Some(walker) => fill(walker, items),
                None => Walker::new(rows, signatures)
                    .and_then(|built| fill(walker.insert(built), items)),
            };
            if result.is_err() {
                next.fetch_max(chunks, Ordering::Relaxed);
                return result;
            }
        }
    })
}

/// Pairs per chunk of a [`pair_kinship`] call.
const PAIR_CHUNK: usize = 4096;

/// Columns per chunk of a [`support_values`] call.
const COLUMN_CHUNK: usize = 256;

fn check_pairs(ped: &KinshipPedigree<'_>, first: &[i32], second: &[i32]) -> Result<(), Error> {
    check_column_length("second", second.len(), first.len())?;
    let n = ped.len();
    let maximum = n as i64 - 1;
    for (field, rows) in [("first", first), ("second", second)] {
        if let Some(position) = rows
            .iter()
            .position(|&row| row < 0 || i64::from(row) > maximum)
        {
            return Err(Error::ValueOutOfRange {
                field,
                position: position.into(),
                value: i64::from(rows[position]),
                minimum: 0,
                maximum,
            });
        }
    }
    Ok(())
}

/// Kinship per requested pair, positionally aligned to `first` and `second`.
///
/// # Errors
///
/// [`Error::LengthMismatch`] when `second` is not as long as `first`,
/// [`Error::ValueOutOfRange`] at the first endpoint outside `0..n`, and
/// [`Error::AllocationFailed`] for the memo, the stack or the output.
pub fn pair_kinship(
    ped: KinshipPedigree<'_>,
    first: &[i32],
    second: &[i32],
) -> Result<Vec<f32>, Error> {
    check_pairs(&ped, first, second)?;
    if first.is_empty() {
        return Ok(Vec::new());
    }
    let out = SharedOutput::new(first.len())?;
    let signatures = AncestorSignatures::build(&ped)?;
    let rows = KinshipRows::new(&ped)?;
    run_chunks(
        &rows,
        &signatures,
        first.len(),
        PAIR_CHUNK,
        |walker, pairs| {
            for k in pairs {
                out.set(k, walker.resolve(first[k], second[k])?);
            }
            Ok(())
        },
    )?;
    Ok(out.into_values())
}

/// Where column `column`'s sorted index range holds `row`, if it does.
fn mirror(indptr: &[i64], indices: &[i32], column: usize, row: i32) -> Option<usize> {
    let start = indptr[column] as usize;
    let end = indptr[column + 1] as usize;
    indices[start..end]
        .binary_search(&row)
        .ok()
        .map(|offset| start + offset)
}

fn support_with(
    ped: KinshipPedigree<'_>,
    indptr: &[i64],
    indices: &[i32],
) -> Result<Vec<f32>, Error> {
    if indices.is_empty() {
        return Ok(Vec::new());
    }
    let n = ped.len();
    let data = SharedOutput::new(indices.len())?;
    let signatures = AncestorSignatures::build(&ped)?;
    let rows = KinshipRows::new(&ped)?;
    run_chunks(&rows, &signatures, n, COLUMN_CHUNK, |walker, columns| {
        for column in columns {
            let start = indptr[column] as usize;
            let end = indptr[column + 1] as usize;
            for position in start..end {
                let row = indices[position];
                if row as usize > column {
                    break;
                }
                let value = walker.resolve(row, column as i32)?;
                data.set(position, value);
                if row as usize == column {
                    continue;
                }
                let mirrored = mirror(indptr, indices, row as usize, column as i32).ok_or(
                    Error::KinshipSupportAsymmetric {
                        row: (row as usize).into(),
                        column: column.into(),
                    },
                )?;
                data.set(mirrored, value);
            }
        }
        Ok(())
    })?;
    Ok(data.into_values())
}

fn check_support(ped: &KinshipPedigree<'_>, indptr: &[i64], indices: &[i32]) -> Result<(), Error> {
    let n = ped.len();
    check_column_length("indptr", indptr.len(), n + 1)?;
    let nnz = indices.len() as i64;
    if let Some(position) = indptr
        .windows(2)
        .position(|w| w[0] < 0 || w[1] < w[0] || w[1] > nnz)
    {
        return Err(Error::ValueOutOfRange {
            field: "indptr",
            position: (position + 1).into(),
            value: indptr[position + 1],
            minimum: indptr[position].max(0),
            maximum: nnz,
        });
    }
    if indptr.first().is_some_and(|&v| v != 0) {
        return Err(Error::ValueOutOfRange {
            field: "indptr",
            position: 0.into(),
            value: indptr[0],
            minimum: 0,
            maximum: 0,
        });
    }
    let maximum = n as i64 - 1;
    if let Some(position) = indices
        .iter()
        .position(|&row| row < 0 || i64::from(row) > maximum)
    {
        return Err(Error::ValueOutOfRange {
            field: "indices",
            position: position.into(),
            value: i64::from(indices[position]),
            minimum: 0,
            maximum,
        });
    }
    if indptr[n] != nnz {
        return Err(Error::ValueOutOfRange {
            field: "indptr",
            position: n.into(),
            value: indptr[n],
            minimum: nnz,
            maximum: nnz,
        });
    }
    for column in 0..n {
        let start = indptr[column] as usize;
        let end = indptr[column + 1] as usize;
        if indices[start..end].windows(2).any(|w| w[0] >= w[1]) {
            return Err(Error::KinshipSupportUnsorted {
                column: column.into(),
            });
        }
    }
    // The walk visits upper entries and writes their mirrors, so a lower
    // entry with no upper mirror would never be written; reject it here
    // rather than leave a zero in the output.
    for column in 0..n {
        let start = indptr[column] as usize;
        let end = indptr[column + 1] as usize;
        for &row in &indices[start..end] {
            if row as usize > column
                && mirror(indptr, indices, row as usize, column as i32).is_none()
            {
                return Err(Error::KinshipSupportAsymmetric {
                    row: (row as usize).into(),
                    column: column.into(),
                });
            }
        }
    }
    Ok(())
}

/// The kinship of every entry of a symmetric CSC support, in `data` order.
///
/// Each upper entry (`row < column`) and each diagonal entry is evaluated
/// once through the call's memo; the upper value is also written at its
/// mirror, found by binary search in column `row`, so a lower entry is never
/// walked.  The support is `n` columns of `indptr` (length `n + 1`, ending
/// at `indices.len()`) over strictly increasing `indices` per column, as a
/// canonical CSC matrix has, and every entry must have its mirror in either
/// direction.
///
/// # Errors
///
/// [`Error::LengthMismatch`] and [`Error::ValueOutOfRange`] for a malformed
/// CSC, [`Error::KinshipSupportUnsorted`] for a column out of order,
/// [`Error::KinshipSupportAsymmetric`] for an upper entry with no mirror, and
/// [`Error::AllocationFailed`] for the memo, the stack or the output.
pub fn support_values(
    ped: KinshipPedigree<'_>,
    indptr: &[i64],
    indices: &[i32],
) -> Result<Vec<f32>, Error> {
    check_support(&ped, indptr, indices)?;
    support_with(ped, indptr, indices)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology::structural_depth;

    /// `(mother, father)` per row, `-1` missing.
    fn ped(rows: &[(i32, i32)], twins: &[(i32, i32)]) -> (Vec<i32>, Vec<i32>, Vec<i32>, Vec<i32>) {
        let mother: Vec<i32> = rows.iter().map(|r| r.0).collect();
        let father: Vec<i32> = rows.iter().map(|r| r.1).collect();
        let mut twin = vec![-1; rows.len()];
        for &(a, b) in twins {
            twin[a as usize] = b;
            twin[b as usize] = a;
        }
        let depth = structural_depth(&mother, &father);
        (mother, father, twin, depth)
    }

    fn values(cols: &(Vec<i32>, Vec<i32>, Vec<i32>, Vec<i32>), pairs: &[(i32, i32)]) -> Vec<f32> {
        let ped = KinshipPedigree::try_new(&cols.0, &cols.1, &cols.2, &cols.3).unwrap();
        let first: Vec<i32> = pairs.iter().map(|p| p.0).collect();
        let second: Vec<i32> = pairs.iter().map(|p| p.1).collect();
        pair_kinship(ped, &first, &second).unwrap()
    }

    #[test]
    fn nuclear_family_has_the_textbook_values() {
        // 0, 1 founders; 2, 3 full sibs; 4 child of 2 and an outsider 5.
        let cols = ped(&[(-1, -1), (-1, -1), (0, 1), (0, 1), (2, 5), (-1, -1)], &[]);
        let got = values(
            &cols,
            &[(0, 0), (0, 1), (0, 2), (2, 3), (3, 2), (3, 4), (4, 4)],
        );
        assert_eq!(got, vec![0.5, 0.0, 0.25, 0.25, 0.25, 0.125, 0.5]);
    }

    #[test]
    fn sib_mating_child_is_inbred() {
        // 2 and 3 are full sibs; 4 is their child: F = 0.25, self-kinship 0.625.
        let cols = ped(&[(-1, -1), (-1, -1), (0, 1), (0, 1), (2, 3)], &[]);
        assert_eq!(values(&cols, &[(4, 4), (2, 4)]), vec![0.625, 0.375]);
    }

    #[test]
    fn mz_twins_take_the_self_formula() {
        let cols = ped(&[(-1, -1), (-1, -1), (0, 1), (0, 1)], &[(2, 3)]);
        assert_eq!(values(&cols, &[(2, 3), (3, 2)]), vec![0.5, 0.5]);
    }

    #[test]
    fn support_fills_both_triangles_from_the_upper() {
        let cols = ped(&[(-1, -1), (-1, -1), (0, 1), (0, 1)], &[]);
        let ped = KinshipPedigree::try_new(&cols.0, &cols.1, &cols.2, &cols.3).unwrap();
        // Dense 4x4 in CSC: column c holds rows 0..4.
        let indptr = [0i64, 4, 8, 12, 16];
        let indices: Vec<i32> = (0..4).flat_map(|_| 0..4).collect();
        let data = support_values(ped, &indptr, &indices).unwrap();
        let expect = [
            0.5, 0.0, 0.25, 0.25, //
            0.0, 0.5, 0.25, 0.25, //
            0.25, 0.25, 0.5, 0.25, //
            0.25, 0.25, 0.25, 0.5,
        ];
        assert_eq!(data, expect);
    }

    #[test]
    fn support_without_a_mirror_is_asymmetric() {
        let cols = ped(&[(-1, -1), (-1, -1), (0, 1)], &[]);
        let ped = KinshipPedigree::try_new(&cols.0, &cols.1, &cols.2, &cols.3).unwrap();
        // Column 2 holds row 0, but column 0 holds only its diagonal.
        let indptr = [0i64, 1, 2, 4];
        let indices = [0, 1, 0, 2];
        let err = support_values(ped, &indptr, &indices).unwrap_err();
        assert_eq!(
            err,
            Error::KinshipSupportAsymmetric {
                row: 0.into(),
                column: 2.into()
            }
        );
    }

    #[test]
    fn support_with_a_lower_entry_and_no_upper_mirror_is_asymmetric() {
        let cols = ped(&[(-1, -1), (-1, -1), (0, 1)], &[]);
        let ped = KinshipPedigree::try_new(&cols.0, &cols.1, &cols.2, &cols.3).unwrap();
        // Column 0 holds row 2, but column 2 holds only its diagonal.
        let indptr = [0i64, 2, 3, 4];
        let indices = [0, 2, 1, 2];
        let err = support_values(ped, &indptr, &indices).unwrap_err();
        assert_eq!(
            err,
            Error::KinshipSupportAsymmetric {
                row: 2.into(),
                column: 0.into()
            }
        );
    }

    #[test]
    fn support_whose_indptr_stops_short_of_nnz_is_rejected() {
        let cols = ped(&[(-1, -1), (-1, -1), (0, 1)], &[]);
        let ped = KinshipPedigree::try_new(&cols.0, &cols.1, &cols.2, &cols.3).unwrap();
        let indptr = [0i64, 1, 2, 2];
        let indices = [0, 1, 2];
        let err = support_values(ped, &indptr, &indices).unwrap_err();
        assert!(matches!(
            err,
            Error::ValueOutOfRange {
                field: "indptr",
                position,
                value: 2,
                minimum: 3,
                maximum: 3
            } if position == 3.into()
        ));
    }

    #[test]
    fn support_rejects_an_unsorted_column() {
        let cols = ped(&[(-1, -1), (-1, -1), (0, 1)], &[]);
        let ped = KinshipPedigree::try_new(&cols.0, &cols.1, &cols.2, &cols.3).unwrap();
        let indptr = [0i64, 1, 2, 5];
        let indices = [0, 1, 2, 0, 1];
        let err = support_values(ped, &indptr, &indices).unwrap_err();
        assert_eq!(err, Error::KinshipSupportUnsorted { column: 2.into() });
    }

    const KINSHIP_FAMILIES: [Family; 5] = [
        Family::KinshipMemo,
        Family::KinshipStack,
        Family::KinshipSignatures,
        Family::KinshipColumns,
        Family::KinshipOutput,
    ];

    /// Every kinship family reports `allocation_failed` on both entries and
    /// instead of aborting.  The seam is process-wide, so each
    /// case runs [`seam_child`] in a fresh copy of this test binary.
    #[test]
    fn a_refused_allocation_of_any_kinship_family_is_an_error() {
        let exe = std::env::current_exe().unwrap();
        for family in KINSHIP_FAMILIES {
            for mode in ["pairs", "support"] {
                let out = std::process::Command::new(&exe)
                    .args([
                        "--exact",
                        "kinship::pairwise::tests::seam_child",
                        "--nocapture",
                    ])
                    .env("PG_KINSHIP_SEAM_FAMILY", family.name())
                    .env("PG_KINSHIP_SEAM_MODE", mode)
                    .output()
                    .unwrap();
                assert!(
                    out.status.success(),
                    "{}/{mode}:\n{}",
                    family.name(),
                    String::from_utf8_lossy(&out.stderr)
                );
            }
        }
    }

    /// The body of one seam case; a no-op unless `PG_KINSHIP_SEAM_FAMILY` is set.
    #[test]
    fn seam_child() {
        use crate::alloc::fail_next;
        let Ok(name) = std::env::var("PG_KINSHIP_SEAM_FAMILY") else {
            return;
        };
        let family = Family::parse(&name).unwrap();
        let mode = std::env::var("PG_KINSHIP_SEAM_MODE").unwrap();
        let cols = crate::relationships::testing::random_pedigree(300, 5);
        let depth = structural_depth(&cols.mother, &cols.father);
        let ped = KinshipPedigree::try_new(&cols.mother, &cols.father, &cols.twin, &depth).unwrap();
        let rows: Vec<i32> = (0..300).collect();
        let indptr: Vec<i64> = (0..=300).map(|c| c * 300).collect();
        let indices: Vec<i32> = (0..300).flat_map(|_| 0..300).collect();
        let run = |ped| {
            if mode == "pairs" {
                pair_kinship(ped, &rows, &rows).map(|v| v.len())
            } else {
                support_values(ped, &indptr, &indices).map(|v| v.len())
            }
        };
        fail_next(Some(family));
        match run(ped) {
            Err(Error::AllocationFailed { operation, .. }) => assert_eq!(operation, family.name()),
            other => panic!("{name}/{mode} gave {other:?}"),
        }
        fail_next(None);
        assert!(run(ped).is_ok());
    }

    #[test]
    fn shared_output_converts_in_place() {
        let out = SharedOutput::new(1000).unwrap();
        out.set(7, 0.25);
        let before = out.0.as_ptr() as usize;
        let values = out.into_values();
        assert_eq!(values.as_ptr() as usize, before);
        assert_eq!((values.len(), values[7], values[8]), (1000, 0.25, 0.0));
    }

    #[test]
    fn endpoints_outside_the_pedigree_are_rejected() {
        let cols = ped(&[(-1, -1), (-1, -1), (0, 1)], &[]);
        let ped = KinshipPedigree::try_new(&cols.0, &cols.1, &cols.2, &cols.3).unwrap();
        let err = pair_kinship(ped, &[0, 3], &[1, 1]).unwrap_err();
        assert!(matches!(
            err,
            Error::ValueOutOfRange {
                field: "first",
                position,
                value: 3,
                ..
            } if position == 1.into()
        ));
        let err = pair_kinship(ped, &[0], &[1, 1]).unwrap_err();
        assert!(matches!(
            err,
            Error::LengthMismatch {
                field: "second",
                ..
            }
        ));
    }
}
