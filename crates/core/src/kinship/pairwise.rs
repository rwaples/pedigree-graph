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
//! them).  LIFO order resolves a node's whole subtree before its compute
//! marker, so every distinct key is computed exactly once even when shared
//! across requested pairs, and endpoint order cannot change a bit.

use super::memo::PairMemo;
use crate::alloc::{self, Family};
use crate::error::Error;
use crate::relationships::{check_column_length, check_row_range};

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

/// Bit 63 marks a compute item; the low 64 bits otherwise pack `lo << 32 | hi`.
const COMPUTE: u64 = 1 << 63;

#[inline]
fn pack(lo: u32, hi: u32) -> u64 {
    (u64::from(lo) << 32) | u64::from(hi)
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

/// One memo, one stack, one pedigree: the state of a single call.
pub struct Walker<'a> {
    ped: KinshipPedigree<'a>,
    memo: PairMemo,
    stack: Vec<u64>,
}

impl<'a> Walker<'a> {
    pub fn new(ped: KinshipPedigree<'a>) -> Result<Self, Error> {
        Ok(Walker {
            ped,
            memo: PairMemo::new(ped.len())?,
            stack: Vec::new(),
        })
    }

    pub fn memo(&self) -> &PairMemo {
        &self.memo
    }

    #[inline]
    fn push(&mut self, item: u64) -> Result<(), Error> {
        alloc::push(&mut self.stack, item, Family::KinshipStack, "uint64")
    }

    /// `(peeled, other)` for a canonical pair: the deeper endpoint, ties to
    /// the greater row.
    #[inline]
    fn peel(&self, lo: u32, hi: u32) -> (usize, usize) {
        let (lo, hi) = (lo as usize, hi as usize);
        if self.ped.depth[lo] > self.ped.depth[hi] {
            (lo, hi)
        } else {
            (hi, lo)
        }
    }

    /// Kinship of rows `a` and `b`, walking whatever the memo lacks.
    pub fn resolve(&mut self, a: i32, b: i32) -> Result<f32, Error> {
        let (rlo, rhi) = canon(a, b);
        if let Some(v) = self.memo.get(rlo, rhi) {
            return Ok(v);
        }
        self.push(pack(rlo, rhi))?;
        while let Some(item) = self.stack.pop() {
            let lo = ((item & !COMPUTE) >> 32) as u32;
            let hi = item as u32;
            let (peeled, other) = self.peel(lo, hi);
            let m = self.ped.mother[peeled];
            let f = self.ped.father[peeled];
            let twin = self.ped.twin;
            let self_like =
                other == peeled || twin[other] == peeled as i32 || twin[peeled] == other as i32;
            let other = other as i32;

            if item & COMPUTE == 0 {
                if self.memo.get(lo, hi).is_some() {
                    continue;
                }
                if self_like {
                    if m < 0 || f < 0 {
                        self.memo.insert(lo, hi, 0.5)?;
                    } else {
                        self.push(item | COMPUTE)?;
                        let (dlo, dhi) = canon(m, f);
                        if self.memo.get(dlo, dhi).is_none() {
                            self.push(pack(dlo, dhi))?;
                        }
                    }
                } else if m < 0 && f < 0 {
                    self.memo.insert(lo, hi, 0.0)?;
                } else {
                    self.push(item | COMPUTE)?;
                    for parent in [m, f] {
                        if parent >= 0 {
                            let (dlo, dhi) = canon(parent, other);
                            if self.memo.get(dlo, dhi).is_none() {
                                self.push(pack(dlo, dhi))?;
                            }
                        }
                    }
                }
            } else {
                let value = if self_like {
                    let (dlo, dhi) = canon(m, f);
                    let v0 = self.memo.get(dlo, dhi).expect("dependency resolved");
                    0.5f32 * (1.0f32 + v0)
                } else {
                    let dep = |memo: &PairMemo, parent: i32| -> f32 {
                        if parent < 0 {
                            0.0
                        } else {
                            let (dlo, dhi) = canon(parent, other);
                            memo.get(dlo, dhi).expect("dependency resolved")
                        }
                    };
                    let v0 = dep(&self.memo, m);
                    let v1 = dep(&self.memo, f);
                    0.5f32 * (v0 + v1)
                };
                self.memo.insert(lo, hi, value)?;
            }
        }
        Ok(self.memo.get(rlo, rhi).expect("root resolved"))
    }
}

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
                position,
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
    let mut out = alloc::with_capacity(first.len(), Family::KinshipOutput, "float32")?;
    let mut walker = Walker::new(ped)?;
    for (&a, &b) in first.iter().zip(second) {
        out.push(walker.resolve(a, b)?);
    }
    Ok(out)
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
    let n = ped.len();
    let mut data = alloc::filled(0.0f32, indices.len(), Family::KinshipOutput, "float32")?;
    let mut walker = Walker::new(ped)?;
    for column in 0..n {
        let start = indptr[column] as usize;
        let end = indptr[column + 1] as usize;
        for position in start..end {
            let row = indices[position];
            if row as usize > column {
                break;
            }
            let value = walker.resolve(row, column as i32)?;
            data[position] = value;
            if row as usize == column {
                continue;
            }
            let mirrored = mirror(indptr, indices, row as usize, column as i32).ok_or(
                Error::KinshipSupportAsymmetric {
                    row: row as usize,
                    column,
                },
            )?;
            data[mirrored] = value;
        }
    }
    Ok(data)
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
            position: position + 1,
            value: indptr[position + 1],
            minimum: indptr[position].max(0),
            maximum: nnz,
        });
    }
    if indptr.first().is_some_and(|&v| v != 0) {
        return Err(Error::ValueOutOfRange {
            field: "indptr",
            position: 0,
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
            position,
            value: i64::from(indices[position]),
            minimum: 0,
            maximum,
        });
    }
    if indptr[n] != nnz {
        return Err(Error::ValueOutOfRange {
            field: "indptr",
            position: n,
            value: indptr[n],
            minimum: nnz,
            maximum: nnz,
        });
    }
    for column in 0..n {
        let start = indptr[column] as usize;
        let end = indptr[column + 1] as usize;
        if indices[start..end].windows(2).any(|w| w[0] >= w[1]) {
            return Err(Error::KinshipSupportUnsorted { column });
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
                    row: row as usize,
                    column,
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
        assert_eq!(err, Error::KinshipSupportAsymmetric { row: 0, column: 2 });
    }

    #[test]
    fn support_with_a_lower_entry_and_no_upper_mirror_is_asymmetric() {
        let cols = ped(&[(-1, -1), (-1, -1), (0, 1)], &[]);
        let ped = KinshipPedigree::try_new(&cols.0, &cols.1, &cols.2, &cols.3).unwrap();
        // Column 0 holds row 2, but column 2 holds only its diagonal.
        let indptr = [0i64, 2, 3, 4];
        let indices = [0, 2, 1, 2];
        let err = support_values(ped, &indptr, &indices).unwrap_err();
        assert_eq!(err, Error::KinshipSupportAsymmetric { row: 2, column: 0 });
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
                position: 3,
                value: 2,
                minimum: 3,
                maximum: 3
            }
        ));
    }

    #[test]
    fn support_rejects_an_unsorted_column() {
        let cols = ped(&[(-1, -1), (-1, -1), (0, 1)], &[]);
        let ped = KinshipPedigree::try_new(&cols.0, &cols.1, &cols.2, &cols.3).unwrap();
        let indptr = [0i64, 1, 2, 5];
        let indices = [0, 1, 2, 0, 1];
        let err = support_values(ped, &indptr, &indices).unwrap_err();
        assert_eq!(err, Error::KinshipSupportUnsorted { column: 2 });
    }

    const KINSHIP_FAMILIES: [Family; 3] = [
        Family::KinshipMemo,
        Family::KinshipStack,
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
    fn endpoints_outside_the_pedigree_are_rejected() {
        let cols = ped(&[(-1, -1), (-1, -1), (0, 1)], &[]);
        let ped = KinshipPedigree::try_new(&cols.0, &cols.1, &cols.2, &cols.3).unwrap();
        let err = pair_kinship(ped, &[0, 3], &[1, 1]).unwrap_err();
        assert!(matches!(
            err,
            Error::ValueOutOfRange {
                field: "first",
                position: 1,
                value: 3,
                ..
            }
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
