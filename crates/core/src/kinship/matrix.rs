//! The depth-major kinship DP behind the three matrix products (ADR 0009).
//!
//! One kernel, three sinks.  In stable depth-major order every row `j` is
//! built from its parents' finished rows by one merge walk, `phi(j, k) =
//! (phi(m, k) + phi(f, k)) / 2` for every relative `k` in either parent row,
//! then its diagonal `(1 + phi(m, f)) / 2`; an MZ pass writes `phi(j, twin)
//! = phi(j, j)` into both rows at every depth, depth 0 included.  Each step
//! is the correctly rounded float32 of a half-sum of two float32 operands,
//! so every entry is the bit [`super::pair_kinship`] returns for the same
//! pair: in depth-major order "deeper endpoint, ties to the greater row" is
//! "greater row", the rule the pairwise walk peels by.
//!
//! * [`kinship_csc`] keeps every row and assembles the complete symmetric
//!   CSC in graph rows.
//! * [`approximate_kinship_csc`] runs the DP twice: a thresholded pass whose
//!   rows are harvested as a one-sided candidate support (each pair once,
//!   under its later endpoint) and then retired, and a complete retiring
//!   pass that captures the exact value of every candidate as its row
//!   finishes.  The support is the 0.7.1 propagated candidate set; the
//!   values are the pinned recurrence.
//! * [`generation_kinship_sums`] retires rows as it goes and accumulates the
//!   within-bucket kinship sum inline, in the order the merge walk emits.
//!
//! Retirement: a row is freed at the end of the depth of its last direct
//! child, after which no merge walk reads it; a later symmetric write to it
//! dissolves in the row store.  Assembly walks graph rows in order and
//! translates stored columns through the permutation, so every output
//! column's rows come out ascending without a sort.

use super::depth_order::DepthOrder;
use super::pairwise::KinshipPedigree;
use super::rows::{Owned, RowStore};
use crate::alloc::{self, Family};
use crate::error::Error;

/// The three arrays of a symmetric CSC matrix in graph rows: `indptr` of
/// `n + 1` int32, `indices` and float32 `data` of `nnz`, rows ascending
/// within each column, the diagonal always present.
#[derive(Debug, Clone, PartialEq)]
pub struct Csc {
    pub indptr: Vec<i32>,
    pub indices: Vec<i32>,
    pub data: Vec<f32>,
}

const SCRATCH: Family = Family::KinshipScratch;

/// The pedigree gathered into stable depth-major space.
struct Topo {
    n: usize,
    /// Graph row at each depth-major position.
    order: Vec<u32>,
    /// Depth-major position of each graph row.
    inverse: Vec<u32>,
    mother: Vec<i32>,
    father: Vec<i32>,
    twin: Vec<i32>,
    /// Rows at depth `d` are `starts[d]..starts[d + 1]`.
    starts: Vec<usize>,
}

impl Topo {
    fn build(ped: &KinshipPedigree<'_>) -> Result<Topo, Error> {
        let n = ped.len();
        let DepthOrder { order, starts } =
            DepthOrder::build(ped.mother(), ped.father(), ped.depth(), SCRATCH)?;
        let mut inverse = alloc::filled(0u32, n, SCRATCH, "uint32")?;
        for (position, &row) in order.iter().enumerate() {
            inverse[row as usize] = position as u32;
        }
        let gather = |rows: &[i32]| -> Result<Vec<i32>, Error> {
            alloc::collect(
                order.iter().map(|&r| {
                    let p = rows[r as usize];
                    if p < 0 {
                        -1
                    } else {
                        inverse[p as usize] as i32
                    }
                }),
                SCRATCH,
                "int32",
            )
        };
        let mother = gather(ped.mother())?;
        let father = gather(ped.father())?;
        let twin = gather(ped.twin())?;
        Ok(Topo {
            n,
            order,
            inverse,
            mother,
            father,
            twin,
            starts,
        })
    }

    fn max_depth(&self) -> usize {
        self.starts.len() - 2
    }

    fn rows_at(&self, d: usize) -> std::ops::Range<usize> {
        self.starts[d]..self.starts[d + 1]
    }

    /// Rows bucketed by the depth of their last direct child, the depth
    /// after which no merge walk reads them.
    fn retirement(&self) -> Result<(Vec<usize>, Vec<u32>), Error> {
        let n = self.n;
        let mut last = alloc::filled(0i32, n, SCRATCH, "int32")?;
        for d in 0..=self.max_depth() {
            for j in self.rows_at(d) {
                last[j] = d as i32;
            }
        }
        for j in 0..n {
            let d = last[j];
            for p in [self.mother[j], self.father[j]] {
                if p >= 0 && d > last[p as usize] {
                    last[p as usize] = d;
                }
            }
        }
        let mut starts = alloc::filled(0usize, self.max_depth() + 2, SCRATCH, "uint64")?;
        for &d in &last {
            starts[d as usize + 1] += 1;
        }
        for d in 1..starts.len() {
            starts[d] += starts[d - 1];
        }
        let mut cursor = starts.clone();
        let mut rows = alloc::filled(0u32, n, SCRATCH, "uint32")?;
        for (j, &d) in last.iter().enumerate() {
            rows[cursor[d as usize]] = j as u32;
            cursor[d as usize] += 1;
        }
        Ok((starts, rows))
    }
}

/// What a DP run produces beyond its rows.
enum Sink {
    /// Keep every row for assembly.
    Rows,
    /// Harvest each finished row's columns `k <= j` as the candidate support.
    Harvest { indptr: Vec<usize>, cols: Vec<u32> },
    /// Write the value of every candidate as its row finishes.
    Capture {
        indptr: Vec<usize>,
        cols: Vec<u32>,
        vals: Vec<f32>,
    },
    /// Accumulate within-bucket sums inline during the merge walk.
    Sums { labels: Vec<i32>, sums: Vec<f64> },
}

struct Dp<'t, S: RowStore> {
    topo: &'t Topo,
    store: S,
    threshold: f64,
    /// `(starts, rows)` of rows to retire at the end of each depth, when
    /// retiring.
    retirement: Option<(Vec<usize>, Vec<u32>)>,
    sink: Sink,
    scratch: Vec<(u32, f32)>,
}

impl<'t, S: RowStore> Dp<'t, S> {
    fn new(topo: &'t Topo, threshold: f64, retire: bool, sink: Sink) -> Result<Self, Error> {
        Ok(Dp {
            topo,
            store: S::new(topo.n)?,
            threshold,
            retirement: if retire {
                Some(topo.retirement()?)
            } else {
                None
            },
            sink,
            scratch: Vec::new(),
        })
    }

    fn run(&mut self) -> Result<(), Error> {
        let topo = self.topo;
        for d in 0..=topo.max_depth() {
            for j in topo.rows_at(d) {
                self.process_row(j)?;
            }
            self.mz_pass(d)?;
            self.depth_done(d)?;
        }
        Ok(())
    }

    /// The merge walk of one row over its parents' finished rows, then its
    /// diagonal.  The emitted relatives are staged so the parent rows are
    /// read as they stood when the row started, then written in emission
    /// order into both rows of each pair.  A parentless row is its
    /// diagonal alone, whatever depth the caller gave it.
    fn process_row(&mut self, j: usize) -> Result<(), Error> {
        let topo = self.topo;
        let m = topo.mother[j];
        let f = topo.father[j];
        if m < 0 && f < 0 {
            return self.store.push(j, j as u32, 0.5);
        }
        let jc = j as u32;
        let (mc, mv) = if m >= 0 {
            (self.store.cols(m as usize), self.store.vals(m as usize))
        } else {
            (&[][..], &[][..])
        };
        let (fc, fv) = if f >= 0 {
            (self.store.cols(f as usize), self.store.vals(f as usize))
        } else {
            (&[][..], &[][..])
        };
        let mut km_f = 0.0f32;
        if m >= 0 && f >= 0 {
            if let Ok(pos) = mc.binary_search(&(f as u32)) {
                km_f = mv[pos];
            }
        }

        self.scratch.clear();
        let (mut pm, mut pf) = (0usize, 0usize);
        while pm < mc.len() || pf < fc.len() {
            let (k, a, b) = if pm < mc.len() && (pf == fc.len() || mc[pm] <= fc[pf]) {
                if pf < fc.len() && fc[pf] == mc[pm] {
                    let out = (mc[pm], mv[pm], fv[pf]);
                    pm += 1;
                    pf += 1;
                    out
                } else {
                    let out = (mc[pm], mv[pm], 0.0f32);
                    pm += 1;
                    out
                }
            } else {
                let out = (fc[pf], 0.0f32, fv[pf]);
                pf += 1;
                out
            };
            if k == jc {
                continue;
            }
            let val = 0.5f32 * (a + b);
            if f64::from(val) <= self.threshold {
                continue;
            }
            alloc::push(&mut self.scratch, (k, val), SCRATCH, "uint64")?;
        }

        if let Sink::Sums { labels, sums } = &mut self.sink {
            let g = labels[j];
            let twin = topo.twin[j];
            for &(k, val) in &self.scratch {
                if labels[k as usize] == g && k as i32 != twin && k < jc {
                    sums[g as usize] += f64::from(val);
                }
            }
        }
        for i in 0..self.scratch.len() {
            let (k, val) = self.scratch[i];
            self.store.push(j, k, val)?;
            self.store.push(k as usize, jc, val)?;
        }
        let self_kin = 0.5f32 * (1.0f32 + km_f);
        self.store.push(j, jc, self_kin)
    }

    /// `phi(j, twin) = phi(j, j)` into both rows of every MZ pair at depth
    /// `d`, overwriting the sib value the merge walk emitted for the pair.
    fn mz_pass(&mut self, d: usize) -> Result<(), Error> {
        let topo = self.topo;
        for j in topo.rows_at(d) {
            let tw = topo.twin[j];
            if tw < 0 || tw as usize == j {
                continue;
            }
            let tw = tw as usize;
            let self_k = self
                .store
                .cols(j)
                .binary_search(&(j as u32))
                .map_or(0.5, |pos| self.store.vals(j)[pos]);
            self.write_sorted(j, tw as u32, self_k)?;
            if !self.store.is_retired(tw) {
                self.write_sorted(tw, j as u32, self_k)?;
            }
        }
        Ok(())
    }

    fn write_sorted(&mut self, row: usize, col: u32, val: f32) -> Result<(), Error> {
        if self.store.is_retired(row) {
            return Ok(());
        }
        match self.store.cols(row).binary_search(&col) {
            Ok(pos) => self.store.row_mut(row).1[pos] = val,
            Err(pos) => {
                self.store.push(row, col, val)?;
                let (cols, vals) = self.store.row_mut(row);
                cols[pos..].rotate_right(1);
                vals[pos..].rotate_right(1);
            }
        }
        Ok(())
    }

    /// After the MZ pass and before retirement: every row at depth `d` is
    /// complete for every column `k <= j`.
    fn depth_done(&mut self, d: usize) -> Result<(), Error> {
        let topo = self.topo;
        match &mut self.sink {
            Sink::Rows | Sink::Sums { .. } => {}
            Sink::Harvest { indptr, cols } => {
                for j in topo.rows_at(d) {
                    let row = self.store.cols(j);
                    let upper = row.partition_point(|&k| k <= j as u32);
                    alloc::extend(
                        cols,
                        row[..upper].iter().copied(),
                        Family::KinshipRows,
                        "uint32",
                    )?;
                    indptr[j + 1] = cols.len();
                }
            }
            Sink::Capture { indptr, cols, vals } => {
                for j in topo.rows_at(d) {
                    let (rc, rv) = (self.store.cols(j), self.store.vals(j));
                    let (mut p, mut q) = (indptr[j], 0usize);
                    let end = indptr[j + 1];
                    while p < end && q < rc.len() {
                        let want = cols[p];
                        if rc[q] < want {
                            q += 1;
                        } else if want < rc[q] {
                            p += 1;
                        } else {
                            vals[p] = rv[q];
                            p += 1;
                            q += 1;
                        }
                    }
                }
            }
        }
        if let Some((starts, rows)) = &self.retirement {
            for &row in &rows[starts[d]..starts[d + 1]] {
                self.store.retire(row as usize);
            }
        }
        Ok(())
    }
}

/// The most entries a CSC may hold: its `indptr` is int32.
pub const MAX_CSC_NNZ: usize = i32::MAX as usize;

/// The int32 `indptr` of per-column counts, refusing before any index array
/// could be sized past `max_nnz` (at most [`MAX_CSC_NNZ`]).
fn checked_indptr(counts: &[usize], max_nnz: usize) -> Result<Vec<i32>, Error> {
    let nnz: u64 = counts.iter().map(|&c| c as u64).sum();
    let max_nnz = max_nnz.min(MAX_CSC_NNZ);
    if nnz > max_nnz as u64 {
        return Err(Error::CscIndexOverflow {
            nnz,
            maximum: max_nnz as i64,
        });
    }
    let mut indptr = alloc::filled(0i32, counts.len() + 1, Family::KinshipCsc, "int32")?;
    let mut total = 0i32;
    for (column, &count) in counts.iter().enumerate() {
        total += count as i32;
        indptr[column + 1] = total;
    }
    Ok(indptr)
}

/// Which entries of the symmetric matrix a CSC keeps.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Triangle {
    /// Both triangles and the diagonal.
    Full,
    /// Row `<=` column in graph rows: the upper triangle and the diagonal.
    Upper,
}

impl Triangle {
    #[inline]
    fn keeps(self, row: usize, column: usize) -> bool {
        self == Triangle::Full || row <= column
    }
}

/// Symmetric row storage in depth-major space, assembled into a CSC in
/// graph rows.  Graph rows are visited in order and each stored column is
/// translated through `order`, so rows within every output column ascend.
fn assemble<'a>(
    topo: &Topo,
    row: impl Fn(usize) -> (&'a [u32], &'a [f32]),
    triangle: Triangle,
    max_nnz: usize,
) -> Result<Csc, Error> {
    let n = topo.n;
    let mut counts = alloc::filled(0usize, n, Family::KinshipCsc, "uint64")?;
    for r in 0..n {
        let graph_row = topo.order[r] as usize;
        for &c in row(r).0 {
            let column = topo.order[c as usize] as usize;
            if triangle.keeps(graph_row, column) {
                counts[column] += 1;
            }
        }
    }
    let indptr = checked_indptr(&counts, max_nnz)?;
    let nnz = indptr[n] as usize;
    let mut cursor = counts;
    for (column, slot) in cursor.iter_mut().enumerate() {
        *slot = indptr[column] as usize;
    }
    let mut indices = alloc::filled(0i32, nnz, Family::KinshipCsc, "int32")?;
    let mut data = alloc::filled(0.0f32, nnz, Family::KinshipCsc, "float32")?;
    for i in 0..n {
        let (cols, vals) = row(topo.inverse[i] as usize);
        for (&c, &v) in cols.iter().zip(vals) {
            let column = topo.order[c as usize] as usize;
            if !triangle.keeps(i, column) {
                continue;
            }
            let pos = cursor[column];
            cursor[column] += 1;
            indices[pos] = i as i32;
            data[pos] = v;
        }
    }
    Ok(Csc {
        indptr,
        indices,
        data,
    })
}

fn complete_with<S: RowStore>(
    topo: &Topo,
    triangle: Triangle,
    max_nnz: usize,
) -> Result<Csc, Error> {
    let mut dp = Dp::<S>::new(topo, 0.0, false, Sink::Rows)?;
    dp.run()?;
    let store = dp.store;
    assemble(topo, |r| (store.cols(r), store.vals(r)), triangle, max_nnz)
}

fn approximate_with<S: RowStore>(topo: &Topo, threshold: f64) -> Result<Csc, Error> {
    let n = topo.n;
    let mut pass1 = Dp::<S>::new(
        topo,
        threshold,
        true,
        Sink::Harvest {
            indptr: alloc::filled(0usize, n + 1, Family::KinshipRows, "uint64")?,
            cols: Vec::new(),
        },
    )?;
    pass1.run()?;
    let Sink::Harvest { indptr, cols } = pass1.sink else {
        unreachable!()
    };
    drop(pass1.store);

    let upper = cols.len();
    let mut pass2 = Dp::<S>::new(
        topo,
        0.0,
        true,
        Sink::Capture {
            indptr,
            cols,
            vals: alloc::filled(f32::NAN, upper, Family::KinshipRows, "float32")?,
        },
    )?;
    pass2.run()?;
    let Sink::Capture { indptr, cols, vals } = pass2.sink else {
        unreachable!()
    };
    drop(pass2.store);
    assert!(
        vals.iter().all(|v| !v.is_nan()),
        "the complete pass captured every propagated candidate"
    );

    // Mirror the one-sided support into symmetric rows, each row's columns
    // in the order they arrive: its own candidates ascending, then the
    // deeper rows that hold it, ascending.
    let mut counts = alloc::filled(0usize, n, Family::KinshipRows, "uint64")?;
    for j in 0..n {
        for &k in &cols[indptr[j]..indptr[j + 1]] {
            counts[j] += 1;
            if k as usize != j {
                counts[k as usize] += 1;
            }
        }
    }
    let mut sym_indptr = alloc::filled(0usize, n + 1, Family::KinshipRows, "uint64")?;
    for j in 0..n {
        sym_indptr[j + 1] = sym_indptr[j] + counts[j];
    }
    let total = sym_indptr[n];
    let mut cursor = counts;
    cursor.copy_from_slice(&sym_indptr[..n]);
    let mut sym_cols = alloc::filled(0u32, total, Family::KinshipRows, "uint32")?;
    let mut sym_vals = alloc::filled(0.0f32, total, Family::KinshipRows, "float32")?;
    for j in 0..n {
        for p in indptr[j]..indptr[j + 1] {
            let (k, v) = (cols[p], vals[p]);
            let pos = cursor[j];
            cursor[j] += 1;
            sym_cols[pos] = k;
            sym_vals[pos] = v;
            if k as usize != j {
                let pos = cursor[k as usize];
                cursor[k as usize] += 1;
                sym_cols[pos] = j as u32;
                sym_vals[pos] = v;
            }
        }
    }
    drop((indptr, cols, vals, cursor));
    assemble(
        topo,
        |r| {
            (
                &sym_cols[sym_indptr[r]..sym_indptr[r + 1]],
                &sym_vals[sym_indptr[r]..sym_indptr[r + 1]],
            )
        },
        Triangle::Full,
        MAX_CSC_NNZ,
    )
}

fn sums_with<S: RowStore>(
    topo: &Topo,
    labels: &[i32],
    n_buckets: usize,
) -> Result<Vec<f64>, Error> {
    let gathered = alloc::collect(
        topo.order.iter().map(|&r| labels[r as usize]),
        SCRATCH,
        "int32",
    )?;
    let mut dp = Dp::<S>::new(
        topo,
        0.0,
        true,
        Sink::Sums {
            labels: gathered,
            sums: alloc::filled(0.0f64, n_buckets, Family::KinshipSums, "float64")?,
        },
    )?;
    dp.run()?;
    let Sink::Sums { sums, .. } = dp.sink else {
        unreachable!()
    };
    Ok(sums)
}

/// The complete kinship matrix: every nonzero pedigree kinship plus the
/// diagonal, as a symmetric CSC in graph rows.
///
/// # Errors
///
/// [`Error::ValueOutOfRange`] on `depth` when a row's depth is negative or
/// not above both parents', [`Error::CscIndexOverflow`] when the entry count
/// exceeds int32, and [`Error::AllocationFailed`] for any buffer.
pub fn kinship_csc(ped: KinshipPedigree<'_>) -> Result<Csc, Error> {
    let topo = Topo::build(&ped)?;
    complete_with::<Owned>(&topo, Triangle::Full, MAX_CSC_NNZ)
}

/// The upper triangle of [`kinship_csc`]: the entries with row `<=` column
/// in graph rows, laid out as it lays out every column, for hosts whose
/// sparse matrix stores one triangle of a symmetric matrix (R's
/// `dsCMatrix`).  The int32 cap applies to the upper entries, so about twice
/// the pairs fit, and the output arrays are about half the size; the DP's
/// row store is the full symmetric matrix either way.  `max_nnz` lowers the
/// cap for tests; hosts pass [`MAX_CSC_NNZ`].
///
/// # Errors
///
/// As [`kinship_csc`], with [`Error::CscIndexOverflow`] above `max_nnz`
/// upper entries.
pub fn kinship_csc_upper(ped: KinshipPedigree<'_>, max_nnz: usize) -> Result<Csc, Error> {
    let topo = Topo::build(&ped)?;
    complete_with::<Owned>(&topo, Triangle::Upper, max_nnz)
}

/// Exact values on the propagation-pruned support: the structure a DP that
/// drops every intermediate `value <= threshold` leaves, every retained
/// entry then recomputed by the complete recurrence.  `threshold` is
/// compared in float64 against the float32 value.
///
/// # Errors
///
/// As [`kinship_csc`], plus [`Error::KinshipThresholdOutOfRange`] for a
/// threshold that is not finite or not in `[0, 1]`.
pub fn approximate_kinship_csc(ped: KinshipPedigree<'_>, threshold: f64) -> Result<Csc, Error> {
    if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
        return Err(Error::KinshipThresholdOutOfRange {
            value: threshold.to_string(),
        });
    }
    let topo = Topo::build(&ped)?;
    approximate_with::<Owned>(&topo, threshold)
}

/// Per bucket, the kinship summed over unordered same-bucket pairs of
/// distinct rows that are not MZ co-twins, as float64 in the order the DP
/// emits them.  `labels` gives each graph row's bucket in `0..n_buckets`.
///
/// # Errors
///
/// As [`kinship_csc`], plus [`Error::LengthMismatch`] when `labels` is not
/// one per row and [`Error::ValueOutOfRange`] on `labels` or `n_buckets`.
pub fn generation_kinship_sums(
    ped: KinshipPedigree<'_>,
    labels: &[i32],
    n_buckets: usize,
) -> Result<Vec<f64>, Error> {
    crate::relationships::check_column_length("labels", labels.len(), ped.len())?;
    if n_buckets == 0 {
        return Err(Error::ValueOutOfRange {
            field: "n_buckets",
            position: 0,
            value: 0,
            minimum: 1,
            maximum: i64::from(i32::MAX),
        });
    }
    if let Some(position) = labels
        .iter()
        .position(|&g| g < 0 || g as usize >= n_buckets)
    {
        return Err(Error::ValueOutOfRange {
            field: "labels",
            position,
            value: i64::from(labels[position]),
            minimum: 0,
            maximum: n_buckets as i64 - 1,
        });
    }
    let topo = Topo::build(&ped)?;
    sums_with::<Owned>(&topo, labels, n_buckets)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kinship::pair_kinship;
    use crate::topology::structural_depth;

    struct Cols {
        mother: Vec<i32>,
        father: Vec<i32>,
        twin: Vec<i32>,
        depth: Vec<i32>,
    }

    fn cols(rows: &[(i32, i32)], twins: &[(i32, i32)]) -> Cols {
        let mother: Vec<i32> = rows.iter().map(|r| r.0).collect();
        let father: Vec<i32> = rows.iter().map(|r| r.1).collect();
        let mut twin = vec![-1; rows.len()];
        for &(a, b) in twins {
            twin[a as usize] = b;
            twin[b as usize] = a;
        }
        let depth = structural_depth(&mother, &father);
        Cols {
            mother,
            father,
            twin,
            depth,
        }
    }

    impl Cols {
        fn ped(&self) -> KinshipPedigree<'_> {
            KinshipPedigree::try_new(&self.mother, &self.father, &self.twin, &self.depth).unwrap()
        }
    }

    fn dense(csc: &Csc, n: usize) -> Vec<Vec<f32>> {
        let mut out = vec![vec![0.0f32; n]; n];
        for (column, window) in csc.indptr.windows(2).enumerate() {
            let (start, end) = (window[0] as usize, window[1] as usize);
            let rows = &csc.indices[start..end];
            assert!(
                rows.windows(2).all(|w| w[0] < w[1]),
                "column {column} sorted"
            );
            for (r, v) in rows.iter().zip(&csc.data[start..end]) {
                out[*r as usize][column] = *v;
            }
        }
        out
    }

    fn pairwise(c: &Cols) -> Vec<Vec<f32>> {
        let n = c.mother.len();
        (0..n)
            .map(|a| {
                (0..n)
                    .map(|b| pair_kinship(c.ped(), &[a as i32], &[b as i32]).unwrap()[0])
                    .collect()
            })
            .collect()
    }

    /// Founders 0..4; 4, 5 full sibs of (0, 1); 6 child of 4 and 2; 7 child of
    /// 6 and 5 (inbred); 8, 9 MZ twins of (3, 5); 10 child of 9 and 7.
    fn family() -> Cols {
        cols(
            &[
                (-1, -1),
                (-1, -1),
                (-1, -1),
                (-1, -1),
                (0, 1),
                (0, 1),
                (4, 2),
                (6, 5),
                (3, 5),
                (3, 5),
                (9, 7),
            ],
            &[(8, 9)],
        )
    }

    /// The same family with its rows shuffled so the permutation is exercised.
    fn shuffled_family() -> (Cols, Vec<usize>) {
        let base = family();
        let n = base.mother.len();
        let perm: Vec<usize> = vec![10, 3, 7, 0, 8, 5, 1, 9, 4, 2, 6];
        let mut inverse = vec![0usize; n];
        for (new, &old) in perm.iter().enumerate() {
            inverse[old] = new;
        }
        let map = |p: i32| {
            if p < 0 {
                -1
            } else {
                inverse[p as usize] as i32
            }
        };
        let mother: Vec<i32> = perm.iter().map(|&old| map(base.mother[old])).collect();
        let father: Vec<i32> = perm.iter().map(|&old| map(base.father[old])).collect();
        let twin: Vec<i32> = perm.iter().map(|&old| map(base.twin[old])).collect();
        let depth = structural_depth(&mother, &father);
        (
            Cols {
                mother,
                father,
                twin,
                depth,
            },
            perm,
        )
    }

    #[test]
    fn the_complete_matrix_is_the_pairwise_recurrence() {
        for c in [family(), shuffled_family().0] {
            let want = pairwise(&c);
            let csc = kinship_csc(c.ped()).unwrap();
            assert_eq!(dense(&csc, c.mother.len()), want);
            assert_eq!(csc.data.len(), csc.indices.len());
        }
    }

    #[test]
    fn a_permuted_graph_gives_the_permuted_matrix() {
        let base = family();
        let (shuffled, perm) = shuffled_family();
        let a = dense(&kinship_csc(base.ped()).unwrap(), 11);
        let b = dense(&kinship_csc(shuffled.ped()).unwrap(), 11);
        for (new_i, &old_i) in perm.iter().enumerate() {
            for (new_j, &old_j) in perm.iter().enumerate() {
                assert_eq!(b[new_i][new_j], a[old_i][old_j]);
            }
        }
    }

    #[test]
    fn approximate_at_zero_is_the_complete_matrix_and_higher_prunes() {
        for c in [family(), shuffled_family().0] {
            let complete = kinship_csc(c.ped()).unwrap();
            {
                assert_eq!(approximate_kinship_csc(c.ped(), 0.0).unwrap(), complete);
                let pruned = approximate_kinship_csc(c.ped(), 0.2).unwrap();
                assert!(pruned.data.len() < complete.data.len());
                let want = pairwise(&c);
                let got = dense(&pruned, c.mother.len());
                for (r, row) in got.iter().enumerate() {
                    assert_eq!(row[r], want[r][r], "diagonal kept and exact");
                    for (k, &v) in row.iter().enumerate() {
                        if v != 0.0 {
                            assert_eq!(v, want[r][k], "retained values are exact");
                        }
                        assert_eq!(v, got[k][r], "symmetric");
                    }
                }
            }
        }
    }

    #[test]
    fn mz_twins_share_the_self_value_on_every_path() {
        let c = family();
        let csc = kinship_csc(c.ped()).unwrap();
        let m = dense(&csc, 11);
        assert_eq!(m[8][9], m[8][8]);
        assert_eq!(m[9][8], m[9][9]);
        let pruned = dense(&approximate_kinship_csc(c.ped(), 0.4).unwrap(), 11);
        assert_eq!(pruned[8][9], m[8][8], "the MZ edge survives any threshold");
    }

    #[test]
    fn generation_sums_match_the_matrix_walk() {
        for c in [family(), shuffled_family().0] {
            let n = c.mother.len();
            let m = dense(&kinship_csc(c.ped()).unwrap(), n);
            let labels: Vec<i32> = c.depth.clone();
            let buckets = *labels.iter().max().unwrap() as usize + 1;
            let mut want = vec![0.0f64; buckets];
            for i in 0..n {
                for j in (i + 1)..n {
                    if labels[i] == labels[j] && c.twin[i] != j as i32 {
                        want[labels[i] as usize] += f64::from(m[i][j]);
                    }
                }
            }
            let got = generation_kinship_sums(c.ped(), &labels, buckets).unwrap();
            for (g, w) in got.iter().zip(&want) {
                assert!((g - w).abs() <= 1e-12, "{got:?} vs {want:?}");
            }
        }
    }

    #[test]
    fn a_retired_row_stays_retired() {
        // 0, 1 founders; 2 their child (0's last direct child is at depth 1);
        // 3 an outsider; 4 = (2, 3) at depth 2 whose merge walk finds 0 in
        // row 2 and writes (0, 4) symmetrically after row 0 has retired.
        let c = cols(&[(-1, -1), (-1, -1), (0, 1), (-1, -1), (2, 3)], &[]);
        let topo = Topo::build(&c.ped()).unwrap();
        let mut dp = Dp::<Owned>::new(&topo, 0.0, true, Sink::Rows).unwrap();
        dp.run().unwrap();
        assert!(dp.store.is_retired(0));
        assert!(dp.store.cols(0).is_empty());
        let m = dense(&kinship_csc(c.ped()).unwrap(), 5);
        assert_eq!(m[0][4], 0.125, "the pair is held on row 4's side");
        let labels = vec![0, 0, 0, 0, 0];
        let sums = generation_kinship_sums(c.ped(), &labels, 1).unwrap();
        let want: f64 = (0..5)
            .flat_map(|i| (i + 1..5).map(move |j| (i, j)))
            .map(|(i, j)| f64::from(m[i][j]))
            .sum();
        assert!((sums[0] - want).abs() <= 1e-12);
    }

    #[test]
    fn nnz_past_int32_is_an_overflow_error() {
        let counts = [usize::try_from(i32::MAX).unwrap(), 1];
        let err = checked_indptr(&counts, MAX_CSC_NNZ).unwrap_err();
        assert_eq!(
            err,
            Error::CscIndexOverflow {
                nnz: i32::MAX as u64 + 1,
                maximum: i64::from(i32::MAX)
            }
        );
        assert_eq!(err.code(), "csc_index_overflow");
        assert_eq!(err.class(), crate::error::ErrorClass::Resource);
        let names: Vec<&str> = err.fields().iter().map(|(name, _)| *name).collect();
        assert_eq!(names, ["nnz", "maximum"]);
        assert_eq!(
            checked_indptr(&[2, 0, 3], MAX_CSC_NNZ).unwrap(),
            vec![0, 2, 2, 5]
        );
    }

    #[test]
    fn malformed_inputs_are_rejected_before_any_work() {
        let c = cols(&[(-1, -1), (-1, -1), (0, 1)], &[]);
        let ped = c.ped();
        let err = approximate_kinship_csc(ped, 1.5).unwrap_err();
        assert!(matches!(err, Error::KinshipThresholdOutOfRange { .. }));
        assert!(matches!(
            approximate_kinship_csc(ped, f64::NAN).unwrap_err(),
            Error::KinshipThresholdOutOfRange { .. }
        ));
        let err = generation_kinship_sums(ped, &[0, 0], 1).unwrap_err();
        assert!(matches!(
            err,
            Error::LengthMismatch {
                field: "labels",
                ..
            }
        ));
        let err = generation_kinship_sums(ped, &[0, 0, 1], 1).unwrap_err();
        assert!(matches!(
            err,
            Error::ValueOutOfRange {
                field: "labels",
                position: 2,
                ..
            }
        ));
        let err = generation_kinship_sums(ped, &[0, 0, 0], 0).unwrap_err();
        assert!(matches!(
            err,
            Error::ValueOutOfRange {
                field: "n_buckets",
                ..
            }
        ));

        let flat = KinshipPedigree::try_new(&c.mother, &c.father, &c.twin, &[0, 0, 0]).unwrap();
        let err = kinship_csc(flat).unwrap_err();
        assert!(matches!(
            err,
            Error::ValueOutOfRange {
                field: "depth",
                position: 2,
                value: 0,
                minimum: 1,
                ..
            }
        ));
        let negative =
            KinshipPedigree::try_new(&c.mother, &c.father, &c.twin, &[-1, 0, 1]).unwrap();
        assert!(matches!(
            kinship_csc(negative).unwrap_err(),
            Error::ValueOutOfRange {
                field: "depth",
                position: 0,
                ..
            }
        ));
    }

    /// The structural check accepts a parentless row above depth 0; it
    /// must still get its diagonal and reach its descendants.
    #[test]
    fn a_founder_above_depth_zero_keeps_its_diagonal() {
        let c = cols(&[(-1, -1), (-1, -1), (0, 1)], &[]);
        let raised = KinshipPedigree::try_new(&c.mother, &c.father, &c.twin, &[0, 1, 2]).unwrap();
        let want = pairwise(&c);
        assert_eq!(dense(&kinship_csc(raised).unwrap(), 3), want);
        assert_eq!(
            dense(&approximate_kinship_csc(raised, 0.1).unwrap(), 3),
            want
        );
        let sums = generation_kinship_sums(raised, &[0, 0, 0], 1).unwrap();
        assert_eq!(sums, vec![0.5]);
    }

    #[test]
    fn an_empty_pedigree_gives_empty_products() {
        let c = cols(&[], &[]);
        let csc = kinship_csc(c.ped()).unwrap();
        assert_eq!(csc.indptr, vec![0]);
        assert!(csc.indices.is_empty());
        assert_eq!(generation_kinship_sums(c.ped(), &[], 1).unwrap(), vec![0.0]);
    }

    const MATRIX_FAMILIES: [Family; 4] = [
        Family::KinshipRows,
        Family::KinshipCsc,
        Family::KinshipSums,
        Family::KinshipScratch,
    ];

    /// Every matrix family reports `allocation_failed` from the product that
    /// reaches it, in a fresh process because the seam is process-wide.
    #[test]
    fn a_refused_allocation_of_any_matrix_family_is_an_error() {
        let exe = std::env::current_exe().unwrap();
        for family in MATRIX_FAMILIES {
            for product in ["complete", "approximate", "sums"] {
                if family == Family::KinshipCsc && product == "sums"
                    || family == Family::KinshipSums && product != "sums"
                {
                    continue;
                }
                let out = std::process::Command::new(&exe)
                    .args([
                        "--exact",
                        "kinship::matrix::tests::seam_child",
                        "--nocapture",
                    ])
                    .env("PG_MATRIX_SEAM_FAMILY", family.name())
                    .env("PG_MATRIX_SEAM_PRODUCT", product)
                    .output()
                    .unwrap();
                assert!(
                    out.status.success(),
                    "{}/{product}:\n{}",
                    family.name(),
                    String::from_utf8_lossy(&out.stderr)
                );
            }
        }
    }

    /// The body of one seam case; a no-op unless `PG_MATRIX_SEAM_FAMILY` is set.
    #[test]
    fn seam_child() {
        use crate::alloc::fail_next;
        let Ok(name) = std::env::var("PG_MATRIX_SEAM_FAMILY") else {
            return;
        };
        let family = Family::parse(&name).unwrap();
        let product = std::env::var("PG_MATRIX_SEAM_PRODUCT").unwrap();
        let c = crate::relationships::testing::random_pedigree(300, 5);
        let depth = structural_depth(&c.mother, &c.father);
        let ped = KinshipPedigree::try_new(&c.mother, &c.father, &c.twin, &depth).unwrap();
        let labels = vec![0i32; 300];
        let run = |ped| match product.as_str() {
            "complete" => kinship_csc(ped).map(|c| c.data.len()),
            "approximate" => approximate_kinship_csc(ped, 0.01).map(|c| c.data.len()),
            _ => generation_kinship_sums(ped, &labels, 1).map(|s| s.len()),
        };
        fail_next(Some(family));
        match run(ped) {
            Err(Error::AllocationFailed { operation, .. }) => assert_eq!(operation, family.name()),
            other => panic!("{name}/{product} gave {other:?}"),
        }
        fail_next(None);
        assert!(run(ped).is_ok());
    }
}
