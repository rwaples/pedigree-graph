//! Compressed sparse rows over int32 row coordinates with saturated multiplicity.

use super::multiplicity::Mult;
use crate::alloc::{self, Family};
use crate::error::Error;

/// A square sparse matrix stored by rows; column indices are sorted within a row.
#[derive(Clone, Debug)]
pub struct Csr {
    n: usize,
    indptr: Vec<usize>,
    indices: Vec<u32>,
    data: Vec<Mult>,
}

impl Csr {
    /// Build from `(row, col)` edges; repeated edges add their multiplicity.
    pub fn from_edges(n: usize, mut edges: Vec<(u32, u32)>) -> Result<Csr, Error> {
        edges.sort_unstable();
        let mut indptr = alloc::filled(0usize, n + 1, Family::Csr, "intp")?;
        let mut indices = alloc::with_capacity(edges.len(), Family::Csr, "int32")?;
        let mut data = alloc::with_capacity(edges.len(), Family::Csr, "uint8")?;
        let mut k = 0;
        while k < edges.len() {
            let (r, c) = edges[k];
            let mut count = 0u64;
            while k < edges.len() && edges[k] == (r, c) {
                count += 1;
                k += 1;
            }
            indptr[r as usize + 1] += 1;
            indices.push(c);
            data.push(Mult::from_count(count));
        }
        for i in 0..n {
            indptr[i + 1] += indptr[i];
        }
        Ok(Csr {
            n,
            indptr,
            indices,
            data,
        })
    }

    #[inline]
    pub fn row(&self, i: usize) -> (&[u32], &[Mult]) {
        let (s, e) = (self.indptr[i], self.indptr[i + 1]);
        (&self.indices[s..e], &self.data[s..e])
    }

    pub fn transpose(&self) -> Result<Csr, Error> {
        let n = self.n;
        let mut indptr = alloc::filled(0usize, n + 1, Family::Csr, "intp")?;
        for &j in &self.indices {
            indptr[j as usize + 1] += 1;
        }
        for i in 0..n {
            indptr[i + 1] += indptr[i];
        }
        let mut next = alloc::cloned(&indptr, Family::Csr, "intp")?;
        let mut indices = alloc::filled(0u32, self.indices.len(), Family::Csr, "int32")?;
        let mut data = alloc::filled(Mult::ZERO, self.indices.len(), Family::Csr, "uint8")?;
        for i in 0..n {
            let (cols, vals) = self.row(i);
            for (&j, &v) in cols.iter().zip(vals) {
                let p = next[j as usize];
                indices[p] = i as u32;
                data[p] = v;
                next[j as usize] += 1;
            }
        }
        Ok(Csr {
            n,
            indptr,
            indices,
            data,
        })
    }
}
