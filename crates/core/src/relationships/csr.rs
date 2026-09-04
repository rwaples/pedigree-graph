//! Compressed sparse rows over int32 row coordinates with saturated multiplicity.

use super::multiplicity::Mult;

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
    pub fn from_edges(n: usize, mut edges: Vec<(u32, u32)>) -> Csr {
        edges.sort_unstable();
        let mut indptr = vec![0usize; n + 1];
        let mut indices = Vec::with_capacity(edges.len());
        let mut data = Vec::with_capacity(edges.len());
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
        Csr {
            n,
            indptr,
            indices,
            data,
        }
    }

    pub fn nnz(&self) -> usize {
        self.indices.len()
    }

    #[inline]
    pub fn row(&self, i: usize) -> (&[u32], &[Mult]) {
        let (s, e) = (self.indptr[i], self.indptr[i + 1]);
        (&self.indices[s..e], &self.data[s..e])
    }

    pub fn transpose(&self) -> Csr {
        let n = self.n;
        let mut indptr = vec![0usize; n + 1];
        for &j in &self.indices {
            indptr[j as usize + 1] += 1;
        }
        for i in 0..n {
            indptr[i + 1] += indptr[i];
        }
        let mut next = indptr.clone();
        let mut indices = vec![0u32; self.nnz()];
        let mut data = vec![Mult::ZERO; self.nnz()];
        for i in 0..n {
            let (cols, vals) = self.row(i);
            for (&j, &v) in cols.iter().zip(vals) {
                let p = next[j as usize];
                indices[p] = i as u32;
                data[p] = v;
                next[j as usize] += 1;
            }
        }
        Csr {
            n,
            indptr,
            indices,
            data,
        }
    }
}
