//! Throwaway spike: port of pedigree_graph's matrix pair engine (degree 5, min_kinship 0).
//! Input TSV columns: mother father twin orig_mother orig_father (row indices / ids, -1 = unknown).
//! Output: `code\ti\tj` per pair, same orientation as the Python engine.

use std::io::{BufRead, BufWriter, Write};
use std::time::Instant;

type Pairs = Vec<(u32, u32)>;

struct Csr {
    n: usize,
    indptr: Vec<usize>,
    indices: Vec<u32>,
    data: Vec<i32>,
}

impl Csr {
    fn from_edges(n: usize, rows: &[u32], cols: &[u32]) -> Csr {
        let mut trip: Vec<(u32, u32)> = rows.iter().copied().zip(cols.iter().copied()).collect();
        trip.sort_unstable();
        let mut indptr = vec![0usize; n + 1];
        let mut indices = Vec::with_capacity(trip.len());
        let mut data = Vec::with_capacity(trip.len());
        let mut k = 0;
        while k < trip.len() {
            let (r, c) = trip[k];
            let mut cnt = 0;
            while k < trip.len() && trip[k] == (r, c) {
                cnt += 1;
                k += 1;
            }
            indptr[r as usize + 1] += 1;
            indices.push(c);
            data.push(cnt);
        }
        for i in 0..n {
            indptr[i + 1] += indptr[i];
        }
        Csr { n, indptr, indices, data }
    }

    fn identity(n: usize) -> Csr {
        let rows: Vec<u32> = (0..n as u32).collect();
        Csr::from_edges(n, &rows, &rows)
    }

    fn nnz(&self) -> usize {
        self.indices.len()
    }

    fn row(&self, i: usize) -> (&[u32], &[i32]) {
        let (s, e) = (self.indptr[i], self.indptr[i + 1]);
        (&self.indices[s..e], &self.data[s..e])
    }

    fn has(&self, i: usize, j: u32) -> bool {
        self.row(i).0.binary_search(&j).is_ok()
    }

    fn transpose(&self) -> Csr {
        let n = self.n;
        let mut counts = vec![0usize; n + 1];
        for &j in &self.indices {
            counts[j as usize + 1] += 1;
        }
        for i in 0..n {
            counts[i + 1] += counts[i];
        }
        let mut pos = counts.clone();
        let mut indices = vec![0u32; self.nnz()];
        let mut data = vec![0i32; self.nnz()];
        for i in 0..n {
            let (cols, vals) = self.row(i);
            for (&j, &v) in cols.iter().zip(vals) {
                let p = pos[j as usize];
                indices[p] = i as u32;
                data[p] = v;
                pos[j as usize] += 1;
            }
        }
        Csr { n, indptr: counts, indices, data }
    }

    /// Gustavson row-wise product with a dense accumulator; sums path multiplicities.
    fn matmul(&self, other: &Csr) -> Csr {
        let n = self.n;
        let mut indptr = vec![0usize; n + 1];
        let mut indices: Vec<u32> = Vec::new();
        let mut data: Vec<i32> = Vec::new();
        let mut acc = vec![0i32; n];
        let mut marker = vec![u32::MAX; n];
        let mut touched: Vec<u32> = Vec::new();
        for i in 0..n {
            touched.clear();
            let (ks, vs) = self.row(i);
            for (&k, &v) in ks.iter().zip(vs) {
                let (js, ws) = other.row(k as usize);
                for (&j, &w) in js.iter().zip(ws) {
                    if marker[j as usize] != i as u32 {
                        marker[j as usize] = i as u32;
                        touched.push(j);
                    }
                    acc[j as usize] += v * w;
                }
            }
            touched.sort_unstable();
            for &j in &touched {
                indices.push(j);
                data.push(acc[j as usize]);
                acc[j as usize] = 0;
            }
            indptr[i + 1] = indices.len();
        }
        Csr { n, indptr, indices, data }
    }


    fn nonzeros(&self) -> Pairs {
        let mut out = Vec::with_capacity(self.nnz());
        for i in 0..self.n {
            for &j in self.row(i).0 {
                out.push((i as u32, j));
            }
        }
        out
    }
}

fn key(a: u32, b: u32) -> u64 {
    let (lo, hi) = if a < b { (a, b) } else { (b, a) };
    ((lo as u64) << 32) | hi as u64
}

fn canon_dedup(pairs: impl Iterator<Item = (u32, u32)>) -> Pairs {
    let mut v: Pairs = pairs.map(|(a, b)| if a < b { (a, b) } else { (b, a) }).collect();
    v.sort_unstable();
    v.dedup();
    v
}

/// Sorted canonical keys of a pair list; computed once per code and reused by every subtract.
fn keys(p: &Pairs) -> Vec<u64> {
    let mut k: Vec<u64> = p.iter().map(|&(a, b)| key(a, b)).collect();
    k.sort_unstable();
    k
}

/// `pairs` must be canonical (lo < hi); each remove list is a sorted key vec.
fn subtract(mut pairs: Pairs, remove: &[&Vec<u64>]) -> Pairs {
    pairs.sort_unstable();
    let cand: Vec<u64> = pairs.iter().map(|&(a, b)| key(a, b)).collect();
    let mut hit = vec![false; cand.len()];
    for rm in remove {
        let (mut i, mut j) = (0, 0);
        while i < cand.len() && j < rm.len() {
            if cand[i] < rm[j] {
                i += 1;
            } else if cand[i] > rm[j] {
                j += 1;
            } else {
                hit[i] = true;
                i += 1;
            }
        }
    }
    pairs.into_iter().zip(hit).filter(|(_, h)| !h).map(|(p, _)| p).collect()
}

/// Off-diagonal nonzeros, canonicalised, minus closer relationships.
fn extract_from_sparse(m: &Csr, remove: &[&Vec<u64>]) -> Pairs {
    let pairs = canon_dedup(m.nonzeros().into_iter().filter(|&(i, j)| i != j));
    subtract(pairs, remove)
}

/// All (lo, hi) pairs of `indices` sharing a group key; a pair sharing several keys is repeated.
fn pairs_from_groups(indices: &[u32], keys: &[i64]) -> Pairs {
    let mut order: Vec<usize> = (0..indices.len()).collect();
    order.sort_by_key(|&k| keys[k]);
    let mut out = Vec::new();
    let mut s = 0;
    while s < order.len() {
        let mut e = s + 1;
        while e < order.len() && keys[order[e]] == keys[order[s]] {
            e += 1;
        }
        for a in s..e {
            for b in a + 1..e {
                let (x, y) = (indices[order[a]], indices[order[b]]);
                out.push(if x < y { (x, y) } else { (y, x) });
            }
        }
        s = e;
    }
    out
}

fn lineal(ak: &Csr) -> Pairs {
    ak.nonzeros()
}

fn symmetric_from_pairs(n: usize, p: &Pairs) -> Csr {
    let rows: Vec<u32> = p.iter().map(|x| x.0).chain(p.iter().map(|x| x.1)).collect();
    let cols: Vec<u32> = p.iter().map(|x| x.1).chain(p.iter().map(|x| x.0)).collect();
    Csr::from_edges(n, &rows, &cols)
}

struct Ped {
    n: usize,
    mother: Vec<i32>,
    father: Vec<i32>,
    twin: Vec<i32>,
    orig_mother: Vec<i64>,
    orig_father: Vec<i64>,
}

fn read_ped(path: &str) -> Ped {
    let f = std::fs::File::open(path).expect("open input");
    let mut p = Ped { n: 0, mother: vec![], father: vec![], twin: vec![], orig_mother: vec![], orig_father: vec![] };
    for (ln, line) in std::io::BufReader::new(f).lines().enumerate() {
        let line = line.unwrap();
        if ln == 0 {
            continue;
        }
        let mut it = line.split('\t').map(|s| s.parse::<i64>().unwrap());
        p.mother.push(it.next().unwrap() as i32);
        p.father.push(it.next().unwrap() as i32);
        p.twin.push(it.next().unwrap() as i32);
        p.orig_mother.push(it.next().unwrap());
        p.orig_father.push(it.next().unwrap());
    }
    p.n = p.mother.len();
    p
}

fn sibling_pairs(p: &Ped) -> (Pairs, Pairs, Pairs) {
    let nt: Vec<u32> = (0..p.n)
        .filter(|&i| (p.orig_mother[i] >= 0 || p.orig_father[i] >= 0) && p.twin[i] < 0)
        .map(|i| i as u32)
        .collect();
    let bk: Vec<u32> = nt.iter().copied().filter(|&i| p.orig_mother[i as usize] >= 0 && p.orig_father[i as usize] >= 0).collect();
    let max_parent = bk
        .iter()
        .map(|&i| p.orig_mother[i as usize].max(p.orig_father[i as usize]))
        .max()
        .unwrap_or(0)
        + 1;
    let fam: Vec<i64> = bk.iter().map(|&i| p.orig_mother[i as usize] * max_parent + p.orig_father[i as usize]).collect();
    let full = pairs_from_groups(&bk, &fam);
    let m_idx: Vec<u32> = nt.iter().copied().filter(|&i| p.orig_mother[i as usize] >= 0).collect();
    let m_key: Vec<i64> = m_idx.iter().map(|&i| p.orig_mother[i as usize]).collect();
    let full_k = keys(&full);
    let mat = subtract(pairs_from_groups(&m_idx, &m_key), &[&full_k]);
    let f_idx: Vec<u32> = nt.iter().copied().filter(|&i| p.orig_father[i as usize] >= 0).collect();
    let f_key: Vec<i64> = f_idx.iter().map(|&i| p.orig_father[i as usize]).collect();
    let pat = subtract(pairs_from_groups(&f_idx, &f_key), &[&full_k]);
    (full, mat, pat)
}

/// Full 1C (>= 2 distinct shared grandparents) and half 1C (exactly 1), sibs excluded.
fn cousin_pairs(p: &Ped, a2: &Csr) -> (Pairs, Pairs) {
    let nz = a2.nonzeros();
    let gc: Vec<u32> = nz.iter().map(|x| x.0).collect();
    let gp: Vec<i64> = nz.iter().map(|x| x.1 as i64).collect();
    let cand = pairs_from_groups(&gc, &gp);
    let mut ks: Vec<u64> = cand
        .iter()
        .filter(|&&(a, b)| {
            let (ma, mb) = (p.orig_mother[a as usize], p.orig_mother[b as usize]);
            let (fa, fb) = (p.orig_father[a as usize], p.orig_father[b as usize]);
            !((ma >= 0 && ma == mb) || (fa >= 0 && fa == fb))
        })
        .map(|&(a, b)| key(a, b))
        .collect();
    ks.sort_unstable();
    let mut full = Vec::new();
    let mut half = Vec::new();
    let mut s = 0;
    while s < ks.len() {
        let mut e = s + 1;
        while e < ks.len() && ks[e] == ks[s] {
            e += 1;
        }
        let pair = ((ks[s] >> 32) as u32, (ks[s] & 0xffff_ffff) as u32);
        if e - s >= 2 { full.push(pair) } else { half.push(pair) }
        s = e;
    }
    (full, half)
}

fn avuncular(p: &Ped, a: &Csr, fsm: &Csr) -> Pairs {
    if fsm.nnz() == 0 {
        return vec![];
    }
    let av = a.matmul(fsm);
    let is_pc = |i: u32, j: u32| {
        let (i, j) = (i as usize, j as usize);
        p.mother[i] == j as i32 || p.father[i] == j as i32 || p.mother[j] == i as i32 || p.father[j] == i as i32
    };
    canon_dedup(av.nonzeros().into_iter().filter(|&(i, j)| i != j && !is_pc(i, j)))
}

fn collateral(a_down_1: &Csr, sib: &Csr, remove: &[&Vec<u64>]) -> Pairs {
    if sib.nnz() == 0 {
        return vec![];
    }
    extract_from_sparse(&a_down_1.matmul(sib), remove)
}

fn threshold(m: &Csr, keep: impl Fn(i32) -> bool) -> Csr {
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    for i in 0..m.n {
        let (js, vs) = m.row(i);
        for (&j, &v) in js.iter().zip(vs) {
            if j as usize != i && keep(v) {
                rows.push(i as u32);
                cols.push(j);
            }
        }
    }
    Csr::from_edges(m.n, &rows, &cols)
}

fn second_cousins(a3: &Csr, a3t: &Csr, a2_shared: &Csr) -> Pairs {
    let d = threshold(&a3.matmul(a3t), |v| v >= 2);
    let mut out = Vec::new();
    for i in 0..d.n {
        for &j in d.row(i).0 {
            if (i as u32) < j && !a2_shared.has(i, j) {
                out.push((i as u32, j));
            }
        }
    }
    out
}

fn concat(parts: &[&Pairs]) -> Pairs {
    parts.iter().flat_map(|p| p.iter().copied()).collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (inp, outp) = (&args[1], &args[2]);
    let p = read_ped(inp);
    let t0 = Instant::now();
    let n = p.n;
    let lap = |label: &str| eprintln!("{:>8.3}s  {}", t0.elapsed().as_secs_f64(), label);

    let mut pairs: Vec<(&str, Pairs)> = Vec::new();

    let mz: Pairs = (0..n).filter(|&i| p.twin[i] >= 0 && (i as i32) < p.twin[i]).map(|i| (i as u32, p.twin[i] as u32)).collect();
    let mo: Pairs = (0..n).filter(|&i| p.mother[i] >= 0).map(|i| (i as u32, p.mother[i] as u32)).collect();
    let fo: Pairs = (0..n).filter(|&i| p.father[i] >= 0).map(|i| (i as u32, p.father[i] as u32)).collect();
    let (fs, mhs, phs) = sibling_pairs(&p);
    lap("degree 0-1");

    let m_rows: Vec<u32> = mo.iter().map(|x| x.0).collect();
    let m_cols: Vec<u32> = mo.iter().map(|x| x.1).collect();
    let f_rows: Vec<u32> = fo.iter().map(|x| x.0).collect();
    let f_cols: Vec<u32> = fo.iter().map(|x| x.1).collect();
    let a = Csr::from_edges(n, &concat_u32(&m_rows, &f_rows), &concat_u32(&m_cols, &f_cols));
    let a2 = a.matmul(&a);
    let a3 = a2.matmul(&a);
    let a4 = a3.matmul(&a);
    let a5 = a4.matmul(&a);
    let eye = Csr::identity(n);
    lap(&format!("ancestor powers (nnz A2={} A3={} A4={} A5={})", a2.nnz(), a3.nnz(), a4.nnz(), a5.nnz()));

    let fsm = symmetric_from_pairs(n, &fs);
    let hsm = symmetric_from_pairs(n, &concat(&[&mhs, &phs]));

    let gp = lineal(&a2);
    let av = avuncular(&p, &a, &fsm);
    lap("degree 2");

    let po = concat(&[&mo, &fo]);
    let (po_k, gp_k, av_k) = (keys(&po), keys(&gp), keys(&av));
    let ggp = lineal(&a3);
    let hav = collateral(&a, &hsm, &[&po_k, &gp_k]);
    let gav = collateral(&a2, &fsm, &[&po_k, &gp_k, &av_k]);
    let (c1, h1c) = cousin_pairs(&p, &a2);
    let (ggp_k, hav_k, gav_k, c1_k, h1c_k) = (keys(&ggp), keys(&hav), keys(&gav), keys(&c1), keys(&h1c));
    lap("degree 3");

    let sib_all = concat(&[&fs, &mhs, &phs]);
    let sib_k = keys(&sib_all);
    let gggp = lineal(&a4);
    let hgav = collateral(&a2, &hsm, &[&po_k, &gp_k, &ggp_k, &hav_k]);
    let ggav = collateral(&a3, &fsm, &[&po_k, &gp_k, &ggp_k, &av_k, &gav_k]);
    let a3t = a3.transpose();
    let a2_a3t = a2.matmul(&a3t);
    lap("  A2 A3^T");
    let c1r1 = extract_from_sparse(&threshold(&a2_a3t, |v| v >= 2), &[&po_k, &gp_k, &ggp_k, &av_k, &gav_k, &sib_k, &c1_k]);
    let (gggp_k, hgav_k, ggav_k, c1r1_k) = (keys(&gggp), keys(&hgav), keys(&ggav), keys(&c1r1));
    lap("degree 4");

    let a2_shared = a2.matmul(&a2.transpose());
    lap("  A2 A2^T");
    let c2 = second_cousins(&a3, &a3t, &a2_shared);
    lap("  2C");
    let g3gp = lineal(&a5);
    let hggav = collateral(&a3, &hsm, &[&po_k, &gp_k, &ggp_k, &gggp_k, &hav_k, &hgav_k]);
    let g3av = collateral(&a4, &fsm, &[&po_k, &gp_k, &ggp_k, &gggp_k, &av_k, &gav_k, &ggav_k]);
    lap("  HGGAv+G3Av");
    let h1c1r = extract_from_sparse(
        &threshold(&a2_a3t, |v| v == 1),
        &[&po_k, &gp_k, &ggp_k, &gggp_k, &hav_k, &hgav_k, &sib_k, &c1_k, &h1c_k, &c1r1_k],
    );
    lap("  H1C1R");
    let c1r2 = extract_from_sparse(
        &threshold(&a2.matmul(&a4.transpose()), |v| v >= 2),
        &[&po_k, &gp_k, &ggp_k, &gggp_k, &av_k, &gav_k, &ggav_k, &sib_k, &c1_k, &h1c_k, &c1r1_k],
    );
    lap("degree 5");
    let _ = eye;

    for (code, v) in [
        ("MZ", mz), ("MO", mo), ("FO", fo), ("FS", fs), ("MHS", mhs), ("PHS", phs),
        ("GP", gp), ("Av", av), ("GGP", ggp), ("HAv", hav), ("GAv", gav), ("1C", c1),
        ("GGGP", gggp), ("HGAv", hgav), ("GGAv", ggav), ("H1C", h1c), ("1C1R", c1r1),
        ("G3GP", g3gp), ("HGGAv", hggav), ("G3Av", g3av), ("H1C1R", h1c1r), ("1C2R", c1r2), ("2C", c2),
    ] {
        pairs.push((code, v));
    }
    let mut w = BufWriter::new(std::fs::File::create(outp).expect("create output"));
    for (code, v) in &pairs {
        for &(i, j) in v {
            writeln!(w, "{}\t{}\t{}", code, i, j).unwrap();
        }
    }
    eprintln!("{:>8.3}s  total (excluding read)", t0.elapsed().as_secs_f64());
    for (code, v) in &pairs {
        eprint!("{}={} ", code, v.len());
    }
    eprintln!();
}

fn concat_u32(a: &[u32], b: &[u32]) -> Vec<u32> {
    a.iter().chain(b.iter()).copied().collect()
}
