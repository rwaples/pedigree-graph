//! The native walk matches a recursive float32 oracle bit for bit on every
//! dumped fixture, in graph space with structural depth as the peel input.
//!
//! The oracle is the ADR 0009 recurrence as written; it is deliberately the
//! `tests/oracle/pair_kinship.py` statement in Rust so the core is held to
//! the same contract the Python differential test holds the binding to.

use pedigree_graph_core::kinship::{pair_kinship, support_values, KinshipPedigree};
use pedigree_graph_core::relationships::PedigreeColumns;
use pedigree_graph_core::topology::structural_depth;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

struct Oracle<'a> {
    mother: &'a [i32],
    father: &'a [i32],
    twin: &'a [i32],
    depth: &'a [i32],
    memo: HashMap<(i32, i32), f32>,
}

impl Oracle<'_> {
    fn phi(&mut self, a: i32, b: i32) -> f32 {
        let (a, b) = if a > b { (b, a) } else { (a, b) };
        if let Some(&v) = self.memo.get(&(a, b)) {
            return v;
        }
        let (other, peeled) = if self.depth[a as usize] > self.depth[b as usize] {
            (b, a)
        } else {
            (a, b)
        };
        let m = self.mother[peeled as usize];
        let f = self.father[peeled as usize];
        let v = if other == peeled
            || self.twin[other as usize] == peeled
            || self.twin[peeled as usize] == other
        {
            if m < 0 || f < 0 {
                0.5f32
            } else {
                0.5f32 * (1.0f32 + self.phi(m, f))
            }
        } else {
            let left = if m >= 0 { self.phi(m, other) } else { 0.0f32 };
            let right = if f >= 0 { self.phi(f, other) } else { 0.0f32 };
            0.5f32 * (left + right)
        };
        self.memo.insert((a, b), v);
        v
    }
}

fn fixtures() -> Vec<PathBuf> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let mut names: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "tsv"))
        .collect();
    names.sort();
    names
}

/// Every pair of a small fixture; a seeded sample plus every self pair otherwise.
fn query(n: usize) -> (Vec<i32>, Vec<i32>) {
    let mut first = Vec::new();
    let mut second = Vec::new();
    if n <= 1100 {
        for a in 0..n {
            for b in a..n {
                first.push(a as i32);
                second.push(b as i32);
            }
        }
        return (first, second);
    }
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15 ^ n as u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state % n as u64) as i32
    };
    for row in 0..n {
        first.push(row as i32);
        second.push(row as i32);
    }
    for _ in 0..2_000 {
        first.push(next());
        second.push(next());
    }
    (first, second)
}

#[test]
fn the_walk_matches_the_oracle_on_every_fixture() {
    let mut checked = 0usize;
    for tsv in fixtures() {
        let name = tsv.file_stem().unwrap().to_string_lossy().to_string();
        let started = std::time::Instant::now();
        let cols = PedigreeColumns::read_tsv(&tsv).unwrap();
        if cols.mother.len() > 10_000 {
            // Self pairs alone walk the whole ancestral closure, 75 s on
            // random_30k; the Python golden lock holds that fixture by hash
            // under the slow marker.
            continue;
        }
        let depth = structural_depth(&cols.mother, &cols.father);
        let ped = KinshipPedigree::try_new(&cols.mother, &cols.father, &cols.twin, &depth).unwrap();
        let (first, second) = query(cols.mother.len());
        let mut oracle = Oracle {
            mother: &cols.mother,
            father: &cols.father,
            twin: &cols.twin,
            depth: &depth,
            memo: HashMap::new(),
        };
        let expected: Vec<u32> = first
            .iter()
            .zip(&second)
            .map(|(&a, &b)| oracle.phi(a, b).to_bits())
            .collect();
        let forward = pair_kinship(ped, &first, &second).unwrap();
        let reverse = pair_kinship(ped, &second, &first).unwrap();
        let got: Vec<u32> = forward.iter().map(|v| v.to_bits()).collect();
        let rev: Vec<u32> = reverse.iter().map(|v| v.to_bits()).collect();
        assert_eq!(got, expected, "{name} forward");
        assert_eq!(rev, expected, "{name} reverse");
        checked += first.len();
        eprintln!("{name}: {} pairs in {:.1?}", first.len(), started.elapsed());
    }
    assert!(checked > 0);
}

#[test]
fn support_values_match_pair_kinship_on_a_dense_small_fixture() {
    let tsv = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/deep_inbred_60g.tsv");
    let cols = PedigreeColumns::read_tsv(&tsv).unwrap();
    let depth = structural_depth(&cols.mother, &cols.father);
    let ped = KinshipPedigree::try_new(&cols.mother, &cols.father, &cols.twin, &depth).unwrap();
    let n = cols.mother.len();
    // Every third pair of the full matrix, symmetric, plus the diagonal.
    let mut columns: Vec<Vec<i32>> = vec![Vec::new(); n];
    for a in 0..n {
        for b in a..n {
            if a == b || (a + b) % 3 == 0 {
                columns[b].push(a as i32);
                if a != b {
                    columns[a].push(b as i32);
                }
            }
        }
    }
    let mut indptr = vec![0i64];
    let mut indices = Vec::new();
    let mut first = Vec::new();
    let mut second = Vec::new();
    for (column, rows) in columns.iter_mut().enumerate() {
        rows.sort_unstable();
        for &row in rows.iter() {
            first.push(row);
            second.push(column as i32);
        }
        indices.extend_from_slice(rows);
        indptr.push(indices.len() as i64);
    }
    let expected = pair_kinship(ped, &first, &second).unwrap();
    let data = support_values(ped, &indptr, &indices).unwrap();
    let got: Vec<u32> = data.iter().map(|v| v.to_bits()).collect();
    let want: Vec<u32> = expected.iter().map(|v| v.to_bits()).collect();
    assert_eq!(got, want);
}
