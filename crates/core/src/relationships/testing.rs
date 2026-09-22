//! Pedigree builders shared by the relationship unit tests.

use super::PedigreeColumns;

/// Rows with parent rows; original ids equal rows, `-1` missing.
pub(super) fn pedigree(parents: &[(i32, i32)], twins: &[(usize, usize)]) -> PedigreeColumns {
    let n = parents.len();
    let mut twin = vec![-1i32; n];
    for &(a, b) in twins {
        twin[a] = b as i32;
        twin[b] = a as i32;
    }
    PedigreeColumns {
        mother: parents.iter().map(|p| p.0).collect(),
        father: parents.iter().map(|p| p.1).collect(),
        twin,
        orig_mother: parents.iter().map(|p| p.0 as i64).collect(),
        orig_father: parents.iter().map(|p| p.1 as i64).collect(),
    }
}

/// A random overlapping-generation pedigree of `n` rows: each row after the
/// first twenty takes two distinct parents from the preceding sixty rows,
/// with a few MZ twin pairs and a few parents outside the pedigree.
pub(crate) fn random_pedigree(n: usize, seed: u64) -> PedigreeColumns {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let mut cols = PedigreeColumns::default();
    for i in 0..n {
        let (mut m, mut f) = (-1i32, -1i32);
        let (mut om, mut of) = (-1i64, -1i64);
        if i >= 20 {
            let lo = i.saturating_sub(60);
            m = (lo + (next() as usize) % (i - lo)) as i32;
            f = (lo + (next() as usize) % (i - lo)) as i32;
            if f == m {
                f = -1;
            }
            om = i64::from(m);
            of = if f >= 0 {
                i64::from(f)
            } else if next() % 4 == 0 {
                1_000_000 + (next() % 50) as i64
            } else {
                -1
            };
        }
        cols.mother.push(m);
        cols.father.push(f);
        cols.twin.push(-1);
        cols.orig_mother.push(om);
        cols.orig_father.push(of);
    }
    for i in (25..n).step_by(97) {
        let j = i - 1;
        cols.twin[i] = j as i32;
        cols.twin[j] = i as i32;
        cols.mother[i] = cols.mother[j];
        cols.father[i] = cols.father[j];
        cols.orig_mother[i] = cols.orig_mother[j];
        cols.orig_father[i] = cols.orig_father[j];
    }
    cols
}
