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

/// Four generations of sibships, rows shuffled so that sibs stand apart.
///
/// Each generation after the founders is twenty couples drawn from the one
/// before, so remarriage makes half sibs and repeated draws inbreed, each
/// couple with one to four children.  A few fathers are ids outside the
/// pedigree, distinct ones for the same mother, and a few unknown.  Two
/// couples a generation share a father and know their mothers' ids but not
/// their rows, so their children have the same parent rows.  The first two
/// children of some sibships are MZ co-twins.
pub(crate) fn shuffled_sibships(seed: u64) -> PedigreeColumns {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let mut cols = PedigreeColumns::default();
    let push = |cols: &mut PedigreeColumns, m: i32, f: i32, om: i64, of: i64| {
        cols.mother.push(m);
        cols.father.push(f);
        cols.twin.push(-1);
        cols.orig_mother.push(om);
        cols.orig_father.push(of);
        cols.mother.len() - 1
    };
    let mut previous: Vec<usize> = (0..30).map(|_| push(&mut cols, -1, -1, -1, -1)).collect();
    for _ in 0..3 {
        let mut generation = Vec::new();
        let mut unresolved_father = None;
        let mut external_mother = None;
        for couple in 0..20u64 {
            let mut m = previous[next() as usize % previous.len()];
            if couple % 7 == 3 {
                m = *external_mother.get_or_insert(m);
            }
            let mut f = previous[next() as usize % previous.len()];
            if f == m {
                f = previous[(previous.iter().position(|&p| p == m).unwrap() + 1) % previous.len()];
            }
            let unresolved_mother = couple % 11 == 5;
            if unresolved_mother {
                f = *unresolved_father.get_or_insert(f);
            }
            let (fr, of) = match couple % 7 {
                3 => (-1, 1_000_000 + couple as i64),
                6 => (-1, -1),
                _ => (f as i32, f as i64),
            };
            let mr = if unresolved_mother { -1 } else { m as i32 };
            let first = cols.mother.len();
            for _ in 0..1 + next() % 4 {
                generation.push(push(&mut cols, mr, fr, m as i64, of));
            }
            if couple % 5 == 1 && cols.mother.len() - first >= 2 {
                cols.twin[first] = first as i32 + 1;
                cols.twin[first + 1] = first as i32;
            }
        }
        previous = generation;
    }

    // Shuffle the rows, relabelling the row columns; ids stay as they are.
    let n = cols.mother.len();
    let mut order: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        order.swap(i, next() as usize % (i + 1));
    }
    let mut new_row = vec![0i32; n];
    for (to, &from) in order.iter().enumerate() {
        new_row[from] = to as i32;
    }
    let relabel = |r: i32| if r < 0 { r } else { new_row[r as usize] };
    PedigreeColumns {
        mother: order.iter().map(|&i| relabel(cols.mother[i])).collect(),
        father: order.iter().map(|&i| relabel(cols.father[i])).collect(),
        twin: order.iter().map(|&i| relabel(cols.twin[i])).collect(),
        orig_mother: order.iter().map(|&i| cols.orig_mother[i]).collect(),
        orig_father: order.iter().map(|&i| cols.orig_father[i]).collect(),
    }
}
