//! Ancestor signatures: a sound test that two rows share no ancestor, so
//! their kinship is exactly zero and the walk need not reach the founders to
//! learn it.
//!
//! Each row's signature is a 256-bit set: one hashed bit for the row itself,
//! one for its MZ co-twin if it has one, and every bit of both parents'
//! signatures.  The signature of a row therefore holds the bit of every row
//! in its ancestor-or-self set, and of every co-twin of one.
//!
//! Disjoint signatures mean a recurrence value of exactly `+0.0`.  The only
//! keys that add anything are self-like ones, `(u, v)` with `u == v` or the
//! two MZ co-twins; every other key is a half-sum of keys one parent up.  So
//! a nonzero `phi(x, y)` reaches a self-like `(u, v)` with `u` in the
//! ancestor-or-self set of one endpoint and `v` in the other's, and `v`'s bit
//! is in both signatures.  A hash collision can only make two unrelated rows
//! look related, which costs a walk, never a value.
//!
//! On the PA-FGRS analysis pedigree of simACE `cure_rA50_200k` (357k rows,
//! six generations) 32.4M of the 35.5M keys a degree-2 support walks are
//! such zeros, almost all pairs of unrelated ancestors (issue #34).

use super::pairwise::KinshipPedigree;
use crate::alloc::{self, Family};
use crate::error::Error;

const WORDS: usize = 4;
const BITS: u64 = (WORDS * 64) as u64;

type Signature = [u64; WORDS];

/// A built signature holds at least its own bit, so all-zero marks a row not
/// yet built.
const UNBUILT: Signature = [0; WORDS];

/// One ancestor signature per row.
pub struct AncestorSignatures {
    sets: Vec<Signature>,
}

impl AncestorSignatures {
    /// The signature of every row, each built after its parents'.
    ///
    /// # Errors
    ///
    /// [`Error::AllocationFailed`] (`kinship_signatures`) for the signatures
    /// or the stack they are built with.
    pub fn build(ped: &KinshipPedigree<'_>) -> Result<Self, Error> {
        let (mother, father, twin) = (ped.mother(), ped.father(), ped.twin());
        let mut sets = alloc::filled(UNBUILT, ped.len(), Family::KinshipSignatures, "uint64")?;
        let mut stack: Vec<usize> = Vec::new();
        for root in 0..ped.len() {
            if sets[root] != UNBUILT {
                continue;
            }
            alloc::push(&mut stack, root, Family::KinshipSignatures, "uint64")?;
            while let Some(&row) = stack.last() {
                if sets[row] != UNBUILT {
                    stack.pop();
                    continue;
                }
                let parents = [mother[row], father[row]];
                let mut waiting = false;
                for parent in parents.into_iter().filter(|&p| p >= 0) {
                    if sets[parent as usize] == UNBUILT {
                        alloc::push(
                            &mut stack,
                            parent as usize,
                            Family::KinshipSignatures,
                            "uint64",
                        )?;
                        waiting = true;
                    }
                }
                if waiting {
                    continue;
                }
                stack.pop();
                let mut set = UNBUILT;
                for genome in [row as i32, twin[row]].into_iter().filter(|&g| g >= 0) {
                    let bit = bit_of(genome as u32);
                    set[bit / 64] |= 1 << (bit % 64);
                }
                for parent in parents.into_iter().filter(|&p| p >= 0) {
                    for (word, &theirs) in set.iter_mut().zip(&sets[parent as usize]) {
                        *word |= theirs;
                    }
                }
                sets[row] = set;
            }
        }
        Ok(AncestorSignatures { sets })
    }

    /// Whether rows `a` and `b` provably share no ancestor-or-self genome.
    #[inline]
    pub fn disjoint(&self, a: u32, b: u32) -> bool {
        let (a, b) = (&self.sets[a as usize], &self.sets[b as usize]);
        a.iter().zip(b).fold(0, |acc, (x, y)| acc | (x & y)) == 0
    }
}

/// A row's bit: SplitMix64's finaliser of the row index, so related rows
/// with nearby indices land on unrelated bits.
#[inline]
fn bit_of(row: u32) -> usize {
    let mut h = u64::from(row).wrapping_add(0x9E37_79B9_7F4A_7C15);
    h = (h ^ (h >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h = (h ^ (h >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    h ^= h >> 31;
    (h % BITS) as usize
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::relationships::testing::random_pedigree;
    use crate::topology::structural_depth;

    #[test]
    fn a_shared_ancestor_or_twin_always_overlaps() {
        // 0, 1 founders; 2 child of 0; 3 founder; 4 child of 3; 5 and 6
        // founder MZ twins, recorded on one side only; 7 child of 6.
        let mother = [-1, -1, 0, -1, 3, -1, -1, 6];
        let father = [-1, -1, 1, -1, -1, -1, -1, -1];
        let twin = [-1, -1, -1, -1, -1, 6, -1, -1];
        let depth = structural_depth(&mother, &father);
        let ped = KinshipPedigree::try_new(&mother, &father, &twin, &depth).unwrap();
        let sigs = AncestorSignatures::build(&ped).unwrap();
        for (a, b) in [(0, 2), (2, 2), (3, 4), (5, 6), (5, 7), (6, 7)] {
            assert!(!sigs.disjoint(a, b), "({a}, {b})");
            assert!(!sigs.disjoint(b, a), "({b}, {a})");
        }
    }

    #[test]
    fn every_ancestor_pair_of_a_random_pedigree_overlaps() {
        let cols = random_pedigree(2000, 7);
        let depth = structural_depth(&cols.mother, &cols.father);
        let ped = KinshipPedigree::try_new(&cols.mother, &cols.father, &cols.twin, &depth).unwrap();
        let sigs = AncestorSignatures::build(&ped).unwrap();
        for row in 0..2000usize {
            let mut seen = vec![false; 2000];
            let mut frontier = vec![row];
            while let Some(r) = frontier.pop() {
                if std::mem::replace(&mut seen[r], true) {
                    continue;
                }
                assert!(!sigs.disjoint(row as u32, r as u32), "({row}, {r})");
                frontier.extend(
                    [cols.mother[r], cols.father[r]]
                        .into_iter()
                        .filter(|&p| p >= 0)
                        .map(|p| p as usize),
                );
            }
        }
    }
}
