//! Inbreeding by the Meuwissen-Luo walk over the genome-node pedigree
//! (ADR 0008).
//!
//! MZ co-twins share one genome node, the lower graph row; a parent step
//! follows the parent's node, and a non-canonical twin copies its node's
//! `F` and Mendelian sampling variance `D`.  Co-twins share their parents
//! and so their depth, and within a depth the sweep visits ascending graph
//! rows, so every node is finished before its co-twin copies it.
//!
//! For each node `i` with both parents, a forward path-sum walk from
//! `t[i] = 1` halves along every parent edge, one depth at a time from
//! `i`'s depth down to the founders, and
//! `F[i] = sum over the touched ancestors j of t[j]^2 D[j] - 1`, summed in
//! touch order.  The frontier of each depth is a linked list threaded
//! through the touched list, newest first.  Neither the touch order nor the
//! order of the adds depends on row labels, so the values are the ones the
//! 0.9.3 Numba walk produced in its topological coordinates.
//!
//! The touched list is `i`'s closed ancestor set of genome nodes, so the walk
//! also counts `i`'s distinct strict ancestors: one per touched node but
//! `i`, and one more for each node both of whose co-twin rows are ancestors.
//! Co-twins name the same parents, so the ancestor rows are the graph-row
//! parents of the touched nodes.

use super::depth_order::DepthOrder;
use super::pairwise::KinshipPedigree;
use crate::alloc::{self, Family};
use crate::error::Error;
use std::borrow::Cow;

const WALK: Family = Family::InbreedingWalk;

/// A row's mother and father columns, borrowed or rewritten to genome nodes.
type Parents<'a> = (Cow<'a, [i32]>, Cow<'a, [i32]>);

/// The end of a frontier chain.
const NIL: u32 = u32::MAX;

/// `F` for every graph row, as float64.
///
/// # Errors
///
/// [`Error::ValueOutOfRange`] on `depth` when a row's depth is negative, not
/// above both parents', or not its MZ co-twin's, and
/// [`Error::AllocationFailed`] for any buffer.
pub fn inbreeding(ped: KinshipPedigree<'_>) -> Result<Vec<f64>, Error> {
    Ok(inbreeding_and_ancestor_counts(ped)?.0)
}

/// `F` for every graph row, as float64, and the distinct strict ancestors of
/// every graph row, as int32, from one walk.  The counts equal
/// [`crate::lineage::distinct_ancestor_counts`]; the walk holds O(rows)
/// working memory where that sweep holds every live row's ancestor set.
///
/// # Errors
///
/// As [`inbreeding()`].
pub fn inbreeding_and_ancestor_counts(
    ped: KinshipPedigree<'_>,
) -> Result<(Vec<f64>, Vec<i32>), Error> {
    let n = ped.len();
    let genome = GenomePedigree::build(&ped, WALK)?;
    let (sweep, mother, father) = (&genome.sweep, &genome.mother, &genome.father);
    let depth = ped.depth();
    let twin = ped.twin();
    let (row_mother, row_father) = (ped.mother(), ped.father());

    let mut f = alloc::filled(0.0f64, n, WALK)?;
    let mut d_var = alloc::filled(0.0f64, n, WALK)?;
    let mut ancestors = alloc::filled(0i32, n, WALK)?;
    let mut t = alloc::filled(0.0f64, n, WALK)?;
    let mut in_frontier = alloc::filled(false, n, WALK)?;
    // Which co-twin rows the current walk has reached as a graph-row parent.
    let mut reached = alloc::filled(false, if genome.has_twins { n } else { 0 }, WALK)?;
    let mut head = alloc::filled(NIL, sweep.max_depth() + 1, WALK)?;
    let capacity = initial_capacity(n, sweep.max_depth());
    let mut touched: Vec<u32> = alloc::with_capacity(capacity, WALK)?;
    let mut next: Vec<u32> = alloc::with_capacity(capacity, WALK)?;

    for &row in &sweep.order {
        let i = row as usize;
        let c = genome_node(twin, i);
        if c != i {
            f[i] = f[c];
            d_var[i] = d_var[c];
            ancestors[i] = ancestors[c];
            continue;
        }
        let (s, d) = (mother[i], father[i]);
        d_var[i] = mendelian_variance(s, d, &f);
        match (s >= 0, d >= 0) {
            (false, false) => continue,
            (true, false) | (false, true) => {
                ancestors[i] = ancestors[s.max(d) as usize] + 1;
                continue;
            }
            (true, true) => {}
        }

        t[i] = 1.0;
        in_frontier[i] = true;
        let di = depth[i] as usize;
        alloc::push(&mut next, head[di], WALK)?;
        head[di] = 0;
        alloc::push(&mut touched, row, WALK)?;
        let mut both_twins = 0usize;

        for k in (0..=di).rev() {
            let mut pos = head[k];
            while pos != NIL {
                let a = touched[pos as usize] as usize;
                let t_a = t[a];
                if genome.has_twins {
                    for r in [row_mother[a], row_father[a]] {
                        if r >= 0 && twin[r as usize] >= 0 && !reached[r as usize] {
                            reached[r as usize] = true;
                            both_twins += usize::from(reached[twin[r as usize] as usize]);
                        }
                    }
                }
                for p in [mother[a], father[a]] {
                    if p < 0 {
                        continue;
                    }
                    let p = p as usize;
                    if !in_frontier[p] {
                        in_frontier[p] = true;
                        t[p] = 0.0;
                        let dp = depth[p] as usize;
                        alloc::push(&mut next, head[dp], WALK)?;
                        head[dp] = touched.len() as u32;
                        alloc::push(&mut touched, p as u32, WALK)?;
                    }
                    t[p] += 0.5 * t_a;
                }
                pos = next[pos as usize];
            }
            head[k] = NIL;
        }

        let mut sum = 0.0f64;
        for &j in &touched {
            let tj = t[j as usize];
            sum += tj * tj * d_var[j as usize];
        }
        f[i] = sum - 1.0;
        ancestors[i] = (touched.len() - 1 + both_twins) as i32;

        for &j in &touched {
            let j = j as usize;
            t[j] = 0.0;
            in_frontier[j] = false;
            if genome.has_twins && twin[j] >= 0 {
                reached[j] = false;
                reached[twin[j] as usize] = false;
            }
        }
        touched.clear();
        next.clear();
    }
    Ok((f, ancestors))
}

/// The Mendelian sampling variance of a genome node whose parent nodes are
/// `s` and `d`, on the `A = 2φ` scale, from the parents' `F`.
fn mendelian_variance(s: i32, d: i32, f: &[f64]) -> f64 {
    match (s >= 0, d >= 0) {
        (false, false) => 1.0,
        (false, true) => 0.75 - 0.25 * f[d as usize],
        (true, false) => 0.75 - 0.25 * f[s as usize],
        (true, true) => 0.5 - 0.25 * (f[s as usize] + f[d as usize]),
    }
}

/// The genome-node pedigree the walk and the kinship sums sweep.
pub(super) struct GenomePedigree<'a> {
    pub(super) sweep: DepthOrder,
    /// Each row's parents as genome nodes.
    pub(super) mother: Cow<'a, [i32]>,
    pub(super) father: Cow<'a, [i32]>,
    has_twins: bool,
}

impl<'a> GenomePedigree<'a> {
    /// # Errors
    ///
    /// As [`inbreeding()`], with every buffer reserved under `family`.
    pub(super) fn build(ped: &KinshipPedigree<'a>, family: Family) -> Result<Self, Error> {
        check_twin_depths(ped.twin(), ped.depth())?;
        let sweep = DepthOrder::build(ped.mother(), ped.father(), ped.depth(), family)?;
        let has_twins = ped.twin().iter().any(|&t| t >= 0);
        let (mother, father) = genome_parents(ped, has_twins, family)?;
        Ok(GenomePedigree {
            sweep,
            mother,
            father,
            has_twins,
        })
    }

    /// The Mendelian sampling variance `D` of every graph row from `F`, the
    /// value the walk computed for it.  A non-canonical twin has its node's
    /// parents and so its node's `D`.
    ///
    /// # Errors
    ///
    /// [`Error::AllocationFailed`] under `family`.
    pub(super) fn mendelian_variances(&self, f: &[f64], family: Family) -> Result<Vec<f64>, Error> {
        alloc::collect(
            self.mother
                .iter()
                .zip(self.father.iter())
                .map(|(&s, &d)| mendelian_variance(s, d, f)),
            family,
        )
    }
}

/// A row's genome node: the lower graph row of an MZ pair, else the row.
pub(super) fn genome_node(twin: &[i32], row: usize) -> usize {
    let t = twin[row];
    if t >= 0 && (t as usize) < row {
        t as usize
    } else {
        row
    }
}

/// Co-twins are one node, swept at one depth.  Structural depth always puts
/// them there, since they share their parents; a raised depth handed to the
/// bindings need not.
fn check_twin_depths(twin: &[i32], depth: &[i32]) -> Result<(), Error> {
    for (row, &t) in twin.iter().enumerate() {
        if t >= 0 && depth[t as usize] != depth[row] {
            let partner = i64::from(depth[t as usize]);
            return Err(Error::ValueOutOfRange {
                field: "depth",
                position: row,
                value: i64::from(depth[row]),
                minimum: partner,
                maximum: partner,
            });
        }
    }
    Ok(())
}

/// Each row's parents as genome nodes; the input columns themselves when
/// the pedigree has no twins.
fn genome_parents<'a>(
    ped: &KinshipPedigree<'a>,
    has_twins: bool,
    family: Family,
) -> Result<Parents<'a>, Error> {
    if !has_twins {
        return Ok((Cow::Borrowed(ped.mother()), Cow::Borrowed(ped.father())));
    }
    let twin = ped.twin();
    let canonical = |rows: &[i32]| {
        alloc::collect(
            rows.iter().map(|&p| {
                if p < 0 {
                    -1
                } else {
                    genome_node(twin, p as usize) as i32
                }
            }),
            family,
        )
    };
    Ok((
        Cow::Owned(canonical(ped.mother())?),
        Cow::Owned(canonical(ped.father())?),
    ))
}

/// The touched list's starting room: the full binary ancestor set of the
/// deepest row plus slack, `2^(max_depth + 1) - 2 + 64` with a floor of 128,
/// capped at the pedigree size.  The list doubles past it.
fn initial_capacity(n: usize, max_depth: usize) -> usize {
    let full = 1usize
        .checked_shl(max_depth as u32 + 1)
        .unwrap_or(usize::MAX)
        .saturating_sub(2);
    full.max(64).saturating_add(64).min(n).max(4)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lineage::ParentColumns;
    use crate::topology::structural_depth;

    struct Cols {
        mother: Vec<i32>,
        father: Vec<i32>,
        twin: Vec<i32>,
        depth: Vec<i32>,
    }

    impl Cols {
        fn new(rows: &[(i32, i32)], twins: &[(i32, i32)]) -> Cols {
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

        fn ped(&self) -> KinshipPedigree<'_> {
            KinshipPedigree::try_new(&self.mother, &self.father, &self.twin, &self.depth).unwrap()
        }
    }

    #[test]
    fn full_sib_offspring_is_a_quarter_inbred() {
        let c = Cols::new(&[(-1, -1), (-1, -1), (0, 1), (0, 1), (2, 3)], &[]);
        assert_eq!(inbreeding(c.ped()).unwrap(), vec![0.0, 0.0, 0.0, 0.0, 0.25]);
    }

    #[test]
    fn parent_offspring_mating_is_a_quarter_inbred() {
        let c = Cols::new(&[(-1, -1), (-1, -1), (0, 1), (0, 2)], &[]);
        assert_eq!(inbreeding(c.ped()).unwrap()[3], 0.25);
    }

    #[test]
    fn a_child_of_mz_co_twins_counts_them_as_one_genome() {
        // Rows 2 and 3 are MZ; 4 is a child of 3 with 5, and 6 of 2 with 4.
        let c = Cols::new(
            &[(-1, -1), (-1, -1), (0, 1), (0, 1), (3, -1), (4, -1), (2, 5)],
            &[(2, 3)],
        );
        let f = inbreeding(c.ped()).unwrap();
        // 6's parents are 2 and 5, and 5 descends from 3, 2's co-twin: the
        // same genome as a grandparent, phi(2, 5) = 1/4 * phi(3, 3) = 1/8.
        assert_eq!(f[6], 0.125);
        assert_eq!(f[2], f[3]);
    }

    fn ancestor_counts(c: &Cols) -> Vec<i32> {
        inbreeding_and_ancestor_counts(c.ped()).unwrap().1
    }

    #[test]
    fn a_descendant_of_both_co_twins_counts_both_rows() {
        // 2, 3 MZ of (0, 1); 4 = (2, -1), 5 = (3, -1), 6 = (4, 5).  Genome
        // node 2 stands for both twins, and both are 6's ancestors.
        let c = Cols::new(
            &[(-1, -1), (-1, -1), (0, 1), (0, 1), (2, -1), (3, -1), (4, 5)],
            &[(2, 3)],
        );
        assert_eq!(ancestor_counts(&c), vec![0, 0, 2, 2, 3, 3, 6]);
    }

    #[test]
    fn walk_counts_match_the_ancestor_set_sweep() {
        for seed in [3, 11, 29] {
            let c = crate::relationships::testing::random_pedigree(800, seed);
            let depth = structural_depth(&c.mother, &c.father);
            assert!(c.twin.iter().any(|&t| t >= 0));
            let no_twins = vec![-1; c.twin.len()];
            let cols = ParentColumns::try_new(&c.mother, &c.father, Some(&depth)).unwrap();
            let want = crate::lineage::distinct_ancestor_counts(cols).unwrap();
            for twin in [&c.twin, &no_twins] {
                let ped = KinshipPedigree::try_new(&c.mother, &c.father, twin, &depth).unwrap();
                let (f, got) = inbreeding_and_ancestor_counts(ped).unwrap();
                assert_eq!(got, want, "seed {seed}");
                assert_eq!(f, inbreeding(ped).unwrap());
            }
        }
    }

    #[test]
    fn row_labels_do_not_change_a_value() {
        let c = crate::relationships::testing::random_pedigree(400, 3);
        let depth = structural_depth(&c.mother, &c.father);
        let ped = KinshipPedigree::try_new(&c.mother, &c.father, &c.twin, &depth).unwrap();
        let want = inbreeding(ped).unwrap();
        let n = c.mother.len();
        let perm: Vec<usize> = (0..n).rev().collect();
        let mut inv = vec![0usize; n];
        for (new, &old) in perm.iter().enumerate() {
            inv[old] = new;
        }
        let relabel = |col: &[i32]| -> Vec<i32> {
            perm.iter()
                .map(|&old| {
                    let p = col[old];
                    if p < 0 {
                        -1
                    } else {
                        inv[p as usize] as i32
                    }
                })
                .collect()
        };
        let (m, f, t) = (relabel(&c.mother), relabel(&c.father), relabel(&c.twin));
        let d = structural_depth(&m, &f);
        let got = inbreeding(KinshipPedigree::try_new(&m, &f, &t, &d).unwrap()).unwrap();
        for (new, &old) in perm.iter().enumerate() {
            assert_eq!(got[new].to_bits(), want[old].to_bits(), "row {old}");
        }
    }

    #[test]
    fn f_is_twice_the_self_kinship_less_one() {
        let c = crate::relationships::testing::random_pedigree(300, 11);
        let depth = structural_depth(&c.mother, &c.father);
        let ped = KinshipPedigree::try_new(&c.mother, &c.father, &c.twin, &depth).unwrap();
        let f = inbreeding(ped).unwrap();
        let rows: Vec<i32> = (0..300).collect();
        let phi = crate::kinship::pair_kinship(ped, &rows, &rows).unwrap();
        for (row, (&fi, &p)) in f.iter().zip(&phi).enumerate() {
            assert!((fi - (2.0 * f64::from(p) - 1.0)).abs() < 1e-6, "row {row}");
        }
    }

    #[test]
    fn the_touched_list_starts_at_the_numba_walks_room() {
        assert_eq!(initial_capacity(10_000, 1), 128);
        assert_eq!(initial_capacity(10_000, 7), 318);
        assert_eq!(initial_capacity(50, 1), 50);
        assert_eq!(initial_capacity(3, 0), 4);
        assert_eq!(initial_capacity(1 << 20, 200), 1 << 20);
    }

    #[test]
    fn repeated_full_sib_mating_approaches_one() {
        let mut rows = vec![(-1, -1), (-1, -1)];
        for g in 0..40 {
            let (a, b) = (2 * g, 2 * g + 1);
            rows.push((a, b));
            rows.push((a, b));
        }
        let c = Cols::new(&rows, &[]);
        let f = inbreeding(c.ped()).unwrap();
        assert_eq!(f[2], 0.0);
        assert_eq!(f[4], 0.25);
        assert_eq!(f[6], 0.375);
        assert!(f[rows.len() - 1] > 0.99);
    }

    #[test]
    fn a_shallow_depth_is_refused() {
        let c = Cols::new(&[(-1, -1), (-1, -1), (0, 1)], &[]);
        let depth = vec![0, 0, 0];
        let ped = KinshipPedigree::try_new(&c.mother, &c.father, &c.twin, &depth).unwrap();
        assert!(matches!(
            inbreeding(ped),
            Err(Error::ValueOutOfRange {
                field: "depth",
                position: 2,
                ..
            })
        ));
    }
}
