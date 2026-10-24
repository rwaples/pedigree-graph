//! Inbreeding by the Meuwissen-Luo walk over the genome-node pedigree
//! (ADR 0008).
//!
//! MZ co-twins share one genome node, the lower graph row; a parent step
//! follows the parent's node.  Co-twins share their parents and so their
//! depth.
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
//! parents of the touched nodes.  A genome parent whose graph-row parent has
//! a co-twin is flagged in the parent column itself, so a pedigree with few
//! twins reads the twin columns only at those parents.
//!
//! A walk reads only `D` of its own node and of shallower ones, so the walks
//! of one depth are independent: given a Rayon pool they run on it, one
//! walker per worker, and each value is the one a serial sweep gives.  Rows
//! whose genome nodes have the same graph-row parents (full sibs, co-twins)
//! have the same `F`, `D` and ancestor count, so each depth is sorted by those
//! parents and only the first row of each family is walked.  The key is the
//! graph-row parents, not the genome nodes: children of co-twin mothers share
//! genome parents but can count different ancestor rows.

use super::depth_order::DepthOrder;
use super::pairwise::KinshipPedigree;
use crate::alloc::{self, Family};
use crate::error::Error;
use rayon::prelude::*;
use std::borrow::Cow;
use std::sync::atomic::{AtomicUsize, Ordering};

const WALK: Family = Family::InbreedingWalk;

/// A row's mother and father columns, borrowed or rewritten to genome nodes.
type Parents<'a> = (Cow<'a, [i32]>, Cow<'a, [i32]>);

/// The end of a frontier chain.
const NIL: u32 = u32::MAX;

/// Rows of one depth classified and walked together, which bounds the
/// per-block buffers whatever the depth's width.
const BLOCK: usize = 1 << 14;

/// The most walks a worker takes from a block at a time; a narrow depth is
/// cut finer, about eight chunks per worker, since deep walks are long.
const CHUNK: usize = 64;

/// `F` for every graph row, as float64.
///
/// # Errors
///
/// [`Error::ValueOutOfRange`] on `twin` when a row's MZ co-twin does not name
/// it back, on `depth` when a row's depth is negative, not above both
/// parents', or not its MZ co-twin's, and [`Error::AllocationFailed`] for any
/// buffer.
pub fn inbreeding(ped: KinshipPedigree<'_>) -> Result<Vec<f64>, Error> {
    Ok(inbreeding_and_ancestor_counts(ped)?.0)
}

/// `F` for every graph row, as float64, and the distinct strict ancestors of
/// every graph row, as int32, from one serial walk.  The counts equal
/// [`crate::lineage::distinct_ancestor_counts`]; the walk holds O(rows)
/// working memory where that sweep holds every live row's ancestor set.
///
/// # Errors
///
/// As [`inbreeding()`].
pub fn inbreeding_and_ancestor_counts(
    ped: KinshipPedigree<'_>,
) -> Result<(Vec<f64>, Vec<i32>), Error> {
    sweep(ped, None)
}

/// As [`inbreeding_and_ancestor_counts`], each depth's walks split across
/// `pool`, one walker and its O(rows) scratch per worker.  Both outputs are
/// the same for every pool size.  Everything but the walks runs on the
/// calling thread, so a one-thread pool is never entered and its buffers are
/// freed where they were allocated.
///
/// # Errors
///
/// As [`inbreeding()`].
pub fn inbreeding_and_ancestor_counts_in(
    ped: KinshipPedigree<'_>,
    pool: &rayon::ThreadPool,
) -> Result<(Vec<f64>, Vec<i32>), Error> {
    sweep(ped, Some(pool))
}

fn sweep(
    ped: KinshipPedigree<'_>,
    pool: Option<&rayon::ThreadPool>,
) -> Result<(Vec<f64>, Vec<i32>), Error> {
    let n = ped.len();
    let mut genome = GenomePedigree::build(&ped, WALK)?;
    let (twin, row_mother, row_father) = (ped.twin(), ped.mother(), ped.father());
    if genome.has_twins {
        flag_co_twin_parents(genome.mother.to_mut(), row_mother, twin);
        flag_co_twin_parents(genome.father.to_mut(), row_father, twin);
    }
    // A row's family: the graph-row parents of its genome node.
    let family = |row: u32| {
        let c = genome_node(twin, row as usize);
        (row_mother[c], row_father[c])
    };
    let sweep = &mut genome.sweep;
    for d in 0..=sweep.max_depth() {
        let (start, end) = (sweep.starts[d], sweep.starts[d + 1]);
        sweep.order[start..end].sort_unstable_by_key(|&row| (family(row), row));
    }
    let nodes = Nodes {
        mother: &genome.mother,
        father: &genome.father,
        row_mother,
        row_father,
        twin,
        depth: ped.depth(),
    };

    let mut f = alloc::filled(0.0f64, n, WALK)?;
    let mut d_var = alloc::filled(0.0f64, n, WALK)?;
    let mut ancestors = alloc::filled(0i32, n, WALK)?;
    let threads = pool
        .map_or(1, rayon::ThreadPool::current_num_threads)
        .max(1);
    let mut walkers = alloc::with_capacity(threads, WALK)?;
    for _ in 0..threads {
        walkers.push(Walker::new(n, genome.sweep.max_depth(), genome.has_twins)?);
    }
    let mut targets: Vec<u32> = alloc::with_capacity(BLOCK.min(n), WALK)?;

    for d in 0..=genome.sweep.max_depth() {
        let rows = genome.sweep.rows_at(d);
        for start in (0..rows.len()).step_by(BLOCK) {
            let end = (start + BLOCK).min(rows.len());
            // A row of the previous row's family copies it once that row is done.
            let copies = |k: usize| k > 0 && family(rows[k]) == family(rows[k - 1]);
            targets.clear();
            for k in (start..end).filter(|&k| !copies(k)) {
                let i = rows[k] as usize;
                let (s, p) = (
                    genome_parent(nodes.mother[i]),
                    genome_parent(nodes.father[i]),
                );
                d_var[i] = mendelian_variance(s, p, &f);
                match (s >= 0, p >= 0) {
                    (false, false) => {}
                    (true, false) | (false, true) => {
                        ancestors[i] = ancestors[s.max(p) as usize] + 1;
                    }
                    (true, true) => targets.push(rows[k]),
                }
            }
            walk_all(
                pool,
                &mut walkers,
                &nodes,
                &targets,
                &d_var,
                genome.has_twins,
            )?;
            for walked in walkers.iter_mut().flat_map(|w| w.out.drain(..)) {
                f[walked.row as usize] = walked.f;
                ancestors[walked.row as usize] = walked.ancestors;
            }
            for k in (start..end).filter(|&k| copies(k)) {
                let (i, sib) = (rows[k] as usize, rows[k - 1] as usize);
                f[i] = f[sib];
                d_var[i] = d_var[sib];
                ancestors[i] = ancestors[sib];
            }
        }
    }
    Ok((f, ancestors))
}

/// Walk every node of `targets`, all at one depth, splitting them across the
/// walkers on `pool`; each walker leaves its results in its `out`.
fn walk_all(
    pool: Option<&rayon::ThreadPool>,
    walkers: &mut [Walker],
    nodes: &Nodes<'_>,
    targets: &[u32],
    d_var: &[f64],
    twins: bool,
) -> Result<(), Error> {
    let run = |walker: &mut Walker, rows: &[u32]| {
        rows.iter().try_for_each(|&row| {
            let walked = if twins {
                walker.walk::<true>(nodes, row, d_var)?
            } else {
                walker.walk::<false>(nodes, row, d_var)?
            };
            alloc::push(&mut walker.out, walked, WALK)
        })
    };
    let pool = match pool {
        Some(pool) if walkers.len() > 1 && targets.len() > 1 => pool,
        _ => return run(&mut walkers[0], targets),
    };
    let chunk = targets.len().div_ceil(8 * walkers.len()).min(CHUNK);
    let chunks = targets.len().div_ceil(chunk);
    let next = AtomicUsize::new(0);
    pool.install(|| {
        walkers.par_iter_mut().try_for_each(|walker| loop {
            let index = next.fetch_add(1, Ordering::Relaxed);
            if index >= chunks {
                return Ok(());
            }
            let rows = &targets[index * chunk..((index + 1) * chunk).min(targets.len())];
            if let Err(e) = run(walker, rows) {
                next.fetch_max(chunks, Ordering::Relaxed);
                return Err(e);
            }
        })
    })
}

/// The columns a walk reads: genome parents, co-twin parents flagged
/// (see [`flag_co_twin_parents`]), and the graph-row parents and twins the
/// flagged ones lead to.
struct Nodes<'a> {
    mother: &'a [i32],
    father: &'a [i32],
    row_mother: &'a [i32],
    row_father: &'a [i32],
    twin: &'a [i32],
    depth: &'a [i32],
}

/// One walk's result.
struct Walked {
    row: u32,
    f: f64,
    ancestors: i32,
}

impl alloc::Dtype for Walked {
    const NAME: &'static str = "object";
}

/// One worker's walk scratch, every per-row entry cleared between walks.
struct Walker {
    t: Vec<f64>,
    in_frontier: Vec<bool>,
    /// Which co-twin rows the current walk has reached as a graph-row parent.
    reached: Vec<bool>,
    /// The rows set in `reached`, to clear.
    marked: Vec<u32>,
    head: Vec<u32>,
    touched: Vec<u32>,
    next: Vec<u32>,
    out: Vec<Walked>,
}

impl alloc::Dtype for Walker {
    const NAME: &'static str = "object";
}

impl Walker {
    fn new(n: usize, max_depth: usize, twins: bool) -> Result<Walker, Error> {
        let capacity = initial_capacity(n, max_depth);
        Ok(Walker {
            t: alloc::filled(0.0f64, n, WALK)?,
            in_frontier: alloc::filled(false, n, WALK)?,
            reached: alloc::filled(false, if twins { n } else { 0 }, WALK)?,
            marked: Vec::new(),
            head: alloc::filled(NIL, max_depth + 1, WALK)?,
            touched: alloc::with_capacity(capacity, WALK)?,
            next: alloc::with_capacity(capacity, WALK)?,
            out: Vec::new(),
        })
    }

    /// `F` and the distinct strict ancestor count of genome node `row`, which
    /// has both parents.  `TWINS` is whether any parent is flagged.
    fn walk<const TWINS: bool>(
        &mut self,
        nodes: &Nodes<'_>,
        row: u32,
        d_var: &[f64],
    ) -> Result<Walked, Error> {
        let Walker {
            t,
            in_frontier,
            reached,
            marked,
            head,
            touched,
            next,
            ..
        } = self;
        let i = row as usize;
        t[i] = 1.0;
        in_frontier[i] = true;
        let di = nodes.depth[i] as usize;
        alloc::push(next, head[di], WALK)?;
        head[di] = 0;
        alloc::push(touched, row, WALK)?;
        let mut both_twins = 0usize;

        for k in (0..=di).rev() {
            let mut pos = head[k];
            while pos != NIL {
                let a = touched[pos as usize] as usize;
                let t_a = t[a];
                for (p, row_parent) in [
                    (nodes.mother[a], nodes.row_mother),
                    (nodes.father[a], nodes.row_father),
                ] {
                    let p = if TWINS && p < -1 {
                        let r = row_parent[a] as usize;
                        if !reached[r] {
                            reached[r] = true;
                            alloc::push(marked, r as u32, WALK)?;
                            both_twins += usize::from(reached[nodes.twin[r] as usize]);
                        }
                        genome_parent(p)
                    } else {
                        p
                    };
                    if p < 0 {
                        continue;
                    }
                    let p = p as usize;
                    if !in_frontier[p] {
                        in_frontier[p] = true;
                        t[p] = 0.0;
                        let dp = nodes.depth[p] as usize;
                        alloc::push(next, head[dp], WALK)?;
                        head[dp] = touched.len() as u32;
                        alloc::push(touched, p as u32, WALK)?;
                    }
                    t[p] += 0.5 * t_a;
                }
                pos = next[pos as usize];
            }
            head[k] = NIL;
        }

        let mut sum = 0.0f64;
        for &j in touched.iter() {
            let tj = t[j as usize];
            sum += tj * tj * d_var[j as usize];
        }
        let walked = Walked {
            row,
            f: sum - 1.0,
            ancestors: (touched.len() - 1 + both_twins) as i32,
        };

        for &j in touched.iter() {
            t[j as usize] = 0.0;
            in_frontier[j as usize] = false;
        }
        for &r in marked.iter() {
            reached[r as usize] = false;
        }
        touched.clear();
        next.clear();
        marked.clear();
        Ok(walked)
    }
}

/// Flag each genome parent whose graph-row parent has an MZ co-twin by
/// storing node `p` as `-2 - p`, below the `-1` of a missing parent.
fn flag_co_twin_parents(genome: &mut [i32], rows: &[i32], twin: &[i32]) {
    for (p, &r) in genome.iter_mut().zip(rows) {
        if r >= 0 && twin[r as usize] >= 0 {
            *p = -2 - *p;
        }
    }
}

/// A genome parent with its co-twin flag removed: the node, or `-1`.
fn genome_parent(p: i32) -> i32 {
    if p < -1 {
        -2 - p
    } else {
        p
    }
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
        check_twins(ped.twin(), ped.depth())?;
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

/// Co-twins are one node, swept at one depth: each names the other, and
/// they share a depth.  Graph construction refuses a non-reciprocal pair,
/// and structural depth always puts co-twins at one depth, since they share
/// their parents; a raised depth handed to the bindings need not.
fn check_twins(twin: &[i32], depth: &[i32]) -> Result<(), Error> {
    for (row, &t) in twin.iter().enumerate() {
        if t < 0 {
            continue;
        }
        let partner = t as usize;
        if twin[partner] != row as i32 {
            return Err(Error::ValueOutOfRange {
                field: "twin",
                position: partner.into(),
                value: i64::from(twin[partner]),
                minimum: row as i64,
                maximum: row as i64,
            });
        }
        if depth[partner] != depth[row] {
            let partner = i64::from(depth[partner]);
            return Err(Error::ValueOutOfRange {
                field: "depth",
                position: row.into(),
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

    /// The 0.13 serial walk, row by row in ascending rows within a depth,
    /// every node walked: the bits every sweep order and pool size must give.
    fn reference(ped: KinshipPedigree<'_>) -> Result<(Vec<f64>, Vec<i32>), Error> {
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

    fn walk_in(threads: usize, ped: KinshipPedigree<'_>) -> (Vec<f64>, Vec<i32>) {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        inbreeding_and_ancestor_counts_in(ped, &pool).unwrap()
    }

    fn bits(f: &[f64]) -> Vec<u64> {
        f.iter().map(|v| v.to_bits()).collect()
    }

    /// `F` and the counts are the reference walk's bits on one and four
    /// threads.
    fn assert_matches_reference(ped: KinshipPedigree<'_>) {
        let (want_f, want_counts) = reference(ped).unwrap();
        for threads in [1, 4] {
            let (f, counts) = walk_in(threads, ped);
            assert_eq!(bits(&f), bits(&want_f), "{threads} threads");
            assert_eq!(counts, want_counts, "{threads} threads");
        }
    }

    /// Generations of sibships whose parents are drawn from every earlier
    /// row, so depths mix; most open with MZ co-twins, and the rows are
    /// shuffled so that sibs stand apart.
    fn twin_sibships(seed: u64) -> Cols {
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        let mut next = move |below: usize| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state % below as u64) as usize
        };
        let mut rows: Vec<(i32, i32)> = vec![(-1, -1); 12];
        let mut twins = Vec::new();
        for _ in 0..6 {
            let earlier = rows.len();
            for _ in 0..25 {
                let m = next(earlier) as i32;
                let f = match next(earlier) as i32 {
                    f if f == m => -1,
                    f => f,
                };
                let first = rows.len();
                for _ in 0..1 + next(4) {
                    rows.push((m, f));
                }
                if rows.len() - first >= 2 && next(3) > 0 {
                    twins.push((first as i32, first as i32 + 1));
                }
            }
        }
        let n = rows.len();
        let mut perm: Vec<usize> = (0..n).collect();
        for k in (1..n).rev() {
            perm.swap(k, next(k + 1));
        }
        let mut place = vec![0i32; n];
        for (new, &old) in perm.iter().enumerate() {
            place[old] = new as i32;
        }
        let moved = |p: i32| if p < 0 { -1 } else { place[p as usize] };
        let shuffled: Vec<(i32, i32)> = perm
            .iter()
            .map(|&old| (moved(rows[old].0), moved(rows[old].1)))
            .collect();
        let twins: Vec<(i32, i32)> = twins.iter().map(|&(a, b)| (moved(a), moved(b))).collect();
        Cols::new(&shuffled, &twins)
    }

    #[test]
    fn children_of_co_twin_mothers_count_their_own_ancestor_rows() {
        // 2 and 3 are MZ; 6 = (2, 5) and 7 = (3, 5) share genome parents, but
        // 5 = (3, 4) makes 3 an ancestor of 6 through two rows of one node.
        let c = Cols::new(
            &[
                (-1, -1),
                (-1, -1),
                (0, 1),
                (0, 1),
                (-1, -1),
                (3, 4),
                (2, 5),
                (3, 5),
            ],
            &[(2, 3)],
        );
        assert_eq!(ancestor_counts(&c)[6..], [6, 5]);
        assert_matches_reference(c.ped());
    }

    #[test]
    fn shared_families_give_the_reference_bits() {
        for seed in [1, 7, 19, 42] {
            let c = twin_sibships(seed);
            assert!(c.twin.iter().any(|&t| t >= 0));
            assert_matches_reference(c.ped());
            let no_twins = vec![-1; c.twin.len()];
            assert_matches_reference(
                KinshipPedigree::try_new(&c.mother, &c.father, &no_twins, &c.depth).unwrap(),
            );
        }
        for seed in [3, 11] {
            let c = crate::relationships::testing::shuffled_sibships(seed);
            let depth = structural_depth(&c.mother, &c.father);
            assert_matches_reference(
                KinshipPedigree::try_new(&c.mother, &c.father, &c.twin, &depth).unwrap(),
            );
        }
    }

    #[test]
    fn a_family_split_across_raised_depths_gives_the_reference_bits() {
        let c = twin_sibships(5);
        // Odd genome nodes one depth deeper: still above both parents and
        // shared by co-twins, but full sibs no longer share a depth.
        let raised: Vec<i32> = (0..c.depth.len())
            .map(|row| 2 * c.depth[row] + (genome_node(&c.twin, row) % 2) as i32)
            .collect();
        assert_matches_reference(
            KinshipPedigree::try_new(&c.mother, &c.father, &c.twin, &raised).unwrap(),
        );
    }

    #[test]
    fn a_depth_wider_than_a_block_gives_the_reference_bits() {
        // Thirty founders, three blocks of their children, so families recur
        // across blocks, then grandchildren of sibs and of half sibs.
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let mut next = move |below: usize| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state % below as u64) as i32
        };
        let mut rows: Vec<(i32, i32)> = vec![(-1, -1); 30];
        for _ in 0..3 * BLOCK {
            let (m, f) = (next(15), 15 + next(15));
            rows.push((m, f));
        }
        for _ in 0..2000 {
            let (m, f) = (30 + next(3 * BLOCK), 30 + next(3 * BLOCK));
            rows.push((m, if m == f { -1 } else { f }));
        }
        rows[31] = rows[30];
        let c = Cols::new(&rows, &[(30, 31)]);
        assert_matches_reference(c.ped());
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
    fn a_co_twin_that_does_not_name_its_partner_is_refused() {
        // 5 names 4 and 6 names 5, but neither is named back.
        let (mother, father) = ([-1, -1, 0, 0, 2, -1, -1], [-1, -1, 1, 1, 3, -1, -1]);
        let twin = [-1, -1, -1, -1, -1, 4, 5];
        let depth = [0, 0, 1, 1, 2, 2, 2];
        let ped = KinshipPedigree::try_new(&mother, &father, &twin, &depth).unwrap();
        assert!(matches!(
            inbreeding(ped),
            Err(Error::ValueOutOfRange {
                field: "twin",
                position,
                value: -1,
                minimum: 5,
                maximum: 5,
            }) if position == 4.into()
        ));
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
                position,
                ..
            }) if position == 2.into()
        ));
    }
}
