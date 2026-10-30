//! Row-streaming relationship classification.
//!
//! Every relationship category in the reference matrix engine is a sparse
//! product whose row `i` depends only on row `i` of its left operand,
//! followed by a multiplicity predicate and a subtraction of closer
//! categories.  Row `i` therefore sees every path that decides pair
//! `(i, j)`, and the pairs of the whole pedigree can be classified one row at
//! a time.  The only global state is the parent matrix `A`, its transpose,
//! and the sibling index, all linear in the pedigree size.
//!
//! A pair is unordered.  A product can be asymmetric, and the reference
//! engine takes nonzeros of the product in both orientations before
//! canonicalising, so each row evaluates both `M[i, :]` and `M^T[i, :]`.  The
//! transposed row of `X @ Y^T` is `Y[i, :] @ X^T`, and every operand chain
//! here reduces to expansions upward through `A` and downward through `A^T`.
//! Each unordered pair is then counted once, at its lower row.
//!
//! The category definitions are ADR 0018's: a collateral category is full
//! when at least two paths of its shape join the pair, summed over both
//! orientations, and half when exactly one does.  A sib link carries one
//! path per shared parent id.  The first cousin sibling exclusion is "shares
//! a known parent id", which is wider than the sibling lists the collateral
//! categories subtract.  Siblings include MZ co-twins (issue #29, ADR 0010 as
//! amended).
//! [`EXCLUSIONS`] is the per-category subtraction table; it is part of each
//! category's definition.
//!
//! Reporting is a separate step (ADR 0010 as amended, `GLOSSARY.md` "Closest
//! category"): after every set of a row is final, each category loses the
//! members an earlier registry category already claims, so a pair is counted
//! under its closest category only, exactly as the Python `_fold_precedence`
//! does on whole blocks.

use super::category::{Category, CategorySet, Counts};
use super::csr::Csr;
use super::multiplicity::Mult;
use super::sets::{self, Accumulator, Weighted};
use super::sibling_index::SiblingIndex;
use super::{MaxDegree, Pedigree};
use crate::alloc::{self, Family};
use crate::error::Error;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

/// Closer categories subtracted from each category's raw candidates.
/// `MO, FO` together are the parent-child-by-row set (`PO`).
pub const EXCLUSIONS: [&[Category]; super::category::N_CATEGORIES] = {
    use Category::*;
    [
        &[],                                           // MZ
        &[],                                           // MO
        &[],                                           // FO
        &[],                                           // FS
        &[FS],                                         // MHS
        &[FS],                                         // PHS
        &[],                                           // GP
        &[MO, FO],                                     // Av
        &[],                                           // GGP
        &[MO, FO, GP],                                 // HAv
        &[MO, FO, GP, Av],                             // GAv
        &[],                                           // 1C: shares-a-parent-id rule, see `cousins`
        &[],                                           // GGGP
        &[MO, FO, GP, GGP, HAv],                       // HGAv
        &[MO, FO, GP, GGP, Av, GAv],                   // GGAv
        &[],                                           // H1C: as 1C
        &[MO, FO, GP, GGP, Av, GAv, FS, MHS, PHS, C1], // 1C1R
        &[],                                           // G3GP
        &[MO, FO, GP, GGP, GGGP, HAv, HGAv],           // HGGAv
        &[MO, FO, GP, GGP, GGGP, Av, GAv, GGAv],       // G3Av
        &[
            MO, FO, GP, GGP, GGGP, HAv, HGAv, FS, MHS, PHS, C1, H1C, C1R1,
        ], // H1C1R
        &[
            MO, FO, GP, GGP, GGGP, Av, GAv, GGAv, FS, MHS, PHS, C1, H1C, C1R1,
        ], // 1C2R
        &[], // 2C: shares-a-grandparent rule, see `cousins`
    ]
};

/// The deepest avuncular shape: `G3Av`, four hops from the niece up to the
/// aunt's sib.
const MAX_AVUNCULAR_DEPTH: usize = 4;

/// Distinct per [`Engine`] built in the process, so a workspace never reuses
/// one engine's sets in another.
static NEXT_ENGINE: AtomicU64 = AtomicU64::new(0);

/// Immutable engine state shared by every row.
pub struct Engine<'p> {
    /// This engine's place in [`NEXT_ENGINE`].
    id: u64,
    ped: Pedigree<'p>,
    up: Csr,
    down: Csr,
    sibs: SiblingIndex,
    max_degree: MaxDegree,
}

/// What the parent-determined part of a row's classification reads: its
/// parent rows and both original parent ids, and the engine, whose degree
/// decides which of those sets exist.  Full sibs share it.
///
/// Only a row with both parent ids known has one.  Then every row with the
/// same key is in its sibling family, so its half sibs are the same set and
/// the cousin rule's "shares a known parent id" removes the same rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FamilyKey {
    engine: u64,
    mother: i32,
    father: i32,
    orig_mother: i64,
    orig_father: i64,
}

/// Per-thread scratch: the accumulator plus every per-row set, reused across rows.
///
/// A workspace also keeps the parent-determined sets of the last row it
/// classified, and the next row of the same family reuses them, so it
/// serves the one [`Engine`] its pool was built for.
pub struct Workspace {
    acc: Accumulator,
    /// Ancestors at exactly k hops (index k, 0 unused), with path multiplicity.
    up: Vec<Weighted>,
    /// Descendants at exactly k hops (index k, 0 unused), support only.
    down: Vec<Vec<u32>>,
    /// Final symmetric relative set per category for the current row.
    sets: Vec<Vec<u32>>,
    /// Pairs sharing at least one grandparent (any path), for the 2C rule.
    shares_grandparent: Vec<u32>,
    /// Per avuncular depth (index k, 0 unused): the sibs of the row's k-hop
    /// ancestors, with the paths through them.  The row is the niece or
    /// nephew.  Depends only on the parents.
    niece: Vec<Weighted>,
    /// Per avuncular depth: the descendants k hops below the row's half
    /// sibs, with path multiplicity.  The same for every full sib.
    half_aunt: Vec<Weighted>,
    /// Per avuncular depth: the descendants k hops below the row's full
    /// sibs, with path multiplicity.
    full_aunt: Vec<Weighted>,
    /// Per avuncular depth: the current row's paths of that shape, both
    /// orientations summed.
    avuncular: Vec<Weighted>,
    scratch: [Vec<u32>; 4],
    weighted: [Weighted; 3],
    /// Per asymmetric category, the sorted members for whom the current row
    /// carries `first_role`; captured after multiplicity filtering and before
    /// the two arms of a product are unioned.  The avuncular and
    /// removed-cousin arms are filled on every workspace, for the next sib
    /// to reuse; the others only when `orient`.
    first_arm: Vec<Vec<u32>>,
    /// Whether this workspace records orientation, which only pair emission needs.
    orient: bool,
    /// The family whose parent-determined sets the workspace holds: `up`
    /// beyond the parents, `shares_grandparent`, `shared`, and the
    /// collateral and removed-cousin arms of `first_arm`.  `None` while a
    /// row is being classified, so a row that fails leaves nothing to reuse.
    family: Option<FamilyKey>,
    /// Per category, the candidates that depend only on the row's family,
    /// before the row itself and closer categories are removed.
    shared: Vec<Vec<u32>>,
}

impl Workspace {
    fn build(n: usize, orient: bool) -> Result<Workspace, Error> {
        Ok(Workspace {
            acc: Accumulator::new(n)?,
            up: vec![Vec::new(); MaxDegree::MAX.get() as usize + 1],
            down: vec![Vec::new(); MaxDegree::MAX.get() as usize + 1],
            sets: vec![Vec::new(); super::category::N_CATEGORIES],
            shares_grandparent: Vec::new(),
            niece: vec![Vec::new(); MAX_AVUNCULAR_DEPTH + 1],
            half_aunt: vec![Vec::new(); MAX_AVUNCULAR_DEPTH + 1],
            full_aunt: vec![Vec::new(); MAX_AVUNCULAR_DEPTH + 1],
            avuncular: vec![Vec::new(); MAX_AVUNCULAR_DEPTH + 1],
            scratch: Default::default(),
            weighted: Default::default(),
            first_arm: vec![Vec::new(); super::category::N_CATEGORIES],
            orient,
            family: None,
            shared: vec![Vec::new(); super::category::N_CATEGORIES],
        })
    }
}

/// The workspaces the tasks of one sweep share, never more than there are
/// threads: a task takes one and gives it back, and a task that finds none
/// free builds its own.
///
/// Counting and emission both use it; `orient` is what a fresh workspace is
/// built for, and is the only difference between the two.
pub struct WorkspacePool {
    n: usize,
    orient: bool,
    free: Mutex<Vec<Workspace>>,
}

impl WorkspacePool {
    /// A pool for counting; its workspaces never record orientation.
    pub fn for_counts(n: usize) -> WorkspacePool {
        WorkspacePool::build(n, false)
    }

    /// A pool for pair emission, whose workspaces also record orientation.
    pub fn for_pairs(n: usize) -> WorkspacePool {
        WorkspacePool::build(n, true)
    }

    fn build(n: usize, orient: bool) -> WorkspacePool {
        WorkspacePool {
            n,
            orient,
            free: Mutex::new(Vec::new()),
        }
    }

    /// A free workspace, or a new one.
    ///
    /// A workspace handed back forgets the family it last classified, so a
    /// pool shared by several engines never reuses one engine's sets in
    /// another.
    ///
    /// # Errors
    ///
    /// [`Error::AllocationFailed`] when a new workspace cannot be built.
    pub fn take(&self) -> Result<Workspace, Error> {
        match self.free.lock().unwrap_or_else(|e| e.into_inner()).pop() {
            Some(mut ws) => {
                ws.family = None;
                Ok(ws)
            }
            None => Workspace::build(self.n, self.orient),
        }
    }

    /// Return a workspace for another task to reuse.
    pub fn give(&self, ws: Workspace) {
        self.free.lock().unwrap_or_else(|e| e.into_inner()).push(ws);
    }
}

impl<'p> Engine<'p> {
    /// Build the shared state: the parent CSR, its transpose, and the sibling index.
    ///
    /// # Errors
    ///
    /// [`Error::AllocationFailed`] for any of those linear buffers.
    pub fn new(ped: &Pedigree<'p>, max_degree: MaxDegree) -> Result<Engine<'p>, Error> {
        let n = ped.len();
        let edges: Vec<(u32, u32)> = alloc::collect(
            (0..n).flat_map(|i| {
                [ped.mother[i], ped.father[i]]
                    .into_iter()
                    .filter(|&p| p >= 0)
                    .map(move |p| (i as u32, p as u32))
            }),
            Family::ParentEdges,
        )?;
        let up = Csr::from_edges(n, edges)?;
        let down = up.transpose()?;
        let sibs = SiblingIndex::build(ped.orig_mother, ped.orig_father)?;
        Ok(Engine {
            id: NEXT_ENGINE.fetch_add(1, Ordering::Relaxed),
            ped: *ped,
            up,
            down,
            sibs,
            max_degree,
        })
    }

    pub fn len(&self) -> usize {
        self.ped.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ped.len() == 0
    }

    /// The rows with every sibling family contiguous, for a walk whose
    /// result does not depend on row order: each workspace then reuses the
    /// parent-determined sets of one sib for the next.  `None` when the rows
    /// are in that order already, below degree 2, where a sib shares
    /// nothing but its parents, or when the order cannot be allocated: it
    /// only saves time, so the walk then keeps row order.
    pub fn family_order(&self) -> Option<Vec<u32>> {
        if self.max_degree.get() < 2 {
            return None;
        }
        self.sibs.family_order().ok().flatten()
    }

    /// Classify every pair involving `row`, fold precedence, then add the
    /// pairs `row` owns to `counts`.
    ///
    /// Every category is unordered here and a pair belongs to its lower row.
    /// With `selected`, only members that are themselves selected count.
    pub fn count_row(
        &self,
        row: usize,
        selected: Option<&[bool]>,
        ws: &mut Workspace,
        counts: &mut Counts,
    ) -> Result<(), Error> {
        self.classify_row(row, ws)?;
        self.fold_row(ws);
        for cat in Category::ALL {
            if cat.degree() > self.max_degree.get() {
                break;
            }
            let set = &ws.sets[cat.index()];
            let owned = match selected {
                None => sets::count_above(set, row),
                Some(mask) => set
                    .iter()
                    .filter(|&&j| j as usize > row && mask[j as usize])
                    .count() as u64,
            };
            counts.add(cat, owned);
        }
        Ok(())
    }

    /// Classify `row`, fold precedence, and hand every pair `row` owns in a
    /// requested category to `sink` in the category's semantic orientation.
    ///
    /// The owner is the lower graph row, so for a symmetric category the pair
    /// arrives as `(row, j)`; for an asymmetric one `row` is `first` exactly
    /// when `j` lies in the row's first arm.  A pair valid in both
    /// orientations is in both arms, hence in the first arm, which reproduces
    /// the Python lower-row tie break.  With `view`, the int32 view row of
    /// every graph row (`-1` unselected), a pair is emitted only when both
    /// rows are selected, relabelled, and a symmetric pair reordered to
    /// `first < second` in view rows.  Without a view, the pairs of one row
    /// arrive in canonical-key order.
    pub fn emit_row(
        &self,
        row: usize,
        requested: &CategorySet,
        view: Option<&[i32]>,
        ws: &mut Workspace,
        mut sink: impl FnMut(Category, u32, u32) -> Result<(), Error>,
    ) -> Result<(), Error> {
        assert!(
            ws.orient,
            "pair emission needs a workspace from WorkspacePool::for_pairs"
        );
        self.classify_row(row, ws)?;
        self.fold_row(ws);
        let r = row as u32;
        for cat in Category::ALL {
            if cat.degree() > self.max_degree.get() {
                break;
            }
            if !requested.contains(cat) {
                continue;
            }
            let set = &ws.sets[cat.index()];
            let arm = &ws.first_arm[cat.index()];
            let symmetric = cat.symmetric();
            for &j in &set[set.partition_point(|&j| j <= r)..] {
                let (mut a, mut b) = if symmetric || arm.binary_search(&j).is_ok() {
                    (r, j)
                } else {
                    (j, r)
                };
                if let Some(map) = view {
                    let (va, vb) = (map[a as usize], map[b as usize]);
                    if va < 0 || vb < 0 {
                        continue;
                    }
                    (a, b) = (va as u32, vb as u32);
                    if symmetric && a > b {
                        std::mem::swap(&mut a, &mut b);
                    }
                }
                sink(cat, a, b)?;
            }
        }
        Ok(())
    }

    /// Keep each member of `ws.sets` only in its closest category.
    ///
    /// Categories are visited in registry order (degree ascending, then
    /// precedence); each loses the members every earlier category claims.
    pub fn fold_row(&self, ws: &mut Workspace) {
        let computed = Category::ALL
            .iter()
            .take_while(|cat| cat.degree() <= self.max_degree.get())
            .count();
        ws.acc.claim_in_order(ws.sets[..computed].iter_mut());
    }

    /// The family key of `row`, when both its parent ids are known.
    fn family_key(&self, row: usize) -> Option<FamilyKey> {
        let ped = &self.ped;
        (ped.orig_mother[row] >= 0 && ped.orig_father[row] >= 0).then(|| FamilyKey {
            engine: self.id,
            mother: ped.mother[row],
            father: ped.father[row],
            orig_mother: ped.orig_mother[row],
            orig_father: ped.orig_father[row],
        })
    }

    /// Fill `ws.sets` with the final symmetric relative set of `row` for every category.
    ///
    /// When the workspace's last row was a full sib of `row`, the sets that
    /// depend only on the parents are reused rather than recomputed: the
    /// ancestors beyond the parents, the siblings of those ancestors, the
    /// half-sib collaterals, and every cousin product.
    pub fn classify_row(&self, row: usize, ws: &mut Workspace) -> Result<(), Error> {
        let key = self.family_key(row);
        let reuse = key.is_some() && ws.family == key;
        ws.family = None;
        self.classify(row, ws, reuse)?;
        ws.family = key;
        Ok(())
    }

    fn classify(&self, row: usize, ws: &mut Workspace, reuse: bool) -> Result<(), Error> {
        use Category::*;
        let deg = self.max_degree.get();
        for s in &mut ws.sets {
            s.clear();
        }

        // Lineal expansions: up with path multiplicity, down as support.
        if !reuse {
            let (parents, mults) = self.up.row(row);
            ws.up[1].clear();
            alloc::extend(
                &mut ws.up[1],
                parents.iter().copied().zip(mults.iter().copied()),
                Family::RowSet,
            )?;
            for k in 2..=deg.max(1) as usize {
                let (below, level) = ws.up.split_at_mut(k);
                ws.acc.hop(&self.up, &below[k - 1], &mut level[0])?;
            }
        }
        ws.acc
            .hop_support(&self.down, &[row as u32], &mut ws.down[1])?;
        for k in 2..=deg.max(1) as usize {
            let (below, level) = ws.down.split_at_mut(k);
            ws.acc
                .hop_support(&self.down, &below[k - 1], &mut level[0])?;
        }

        // Degree 0 and 1: the co-twin, each parent role by row in both
        // orientations, sibs by original parent id.
        if self.ped.twin[row] >= 0 {
            ws.sets[MZ.index()].push(self.ped.twin[row] as u32);
        }
        if deg < 1 {
            return Ok(());
        }
        let ped = &self.ped;
        let [mut mo, mut fo] = [MO, FO].map(|c| std::mem::take(&mut ws.sets[c.index()]));
        let roles = parent_role(row, ped.mother, &ws.down[1], &mut mo)
            .and_then(|()| parent_role(row, ped.father, &ws.down[1], &mut fo));
        ws.sets[MO.index()] = mo;
        ws.sets[FO.index()] = fo;
        roles?;
        if ws.orient {
            // The row is the offspring, hence `first`, towards its own parent.
            ws.first_arm[MO.index()] = parent_arm(ped.mother[row]);
            ws.first_arm[FO.index()] = parent_arm(ped.father[row]);
        }
        self.sibs.full_sibs(row, &mut ws.sets[FS.index()])?;
        self.sibs
            .maternal_half_sibs(row, &mut ws.sets[MHS.index()])?;
        self.sibs
            .paternal_half_sibs(row, &mut ws.sets[PHS.index()])?;
        if deg < 2 {
            return Ok(());
        }

        // Degree 2: GP is lineal (up[2] ∪ down[2] as an exclusion set); Av.
        // Each avuncular depth's paths are counted at its full category's
        // degree and read again at its half category's, one degree on.
        let full = |m: Mult| m.at_least_two();
        let half = |m: Mult| m.is_one();
        lineal(ws, GP, 2)?;
        self.avuncular_paths(row, ws, 1, reuse)?;
        self.collateral(row, ws, Av, 1, full)?;
        if deg < 3 {
            return Ok(());
        }

        // Degree 3.
        lineal(ws, GGP, 3)?;
        self.collateral(row, ws, HAv, 1, half)?;
        self.avuncular_paths(row, ws, 2, reuse)?;
        self.collateral(row, ws, GAv, 2, full)?;
        self.cousins(row, ws, reuse)?;
        if deg < 4 {
            return Ok(());
        }

        // Degree 4.
        lineal(ws, GGGP, 4)?;
        self.collateral(row, ws, HGAv, 2, half)?;
        self.avuncular_paths(row, ws, 3, reuse)?;
        self.collateral(row, ws, GGAv, 3, full)?;
        self.removed_cousins(row, ws, 2, 3, C1R1, reuse, full)?;
        if deg < 5 {
            return Ok(());
        }

        // Degree 5.
        lineal(ws, G3GP, 5)?;
        self.collateral(row, ws, HGGAv, 3, half)?;
        self.avuncular_paths(row, ws, 4, reuse)?;
        self.collateral(row, ws, G3Av, 4, full)?;
        self.removed_cousins(row, ws, 2, 3, H1C1R, reuse, half)?;
        self.removed_cousins(row, ws, 2, 4, C1R2, reuse, full)?;
        self.second_cousins(row, ws, reuse)
    }

    /// The avuncular family's paths at `depth` (1 for Av and HAv, 2 for GAv
    /// and HGAv, and so on), into `ws.avuncular[depth]`.
    ///
    /// The row is the niece or nephew of each sib of its `depth`-hop
    /// ancestors, and the aunt or uncle of the descendants `depth` hops below
    /// its own sibs.  A sib link carries one path per parent id the two
    /// share, so two for full sibs, and the two orientations are summed
    /// (ADR 0018).  Depths are visited in order, each sib arm one hop below
    /// the last.  The niece arm depends only on the parents and the half sib
    /// arm only on the family, so `reuse` keeps both from the last row.
    fn avuncular_paths(
        &self,
        row: usize,
        ws: &mut Workspace,
        depth: usize,
        reuse: bool,
    ) -> Result<(), Error> {
        let [sib, scratch, ..] = &mut ws.scratch;
        let [terms, tmp, _] = &mut ws.weighted;
        if !reuse {
            // One path per shared parent id: each id group of the ancestor.
            terms.clear();
            for &(a, m) in &ws.up[depth] {
                for group in self.sibs.by_parent_id(a as usize) {
                    let others = group.iter().filter(|&&j| j != a);
                    alloc::extend(terms, others.map(|&j| (j, m)), Family::RowSet)?;
                }
            }
            ws.acc.sum(terms.iter().copied(), &mut ws.niece[depth])?;
            if depth == 1 {
                self.sibs.half_sibs(row, scratch, sib)?;
                weighted_rows(sib, Mult::ONE, terms)?;
            }
            sibs_below(&self.down, &mut ws.acc, &mut ws.half_aunt, depth, terms)?;
        }
        if depth == 1 {
            self.sibs.full_sibs(row, sib)?;
            weighted_rows(sib, Mult::TWO, terms)?;
        }
        sibs_below(&self.down, &mut ws.acc, &mut ws.full_aunt, depth, terms)?;
        sets::sum_into(&ws.niece[depth], &ws.half_aunt[depth], tmp)?;
        sets::sum_into(tmp, &ws.full_aunt[depth], &mut ws.avuncular[depth])
    }

    /// One avuncular category: the members of `ws.avuncular[depth]` whose
    /// path count `keep` accepts, minus the row and closer categories.
    fn collateral(
        &self,
        row: usize,
        ws: &mut Workspace,
        cat: Category,
        depth: usize,
        keep: impl Fn(Mult) -> bool,
    ) -> Result<(), Error> {
        let result = &mut ws.scratch[0];
        sets::select_into(&ws.avuncular[depth], keep, result)?;
        if ws.orient {
            // The row is the niece or nephew of every member the niece arm
            // reaches, including a pair joined in both orientations.
            sets::support_into(&ws.niece[depth], &mut ws.first_arm[cat.index()])?;
        }
        sets::drop_self(result, row);
        self.finalize(cat, result, &ws.sets);
        std::mem::swap(&mut ws.sets[cat.index()], result);
        Ok(())
    }

    /// 1C, H1C, and the shared-grandparent support used by 2C.
    ///
    /// 1C and H1C threshold the paths through shared grandparents,
    /// `A^2 @ (A^2)^T`.  All three depend only on the row's grandparents
    /// and parent ids, so `reuse` keeps them from the last row.  The support keeps the row
    /// itself, which shares every one of its grandparents, so that it is
    /// the same for every full sib; 2C drops the row before subtracting it.
    fn cousins(&self, row: usize, ws: &mut Workspace, reuse: bool) -> Result<(), Error> {
        use Category::*;
        if !reuse {
            let ped = &self.ped;
            let [counted, _, tmp] = &mut ws.weighted;
            self.chain_down(&mut ws.acc, &ws.up[2], 2, counted, tmp)?;
            sets::support_into(counted, &mut ws.shares_grandparent)?;
            // The row and every full sib of it share a known parent id, so
            // the same rows leave whichever sib computes the sets.
            let shares_parent_id = |j: u32| {
                let j = j as usize;
                (ped.orig_mother[row] >= 0 && ped.orig_mother[row] == ped.orig_mother[j])
                    || (ped.orig_father[row] >= 0 && ped.orig_father[row] == ped.orig_father[j])
            };
            counted.retain(|&(j, _)| j as usize != row && !shares_parent_id(j));
            sets::select_into(counted, |m| m.at_least_two(), &mut ws.shared[C1.index()])?;
            sets::select_into(counted, |m| m.is_one(), &mut ws.shared[H1C.index()])?;
        }
        for cat in [C1, H1C] {
            sets::copy_into(&mut ws.sets[cat.index()], &ws.shared[cat.index()])?;
        }
        Ok(())
    }

    /// `A^a @ (A^b)^T` plus its transpose, with path multiplicity, thresholded by `keep`.
    ///
    /// The two orientations of a pair are summed before the threshold
    /// (ADR 0018).  Both products start from the row's ancestors, so `reuse`
    /// keeps them from the last row.
    #[allow(clippy::too_many_arguments)]
    fn removed_cousins(
        &self,
        row: usize,
        ws: &mut Workspace,
        a: usize,
        b: usize,
        cat: Category,
        reuse: bool,
        keep: impl Fn(Mult) -> bool,
    ) -> Result<(), Error> {
        if !reuse {
            let [forward, backward, tmp] = &mut ws.weighted;
            self.chain_down(&mut ws.acc, &ws.up[a], b, forward, tmp)?;
            self.chain_down(&mut ws.acc, &ws.up[b], a, backward, tmp)?;
            // Through `backward` the row sits `b > a` meioses from the shared
            // ancestor: it is the junior cousin, which the registry puts first.
            // A pair joined both ways is in this arm too.  Kept whether or not
            // orientation is wanted, for the next sib to reuse; the counting
            // path never reads it.
            sets::support_into(backward, &mut ws.first_arm[cat.index()])?;
            sets::sum_into(forward, backward, tmp)?;
            sets::select_into(tmp, &keep, &mut ws.shared[cat.index()])?;
        }
        let set = &mut ws.sets[cat.index()];
        sets::copy_into(set, &ws.shared[cat.index()])?;
        sets::drop_self(set, row);
        let mut result = std::mem::take(set);
        self.finalize(cat, &mut result, &ws.sets);
        ws.sets[cat.index()] = result;
        Ok(())
    }

    /// `A^3 @ (A^3)^T >= 2`, minus pairs sharing any grandparent.
    fn second_cousins(&self, row: usize, ws: &mut Workspace, reuse: bool) -> Result<(), Error> {
        let cat = Category::C2;
        if !reuse {
            let [product, _, tmp] = &mut ws.weighted;
            self.chain_down(&mut ws.acc, &ws.up[3], 3, product, tmp)?;
            sets::select_into(product, |m| m.at_least_two(), &mut ws.shared[cat.index()])?;
        }
        let set = &mut ws.sets[cat.index()];
        sets::copy_into(set, &ws.shared[cat.index()])?;
        sets::drop_self(set, row);
        sets::subtract(set, &ws.shares_grandparent);
        Ok(())
    }

    /// `src @ (A^T)^k` with saturated path multiplicity.
    fn chain_down(
        &self,
        acc: &mut Accumulator,
        src: &Weighted,
        k: usize,
        out: &mut Weighted,
        tmp: &mut Weighted,
    ) -> Result<(), Error> {
        out.clear();
        alloc::reserve(out, src.len(), Family::RowSet)?;
        out.extend_from_slice(src);
        for _ in 0..k {
            acc.hop(&self.down, out, tmp)?;
            std::mem::swap(out, tmp);
        }
        Ok(())
    }

    fn finalize(&self, cat: Category, candidates: &mut Vec<u32>, sets: &[Vec<u32>]) {
        for &closer in EXCLUSIONS[cat.index()] {
            sets::subtract(candidates, &sets[closer.index()]);
        }
    }
}

/// `levels[depth]`: one hop down from `sibs` at depth 1, else from `levels[depth - 1]`.
fn sibs_below(
    down: &Csr,
    acc: &mut Accumulator,
    levels: &mut [Weighted],
    depth: usize,
    sibs: &Weighted,
) -> Result<(), Error> {
    let (above, level) = levels.split_at_mut(depth);
    let src = if depth == 1 { sibs } else { &above[depth - 1] };
    acc.hop(down, src, &mut level[0])
}

/// `rows`, each with multiplicity `m`, into a buffer the caller reuses.
fn weighted_rows(rows: &[u32], m: Mult, out: &mut Weighted) -> Result<(), Error> {
    out.clear();
    alloc::extend(out, rows.iter().map(|&j| (j, m)), Family::RowSet)
}

/// Lineal pairs in both orientations: ancestors at k hops and descendants at
/// k hops.  Towards an ancestor the row is the descendant, hence `first`.
fn lineal(ws: &mut Workspace, cat: Category, k: usize) -> Result<(), Error> {
    let mut set = std::mem::take(&mut ws.sets[cat.index()]);
    let built = (|| {
        sets::support_into(&ws.up[k], &mut set)?;
        if ws.orient {
            // The up arm alone: towards an ancestor the row is the descendant.
            sets::copy_into(&mut ws.first_arm[cat.index()], &set)?;
        }
        sets::union_into(&mut set, &ws.down[k])
    })();
    ws.sets[cat.index()] = set;
    built
}

/// The first arm of a parent role: the row's own parent, if any.
fn parent_arm(parent: i32) -> Vec<u32> {
    if parent >= 0 {
        vec![parent as u32]
    } else {
        Vec::new()
    }
}

/// One parent role in both orientations: `row`'s parent of that role, and
/// the children (`down1`, sorted) for whom `row` fills that role.
///
/// Fills `set` in place, so the category keeps one buffer across rows.
fn parent_role(row: usize, parent: &[i32], down1: &[u32], set: &mut Vec<u32>) -> Result<(), Error> {
    set.clear();
    alloc::extend(
        set,
        down1
            .iter()
            .copied()
            .filter(|&j| parent[j as usize] == row as i32),
        Family::RowSet,
    )?;
    if parent[row] >= 0 {
        sets::union_into(set, &[parent[row] as u32])?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::testing::{pedigree, random_pedigree, shuffled_sibships};
    use super::super::{count_pairs, Category, CategorySet, Counts, MaxDegree, Progress};
    use super::{Engine, Workspace};

    /// Every pair `row` owns, as `emit_row` hands them out.
    fn emitted(engine: &Engine, row: usize, ws: &mut Workspace) -> Vec<(Category, u32, u32)> {
        let mut pairs = Vec::new();
        engine
            .emit_row(
                row,
                &CategorySet::up_to_degree(MaxDegree::MAX.get()),
                None,
                ws,
                |cat, a, b| {
                    pairs.push((cat, a, b));
                    Ok(())
                },
            )
            .unwrap();
        pairs
    }

    /// The pairs `row` owns per category, as `count_row` adds them.
    fn counted(engine: &Engine, row: usize, ws: &mut Workspace) -> Counts {
        let mut counts = Counts::default();
        engine.count_row(row, None, ws, &mut counts).unwrap();
        counts
    }

    #[test]
    fn a_sib_reusing_the_family_sets_classifies_as_a_fresh_workspace_does() {
        for (cols, family_rows) in [
            (shuffled_sibships(3), true),
            (shuffled_sibships(8), true),
            (random_pedigree(300, 5), false),
        ] {
            let ped = cols.try_borrow().unwrap();
            let engine = Engine::new(&ped, MaxDegree::MAX).unwrap();
            let order = engine
                .family_order()
                .unwrap_or_else(|| (0..ped.len() as u32).collect());
            let mut sorted = order.clone();
            sorted.sort_unstable();
            assert!(sorted.iter().copied().eq(0..ped.len() as u32));
            // Sorted by parent rows, rows of different families with the
            // same parent rows meet, and those must not reuse each other.
            let mut by_parent_rows: Vec<u32> = (0..ped.len() as u32).collect();
            by_parent_rows.sort_by_key(|&r| (ped.mother[r as usize], ped.father[r as usize]));
            for walk in [&order, &by_parent_rows] {
                let mut shared = Workspace::build(ped.len(), true).unwrap();
                let mut counting = Workspace::build(ped.len(), false).unwrap();
                let mut reused = 0;
                for &row in walk {
                    let row = row as usize;
                    reused += usize::from(
                        engine.family_key(row).is_some() && shared.family == engine.family_key(row),
                    );
                    let mut fresh = Workspace::build(ped.len(), true).unwrap();
                    assert_eq!(
                        emitted(&engine, row, &mut shared),
                        emitted(&engine, row, &mut fresh),
                        "row {row}"
                    );
                    // The counting path fills no orientation arms of its own.
                    let mut fresh = Workspace::build(ped.len(), false).unwrap();
                    assert_eq!(
                        counted(&engine, row, &mut counting),
                        counted(&engine, row, &mut fresh),
                        "row {row}"
                    );
                }
                if family_rows {
                    assert!(
                        reused > ped.len() / 3,
                        "{reused} of {} rows reused",
                        ped.len()
                    );
                }
            }
        }
    }

    #[test]
    fn a_workspace_moved_between_engines_reuses_nothing() {
        let cols = shuffled_sibships(3);
        let ped = cols.try_borrow().unwrap();
        let shallow = Engine::new(&ped, MaxDegree::try_new(2).unwrap()).unwrap();
        let deep = Engine::new(&ped, MaxDegree::MAX).unwrap();
        let order = deep.family_order().unwrap();
        let mut shared = Workspace::build(ped.len(), true).unwrap();
        for pair in order.chunks(2) {
            // Each sib after the first meets the other engine's family sets.
            for (engine, &row) in [&shallow, &deep].into_iter().zip(pair) {
                let row = row as usize;
                let mut fresh = Workspace::build(ped.len(), true).unwrap();
                assert_eq!(
                    emitted(engine, row, &mut shared),
                    emitted(engine, row, &mut fresh),
                    "row {row}"
                );
            }
        }
    }

    #[test]
    fn contiguous_families_need_no_order_and_scattered_ones_are_gathered() {
        // Sibs 2 and 4 of parents 0 and 1, with 3 between them.
        let scattered = pedigree(&[(-1, -1), (-1, -1), (0, 1), (-1, -1), (0, 1)], &[]);
        let ped = scattered.try_borrow().unwrap();
        let order = Engine::new(&ped, MaxDegree::MAX).unwrap().family_order();
        assert_eq!(order, Some(vec![0, 1, 2, 4, 3]));
        let contiguous = pedigree(&[(-1, -1), (-1, -1), (0, 1), (0, 1), (-1, -1)], &[]);
        let ped = contiguous.try_borrow().unwrap();
        let order = Engine::new(&ped, MaxDegree::MAX).unwrap().family_order();
        assert_eq!(order, None);
    }

    #[test]
    fn counts_do_not_depend_on_row_order() {
        let shuffled = shuffled_sibships(3);
        let ped = shuffled.try_borrow().unwrap();
        let engine = Engine::new(&ped, MaxDegree::MAX).unwrap();
        let mut in_row_order = Counts::default();
        for row in 0..ped.len() {
            // A fresh workspace a row, so nothing is reused.
            let mut ws = Workspace::build(ped.len(), true).unwrap();
            for (cat, _, _) in emitted(&engine, row, &mut ws) {
                in_row_order.add(cat, 1);
            }
        }
        let got = count_pairs(&ped, MaxDegree::MAX, None, &Progress::default()).unwrap();
        assert_eq!(got, in_row_order);
        assert!(got.get(Category::C1) > 0 && got.get(Category::C2) > 0);
    }

    fn expect(pairs: &[(Category, u64)]) -> Counts {
        let mut counts = Counts::default();
        for &(cat, n) in pairs {
            counts.add(cat, n);
        }
        counts
    }

    #[test]
    fn nuclear_family_splits_parent_roles() {
        let ped = pedigree(&[(-1, -1), (-1, -1), (0, 1), (0, 1)], &[]);
        let got = count_pairs(
            &ped.try_borrow().unwrap(),
            MaxDegree::MAX,
            None,
            &Progress::default(),
        )
        .unwrap();
        assert_eq!(
            got,
            expect(&[(Category::MO, 2), (Category::FO, 2), (Category::FS, 1)])
        );
    }

    #[test]
    fn a_mother_who_is_also_a_grandmother_counts_once_as_mother() {
        // g(0) and h(1) have p(2); g and p have i(3).  Pair (g, i) is MO and
        // GP; pair (p, i) is FO and MHS through g.  The closest category wins.
        let ped = pedigree(&[(-1, -1), (-1, -1), (0, 1), (0, 2)], &[]);
        let got = count_pairs(
            &ped.try_borrow().unwrap(),
            MaxDegree::MAX,
            None,
            &Progress::default(),
        )
        .unwrap();
        assert_eq!(
            got,
            expect(&[(Category::MO, 2), (Category::FO, 2), (Category::GP, 1)])
        );
    }

    #[test]
    fn mz_co_twins_are_twins_not_sibs() {
        let ped = pedigree(&[(-1, -1), (-1, -1), (0, 1), (0, 1)], &[(2, 3)]);
        let got = count_pairs(
            &ped.try_borrow().unwrap(),
            MaxDegree::MAX,
            None,
            &Progress::default(),
        )
        .unwrap();
        assert_eq!(
            got,
            expect(&[(Category::MZ, 1), (Category::MO, 2), (Category::FO, 2)])
        );
    }

    /// Mother 0; fathers 1 and 2; co-twins 3, 4 and full sib 6 (0×1); half
    /// sib 5 (0×2); spouses 7, 8; child 9 of twin 3 (×7); child 10 of sib 6 (×8).
    const TWINS_WITH_SIBS: [(i32, i32); 11] = [
        (-1, -1),
        (-1, -1),
        (-1, -1),
        (0, 1),
        (0, 1),
        (0, 2),
        (0, 1),
        (-1, -1),
        (-1, -1),
        (3, 7),
        (8, 6),
    ];

    #[test]
    fn mz_co_twins_keep_their_sibs_and_collaterals() {
        use Category::*;
        let ped = pedigree(&TWINS_WITH_SIBS, &[(3, 4)]);
        let got = count_pairs(
            &ped.try_borrow().unwrap(),
            MaxDegree::try_new(3).unwrap(),
            None,
            &Progress::default(),
        )
        .unwrap();
        // FS (3,6), (4,6); MHS (3,5), (4,5), (5,6); Av (10,3), (9,4), (10,4),
        // (9,6); HAv (9,5), (10,5): issue #29.
        assert_eq!(
            got,
            expect(&[
                (MZ, 1),
                (MO, 6),
                (FO, 6),
                (FS, 2),
                (MHS, 3),
                (GP, 4),
                (Av, 4),
                (HAv, 2),
                (C1, 1),
            ])
        );
    }

    #[test]
    fn a_twin_link_moves_only_the_co_twin_pair_from_fs_to_mz() {
        let untwinned = pedigree(&TWINS_WITH_SIBS, &[]);
        let twinned = pedigree(&TWINS_WITH_SIBS, &[(3, 4)]);
        let base = count_pairs(
            &untwinned.try_borrow().unwrap(),
            MaxDegree::MAX,
            None,
            &Progress::default(),
        )
        .unwrap();
        let got = count_pairs(
            &twinned.try_borrow().unwrap(),
            MaxDegree::MAX,
            None,
            &Progress::default(),
        )
        .unwrap();
        for cat in Category::ALL {
            let delta = match cat {
                Category::MZ => 1,
                Category::FS => -1,
                _ => 0,
            };
            assert_eq!(
                got.get(cat) as i64,
                base.get(cat) as i64 + delta,
                "{}",
                cat.code()
            );
        }
    }

    #[test]
    fn degree_cutoff_stops_the_fold_and_the_count() {
        let ped = pedigree(&[(-1, -1), (-1, -1), (0, 1), (0, 2)], &[]);
        let got = count_pairs(
            &ped.try_borrow().unwrap(),
            MaxDegree::try_new(1).unwrap(),
            None,
            &Progress::default(),
        )
        .unwrap();
        assert_eq!(got, expect(&[(Category::MO, 2), (Category::FO, 2)]));
        assert_eq!(
            count_pairs(
                &ped.try_borrow().unwrap(),
                MaxDegree::try_new(0).unwrap(),
                None,
                &Progress::default()
            )
            .unwrap(),
            Counts::default()
        );
    }

    #[test]
    fn a_selection_counts_only_pairs_inside_it_but_classifies_through_everyone() {
        // Grandmother g(0), her child p(2) with h(1), and p's child c(3) with
        // k(4).  Selecting g and c keeps their GP pair although p is unselected.
        let ped = pedigree(&[(-1, -1), (-1, -1), (0, 1), (2, 4), (-1, -1)], &[]);
        let selected = [true, false, false, true, false];
        let got = count_pairs(
            &ped.try_borrow().unwrap(),
            MaxDegree::MAX,
            Some(&selected),
            &Progress::default(),
        )
        .unwrap();
        assert_eq!(got, expect(&[(Category::GP, 1)]));
        let all = count_pairs(
            &ped.try_borrow().unwrap(),
            MaxDegree::MAX,
            None,
            &Progress::default(),
        )
        .unwrap();
        assert_eq!(
            all,
            expect(&[(Category::MO, 2), (Category::FO, 2), (Category::GP, 2)])
        );
    }
}
