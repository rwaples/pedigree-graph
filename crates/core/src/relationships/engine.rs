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
//! The category definitions below reproduce the matrix engine of
//! `pedigree_graph/_pair_extractor.py` bit for bit, including its two
//! idiosyncrasies: first cousins count *distinct* shared grandparents while
//! the once/twice-removed cousins and second cousins count *paths*, and the
//! cousin sibling exclusion is "shares a known parent id", which is wider than
//! the twin-filtered sibling lists the collateral categories subtract.
//! [`EXCLUSIONS`] is the reference engine's per-category subtraction table;
//! it is part of each category's definition.
//!
//! Reporting is a separate step (ADR 0010 as amended, `CONTEXT.md` "Closest
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

/// Closer categories subtracted from each category's raw candidates
/// (`_pair_extractor.py` subtract lists).  `MO, FO` together are the
/// reference engine's parent-child-by-row set (`PO`).
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

/// Immutable engine state shared by every row.
pub struct Engine<'p> {
    ped: Pedigree<'p>,
    up: Csr,
    down: Csr,
    sibs: SiblingIndex,
    max_degree: MaxDegree,
}

/// Per-thread scratch: the accumulator plus every per-row set, reused across rows.
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
    /// Distinct grandchildren of each grandparent of the current row.
    grandchildren: Vec<Vec<u32>>,
    scratch: [Vec<u32>; 4],
    weighted: [Weighted; 3],
    /// Per asymmetric category, the sorted members for whom the current row
    /// carries `first_role`; captured after multiplicity filtering and before
    /// the two arms of a product are unioned.  Only filled when `orient`.
    first_arm: Vec<Vec<u32>>,
    /// Whether this workspace records orientation, which only pair emission needs.
    orient: bool,
}

impl Workspace {
    /// A workspace for counting; orientation is never recorded.
    pub fn new(n: usize) -> Result<Workspace, Error> {
        Workspace::build(n, false)
    }

    /// A workspace for pair emission, which also records orientation.
    pub fn for_pairs(n: usize) -> Result<Workspace, Error> {
        Workspace::build(n, true)
    }

    fn build(n: usize, orient: bool) -> Result<Workspace, Error> {
        Ok(Workspace {
            acc: Accumulator::new(n)?,
            up: vec![Vec::new(); MaxDegree::MAX.get() as usize + 1],
            down: vec![Vec::new(); MaxDegree::MAX.get() as usize + 1],
            sets: vec![Vec::new(); super::category::N_CATEGORIES],
            shares_grandparent: Vec::new(),
            grandchildren: Vec::new(),
            scratch: Default::default(),
            weighted: Default::default(),
            first_arm: vec![Vec::new(); super::category::N_CATEGORIES],
            orient,
        })
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
            "int32",
        )?;
        let up = Csr::from_edges(n, edges)?;
        let down = up.transpose()?;
        let sibs = SiblingIndex::build(ped.twin, ped.orig_mother, ped.orig_father)?;
        Ok(Engine {
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
        assert!(ws.orient, "pair emission needs Workspace::for_pairs");
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

    /// Fill `ws.sets` with the final symmetric relative set of `row` for every category.
    pub fn classify_row(&self, row: usize, ws: &mut Workspace) -> Result<(), Error> {
        use Category::*;
        let deg = self.max_degree.get();
        for s in &mut ws.sets {
            s.clear();
        }

        // Lineal expansions: up with path multiplicity, down as support.
        let (parents, mults) = self.up.row(row);
        ws.up[1].clear();
        alloc::extend(
            &mut ws.up[1],
            parents.iter().copied().zip(mults.iter().copied()),
            Family::RowSet,
            "int32",
        )?;
        ws.acc
            .hop_support(&self.down, &[row as u32], &mut ws.down[1])?;
        for k in 2..=deg.max(1) as usize {
            let (below, level) = ws.up.split_at_mut(k);
            ws.acc.hop(&self.up, &below[k - 1], &mut level[0])?;
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
        ws.sets[MO.index()] = parent_role(row, ped.mother, &ws.down[1])?;
        ws.sets[FO.index()] = parent_role(row, ped.father, &ws.down[1])?;
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
        lineal(ws, GP, 2)?;
        self.collateral(row, ws, Av, SibKind::Full, 2)?;
        if deg < 3 {
            return Ok(());
        }

        // Degree 3.
        lineal(ws, GGP, 3)?;
        self.collateral(row, ws, HAv, SibKind::Half, 2)?;
        self.collateral(row, ws, GAv, SibKind::Full, 3)?;
        self.cousins(row, ws)?;
        if deg < 4 {
            return Ok(());
        }

        // Degree 4.
        lineal(ws, GGGP, 4)?;
        self.collateral(row, ws, HGAv, SibKind::Half, 3)?;
        self.collateral(row, ws, GGAv, SibKind::Full, 4)?;
        self.removed_cousins(row, ws, 2, 3, C1R1, |m| m.at_least_two())?;
        if deg < 5 {
            return Ok(());
        }

        // Degree 5.
        lineal(ws, G3GP, 5)?;
        self.collateral(row, ws, HGGAv, SibKind::Half, 4)?;
        self.collateral(row, ws, G3Av, SibKind::Full, 5)?;
        self.removed_cousins(row, ws, 2, 3, H1C1R, |m| m.is_one())?;
        self.removed_cousins(row, ws, 2, 4, C1R2, |m| m.at_least_two())?;
        self.second_cousins(row, ws)
    }

    /// `A^(down-1) @ S` and its transpose `S @ (A^T)^(down-1)` for a sibling matrix `S`.
    fn collateral(
        &self,
        row: usize,
        ws: &mut Workspace,
        cat: Category,
        kind: SibKind,
        down: usize,
    ) -> Result<(), Error> {
        let [sib, result, tmp, tmp2] = &mut ws.scratch;
        result.clear();
        for &(p, _) in &ws.up[down - 1] {
            kind.sibs(&self.sibs, p as usize, tmp2, sib)?;
            sets::union_into(result, sib)?;
        }
        if ws.orient {
            // Sibs of the row's ancestors: the row is the niece or nephew.
            ws.first_arm[cat.index()] = alloc::cloned(result, Family::RowSet, "int32")?;
        }
        kind.sibs(&self.sibs, row, tmp2, sib)?;
        for _ in 1..down {
            ws.acc.hop_support(&self.down, sib, tmp)?;
            std::mem::swap(sib, tmp);
        }
        sets::union_into(result, sib)?;
        sets::drop_self(result, row);
        self.finalize(cat, result, &ws.sets);
        std::mem::swap(&mut ws.sets[cat.index()], result);
        Ok(())
    }

    /// 1C, H1C, and the shared-grandparent support used by 2C.
    fn cousins(&self, row: usize, ws: &mut Workspace) -> Result<(), Error> {
        use Category::*;
        let ped = &self.ped;
        let [children, ..] = &mut ws.scratch;
        let [counted, ..] = &mut ws.weighted;
        let grandparents = ws.up[2].len();
        ws.grandchildren.resize_with(grandparents, Vec::new);
        for (&(g, _), grandchildren) in ws.up[2].iter().zip(&mut ws.grandchildren) {
            ws.acc.hop_support(&self.down, &[g], children)?;
            ws.acc.hop_support(&self.down, children, grandchildren)?;
        }
        ws.acc.count_memberships(
            ws.grandchildren[..grandparents]
                .iter()
                .map(|v| v.as_slice()),
            counted,
        )?;
        counted.retain(|&(j, _)| j as usize != row);
        ws.shares_grandparent = sets::support(counted)?;
        let shares_parent_id = |j: u32| {
            let j = j as usize;
            (ped.orig_mother[row] >= 0 && ped.orig_mother[row] == ped.orig_mother[j])
                || (ped.orig_father[row] >= 0 && ped.orig_father[row] == ped.orig_father[j])
        };
        ws.sets[C1.index()] = sets::select(counted, |m| m.at_least_two())?;
        ws.sets[C1.index()].retain(|&j| !shares_parent_id(j));
        ws.sets[H1C.index()] = sets::select(counted, |m| m.is_one())?;
        ws.sets[H1C.index()].retain(|&j| !shares_parent_id(j));
        Ok(())
    }

    /// `A^a @ (A^b)^T` in both orientations with path multiplicity, thresholded by `keep`.
    fn removed_cousins(
        &self,
        row: usize,
        ws: &mut Workspace,
        a: usize,
        b: usize,
        cat: Category,
        keep: impl Fn(Mult) -> bool,
    ) -> Result<(), Error> {
        let [forward, backward, tmp] = &mut ws.weighted;
        self.chain_down(&mut ws.acc, &ws.up[a], b, forward, tmp)?;
        self.chain_down(&mut ws.acc, &ws.up[b], a, backward, tmp)?;
        let mut result = sets::select(forward, &keep)?;
        let backward = sets::select(backward, &keep)?;
        sets::union_into(&mut result, &backward)?;
        if ws.orient {
            // Through `backward` the row sits `b > a` meioses from the shared
            // ancestor: it is the junior cousin, which the registry puts first
            // (`_pair_extractor.py` reads these products with `row_is_first=False`).
            ws.first_arm[cat.index()] = backward;
        }
        sets::drop_self(&mut result, row);
        self.finalize(cat, &mut result, &ws.sets);
        ws.sets[cat.index()] = result;
        Ok(())
    }

    /// `A^3 @ (A^3)^T >= 2`, minus pairs sharing any grandparent.
    fn second_cousins(&self, row: usize, ws: &mut Workspace) -> Result<(), Error> {
        let [product, _, tmp] = &mut ws.weighted;
        self.chain_down(&mut ws.acc, &ws.up[3], 3, product, tmp)?;
        let mut result = sets::select(product, |m| m.at_least_two())?;
        sets::drop_self(&mut result, row);
        sets::subtract(&mut result, &ws.shares_grandparent);
        ws.sets[Category::C2.index()] = result;
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
        alloc::reserve(out, src.len(), Family::RowSet, "int32")?;
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

#[derive(Clone, Copy)]
enum SibKind {
    Full,
    Half,
}

impl SibKind {
    fn sibs(
        self,
        index: &SiblingIndex,
        row: usize,
        scratch: &mut Vec<u32>,
        out: &mut Vec<u32>,
    ) -> Result<(), Error> {
        match self {
            SibKind::Full => index.full_sibs(row, out),
            SibKind::Half => index.half_sibs(row, scratch, out),
        }
    }
}

/// Lineal pairs in both orientations: ancestors at k hops and descendants at
/// k hops.  Towards an ancestor the row is the descendant, hence `first`.
fn lineal(ws: &mut Workspace, cat: Category, k: usize) -> Result<(), Error> {
    let mut set = sets::support(&ws.up[k])?;
    if ws.orient {
        ws.first_arm[cat.index()] = alloc::cloned(&set, Family::RowSet, "int32")?;
    }
    sets::union_into(&mut set, &ws.down[k])?;
    ws.sets[cat.index()] = set;
    Ok(())
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
fn parent_role(row: usize, parent: &[i32], down1: &[u32]) -> Result<Vec<u32>, Error> {
    let mut set = alloc::collect(
        down1
            .iter()
            .copied()
            .filter(|&j| parent[j as usize] == row as i32),
        Family::RowSet,
        "int32",
    )?;
    if parent[row] >= 0 {
        sets::union_into(&mut set, &[parent[row] as u32])?;
    }
    Ok(set)
}

#[cfg(test)]
mod tests {
    use super::super::testing::pedigree;
    use super::super::{count_pairs, Category, Counts, MaxDegree};

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
        let got = count_pairs(&ped.try_borrow().unwrap(), MaxDegree::MAX, None).unwrap();
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
        let got = count_pairs(&ped.try_borrow().unwrap(), MaxDegree::MAX, None).unwrap();
        assert_eq!(
            got,
            expect(&[(Category::MO, 2), (Category::FO, 2), (Category::GP, 1)])
        );
    }

    #[test]
    fn mz_co_twins_are_twins_not_sibs() {
        let ped = pedigree(&[(-1, -1), (-1, -1), (0, 1), (0, 1)], &[(2, 3)]);
        let got = count_pairs(&ped.try_borrow().unwrap(), MaxDegree::MAX, None).unwrap();
        assert_eq!(
            got,
            expect(&[(Category::MZ, 1), (Category::MO, 2), (Category::FO, 2)])
        );
    }

    #[test]
    fn degree_cutoff_stops_the_fold_and_the_count() {
        let ped = pedigree(&[(-1, -1), (-1, -1), (0, 1), (0, 2)], &[]);
        let got = count_pairs(
            &ped.try_borrow().unwrap(),
            MaxDegree::try_new(1).unwrap(),
            None,
        )
        .unwrap();
        assert_eq!(got, expect(&[(Category::MO, 2), (Category::FO, 2)]));
        assert_eq!(
            count_pairs(
                &ped.try_borrow().unwrap(),
                MaxDegree::try_new(0).unwrap(),
                None
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
        let got = count_pairs(&ped.try_borrow().unwrap(), MaxDegree::MAX, Some(&selected)).unwrap();
        assert_eq!(got, expect(&[(Category::GP, 1)]));
        let all = count_pairs(&ped.try_borrow().unwrap(), MaxDegree::MAX, None).unwrap();
        assert_eq!(
            all,
            expect(&[(Category::MO, 2), (Category::FO, 2), (Category::GP, 2)])
        );
    }
}
