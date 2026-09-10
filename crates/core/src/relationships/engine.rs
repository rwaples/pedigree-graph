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

use super::category::{Category, Counts};
use super::csr::Csr;
use super::multiplicity::Mult;
use super::sets::{self, Accumulator, Weighted};
use super::sibling_index::SiblingIndex;
use super::{MaxDegree, Pedigree};

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
}

impl Workspace {
    pub fn new(n: usize) -> Workspace {
        Workspace {
            acc: Accumulator::new(n),
            up: vec![Vec::new(); MaxDegree::MAX.get() as usize + 1],
            down: vec![Vec::new(); MaxDegree::MAX.get() as usize + 1],
            sets: vec![Vec::new(); super::category::N_CATEGORIES],
            shares_grandparent: Vec::new(),
            grandchildren: Vec::new(),
            scratch: Default::default(),
            weighted: Default::default(),
        }
    }
}

impl<'p> Engine<'p> {
    pub fn new(ped: &Pedigree<'p>, max_degree: MaxDegree) -> Engine<'p> {
        let n = ped.len();
        let edges: Vec<(u32, u32)> = (0..n)
            .flat_map(|i| {
                [ped.mother[i], ped.father[i]]
                    .into_iter()
                    .filter(|&p| p >= 0)
                    .map(move |p| (i as u32, p as u32))
            })
            .collect();
        let up = Csr::from_edges(n, edges);
        let down = up.transpose();
        let sibs = SiblingIndex::build(ped.twin, ped.orig_mother, ped.orig_father);
        Engine {
            ped: *ped,
            up,
            down,
            sibs,
            max_degree,
        }
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
    ) {
        self.classify_row(row, ws);
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
    pub fn classify_row(&self, row: usize, ws: &mut Workspace) {
        use Category::*;
        let deg = self.max_degree.get();
        for s in &mut ws.sets {
            s.clear();
        }

        // Lineal expansions: up with path multiplicity, down as support.
        let (parents, mults) = self.up.row(row);
        ws.up[1].clear();
        ws.up[1].extend(parents.iter().copied().zip(mults.iter().copied()));
        ws.acc
            .hop_support(&self.down, &[row as u32], &mut ws.down[1]);
        for k in 2..=deg.max(1) as usize {
            let (below, level) = ws.up.split_at_mut(k);
            ws.acc.hop(&self.up, &below[k - 1], &mut level[0]);
            let (below, level) = ws.down.split_at_mut(k);
            ws.acc.hop_support(&self.down, &below[k - 1], &mut level[0]);
        }

        // Degree 0 and 1: the co-twin, each parent role by row in both
        // orientations, sibs by original parent id.
        if self.ped.twin[row] >= 0 {
            ws.sets[MZ.index()].push(self.ped.twin[row] as u32);
        }
        if deg < 1 {
            return;
        }
        let ped = &self.ped;
        ws.sets[MO.index()] = parent_role(row, ped.mother, &ws.down[1]);
        ws.sets[FO.index()] = parent_role(row, ped.father, &ws.down[1]);
        self.sibs.full_sibs(row, &mut ws.sets[FS.index()]);
        self.sibs.maternal_half_sibs(row, &mut ws.sets[MHS.index()]);
        self.sibs.paternal_half_sibs(row, &mut ws.sets[PHS.index()]);
        if deg < 2 {
            return;
        }

        // Degree 2: GP is lineal (up[2] ∪ down[2] as an exclusion set); Av.
        ws.sets[GP.index()] = both_ways(&ws.up[2], &ws.down[2]);
        self.collateral(row, ws, Av, SibKind::Full, 2);
        if deg < 3 {
            return;
        }

        // Degree 3.
        ws.sets[GGP.index()] = both_ways(&ws.up[3], &ws.down[3]);
        self.collateral(row, ws, HAv, SibKind::Half, 2);
        self.collateral(row, ws, GAv, SibKind::Full, 3);
        self.cousins(row, ws);
        if deg < 4 {
            return;
        }

        // Degree 4.
        ws.sets[GGGP.index()] = both_ways(&ws.up[4], &ws.down[4]);
        self.collateral(row, ws, HGAv, SibKind::Half, 3);
        self.collateral(row, ws, GGAv, SibKind::Full, 4);
        self.removed_cousins(row, ws, 2, 3, C1R1, |m| m.at_least_two());
        if deg < 5 {
            return;
        }

        // Degree 5.
        ws.sets[G3GP.index()] = both_ways(&ws.up[5], &ws.down[5]);
        self.collateral(row, ws, HGGAv, SibKind::Half, 4);
        self.collateral(row, ws, G3Av, SibKind::Full, 5);
        self.removed_cousins(row, ws, 2, 3, H1C1R, |m| m.is_one());
        self.removed_cousins(row, ws, 2, 4, C1R2, |m| m.at_least_two());
        self.second_cousins(row, ws);
    }

    /// `A^(down-1) @ S` and its transpose `S @ (A^T)^(down-1)` for a sibling matrix `S`.
    fn collateral(
        &self,
        row: usize,
        ws: &mut Workspace,
        cat: Category,
        kind: SibKind,
        down: usize,
    ) {
        let [sib, result, tmp, tmp2] = &mut ws.scratch;
        result.clear();
        for &(p, _) in &ws.up[down - 1] {
            kind.sibs(&self.sibs, p as usize, tmp2, sib);
            sets::union_into(result, sib);
        }
        kind.sibs(&self.sibs, row, tmp2, sib);
        for _ in 1..down {
            ws.acc.hop_support(&self.down, sib, tmp);
            std::mem::swap(sib, tmp);
        }
        sets::union_into(result, sib);
        sets::drop_self(result, row);
        self.finalize(cat, result, &ws.sets);
        std::mem::swap(&mut ws.sets[cat.index()], result);
    }

    /// 1C, H1C, and the shared-grandparent support used by 2C.
    fn cousins(&self, row: usize, ws: &mut Workspace) {
        use Category::*;
        let ped = &self.ped;
        let [children, ..] = &mut ws.scratch;
        let [counted, ..] = &mut ws.weighted;
        let grandparents = ws.up[2].len();
        ws.grandchildren.resize_with(grandparents, Vec::new);
        for (&(g, _), grandchildren) in ws.up[2].iter().zip(&mut ws.grandchildren) {
            ws.acc.hop_support(&self.down, &[g], children);
            ws.acc.hop_support(&self.down, children, grandchildren);
        }
        ws.acc.count_memberships(
            ws.grandchildren[..grandparents]
                .iter()
                .map(|v| v.as_slice()),
            counted,
        );
        counted.retain(|&(j, _)| j as usize != row);
        ws.shares_grandparent = sets::support(counted);
        let shares_parent_id = |j: u32| {
            let j = j as usize;
            (ped.orig_mother[row] >= 0 && ped.orig_mother[row] == ped.orig_mother[j])
                || (ped.orig_father[row] >= 0 && ped.orig_father[row] == ped.orig_father[j])
        };
        ws.sets[C1.index()] = counted
            .iter()
            .filter(|&&(j, m)| m.at_least_two() && !shares_parent_id(j))
            .map(|&(j, _)| j)
            .collect();
        ws.sets[H1C.index()] = counted
            .iter()
            .filter(|&&(j, m)| m.is_one() && !shares_parent_id(j))
            .map(|&(j, _)| j)
            .collect();
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
    ) {
        let [forward, backward, tmp] = &mut ws.weighted;
        self.chain_down(&mut ws.acc, &ws.up[a], b, forward, tmp);
        self.chain_down(&mut ws.acc, &ws.up[b], a, backward, tmp);
        let mut result = sets::select(forward, &keep);
        sets::union_into(&mut result, &sets::select(backward, &keep));
        sets::drop_self(&mut result, row);
        self.finalize(cat, &mut result, &ws.sets);
        ws.sets[cat.index()] = result;
    }

    /// `A^3 @ (A^3)^T >= 2`, minus pairs sharing any grandparent.
    fn second_cousins(&self, row: usize, ws: &mut Workspace) {
        let [product, _, tmp] = &mut ws.weighted;
        self.chain_down(&mut ws.acc, &ws.up[3], 3, product, tmp);
        let mut result = sets::select(product, |m| m.at_least_two());
        sets::drop_self(&mut result, row);
        sets::subtract(&mut result, &ws.shares_grandparent);
        ws.sets[Category::C2.index()] = result;
    }

    /// `src @ (A^T)^k` with saturated path multiplicity.
    fn chain_down(
        &self,
        acc: &mut Accumulator,
        src: &Weighted,
        k: usize,
        out: &mut Weighted,
        tmp: &mut Weighted,
    ) {
        out.clear();
        out.extend_from_slice(src);
        for _ in 0..k {
            acc.hop(&self.down, out, tmp);
            std::mem::swap(out, tmp);
        }
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
    fn sibs(self, index: &SiblingIndex, row: usize, scratch: &mut Vec<u32>, out: &mut Vec<u32>) {
        match self {
            SibKind::Full => index.full_sibs(row, out),
            SibKind::Half => index.half_sibs(row, scratch, out),
        }
    }
}

/// Lineal pairs in both orientations: ancestors at k hops and descendants at k hops.
fn both_ways(up: &Weighted, down: &[u32]) -> Vec<u32> {
    let mut set = sets::support(up);
    sets::union_into(&mut set, down);
    set
}

/// One parent role in both orientations: `row`'s parent of that role, and
/// the children (`down1`, sorted) for whom `row` fills that role.
fn parent_role(row: usize, parent: &[i32], down1: &[u32]) -> Vec<u32> {
    let mut set: Vec<u32> = down1
        .iter()
        .copied()
        .filter(|&j| parent[j as usize] == row as i32)
        .collect();
    if parent[row] >= 0 {
        sets::union_into(&mut set, &[parent[row] as u32]);
    }
    set
}

#[cfg(test)]
mod tests {
    use super::super::{count_pairs, Category, Counts, MaxDegree, PedigreeColumns};

    /// Rows with parent rows; original ids equal rows, `-1` missing.
    fn pedigree(parents: &[(i32, i32)], twins: &[(usize, usize)]) -> PedigreeColumns {
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
        let got = count_pairs(&ped.try_borrow().unwrap(), MaxDegree::MAX, None);
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
        let got = count_pairs(&ped.try_borrow().unwrap(), MaxDegree::MAX, None);
        assert_eq!(
            got,
            expect(&[(Category::MO, 2), (Category::FO, 2), (Category::GP, 1)])
        );
    }

    #[test]
    fn mz_co_twins_are_twins_not_sibs() {
        let ped = pedigree(&[(-1, -1), (-1, -1), (0, 1), (0, 1)], &[(2, 3)]);
        let got = count_pairs(&ped.try_borrow().unwrap(), MaxDegree::MAX, None);
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
        );
        assert_eq!(got, expect(&[(Category::MO, 2), (Category::FO, 2)]));
        assert_eq!(
            count_pairs(
                &ped.try_borrow().unwrap(),
                MaxDegree::try_new(0).unwrap(),
                None
            ),
            Counts::default()
        );
    }

    #[test]
    fn a_selection_counts_only_pairs_inside_it_but_classifies_through_everyone() {
        // Grandmother g(0), her child p(2) with h(1), and p's child c(3) with
        // k(4).  Selecting g and c keeps their GP pair although p is unselected.
        let ped = pedigree(&[(-1, -1), (-1, -1), (0, 1), (2, 4), (-1, -1)], &[]);
        let selected = [true, false, false, true, false];
        let got = count_pairs(&ped.try_borrow().unwrap(), MaxDegree::MAX, Some(&selected));
        assert_eq!(got, expect(&[(Category::GP, 1)]));
        let all = count_pairs(&ped.try_borrow().unwrap(), MaxDegree::MAX, None);
        assert_eq!(
            all,
            expect(&[(Category::MO, 2), (Category::FO, 2), (Category::GP, 2)])
        );
    }
}
