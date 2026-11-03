//! Pedigree strategies and builders for the property tests.
//!
//! A pedigree is built from its size and a list of dice, read in order and
//! 0 once used up, so every shrunk case is still a valid pedigree: shrinking
//! makes it smaller and its choices the first ones.  Rows, ids and
//! topological order all differ, a few parents are external ids, and a few
//! rows are MZ co-twins, so the properties run through the id resolution and
//! depth-major ordering the hosts rely on.

#![allow(dead_code, reason = "each test binary uses a subset")]

use pedigree_graph_core::columns::GraphColumns;
use pedigree_graph_core::graph::{self, Columns, Limits, PedigreeGraph, SexEncoding};
use pedigree_graph_core::kinship::{kinship_csc, KinshipPedigree};
use pedigree_graph_core::relationships::Pedigree;
use pedigree_graph_core::topology::structural_depth;
use proptest::prelude::*;
use proptest::test_runner::TestRunner;
use std::collections::BTreeSet;
use std::sync::atomic::AtomicUsize;

/// External mother ids, then external father ids: above every row id.
const EXTERNAL_MOTHER: i64 = 9_000_000_000;
const EXTERNAL_FATHER: i64 = 9_100_000_000;

/// Host columns: one id, mother id, father id and co-twin id per row.
#[derive(Clone, Debug)]
pub struct Ped {
    pub ids: Vec<i64>,
    pub mother: Vec<i64>,
    pub father: Vec<i64>,
    pub twin: Vec<i64>,
}

impl Ped {
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// The same pedigree with each row id `ids[r]` replaced by
    /// `perm[r] * mult + 3`; external ids are kept and the row order is
    /// unchanged (`tests/conftest.py::relabel_pedigree`).
    pub fn relabelled(&self, perm: &[usize], mult: i64) -> Ped {
        let new: std::collections::HashMap<i64, i64> = self
            .ids
            .iter()
            .zip(perm)
            .map(|(&id, &p)| (id, p as i64 * mult + 3))
            .collect();
        let map = |v: &[i64]| v.iter().map(|x| *new.get(x).unwrap_or(x)).collect();
        Ped {
            ids: map(&self.ids),
            mother: map(&self.mother),
            father: map(&self.father),
            twin: map(&self.twin),
        }
    }

    /// The rows in `order`: row `r` of the result is row `order[r]`, ids kept.
    pub fn permuted(&self, order: &[usize]) -> Ped {
        let pick = |v: &[i64]| order.iter().map(|&r| v[r]).collect();
        Ped {
            ids: pick(&self.ids),
            mother: pick(&self.mother),
            father: pick(&self.father),
            twin: pick(&self.twin),
        }
    }
}

/// A built pedigree with its structural depth.
pub struct Built {
    pub graph: PedigreeGraph,
    pub depth: Vec<i32>,
}

impl Built {
    pub fn new(ped: &Ped) -> Built {
        let graph = graph::build(
            Columns {
                ids: &ped.ids,
                mother: &ped.mother,
                father: &ped.father,
                twin: Some(&ped.twin),
                sex: None,
                generation: None,
                birth_year: None,
            },
            SexEncoding::Simace,
            Limits::default(),
        )
        .expect("the strategy makes valid pedigrees");
        let depth = structural_depth(&graph.mother_rows, &graph.father_rows);
        Built { graph, depth }
    }

    pub fn len(&self) -> usize {
        self.graph.len()
    }

    fn columns(&self) -> GraphColumns<'_> {
        GraphColumns {
            mother: &self.graph.mother_rows,
            father: &self.graph.father_rows,
            twin: &self.graph.twin_rows,
        }
    }

    pub fn kinship(&self) -> KinshipPedigree<'_> {
        self.columns()
            .kinship(&self.depth)
            .expect("built columns are a kinship input")
    }

    pub fn relationships(&self) -> Pedigree<'_> {
        self.columns()
            .relationships(&self.graph.mother_ids, &self.graph.father_ids)
            .expect("built columns are an engine input")
    }

    /// The complete kinship matrix, dense, `k[a][b]`.
    pub fn dense_kinship(&self) -> Vec<Vec<f32>> {
        let csc = kinship_csc(self.kinship()).expect("a small pedigree's matrix");
        let n = self.len();
        let mut k = vec![vec![0f32; n]; n];
        for (col, span) in csc.indptr.windows(2).enumerate() {
            for at in span[0] as usize..span[1] as usize {
                k[csc.indices[at] as usize][col] = csc.data[at];
            }
        }
        k
    }

    /// The row's parent rows, missing and external ones left out.
    pub fn parent_rows(&self, row: usize) -> impl Iterator<Item = usize> {
        [self.graph.mother_rows[row], self.graph.father_rows[row]]
            .into_iter()
            .filter(|&p| p >= 0)
            .map(|p| p as usize)
    }

    pub fn twin_of(&self, row: usize) -> Option<usize> {
        let t = self.graph.twin_rows[row];
        (t >= 0).then_some(t as usize)
    }
}

struct Dice<'a> {
    values: &'a [u32],
    at: usize,
}

impl Dice<'_> {
    /// The next die in `0..n`.
    fn roll(&mut self, n: usize) -> usize {
        let v = self.values.get(self.at).copied().unwrap_or(0);
        self.at += 1;
        v as usize % n
    }

    fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            v.swap(i, self.roll(i + 1));
        }
    }
}

/// A parent while generating: an earlier individual, missing, or the k-th
/// external id.
#[derive(Clone, Copy, PartialEq)]
enum Parent {
    Row(usize),
    Missing,
    External(i64),
}

/// Parents come from the last `WINDOW` individuals, so pedigrees run deep.
const WINDOW: usize = 10;

/// Even individuals are mothers and odd ones fathers, so no id plays both
/// parent roles (ADR 0020).
fn is_mother(i: usize) -> bool {
    i % 2 == 0
}

/// The parent a die picks for individual `i` in one role: one of the last
/// [`WINDOW`] individuals of that role, missing, or (with `external`) one of
/// three external ids of that role.
fn pick(dice: &mut Dice, i: usize, external: bool, mother: bool) -> Parent {
    let rows: Vec<usize> = (i - i.min(WINDOW)..i)
        .rev()
        .filter(|&j| is_mother(j) == mother)
        .collect();
    let w = rows.len();
    let v = dice.roll(w + 1 + if external { 3 } else { 0 });
    match v {
        v if v < w => Parent::Row(rows[v]),
        v if v == w => Parent::Missing,
        v => Parent::External((v - w - 1) as i64),
    }
}

fn generate(n: usize, non_inbred: bool, dice: &[u32]) -> Ped {
    let mut dice = Dice {
        values: dice,
        at: 0,
    };
    let mut mother = vec![Parent::Missing; n];
    let mut father = vec![Parent::Missing; n];
    let mut twin: Vec<Option<usize>> = vec![None; n];
    // Ancestors including self and any co-twin, for the non-inbred arm.
    let mut closed: Vec<BTreeSet<usize>> = (0..n).map(|i| BTreeSet::from([i])).collect();
    for i in 1..n {
        if twin[i - 1].is_none() && dice.roll(12) == 11 {
            // MZ co-twin of the individual before: same parents.
            mother[i] = mother[i - 1];
            father[i] = father[i - 1];
            twin[i] = Some(i - 1);
            twin[i - 1] = Some(i);
            closed[i] = closed[i - 1].clone();
            closed[i].insert(i);
            closed[i - 1].insert(i);
            continue;
        }
        if dice.roll(4) == 0 {
            continue; // founder
        }
        let m = pick(&mut dice, i, !non_inbred, true);
        let f = match m {
            Parent::Row(m) if non_inbred => {
                // A mate unrelated to `m`, so the child is not inbred.
                let mates: Vec<usize> = (0..i)
                    .filter(|&c| !is_mother(c) && closed[m].is_disjoint(&closed[c]))
                    .collect();
                match dice.roll(mates.len() + 1) {
                    0 => Parent::Missing,
                    k => Parent::Row(mates[k - 1]),
                }
            }
            _ => pick(&mut dice, i, !non_inbred, false),
        };
        mother[i] = m;
        father[i] = f;
        for p in [m, f] {
            if let Parent::Row(p) = p {
                let ancestors = closed[p].clone();
                closed[i].extend(ancestors);
            }
        }
    }
    // Individual `i` sits at row `at[i]` and has id `ids[at[i]]`.
    let mut at: Vec<usize> = (0..n).collect();
    dice.shuffle(&mut at);
    let mut ids: Vec<i64> = (0..n as i64).map(|r| r * 7 + 3).collect();
    dice.shuffle(&mut ids);
    let id = |p: Parent, external: i64| match p {
        Parent::Row(j) => ids[at[j]],
        Parent::Missing => -1,
        Parent::External(k) => external + k,
    };
    let mut ped = Ped {
        ids: ids.clone(),
        mother: vec![-1; n],
        father: vec![-1; n],
        twin: vec![-1; n],
    };
    for i in 0..n {
        let r = at[i];
        ped.mother[r] = id(mother[i], EXTERNAL_MOTHER);
        ped.father[r] = id(father[i], EXTERNAL_FATHER);
        ped.twin[r] = twin[i].map_or(-1, |t| ids[at[t]]);
    }
    ped
}

/// Enough dice for 25 rows (at most five each) and both shuffles.
fn dice() -> impl Strategy<Value = Vec<u32>> {
    prop::collection::vec(any::<u32>(), 200)
}

/// A possibly inbred pedigree of 1 to 25 rows, with MZ co-twins and
/// external parents (`tests/conftest.py::random_pedigree`, generalized).
pub fn pedigree() -> impl Strategy<Value = Ped> {
    (1usize..=25, dice()).prop_map(|(n, dice)| generate(n, false, &dice))
}

/// A pedigree whose every child has unrelated parents, so no one is inbred
/// (`tests/conftest.py::non_inbred_pedigree`, with MZ co-twins).
pub fn non_inbred_pedigree() -> impl Strategy<Value = Ped> {
    (1usize..=25, dice()).prop_map(|(n, dice)| generate(n, true, &dice))
}

/// A pedigree and a permutation of its rows.
pub fn shuffled_pedigree() -> impl Strategy<Value = (Ped, Vec<usize>)> {
    pedigree().prop_flat_map(|ped| {
        let order = Just((0..ped.len()).collect::<Vec<_>>()).prop_shuffle();
        (Just(ped), order)
    })
}

/// A pedigree with a row permutation and an id multiplier for relabelling.
pub fn relabelled_pedigree() -> impl Strategy<Value = (Ped, Vec<usize>, i64)> {
    shuffled_pedigree().prop_flat_map(|(ped, perm)| {
        let mult = prop_oneof![Just(1i64), Just(1000), Just(1_000_000)];
        (Just(ped), Just(perm), mult)
    })
}

/// Cases per property in an ordinary run.
const CASES: u32 = 256;

/// Cases per property: `PROPTEST_CASES` when set (proptest's own variable,
/// which its config would ignore here because the runner sets `cases`),
/// else [`CASES`].
fn cases() -> u32 {
    std::env::var("PROPTEST_CASES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(CASES)
}

/// Run `test` on [`cases()`] draws of `strategy`, then require that it found
/// something to check at least `per_case` times per case on average: `test`
/// adds what it checked to the counter, so a strategy that stops making such
/// cases fails instead of passing vacuously.  A failing case is saved beside
/// the calling test file, as `proptest!` does.
#[track_caller]
pub fn check<S: Strategy>(
    name: &str,
    per_case: f64,
    strategy: S,
    test: impl Fn(S::Value, &AtomicUsize) -> Result<(), TestCaseError>,
) {
    let cases = cases();
    let floor = (per_case * f64::from(cases)).ceil() as usize;
    let checked = AtomicUsize::new(0);
    let config = ProptestConfig {
        cases,
        source_file: Some(std::panic::Location::caller().file()),
        ..ProptestConfig::default()
    };
    let mut runner = TestRunner::new(config);
    if let Err(e) = runner.run(&strategy, |value| test(value, &checked)) {
        panic!("{name}: {e}");
    }
    let checked = checked.into_inner();
    assert!(
        checked >= floor,
        "{name}: checked {checked} times in {cases} cases, expected at least {floor}"
    );
}
