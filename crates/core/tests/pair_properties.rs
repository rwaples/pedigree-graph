//! Properties of relationship-pair extraction and counting on random
//! pedigrees: the invariants `tests/test_pair_extraction_properties.py` and
//! `tests/test_pair_engines_properties.py` check through the Python binding,
//! held here against the core every host calls.
//!
//! `close_relative_counts` and the `None` a count above `max_degree` gets
//! are Python, so their properties stay in the Python suite.

#![allow(
    clippy::expect_used,
    clippy::panic,
    reason = "test helpers outside #[test] functions: a panic fails the test"
)]

mod common;

use common::{check, pedigree, relabelled_pedigree, shuffled_pedigree, Built, Ped};
use pedigree_graph_core::relationships::query::{self, Receiver};
use pedigree_graph_core::relationships::{
    Category, CategorySet, Counts, Execution, MaxDegree, PairBlocks, Progress,
};
use proptest::prelude::*;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering};

fn tally(checked: &AtomicUsize, n: usize) {
    checked.fetch_add(n, Ordering::Relaxed);
}

/// Every category's counts up to `degree`, the categories above it zero.
fn counts(b: &Built, degree: u8) -> Counts {
    query::counts(
        &b.relationships(),
        CategorySet::up_to_degree(degree),
        Receiver::Graph,
        &Progress::default(),
    )
    .expect("counts")
}

fn blocks(b: &Built, execution: Execution) -> PairBlocks {
    query::pairs(
        &b.relationships(),
        CategorySet::up_to_degree(MaxDegree::MAX.get()),
        Receiver::Graph,
        execution,
        &Progress::default(),
    )
    .expect("pairs")
}

/// The pairs of `cat`, each as written.
fn pairs(blocks: &PairBlocks, cat: Category) -> impl Iterator<Item = (usize, usize)> + '_ {
    let block = blocks.get(cat);
    block
        .first
        .iter()
        .zip(&block.second)
        .map(|(&x, &y)| (x as usize, y as usize))
}

/// No pair is a row with itself, and no unordered pair appears twice: a
/// pair has one closest category, in one orientation.
#[test]
fn pairs_are_distinct_rows_listed_once() {
    check("pair keys", 10.0, pedigree(), |ped, checked| {
        let b = Built::new(&ped);
        let all = blocks(&b, Execution::Speed);
        let mut seen: HashMap<(usize, usize), &str> = HashMap::new();
        for cat in Category::ALL {
            for (x, y) in pairs(&all, cat) {
                prop_assert_ne!(x, y, "{} self pair", cat.code());
                if let Some(other) = seen.insert((x.min(y), x.max(y)), cat.code()) {
                    prop_assert!(false, "({x}, {y}) in {other} and {}", cat.code());
                }
            }
        }
        tally(checked, seen.len());
        Ok(())
    });
}

/// Siblings are classified by parent ids, external parents included: full
/// sibs share both known parent ids, half sibs one and not the other.  Every
/// pair sharing both known parent ids is FS or MZ: no other category can
/// claim a full sib (a full sib as a parent would be its own parent's
/// child).
#[test]
fn siblings_follow_the_parent_ids() {
    check("siblings", 0.1, pedigree(), |ped, checked| {
        let b = Built::new(&ped);
        let (mo, fa) = (&b.graph.mother_ids, &b.graph.father_ids);
        let all = blocks(&b, Execution::Speed);
        let shared = |p: &[i64], x: usize, y: usize| p[x] >= 0 && p[x] == p[y];
        for (x, y) in pairs(&all, Category::FS) {
            prop_assert!(shared(mo, x, y) && shared(fa, x, y), "FS ({}, {})", x, y);
        }
        for (x, y) in pairs(&all, Category::MHS) {
            prop_assert!(shared(mo, x, y) && !shared(fa, x, y), "MHS ({}, {})", x, y);
        }
        for (x, y) in pairs(&all, Category::PHS) {
            prop_assert!(shared(fa, x, y) && !shared(mo, x, y), "PHS ({}, {})", x, y);
        }
        let full_or_mz: HashSet<(usize, usize)> = pairs(&all, Category::FS)
            .chain(pairs(&all, Category::MZ))
            .map(|(x, y)| (x.min(y), x.max(y)))
            .collect();
        for x in 0..b.len() {
            for y in x + 1..b.len() {
                if shared(mo, x, y) && shared(fa, x, y) {
                    prop_assert!(full_or_mz.contains(&(x, y)), "({}, {}) not FS or MZ", x, y);
                    tally(checked, 1);
                }
            }
        }
        Ok(())
    });
}

/// Relabelling the ids changes no count.
#[test]
fn counts_do_not_depend_on_ids() {
    check(
        "relabel",
        1.0,
        relabelled_pedigree(),
        |(ped, perm, mult), checked| {
            let a = counts(&Built::new(&ped), MaxDegree::MAX.get());
            let b = counts(
                &Built::new(&ped.relabelled(&perm, mult)),
                MaxDegree::MAX.get(),
            );
            prop_assert_eq!(a.0, b.0);
            tally(checked, 1);
            Ok(())
        },
    );
}

/// Reordering the rows moves every pair with its rows: the counts and each
/// category's unordered pairs are unchanged, and a directed pair keeps its
/// orientation.  The one exception is a collateral pair a loop puts in its
/// category both ways round (each a half aunt of the other, say): the lower
/// row leads it (`Engine::emit_row`), so it may turn round with the rows,
/// and then it is lower row first in both orders.  A lineal pair never
/// turns round.
#[test]
fn row_order_does_not_change_pairs() {
    check(
        "row order",
        1.0,
        shuffled_pedigree(),
        |(ped, order), checked| {
            let (a, b) = (Built::new(&ped), Built::new(&ped.permuted(&order)));
            prop_assert_eq!(
                counts(&a, MaxDegree::MAX.get()).0,
                counts(&b, MaxDegree::MAX.get()).0
            );
            let (pa, pb) = (blocks(&a, Execution::Speed), blocks(&b, Execution::Speed));
            for cat in Category::ALL {
                let unordered = |(x, y): (usize, usize)| (x.min(y), x.max(y));
                let want: HashSet<_> = pairs(&pa, cat).collect();
                let got: Vec<_> = pairs(&pb, cat).collect();
                let moved: HashSet<_> = got.iter().map(|&(r, s)| (order[r], order[s])).collect();
                prop_assert_eq!(
                    moved.iter().copied().map(unordered).collect::<HashSet<_>>(),
                    want.iter().copied().map(unordered).collect::<HashSet<_>>(),
                    "{}",
                    cat.code()
                );
                let Some((_, senior)) = cat.roles() else {
                    continue;
                };
                // An ancestor of its own descendant would be a cycle.
                let lineal = matches!(senior, "mother" | "father" | "ancestor");
                for &(r, s) in &got {
                    let (x, y) = (order[r], order[s]);
                    if !want.contains(&(x, y)) {
                        prop_assert!(
                            !lineal && r < s && y < x,
                            "{} ({r}, {s}) was ({y}, {x})",
                            cat.code()
                        );
                    }
                }
            }
            tally(checked, 1);
            Ok(())
        },
    );
}

/// The case proptest found: rows 1 and 5 are each a half aunt of the other
/// (row 5 and row 1's mother share a father, row 1 and row 5's mother share
/// a mother), two half paths that make them `Av`.  The lower row leads in
/// either row order.
#[test]
fn a_pair_avuncular_both_ways_is_led_by_the_lower_row() {
    let ped = Ped {
        ids: vec![10, 3, 17, 45, 38, 24, 31],
        mother: vec![9_000_000_000, 10, 10, -1, -1, 17, -1],
        father: vec![38, 31, 9_100_000_001, 9_100_000_000, -1, 38, -1],
        twin: vec![-1; 7],
    };
    let av = |ped: &Ped| {
        pairs(&blocks(&Built::new(ped), Execution::Speed), Category::Av).collect::<Vec<_>>()
    };
    assert_eq!(av(&ped), vec![(1, 5)]);
    // Rows 1 and 5 swapped.
    assert_eq!(av(&ped.permuted(&[0, 5, 2, 3, 4, 1, 6])), vec![(1, 5)]);
}

/// Each category's pairs, under either execution, are as many as its count.
#[test]
fn pairs_and_counts_agree() {
    check("pairs vs counts", 20.0, pedigree(), |ped, checked| {
        let b = Built::new(&ped);
        let c = counts(&b, MaxDegree::MAX.get());
        for execution in [Execution::Speed, Execution::Memory] {
            let all = blocks(&b, execution);
            for cat in Category::ALL {
                let n = all.get(cat).len() as u64;
                prop_assert_eq!(n, c.get(cat), "{} {}", cat.code(), execution.name());
                tally(checked, n as usize);
            }
        }
        Ok(())
    });
}

/// Counting to degree `d` gives the full counts of every category of degree
/// at most `d` and nothing above it.
#[test]
fn degree_gates_the_counts() {
    check("degree gating", 1.0, pedigree(), |ped, checked| {
        let b = Built::new(&ped);
        let full = counts(&b, MaxDegree::MAX.get());
        for d in 0..=MaxDegree::MAX.get() {
            let c = counts(&b, d);
            for cat in Category::ALL {
                let want = if cat.degree() <= d { full.get(cat) } else { 0 };
                prop_assert_eq!(c.get(cat), want, "degree {} {}", d, cat.code());
            }
        }
        tally(checked, 1);
        Ok(())
    });
}
