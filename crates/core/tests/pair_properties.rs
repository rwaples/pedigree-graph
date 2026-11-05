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

use common::{
    check, non_inbred_pedigree, pedigree, relabelled_pedigree, shuffled_pedigree, Built, Ped,
};
use pedigree_graph_core::kinship::inbreeding;
use pedigree_graph_core::lineage::distinct_ancestor_counts;
use pedigree_graph_core::relationships::query::{self, Compaction, Receiver};
use pedigree_graph_core::relationships::{
    relationship_burden, Category, CategorySet, Counts, Execution, MaxDegree, MomentsInput,
    MomentsTable, Operand, PairBlocks, Product, Progress, RelativesPerPerson, Side, Statistic,
    Symmetric, Threshold, ThresholdColumn,
};
use proptest::prelude::*;
use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::num::NonZeroUsize;
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

/// The sex-specific category a parent-role swap exchanges `cat` with: MO
/// with FO, MHS with PHS, every other category with itself
/// (`pedigree_graph/_registry.py`).
fn role_partner(cat: Category) -> Category {
    match cat {
        Category::MO => Category::FO,
        Category::FO => Category::MO,
        Category::MHS => Category::PHS,
        Category::PHS => Category::MHS,
        other => other,
    }
}

const THREADS: NonZeroUsize = NonZeroUsize::new(2).expect("2 is not zero");

/// Every category's relatives of every row, with one threshold column.
fn relatives(b: &Built, relative: &[f64]) -> RelativesPerPerson {
    let columns = [ThresholdColumn {
        relative,
        threshold: Threshold::Scalar(2.0),
    }];
    query::relatives(
        &b.relationships(),
        CategorySet::up_to_degree(MaxDegree::MAX.get()),
        Receiver::Graph,
        &columns,
        THREADS,
        &Progress::default(),
    )
    .expect("relatives")
}

/// A row's counts in `cat`: its relatives, then those passing the column.
fn credits(r: &RelativesPerPerson, row: usize, cat: Category) -> &[u32] {
    let base = (row * r.n_categories + cat.index()) * r.stride;
    &r.counts[base..base + r.stride]
}

/// Small-integer moments inputs, one entry per row: two first labels, two
/// second labels, two value columns, one equality key.  Integers keep the
/// quantization exponent at zero, so every accumulator is exact (ADR 0013).
struct Inputs {
    first: Vec<i32>,
    second: Vec<i32>,
    values: Vec<i64>,
    same: Vec<i64>,
}

impl Inputs {
    /// Inputs for `n` rows read from `bytes`, cycled.
    fn new(n: usize, bytes: &[u8]) -> Inputs {
        let at = |r: usize| u32::from(bytes[r % bytes.len()]);
        Inputs {
            first: (0..n).map(|r| (at(r) % 2) as i32).collect(),
            second: (0..n).map(|r| (at(r) / 2 % 2) as i32).collect(),
            values: (0..n)
                .flat_map(|r| [i64::from(at(r) % 7) - 3, i64::from(at(r) / 7 % 5) - 2])
                .collect(),
            same: (0..n).map(|r| i64::from(at(r) / 35 % 3)).collect(),
        }
    }

    /// The inputs of the rows in `range`.
    fn rows(&self, range: std::ops::Range<usize>) -> Inputs {
        Inputs {
            first: self.first[range.clone()].to_vec(),
            second: self.second[range.clone()].to_vec(),
            values: self.values[2 * range.start..2 * range.end].to_vec(),
            same: self.same[range].to_vec(),
        }
    }
}

/// The diagonal products and one cross-column product.
fn products() -> Vec<Product> {
    let mut p = Product::diagonal(2);
    p.push(Product {
        a: Operand {
            side: Side::First,
            column: 0,
        },
        b: Operand {
            side: Side::Second,
            column: 1,
        },
    });
    p
}

/// Every category's moments of `inputs` as a table of shape
/// `[categories, cells]`, categories in registry order.
fn moments(
    b: &Built,
    inputs: &Inputs,
    products: &[Product],
    symmetric: Symmetric,
) -> MomentsTable<'static> {
    let requested = CategorySet::up_to_degree(MaxDegree::MAX.get());
    let input = MomentsInput {
        labels_first: &inputs.first,
        n_labels_first: 2,
        labels_second: &inputs.second,
        n_labels_second: 2,
        values: &inputs.values,
        n_columns: 2,
        products,
        same: &inputs.same,
        n_same: 1,
    };
    let m = query::moments(
        &b.relationships(),
        requested,
        Receiver::Graph,
        &input,
        symmetric,
        THREADS,
        1 << 30,
        &Progress::default(),
    )
    .expect("moments");
    MomentsTable::from_bytes(
        vec![requested.iter().count(), m.cells],
        2,
        products.to_vec(),
        symmetric,
        vec![0, 0],
        m.width,
        Cow::Owned(m.table),
    )
    .expect("a moments call's own table")
}

/// The encoded accumulators of `cat`'s cells in a [`moments`] table.
fn category_cells<'t>(t: &'t MomentsTable<'_>, cat: Category) -> &'t [u8] {
    let len = t.shape()[1] * t.stride() * t.width();
    &t.bytes()[cat.index() * len..(cat.index() + 1) * len]
}

/// Swapping the mother and father ids (and flipping sex, so MZ co-twins
/// keep one sex) exchanges MO with FO and MHS with PHS, row for row, and
/// changes no other category: in the pair blocks of both executions, the
/// relatives of every row, the moments under both symmetric rules, and the
/// burden.
#[test]
fn a_parent_role_swap_exchanges_the_sex_specific_categories() {
    let strategy = (pedigree(), prop::collection::vec(any::<u8>(), 25));
    check("role swap", 9.0, strategy, |(ped, bytes), checked| {
        let sex = ped.role_sex();
        let flipped: Vec<i64> = sex.iter().map(|&s| if s < 0 { s } else { 1 - s }).collect();
        let a = Built::with_sex(&ped, Some(&sex));
        let b = Built::with_sex(&ped.roles_swapped(), Some(&flipped));
        prop_assert_eq!(&a.graph.mother_rows, &b.graph.father_rows);
        prop_assert_eq!(&a.graph.father_rows, &b.graph.mother_rows);

        for execution in [Execution::Speed, Execution::Memory] {
            let (pa, pb) = (blocks(&a, execution), blocks(&b, execution));
            for cat in Category::ALL {
                let partner = role_partner(cat);
                prop_assert_eq!(
                    pa.get(cat),
                    pb.get(partner),
                    "{} {}",
                    cat.code(),
                    execution.name()
                );
                if cat != partner {
                    tally(checked, pa.get(cat).len());
                }
            }
        }

        let relative: Vec<f64> = (0..a.len())
            .map(|r| f64::from(bytes[r % bytes.len()] % 5))
            .collect();
        let (ra, rb) = (relatives(&a, &relative), relatives(&b, &relative));
        for row in 0..a.len() {
            for cat in Category::ALL {
                prop_assert_eq!(
                    credits(&ra, row, cat),
                    credits(&rb, row, role_partner(cat)),
                    "row {} {}",
                    row,
                    cat.code()
                );
            }
        }

        let inputs = Inputs::new(a.len(), &bytes);
        let products = products();
        for symmetric in [Symmetric::Canonical, Symmetric::Both] {
            let ma = moments(&a, &inputs, &products, symmetric);
            let mb = moments(&b, &inputs, &products, symmetric);
            // The same accumulators in another category order: one width.
            prop_assert_eq!(ma.width(), mb.width());
            for cat in Category::ALL {
                prop_assert_eq!(
                    category_cells(&ma, cat),
                    category_cells(&mb, role_partner(cat)),
                    "{} {}",
                    cat.code(),
                    symmetric.name()
                );
            }
        }

        let burden = |x: &Built| {
            relationship_burden(&x.relationships(), &x.depth, &Progress::default()).expect("burden")
        };
        let (ba, bb) = (burden(&a), burden(&b));
        for cat in Category::ALL {
            prop_assert_eq!(
                ba.categories[cat.index()],
                bb.categories[role_partner(cat).index()],
                "{}",
                cat.code()
            );
        }
        prop_assert_eq!(ba.per_person, bb.per_person);
        prop_assert_eq!(ba.same_depth, bb.same_depth);
        Ok(())
    });
}

/// The lineal categories; `up()` is each one's meioses up.
const LINEAL: [Category; 6] = [
    Category::MO,
    Category::FO,
    Category::GP,
    Category::GGP,
    Category::GGGP,
    Category::G3GP,
];

/// Each strict ancestor row of `row`, with its fewest meioses up.
fn ancestors(b: &Built, row: usize) -> HashMap<usize, u32> {
    let mut nearest = HashMap::new();
    let mut frontier = vec![(row, 0u32)];
    while let Some((r, d)) = frontier.pop() {
        for p in b.parent_rows(r) {
            let best = nearest.entry(p).or_insert(u32::MAX);
            if d + 1 < *best {
                *best = d + 1;
                frontier.push((p, d + 1));
            }
        }
    }
    nearest
}

/// The row-streaming engine against the lineage engine: a row's lineal
/// credits in its relatives (MO + FO + GP + GGP + GGGP + G3GP, each pair
/// crediting the descendant) are at most its distinct ancestors, each
/// lineal pair's senior is an ancestor of its junior, and an MO or FO
/// senior is the junior's mother or father row.
///
/// When the row is at depth at most 5, neither it nor any ancestor has an
/// external parent, and neither it nor any ancestor is inbred, each
/// ancestor is in the lineal category of its fewest meioses up with the
/// row, so the credits are the distinct ancestor count.  The row's own
/// `F = 0` is needed: an inbred row's ancestor can be a closer relative
/// (`an_inbred_rows_grandmother_can_be_its_half_sib`).
#[test]
fn lineal_credits_are_the_distinct_ancestors() {
    let strategy = prop_oneof![pedigree(), non_inbred_pedigree()];
    check("lineal credits", 3.0, strategy, |ped, checked| {
        let b = Built::new(&ped);
        let distinct = distinct_ancestor_counts(b.lineage()).expect("ancestor counts");
        let rel = relatives(&b, &vec![0.0; b.len()]);
        let all = blocks(&b, Execution::Speed);
        let f = inbreeding(b.kinship()).expect("inbreeding");
        let external = |r: usize| {
            (b.graph.mother_ids[r] >= 0 && b.graph.mother_rows[r] < 0)
                || (b.graph.father_ids[r] >= 0 && b.graph.father_rows[r] < 0)
        };
        let mut lineal_of: Vec<HashMap<usize, Category>> = vec![HashMap::new(); b.len()];
        let mut equal = 0;
        for cat in LINEAL {
            for (junior, senior) in pairs(&all, cat) {
                let parent = match cat {
                    Category::MO => Some(b.graph.mother_rows[junior]),
                    Category::FO => Some(b.graph.father_rows[junior]),
                    _ => None,
                };
                if let Some(parent) = parent {
                    prop_assert_eq!(
                        parent,
                        senior as i32,
                        "{} ({}, {})",
                        cat.code(),
                        junior,
                        senior
                    );
                }
                lineal_of[junior].insert(senior, cat);
            }
        }
        for row in 0..b.len() {
            let lineal: u32 = LINEAL.iter().map(|&cat| credits(&rel, row, cat)[0]).sum();
            let n = i64::from(distinct[row]);
            prop_assert!(
                i64::from(lineal) <= n,
                "row {row}: {lineal} credits, {n} ancestors"
            );
            prop_assert_eq!(lineal as usize, lineal_of[row].len(), "row {}", row);
            let up = ancestors(&b, row);
            prop_assert_eq!(up.len() as i64, n, "row {}", row);
            for senior in lineal_of[row].keys() {
                prop_assert!(
                    up.contains_key(senior),
                    "row {row}: {senior} not an ancestor"
                );
            }
            let clean = b.depth[row] <= 5
                && f[row] == 0.0
                && !external(row)
                && up.keys().all(|&p| !external(p) && f[p] == 0.0);
            if clean {
                let want: HashMap<usize, Category> = up
                    .iter()
                    .map(|(&p, &d)| {
                        let cat = match d {
                            1 if b.graph.mother_rows[row] == p as i32 => Category::MO,
                            1 => Category::FO,
                            _ => *LINEAL
                                .iter()
                                .find(|cat| u32::from(cat.up()) == d)
                                .expect("depth <= 5"),
                        };
                        (p, cat)
                    })
                    .collect();
                prop_assert_eq!(&lineal_of[row], &want, "row {}", row);
                prop_assert_eq!(i64::from(lineal), n, "row {}", row);
                equal += 1;
            }
        }
        tally(checked, equal);
        Ok(())
    });
}

/// The case proptest found against lineal credits equalling the distinct
/// ancestors at depth at most 5 with no external parent and no inbred
/// ancestor, when the row itself may be inbred.  Row 4's father (row 1) is
/// also its great-grandfather, so its maternal grandmother (row 2, a
/// daughter of row 1) is also its paternal half sib: the closer PHS wins,
/// and row 4 has four ancestors but three lineal credits.  Row 4 itself is
/// inbred (F = 1/8).
#[test]
fn an_inbred_rows_grandmother_can_be_its_half_sib() {
    let ped = Ped {
        ids: vec![1, 2, 3, 4, 5],
        mother: vec![-1, -1, 1, 3, 4],
        father: vec![-1, -1, 2, -1, 2],
        twin: vec![-1; 5],
    };
    let b = Built::new(&ped);
    let all = blocks(&b, Execution::Speed);
    assert_eq!(pairs(&all, Category::PHS).collect::<Vec<_>>(), vec![(2, 4)]);
    assert!(!pairs(&all, Category::GP).any(|p| p == (4, 2)));
    assert_eq!(distinct_ancestor_counts(b.lineage()).expect("counts")[4], 4);
    let rel = relatives(&b, &[0.0; 5]);
    let lineal: u32 = LINEAL.iter().map(|&cat| credits(&rel, 4, cat)[0]).sum();
    assert_eq!(lineal, 3);
    assert_eq!(inbreeding(b.kinship()).expect("inbreeding")[4], 0.125);
}

/// The moments of two pedigrees side by side are the merge of each one's
/// moments, under both symmetric rules: every accumulator and every
/// derived float.
#[test]
fn moments_of_a_disjoint_union_are_the_merge_of_the_parts() {
    let strategy = (
        pedigree(),
        pedigree(),
        prop::collection::vec(any::<u8>(), 50),
    );
    check(
        "moments union",
        0.5,
        strategy,
        |(pa, pb, bytes), checked| {
            let union = pa.disjoint_union(&pb);
            let (a, b, u) = (Built::new(&pa), Built::new(&pb), Built::new(&union));
            let inputs = Inputs::new(u.len(), &bytes);
            let products = products();
            for symmetric in [Symmetric::Canonical, Symmetric::Both] {
                let ma = moments(&a, &inputs.rows(0..a.len()), &products, symmetric);
                let mb = moments(&b, &inputs.rows(a.len()..u.len()), &products, symmetric);
                let mu = moments(&u, &inputs, &products, symmetric);
                let merged = ma.merge(&mb).expect("one layout");
                // The counts, and the sums and cross sums below, are every
                // accumulator, exact as floats at these magnitudes.  The
                // exponents are not compared: every input is an integer,
                // so `moments` gives every table exponent 0.
                prop_assert_eq!(merged.counts(), mu.counts(), "{}", symmetric.name());
                let bits = |v: Vec<f64>| v.into_iter().map(f64::to_bits).collect::<Vec<_>>();
                for statistic in Statistic::ALL {
                    let n = if statistic.per_product() {
                        products.len()
                    } else {
                        2
                    };
                    for index in 0..n {
                        let x = merged.derive(statistic, index, "x").expect("derive");
                        let y = mu.derive(statistic, index, "x").expect("derive");
                        prop_assert_eq!(bits(x), bits(y), "{} {}", statistic.name(), index);
                    }
                }
                tally(
                    checked,
                    usize::from(ma.total_pairs() > 0 && mb.total_pairs() > 0),
                );
            }
            Ok(())
        },
    );
}

/// The graph's pairs restricted to the view `graph_rows`, built without
/// the engine's view map: pairs with both rows selected, in view rows, a
/// symmetric category's pairs lower view row first and an asymmetric one's
/// in their semantic orientation, each block in view-space canonical-key
/// order; with every category's kept pair count.
fn restricted(graph: &PairBlocks, graph_rows: &[i32], n: usize) -> (PairBlocks, Counts) {
    let mut to_view: Vec<Option<i32>> = vec![None; n];
    for (view_row, &row) in graph_rows.iter().enumerate() {
        to_view[row as usize] = Some(view_row as i32);
    }
    let mut blocks = PairBlocks::empty();
    let mut counts = Counts::default();
    for cat in Category::ALL {
        let block = graph.get(cat);
        let mut kept: Vec<(i32, i32)> = block
            .first
            .iter()
            .zip(&block.second)
            .filter_map(|(&a, &b)| {
                let (a, b) = (to_view[a as usize]?, to_view[b as usize]?);
                Some(if cat.symmetric() && a > b {
                    (b, a)
                } else {
                    (a, b)
                })
            })
            .collect();
        kept.sort_unstable_by_key(|&(a, b)| (a.min(b), a.max(b)));
        counts.0[cat.index()] = kept.len() as u64;
        let out = &mut blocks.0[cat.index()];
        out.first = kept.iter().map(|p| p.0).collect();
        out.second = kept.iter().map(|p| p.1).collect();
    }
    (blocks, counts)
}

/// A pedigree and a view of it: a shuffled selection of its rows, of any
/// size from none to all.
fn viewed_pedigree() -> impl Strategy<Value = (Ped, Vec<i32>)> {
    pedigree().prop_flat_map(|ped| {
        let n = ped.len();
        let rows = Just((0..n as i32).collect::<Vec<_>>()).prop_shuffle();
        (Just(ped), rows, 0..=n).prop_map(|(ped, mut rows, k)| {
            rows.truncate(k);
            (ped, rows)
        })
    })
}

/// On a random view, both paths' pairs (both executions), counts and
/// relatives are the whole graph's restricted to the view: unselected rows
/// still connect the selected ones, and compaction drops only rows no
/// selected pair needs.
#[test]
fn a_view_is_the_graph_restricted_to_it_on_either_path() {
    check(
        "view subsets",
        3.4,
        viewed_pedigree(),
        |(ped, rows), checked| {
            let b = Built::new(&ped);
            let rel = b.relationships();
            let all = CategorySet::up_to_degree(MaxDegree::MAX.get());
            let progress = Progress::default();
            let graph = blocks(&b, Execution::Speed);
            let (want_blocks, want_counts) = restricted(&graph, &rows, b.len());
            let on = |compaction| Receiver::View {
                graph_rows: &rows,
                compaction,
            };
            // Each pair credits its first member, and a symmetric one both:
            // one relative, and one more when the other member's relative
            // value passes the threshold.
            let relative: Vec<f64> = (0..rows.len()).map(|v| (v % 5) as f64).collect();
            let threshold = 2.0;
            let stride = 2;
            let mut want_relatives = vec![0u32; rows.len() * Category::ALL.len() * stride];
            for cat in Category::ALL {
                let block = want_blocks.get(cat);
                for (&x, &y) in block.first.iter().zip(&block.second) {
                    let (x, y) = (x as usize, y as usize);
                    let credited: &[(usize, usize)] = if cat.symmetric() {
                        &[(x, y), (y, x)]
                    } else {
                        &[(x, y)]
                    };
                    for &(row, other) in credited {
                        let base = (row * Category::ALL.len() + cat.index()) * stride;
                        want_relatives[base] += 1;
                        want_relatives[base + 1] += u32::from(relative[other] <= threshold);
                    }
                }
            }
            let columns = [ThresholdColumn {
                relative: &relative,
                threshold: Threshold::Scalar(threshold),
            }];
            for compaction in [Compaction::Full, Compaction::Compact] {
                for execution in [Execution::Speed, Execution::Memory] {
                    let got = query::pairs(&rel, all, on(compaction), execution, &progress)
                        .expect("view pairs");
                    prop_assert_eq!(&got, &want_blocks, "{:?} {}", compaction, execution.name());
                }
                let counts = query::counts(&rel, all, on(compaction), &progress).expect("counts");
                prop_assert_eq!(counts.0, want_counts.0, "{:?}", compaction);
                let got = query::relatives(&rel, all, on(compaction), &columns, THREADS, &progress)
                    .expect("view relatives");
                prop_assert_eq!(&got.counts, &want_relatives, "relatives {:?}", compaction);
            }
            tally(checked, want_blocks.total());
            Ok(())
        },
    );
}
