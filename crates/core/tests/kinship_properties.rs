//! Properties of the kinship matrix, pairwise kinship and inbreeding on
//! random pedigrees: the invariants `tests/test_kinship_properties.py` and
//! `tests/test_inbreeding_properties.py` check through the Python binding,
//! held here against the core every host calls.
//!
//! Kinship is the pinned float32 recurrence of ADR 0009, so the matrix and
//! the pairwise walk agree to the bit; comparisons against a float64 value
//! use the envelope of `tests/_support.py`.

#![allow(
    clippy::expect_used,
    clippy::panic,
    reason = "test helpers outside #[test] functions: a panic fails the test"
)]
#![allow(
    clippy::needless_range_loop,
    reason = "rows index the matrix both ways round and the parent columns"
)]

mod common;

use common::{
    check, non_inbred_pedigree, pedigree, relabelled_pedigree, shuffled_pedigree, Built, Ped,
};
use pedigree_graph_core::kinship::{inbreeding, pair_kinship};
use pedigree_graph_core::relationships::query::{self, Receiver};
use pedigree_graph_core::relationships::{CategorySet, Progress};
use proptest::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};

/// One float32 rounding unit at kinship scale (`tests/_support.py`).
const ENVELOPE_UNIT: f64 = 1.0 / (1u64 << 25) as f64;

/// The largest `d_a + d_b` at which float32 kinship equals the rational
/// (`tests/_support.py`, derived in `tests/test_kinship_exact.py`).
const EXACT_DEPTH_SUM: i32 = 22;

/// The largest depth at which float64 `F` equals the rational
/// (`tests/_support.py`).
const EXACT_F_DEPTH: i32 = 25;

fn tally(checked: &AtomicUsize, n: usize) {
    checked.fetch_add(n, Ordering::Relaxed);
}

/// The CSC is mirrored from one stored value per pair, so it is symmetric
/// to the bit; a float32 half-sum of values in [0, 1] stays in [0, 1].
#[test]
fn the_matrix_is_symmetric_and_bounded() {
    check("symmetric", 4.0, pedigree(), |ped, checked| {
        let b = Built::new(&ped);
        let k = b.dense_kinship();
        for x in 0..b.len() {
            prop_assert!(k[x][x] >= 0.5, "diagonal {x}: {}", k[x][x]);
            for y in 0..b.len() {
                prop_assert_eq!(k[x][y].to_bits(), k[y][x].to_bits(), "({}, {})", x, y);
                prop_assert!((0.0..=1.0).contains(&k[x][y]), "({x}, {y}): {}", k[x][y]);
            }
        }
        tally(checked, b.len());
        Ok(())
    });
}

/// A row without parent rows has self-kinship 1/2 and no kinship with
/// another such row unless they are co-twins, whose kinship is the
/// self-kinship.
#[test]
fn founders_are_unrelated_with_self_kinship_one_half() {
    check("founders", 2.0, pedigree(), |ped, checked| {
        let b = Built::new(&ped);
        let k = b.dense_kinship();
        let founders: Vec<usize> = (0..b.len())
            .filter(|&r| b.parent_rows(r).next().is_none())
            .collect();
        for (at, &x) in founders.iter().enumerate() {
            prop_assert_eq!(k[x][x], 0.5);
            for &y in &founders[at + 1..] {
                let want = if b.twin_of(x) == Some(y) { 0.5 } else { 0.0 };
                prop_assert_eq!(k[x][y], want, "({}, {})", x, y);
            }
        }
        tally(checked, founders.len());
        Ok(())
    });
}

/// Every co-twin pair has the kinship of either twin with itself.
#[test]
fn co_twins_have_self_kinship() {
    check("co-twins", 0.5, pedigree(), |ped, checked| {
        let b = Built::new(&ped);
        let k = b.dense_kinship();
        for x in 0..b.len() {
            if let Some(t) = b.twin_of(x) {
                prop_assert_eq!(k[x][t].to_bits(), k[x][x].to_bits(), "({}, {})", x, t);
                prop_assert_eq!(k[t][t].to_bits(), k[x][x].to_bits(), "({}, {})", x, t);
                tally(checked, 1);
            }
        }
        Ok(())
    });
}

/// With unrelated parents, a child's kinship with each parent row is 1/4.
#[test]
fn parent_offspring_is_one_quarter_when_not_inbred() {
    check(
        "parent-offspring",
        4.0,
        non_inbred_pedigree(),
        |ped, checked| {
            let b = Built::new(&ped);
            let k = b.dense_kinship();
            for child in 0..b.len() {
                for parent in b.parent_rows(child) {
                    prop_assert_eq!(k[child][parent], 0.25, "({}, {})", child, parent);
                    tally(checked, 1);
                }
            }
            Ok(())
        },
    );
}

/// `phi(i, j) = (phi(m_i, j) + phi(f_i, j)) / 2` for `j` not a descendant of
/// `i`, not `i` and not its co-twin; a missing parent contributes 0.  The
/// matrix stores the correctly rounded float32 half-sum when it peels `i`
/// (deeper, or a depth tie with `i` the greater row, ADR 0009); the float64
/// sum of two float32 values is exact, so the cast is that one rounding.  A
/// tie it peels from `j` reaches the same rational by `j`'s parents: equal
/// while every value is exact, inside the two-path envelope beyond.
#[test]
fn the_kinship_recursion_holds() {
    check("recursion", 30.0, pedigree(), |ped, checked| {
        let b = Built::new(&ped);
        let k = b.dense_kinship();
        let d = &b.depth;
        for i in 0..b.len() {
            let (m, f) = (b.graph.mother_rows[i], b.graph.father_rows[i]);
            if m < 0 && f < 0 {
                continue;
            }
            let of = |p: i32, j: usize| {
                if p < 0 {
                    0.0
                } else {
                    f64::from(k[p as usize][j])
                }
            };
            for j in 0..b.len() {
                if j == i || b.twin_of(i) == Some(j) || d[j] > d[i] {
                    continue;
                }
                let step = (0.5 * (of(m, j) + of(f, j))) as f32;
                let depth_sum = d[i] + d[j];
                if d[i] > d[j] || i > j || depth_sum <= EXACT_DEPTH_SUM {
                    prop_assert_eq!(k[i][j].to_bits(), step.to_bits(), "({}, {})", i, j);
                } else {
                    let bound = 2.0 * f64::from(depth_sum + 1) * ENVELOPE_UNIT;
                    prop_assert!((f64::from(k[i][j]) - f64::from(step)).abs() <= bound);
                }
                tally(checked, 1);
            }
        }
        Ok(())
    });
}

/// The pairwise walk and the matrix DP implement one pinned recurrence, so
/// they agree to the bit on every pair, self pairs included.
#[test]
fn pair_kinship_is_the_matrix_entry() {
    check("pair kinship", 30.0, pedigree(), |ped, checked| {
        let b = Built::new(&ped);
        let k = b.dense_kinship();
        let (mut first, mut second) = (Vec::new(), Vec::new());
        for x in 0..b.len() {
            for y in x..b.len() {
                first.push(x as i32);
                second.push(y as i32);
            }
        }
        let got = pair_kinship(b.kinship(), &first, &second).expect("rows in range");
        for ((&x, &y), v) in first.iter().zip(&second).zip(&got) {
            let want = k[x as usize][y as usize];
            prop_assert_eq!(v.to_bits(), want.to_bits(), "({}, {})", x, y);
        }
        tally(checked, got.len());
        Ok(())
    });
}

/// Relabelling the ids resolves to the same rows, so the matrix and the
/// relationship counts (which group siblings by parent id) are unchanged.
#[test]
fn ids_do_not_change_kinship_or_counts() {
    check(
        "relabel",
        1.0,
        relabelled_pedigree(),
        |(ped, perm, mult), checked| {
            let (a, b) = (Built::new(&ped), Built::new(&ped.relabelled(&perm, mult)));
            prop_assert_eq!(&a.graph.mother_rows, &b.graph.mother_rows);
            prop_assert_eq!(&a.graph.father_rows, &b.graph.father_rows);
            prop_assert_eq!(&a.graph.twin_rows, &b.graph.twin_rows);
            prop_assert_eq!(a.dense_kinship(), b.dense_kinship());
            let count = |x: &Built| {
                query::counts(
                    &x.relationships(),
                    CategorySet::up_to_degree(3),
                    Receiver::Graph,
                    &Progress::default(),
                )
                .expect("counts")
            };
            prop_assert_eq!(count(&a).0, count(&b).0);
            tally(checked, 1);
            Ok(())
        },
    );
}

/// Reordering the rows moves each row's kinship and `F` with it.  Exact
/// values do not depend on which endpoint a depth tie peels, so they move
/// to the bit; past `EXACT_DEPTH_SUM` the two orders may round along
/// different paths, inside the two-path envelope.
#[test]
fn row_order_does_not_change_kinship_or_inbreeding() {
    check(
        "row order",
        1.0,
        shuffled_pedigree(),
        |(ped, order), checked| {
            let (a, b) = (Built::new(&ped), Built::new(&ped.permuted(&order)));
            let (ka, kb) = (a.dense_kinship(), b.dense_kinship());
            let fa = inbreeding(a.kinship()).expect("inbreeding");
            let fb = inbreeding(b.kinship()).expect("inbreeding");
            for (r, &x) in order.iter().enumerate() {
                prop_assert_eq!(b.depth[r], a.depth[x], "depth of row {}", r);
                let d = a.depth[x];
                if d <= EXACT_F_DEPTH {
                    prop_assert_eq!(fb[r], fa[x], "F of row {}", r);
                } else {
                    let bound = 2.0 * f64::from(2 * d + 1) * ENVELOPE_UNIT;
                    prop_assert!((fb[r] - fa[x]).abs() <= bound, "F of row {r}");
                }
                for (s, &y) in order.iter().enumerate() {
                    let depth_sum = a.depth[x] + a.depth[y];
                    if depth_sum <= EXACT_DEPTH_SUM {
                        prop_assert_eq!(kb[r][s].to_bits(), ka[x][y].to_bits(), "({}, {})", r, s);
                    } else {
                        let bound = 2.0 * f64::from(depth_sum + 1) * ENVELOPE_UNIT;
                        prop_assert!((f64::from(kb[r][s]) - f64::from(ka[x][y])).abs() <= bound);
                    }
                }
            }
            tally(checked, 1);
            Ok(())
        },
    );
}

/// `F >= 0` and `F = 2 phi(i, i) - 1`: exact while `2 d <= 22`, within the
/// envelope beyond.
#[test]
fn inbreeding_is_the_matrix_diagonal() {
    check("inbreeding diagonal", 4.0, pedigree(), |ped, checked| {
        let b = Built::new(&ped);
        let k = b.dense_kinship();
        let f = inbreeding(b.kinship()).expect("inbreeding");
        for (i, &fi) in f.iter().enumerate() {
            prop_assert!(fi >= 0.0, "F[{i}] = {fi}");
            let from_diagonal = 2.0 * f64::from(k[i][i]) - 1.0;
            let d = b.depth[i];
            if 2 * d <= EXACT_DEPTH_SUM {
                prop_assert_eq!(fi, from_diagonal, "row {}", i);
            } else {
                let bound = 2.0 * f64::from(2 * d + 1) * ENVELOPE_UNIT;
                prop_assert!((fi - from_diagonal).abs() <= bound, "row {i}");
            }
        }
        tally(checked, f.len());
        Ok(())
    });
}

/// `F` is the kinship of the parents, and 0 without two parent rows.
#[test]
fn inbreeding_is_the_parents_kinship() {
    check("inbreeding parents", 0.8, pedigree(), |ped, checked| {
        let b = Built::new(&ped);
        assert_inbreeding_is_parent_kinship(&b, checked)
    });
}

fn assert_inbreeding_is_parent_kinship(
    b: &Built,
    checked: &AtomicUsize,
) -> Result<(), TestCaseError> {
    let f = inbreeding(b.kinship()).expect("inbreeding");
    let (mut rows, mut mothers, mut fathers) = (Vec::new(), Vec::new(), Vec::new());
    for (i, &fi) in f.iter().enumerate() {
        let (m, fa) = (b.graph.mother_rows[i], b.graph.father_rows[i]);
        if m < 0 || fa < 0 {
            prop_assert_eq!(fi, 0.0, "row {}", i);
        } else {
            rows.push(i);
            mothers.push(m);
            fathers.push(fa);
        }
    }
    let phi = pair_kinship(b.kinship(), &mothers, &fathers).expect("rows in range");
    for (&i, &p) in rows.iter().zip(&phi) {
        // `np.allclose`'s bound, as the Python test has it: float32 phi
        // drifts from float64 F on deep pedigrees.
        let p = f64::from(p);
        prop_assert!(
            (f[i] - p).abs() <= 1e-9 + 1e-5 * p.abs(),
            "row {i}: F {} phi {p}",
            f[i]
        );
    }
    tally(checked, rows.len());
    Ok(())
}

/// Kinship values scaled by `2^SCALE` are integers: a float32 in [0, 1]
/// above `2^-(SCALE - 23)` has no bit below `2^-SCALE`, and the pedigrees
/// here never reach kinship that small.
const SCALE: i32 = 100;

/// `v' Phi v` exactly, for the float32 matrix `k` as stored: every entry is
/// a dyadic rational, so the sum is an integer in units of `2^-SCALE`.
/// With entries at most `2^SCALE`, `|v_i| <= 3` and 25 rows, it is below
/// `625 * 9 * 2^SCALE < 2^113` and fits an i128.
fn quadratic_form(k: &[Vec<f32>], v: &[i32]) -> Result<i128, TestCaseError> {
    let unit = 2f64.powi(SCALE);
    let mut total = 0i128;
    for (x, &vx) in v.iter().enumerate() {
        for (y, &vy) in v.iter().enumerate() {
            let scaled = f64::from(k[x][y]) * unit;
            prop_assert!(
                scaled.fract() == 0.0,
                "({x}, {y}) is not a multiple of 2^-{SCALE}"
            );
            total += i128::from(vx * vy) * scaled as i128;
        }
    }
    Ok(total)
}

/// The kinship matrix is positive semidefinite: `v' Phi v >= 0`, computed
/// exactly on the stored float32 values, for random small-integer `v`, for
/// each child's `2 e_c - e_m - e_f` (whose form is `2 - phi_mm - phi_ff`,
/// the Mendelian sampling term, exactly while every entry it reads is the
/// rational, `2 d_c <= EXACT_DEPTH_SUM`), and `0` for each co-twin pair's
/// `e_x - e_t`, the null direction that makes an MZ pedigree's matrix
/// singular.
#[test]
fn the_matrix_is_positive_semidefinite() {
    let vectors = prop::collection::vec(prop::collection::vec(-3i32..=3, 25), 8);
    check(
        "semidefinite",
        9.0,
        (pedigree(), vectors),
        |(ped, vs), checked| {
            let b = Built::new(&ped);
            let n = b.len();
            let k = b.dense_kinship();
            for v in &vs {
                let q = quadratic_form(&k, &v[..n])?;
                prop_assert!(q >= 0, "v' Phi v = {q} * 2^-{SCALE} for {:?}", &v[..n]);
                tally(checked, 1);
            }
            for c in 0..n {
                let (m, f) = (b.graph.mother_rows[c], b.graph.father_rows[c]);
                if m < 0 || f < 0 {
                    continue;
                }
                let mut v = vec![0i32; n];
                v[c] = 2;
                v[m as usize] -= 1;
                v[f as usize] -= 1;
                let q = quadratic_form(&k, &v)?;
                let (pm, pf) = (
                    f64::from(k[m as usize][m as usize]),
                    f64::from(k[f as usize][f as usize]),
                );
                prop_assert!(q >= 0, "child {c}: {q}");
                // The deepest entry the form reads is the child's own, at
                // depth sum 2 d_c; past the exact depth it may be rounded.
                if 2 * b.depth[c] <= EXACT_DEPTH_SUM {
                    prop_assert_eq!(q as f64 / 2f64.powi(SCALE), 2.0 - pm - pf, "child {}", c);
                }
                tally(checked, 1);
            }
            for x in 0..n {
                if let Some(t) = b.twin_of(x) {
                    let mut v = vec![0i32; n];
                    v[x] = 1;
                    v[t] = -1;
                    prop_assert_eq!(quadratic_form(&k, &v)?, 0, "co-twins ({}, {})", x, t);
                    tally(checked, 1);
                }
            }
            Ok(())
        },
    );
}

/// Full-sib mating: child 4 of sibs 2 and 3 has `F = phi(2, 3) = 1/4`.
#[test]
fn full_sib_mating_inbreeds_one_quarter() {
    let ped = Ped {
        ids: vec![0, 1, 2, 3, 4],
        mother: vec![-1, -1, 0, 0, 2],
        father: vec![-1, -1, 1, 1, 3],
        twin: vec![-1; 5],
    };
    let b = Built::new(&ped);
    assert_eq!(inbreeding(b.kinship()).expect("inbreeding")[4], 0.25);
    let checked = AtomicUsize::new(0);
    assert_inbreeding_is_parent_kinship(&b, &checked).expect("F is the parents' kinship");
    assert_eq!(checked.into_inner(), 3);
}
