//! The private topological row order shared by every order-dependent kernel.
//!
//! Public graph coordinates are input rows in whatever acyclic order the host
//! supplied, but several kernels need parents to precede children in the index
//! space they sweep: the Meuwissen-Luo inbreeding walk, the descendant
//! path-count reverse sweep, the pairwise kinship peel, the kinship DP, and the
//! Caballero-Toro forward sweep.  Giving each one its own order would be five
//! permutations to keep consistent, so this module builds one for all of them,
//! stable depth-major, plus the maps that move rows between graph space and it.
//!
//! Stable depth-major is topological because a child's structural depth
//! strictly exceeds both parents'.  Ties within a depth keep input row order,
//! so no original id ever influences the result.  When the input rows are
//! already depth-major the permutation is the identity, and [`Order::Identity`]
//! records that without allocating either map.
//!
//! Depth and the permutation are built separately.  [`structural_depth`] is
//! useful on its own, and folding the sort into it would tax every graph
//! whether or not an order-dependent kernel is ever called.
//!
//! Host `-1` sentinels are the only absent-parent state here.  The
//! missing/external distinction the domain model draws is a separate concern
//! and does not change any order this module computes.

use crate::error::Error;
use std::collections::VecDeque;

/// The stable depth-major permutation, or the identity when there is none.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Order {
    /// The graph rows are already depth-major; both maps would be `0..n`.
    Identity,
    /// `order[k]` is the graph row at topological position `k`; `inverse[row]`
    /// is that row's position.  `i64` because the Python side exposes both as
    /// `np.intp`.
    Permuted {
        /// Topological position to graph row.
        order: Vec<i64>,
        /// Graph row to topological position.
        inverse: Vec<i64>,
    },
}

/// Whether every represented parent already precedes its child.
///
/// `mother` and `father` must have one entry per row and equal length; the host
/// binding enforces that.  An absent parent is `-1`, which is below every row
/// index, so one comparison covers both the represented and the absent case.
pub fn is_topological(mother: &[i32], father: &[i32]) -> bool {
    mother
        .iter()
        .zip(father)
        .enumerate()
        .all(|(row, (&mother, &father))| {
            i64::from(mother) < row as i64 && i64::from(father) < row as i64
        })
}

/// Structural depth per row: founders 0, otherwise `max(parents) + 1`.
///
/// The parent edges must be acyclic and every represented reference must name a
/// valid row; callers run [`cycle_witness`] first.  `mother` and `father` must
/// have one entry per row and equal length.
///
/// An absent parent contributes 0, so a row whose only represented parent sits
/// at depth `d` gets `d + 1`.
///
/// Each row is resolved once by an explicit-stack DFS over its parents, which
/// is linear in the number of rows rather than the `O(n * depth)` of a
/// fixed-point sweep.  Only unvisited parents are pushed, so cyclic input
/// terminates with a meaningless answer instead of looping forever.
pub fn structural_depth(mother: &[i32], father: &[i32]) -> Vec<i32> {
    const UNVISITED: u8 = 0;
    const ON_STACK: u8 = 1;
    const DONE: u8 = 2;

    let n = mother.len();
    let mut depth = vec![0i32; n];
    let mut state = vec![UNVISITED; n];
    let mut stack: Vec<usize> = Vec::new();

    for root in 0..n {
        if state[root] != UNVISITED {
            continue;
        }
        stack.push(root);
        while let Some(row) = stack.pop() {
            match state[row] {
                UNVISITED => {
                    state[row] = ON_STACK;
                    stack.push(row);
                    for parent in [mother[row], father[row]] {
                        if parent >= 0 && state[parent as usize] == UNVISITED {
                            stack.push(parent as usize);
                        }
                    }
                }
                ON_STACK => {
                    let mut resolved = 0i32;
                    let mut has_parent = false;
                    for parent in [mother[row], father[row]] {
                        if parent >= 0 {
                            has_parent = true;
                            resolved = resolved.max(depth[parent as usize]);
                        }
                    }
                    depth[row] = if has_parent { resolved + 1 } else { 0 };
                    state[row] = DONE;
                }
                _ => {}
            }
        }
    }
    depth
}

/// The stable depth-major permutation of `depth`, ties keeping row order.
///
/// `depth` must be non-negative, which [`structural_depth`] guarantees; the
/// values key the counting sort's buckets.
///
/// A stable sort by depth is the identity exactly when `depth` is
/// non-decreasing, so that one linear scan decides [`Order::Identity`] and the
/// common already-ordered graph allocates nothing.  Otherwise a counting sort
/// keyed by depth is naturally stable and runs in `O(n + max_depth)`.
pub fn depth_major_order(depth: &[i32]) -> Order {
    if depth.windows(2).all(|w| w[0] <= w[1]) {
        return Order::Identity;
    }
    let n = depth.len();
    let max_depth = depth.iter().copied().max().unwrap_or(0) as usize;

    let mut starts = vec![0usize; max_depth + 2];
    for &d in depth {
        starts[d as usize + 1] += 1;
    }
    for bucket in 1..starts.len() {
        starts[bucket] += starts[bucket - 1];
    }

    let mut order = vec![0i64; n];
    let mut inverse = vec![0i64; n];
    for (row, &d) in depth.iter().enumerate() {
        let position = starts[d as usize];
        starts[d as usize] += 1;
        order[position] = row as i64;
        inverse[row] = position as i64;
    }
    Order::Permuted { order, inverse }
}

/// One deterministic cycle witness, or `None` when the parent edges are acyclic.
///
/// `ids`, `mother` and `father` must have one entry per row and equal length;
/// the host binding enforces that.
///
/// The witness is chosen by id rather than by row, so the same graph reports
/// the same cycle whatever order its rows arrive in: Kahn's algorithm peels
/// everything reachable from an indegree-zero row, the walk starts at the
/// smallest id left behind, and at each step takes the smallest-id parent that
/// is still unpeeled.  The returned ids are rotated so the smallest comes
/// first.
///
/// Every unpeeled row has at least one unpeeled represented parent, so the walk
/// cannot dead-end.  A dead end would mean that invariant is broken, and yields
/// `None` rather than a panic.
pub fn cycle_witness(ids: &[i64], mother: &[i32], father: &[i32]) -> Option<Vec<i64>> {
    let n = ids.len();
    let peeled = kahn_peel(mother, father, n);
    let start = (0..n)
        .filter(|&row| !peeled[row])
        .min_by_key(|&row| ids[row])?;

    let mut walk: Vec<usize> = Vec::new();
    let mut visited = vec![-1i32; n];
    let mut row = start;
    let entry = loop {
        if visited[row] >= 0 {
            break Some(visited[row] as usize);
        }
        visited[row] = walk.len() as i32;
        walk.push(row);
        let mut next: Option<usize> = None;
        for parent in [mother[row], father[row]] {
            if parent < 0 || peeled[parent as usize] {
                continue;
            }
            let parent = parent as usize;
            if next.is_none_or(|best| ids[parent] < ids[best]) {
                next = Some(parent);
            }
        }
        match next {
            Some(parent) => row = parent,
            None => break None,
        }
    };

    let cycle = &walk[entry?..];
    let mut smallest = 0usize;
    for (position, &row) in cycle.iter().enumerate() {
        if ids[row] < ids[cycle[smallest]] {
            smallest = position;
        }
    }
    Some(
        cycle[smallest..]
            .iter()
            .chain(&cycle[..smallest])
            .map(|&row| ids[row])
            .collect(),
    )
}

/// [`cycle_witness`] as a validation gate, raising [`Error::Cycle`].
///
/// `ids`, `mother` and `father` must have one entry per row and equal length.
pub fn validate_acyclic(ids: &[i64], mother: &[i32], father: &[i32]) -> Result<(), Error> {
    match cycle_witness(ids, mother, father) {
        Some(ids) => Err(Error::Cycle { ids }),
        None => Ok(()),
    }
}

/// Mask of the rows Kahn's algorithm could peel; the rest are the cyclic core.
///
/// The peeled set is the same whichever order the queue drains, so the
/// child lists below are built in whatever order the rows supply them.
fn kahn_peel(mother: &[i32], father: &[i32], n: usize) -> Vec<bool> {
    let (starts, children) = children_by_parent(mother, father, n);
    let mut indegree = vec![0u8; n];
    for (row, (&mother, &father)) in mother.iter().zip(father).enumerate() {
        indegree[row] = u8::from(mother >= 0) + u8::from(father >= 0);
    }

    let mut peeled = vec![false; n];
    let mut queue: VecDeque<usize> = (0..n).filter(|&row| indegree[row] == 0).collect();
    while let Some(row) = queue.pop_front() {
        peeled[row] = true;
        for &child in &children[starts[row]..starts[row + 1]] {
            let child = child as usize;
            indegree[child] -= 1;
            if indegree[child] == 0 {
                queue.push_back(child);
            }
        }
    }
    peeled
}

/// CSR parent-to-children adjacency: `children[starts[p]..starts[p + 1]]`.
fn children_by_parent(mother: &[i32], father: &[i32], n: usize) -> (Vec<usize>, Vec<i32>) {
    let mut starts = vec![0usize; n + 1];
    for &parent in mother.iter().chain(father) {
        if parent >= 0 {
            starts[parent as usize + 1] += 1;
        }
    }
    for parent in 1..=n {
        starts[parent] += starts[parent - 1];
    }

    let mut cursor = starts.clone();
    let mut children = vec![0i32; starts[n]];
    for (row, (&mother, &father)) in mother.iter().zip(father).enumerate() {
        for parent in [mother, father] {
            if parent >= 0 {
                let slot = cursor[parent as usize];
                cursor[parent as usize] += 1;
                children[slot] = row as i32;
            }
        }
    }
    (starts, children)
}

#[cfg(test)]
mod tests {
    use super::{
        cycle_witness, depth_major_order, is_topological, structural_depth, validate_acyclic, Order,
    };
    use crate::error::Error;

    const DEPTH_MAJOR_MOTHER: [i32; 5] = [-1, -1, -1, 0, 3];
    const DEPTH_MAJOR_FATHER: [i32; 5] = [-1, -1, -1, 1, 1];

    fn permuted(order: &[i64], inverse: &[i64], rows: &[i32]) -> Vec<i32> {
        order
            .iter()
            .map(|&row| {
                let parent = rows[row as usize];
                if parent < 0 {
                    -1
                } else {
                    inverse[parent as usize] as i32
                }
            })
            .collect()
    }

    fn assert_orders_topologically(order: &Order, mother: &[i32], father: &[i32]) {
        let Order::Permuted { order, inverse } = order else {
            panic!("expected a permutation");
        };
        let mother = permuted(order, inverse, mother);
        let father = permuted(order, inverse, father);
        assert!(
            is_topological(&mother, &father),
            "applying the permutation must put both parents before every child"
        );
    }

    #[test]
    fn depth_major_fixture_needs_no_permutation() {
        let depth = structural_depth(&DEPTH_MAJOR_MOTHER, &DEPTH_MAJOR_FATHER);
        assert_eq!(depth, vec![0, 0, 0, 1, 2]);
        assert_eq!(depth_major_order(&depth), Order::Identity);
        assert!(is_topological(&DEPTH_MAJOR_MOTHER, &DEPTH_MAJOR_FATHER));
    }

    #[test]
    fn permuted_fixture_is_reordered_by_depth() {
        let mother = [1, -1, 0, -1, -1];
        let father = [4, -1, 4, -1, -1];
        let depth = structural_depth(&mother, &father);
        assert_eq!(depth, vec![1, 0, 2, 0, 0]);

        let order = depth_major_order(&depth);
        assert_eq!(
            order,
            Order::Permuted {
                order: vec![1, 3, 4, 0, 2],
                inverse: vec![3, 0, 4, 1, 2],
            }
        );
        assert!(!is_topological(&mother, &father));
        assert_orders_topologically(&order, &mother, &father);
    }

    #[test]
    fn children_first_fixture_is_reversed_stably() {
        let mother = [1, 2, -1, -1, -1];
        let father = [3, 3, -1, -1, -1];
        let depth = structural_depth(&mother, &father);
        assert_eq!(depth, vec![2, 1, 0, 0, 0]);

        let order = depth_major_order(&depth);
        assert_eq!(
            order,
            Order::Permuted {
                order: vec![2, 3, 4, 1, 0],
                inverse: vec![4, 3, 0, 1, 2],
            }
        );
        assert!(!is_topological(&mother, &father));
        assert_orders_topologically(&order, &mother, &father);

        let Order::Permuted { order, .. } = &order else {
            panic!("expected a permutation");
        };
        let gathered: Vec<i32> = order.iter().map(|&row| depth[row as usize]).collect();
        assert!(
            gathered.windows(2).all(|w| w[0] <= w[1]),
            "gathered depth must be ascending, got {gathered:?}"
        );
    }

    #[test]
    fn ties_within_a_depth_keep_row_order() {
        let mother = [-1, -1, -1, 0, 1];
        let father = [-1, -1, -1, 2, 2];
        let depth = structural_depth(&mother, &father);
        assert_eq!(depth, vec![0, 0, 0, 1, 1]);
        assert_eq!(depth_major_order(&depth), Order::Identity);
    }

    #[test]
    fn one_absent_parent_still_deepens_the_child() {
        let depth = structural_depth(&[-1, -1, 0], &[-1, -1, -1]);
        assert_eq!(depth, vec![0, 0, 1]);
        assert_eq!(depth_major_order(&depth), Order::Identity);
    }

    #[test]
    fn a_single_parent_chain_deepens_every_step() {
        let depth = structural_depth(&[-1, 0, 1, -1], &[-1, -1, -1, 2]);
        assert_eq!(depth, vec![0, 1, 2, 3]);
        assert_eq!(depth_major_order(&depth), Order::Identity);
    }

    #[test]
    fn an_empty_graph_is_the_identity() {
        let depth = structural_depth(&[], &[]);
        assert_eq!(depth, Vec::<i32>::new());
        assert_eq!(depth_major_order(&depth), Order::Identity);
        assert!(is_topological(&[], &[]));
    }

    #[test]
    fn cyclic_input_terminates_instead_of_looping() {
        assert_eq!(structural_depth(&[0], &[-1]).len(), 1);
        assert_eq!(structural_depth(&[1, 0], &[-1, -1]).len(), 2);
        assert_eq!(
            structural_depth(&[1, 2, 0, 0, -1], &[-1, 4, -1, -1, -1]).len(),
            5
        );
    }

    #[test]
    fn a_self_parent_is_its_own_witness() {
        assert_eq!(cycle_witness(&[7], &[0], &[-1]), Some(vec![7]));
    }

    #[test]
    fn a_two_row_cycle_is_rotated_to_the_smallest_id() {
        assert_eq!(cycle_witness(&[8, 3], &[1, 0], &[-1, -1]), Some(vec![3, 8]));
    }

    #[test]
    fn a_descendant_tail_is_excluded_from_the_witness() {
        let ids = [30, 20, 40, 10, 50];
        let mother = [1, 2, 0, 0, -1];
        let father = [-1, 4, -1, -1, -1];
        assert_eq!(
            cycle_witness(&ids, &mother, &father),
            Some(vec![20, 40, 30])
        );
    }

    #[test]
    fn the_walk_takes_the_smaller_id_parent() {
        let ids = [100, 50, 20, 5];
        let mother = [1, 0, 0, -1];
        let father = [2, 3, -1, -1];
        assert_eq!(cycle_witness(&ids, &mother, &father), Some(vec![20, 100]));
    }

    #[test]
    fn an_acyclic_graph_has_no_witness() {
        let ids = [10, 11, 12, 13, 14];
        assert_eq!(
            cycle_witness(&ids, &DEPTH_MAJOR_MOTHER, &DEPTH_MAJOR_FATHER),
            None
        );
    }

    #[test]
    fn validate_accepts_an_acyclic_graph() {
        let ids = [10, 11, 12, 13, 14];
        assert_eq!(
            validate_acyclic(&ids, &DEPTH_MAJOR_MOTHER, &DEPTH_MAJOR_FATHER),
            Ok(())
        );
    }

    #[test]
    fn validate_reports_the_witness_as_a_cycle_error() {
        let ids = [30, 20, 40, 10, 50];
        let mother = [1, 2, 0, 0, -1];
        let father = [-1, 4, -1, -1, -1];
        assert_eq!(
            validate_acyclic(&ids, &mother, &father),
            Err(Error::Cycle {
                ids: vec![20, 40, 30]
            })
        );
    }
}
