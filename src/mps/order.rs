//! Graph-to-chain mapping.
//!
//! The ordering fixes every edge span, and the sum of spans fixes the cost, so
//! this pass is worth writing: on `advantage2-system1` reverse Cuthill-McKee
//! cuts the span sum from 76.1 million to 7.5 million against the received node
//! order. Spectral ordering by the Fiedler vector halves it again, but it needs
//! a sparse eigensolver, and a factor of two does not rescue a case that misses
//! its budget by three orders of magnitude.

use quip_miner_core::IsingGraph;

/// Reverse Cuthill-McKee ordering. `order[k]` is the original node at chain
/// slot `k`.
///
/// A breadth-first sweep that visits neighbours in increasing degree order,
/// then reverses the result. Disconnected graphs restart at the lowest-degree
/// unvisited vertex, which keeps each component in a contiguous block and the
/// bond between components at 1. Self-loops and out-of-range edges are skipped,
/// matching `CpuGraph::from_base` and `energy_milli`.
pub(crate) fn reverse_cuthill_mckee(n: usize, edges: &[(usize, usize)]) -> Vec<u32> {
    if n == 0 {
        return Vec::new();
    }
    let mut adj: Vec<Vec<u32>> = vec![Vec::new(); n];
    for &(u, v) in edges {
        if u >= n || v >= n || u == v {
            continue;
        }
        adj[u].push(v as u32);
        adj[v].push(u as u32);
    }
    for row in &mut adj {
        row.sort_unstable();
        row.dedup();
    }
    let deg: Vec<usize> = adj.iter().map(Vec::len).collect();
    for row in &mut adj {
        row.sort_by_key(|&x| (deg[x as usize], x));
    }

    let mut visited = vec![false; n];
    let mut out: Vec<u32> = Vec::with_capacity(n);
    let mut head = 0usize;
    while let Some(root) = (0..n).filter(|&i| !visited[i]).min_by_key(|&i| (deg[i], i)) {
        visited[root] = true;
        out.push(root as u32);
        while head < out.len() {
            let cur = out[head] as usize;
            head += 1;
            for &nb in &adj[cur] {
                if !visited[nb as usize] {
                    visited[nb as usize] = true;
                    out.push(nb);
                }
            }
        }
    }
    out.reverse();
    out
}

/// Largest slot distance across any edge, which is the matrix bandwidth of the
/// ordering. Skips the same edges the kernel skips.
///
/// The kernel costs scale with the span sum rather than the bandwidth, so this
/// exists to hold the ordering to account in the tests, not to steer a job.
#[cfg(test)]
pub(crate) fn bandwidth(slot: &[u32], edges: &[(usize, usize)]) -> usize {
    let n = slot.len();
    let mut widest = 0usize;
    for &(u, v) in edges {
        if u >= n || v >= n || u == v {
            continue;
        }
        let a = slot[u] as usize;
        let b = slot[v] as usize;
        widest = widest.max(a.max(b) - a.min(b));
    }
    widest
}

/// The problem in chain order, built once per job.
pub(crate) struct ChainProblem {
    /// `order[k]` is the original node at chain slot `k`. Spins come back by
    /// walking this: chain position `k` scores as node `order[k]`. The inverse
    /// is a construction detail of `from_graph`, so it is not carried here.
    pub(crate) order: Vec<u32>,
    /// Couplings as `(low_slot, high_slot, j)`, sorted by ascending span.
    pub(crate) gates: Vec<(u32, u32, f64)>,
    /// Fields in chain order.
    pub(crate) h: Vec<f64>,
}

impl ChainProblem {
    /// Map a received problem onto the chain.
    ///
    /// The defensive posture matches `CpuGraph::from_base` and `energy_milli`:
    /// self-loops and out-of-range edges are skipped, a `j` shorter than the
    /// edge list reads 0, and a zero coupling is dropped because its gate is
    /// the identity. Duplicate edges are kept and both gates are applied,
    /// because `energy_milli` sums both entries. Non-finite fields and
    /// couplings are dropped from the evolution, because `tanh` of a non-finite
    /// value poisons the whole state; scoring still sees the original values.
    pub(crate) fn from_graph(g: &IsingGraph) -> Self {
        let n = g.h.len();
        let order = reverse_cuthill_mckee(n, &g.edges);
        let mut slot = vec![0u32; n];
        for (k, &node) in order.iter().enumerate() {
            slot[node as usize] = k as u32;
        }
        let h: Vec<f64> = order
            .iter()
            .map(|&node| {
                let x = g.h[node as usize];
                if x.is_finite() {
                    x
                } else {
                    0.0
                }
            })
            .collect();
        let mut gates: Vec<(u32, u32, f64)> = Vec::with_capacity(g.edges.len());
        for (k, &(u, v)) in g.edges.iter().enumerate() {
            if u >= n || v >= n || u == v {
                continue;
            }
            let j = g.j.get(k).copied().unwrap_or(0.0);
            if j == 0.0 || !j.is_finite() {
                continue;
            }
            let a = slot[u];
            let b = slot[v];
            gates.push((a.min(b), a.max(b), j));
        }
        // Ascending span, so the cheap short-range gates run first and a
        // deadline stop loses the least work. `sort_by_key` is stable, so
        // duplicate edges keep their received order.
        gates.sort_by_key(|&(a, b, _)| (b - a, a, b));
        Self { order, gates, h }
    }

    /// Sum of gate spans, the quantity the cost model scales with.
    pub(crate) fn span_sum(&self) -> u64 {
        self.gates.iter().map(|&(a, b, _)| u64::from(b - a)).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inverse(order: &[u32]) -> Vec<u32> {
        let mut slot = vec![0u32; order.len()];
        for (k, &node) in order.iter().enumerate() {
            slot[node as usize] = k as u32;
        }
        slot
    }

    fn assert_permutation(order: &[u32], n: usize) {
        assert_eq!(order.len(), n, "ordering must cover every node");
        let mut seen = order.to_vec();
        seen.sort_unstable();
        let want: Vec<u32> = (0..n as u32).collect();
        assert_eq!(seen, want, "ordering must be a permutation");
    }

    /// The `smoke` preset: an 8-node ring.
    fn ring8() -> Vec<(usize, usize)> {
        vec![
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 0),
        ]
    }

    #[test]
    fn ordering_is_a_permutation_for_every_shape() {
        let cases: Vec<(usize, Vec<(usize, usize)>)> = vec![
            (0, vec![]),
            (1, vec![]),
            (1, vec![(0, 0)]),
            (4, vec![]),
            (8, ring8()),
            (6, vec![(0, 5), (5, 1), (1, 4), (4, 2), (2, 3)]),
            (5, vec![(0, 1), (0, 1), (3, 3), (2, 9)]),
        ];
        for (n, edges) in cases {
            let order = reverse_cuthill_mckee(n, &edges);
            assert_permutation(&order, n);
        }
    }

    #[test]
    fn the_smoke_ring_gets_bandwidth_two() {
        let edges = ring8();
        let order = reverse_cuthill_mckee(8, &edges);
        let slot = inverse(&order);
        assert_eq!(bandwidth(&slot, &edges), 2);
    }

    #[test]
    fn ordering_never_widens_the_bandwidth_of_the_committed_shapes() {
        // `as given` order against the reordered one, on the ring and on a
        // deliberately scrambled path.
        let cases: Vec<(usize, Vec<(usize, usize)>)> = vec![
            (8, ring8()),
            (
                9,
                vec![(0, 8), (8, 3), (3, 5), (5, 1), (1, 7), (7, 2), (2, 6), (6, 4)],
            ),
            (
                12,
                vec![
                    (0, 1),
                    (1, 2),
                    (2, 3),
                    (0, 4),
                    (1, 5),
                    (2, 6),
                    (3, 7),
                    (4, 5),
                    (5, 6),
                    (6, 7),
                    (4, 8),
                    (5, 9),
                    (6, 10),
                    (7, 11),
                    (8, 9),
                    (9, 10),
                    (10, 11),
                ],
            ),
        ];
        for (n, edges) in cases {
            let identity: Vec<u32> = (0..n as u32).collect();
            let given = bandwidth(&identity, &edges);
            let order = reverse_cuthill_mckee(n, &edges);
            let slot = inverse(&order);
            let got = bandwidth(&slot, &edges);
            assert!(
                got <= given,
                "n={n}: reordered bandwidth {got} exceeds the given {given}"
            );
        }
    }

    #[test]
    fn disconnected_components_land_in_contiguous_blocks() {
        // Two triangles and one isolated node.
        let edges = vec![(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)];
        let order = reverse_cuthill_mckee(7, &edges);
        assert_permutation(&order, 7);
        let slot = inverse(&order);
        let block_a: Vec<u32> = (0..3).map(|i| slot[i]).collect();
        let block_b: Vec<u32> = (3..6).map(|i| slot[i]).collect();
        for block in [block_a, block_b] {
            let mut s = block;
            s.sort_unstable();
            assert_eq!(
                s[2] - s[0],
                2,
                "a component must occupy consecutive slots, got {s:?}"
            );
        }
    }

    #[test]
    fn ordering_is_deterministic() {
        let edges = ring8();
        assert_eq!(
            reverse_cuthill_mckee(8, &edges),
            reverse_cuthill_mckee(8, &edges)
        );
    }

    #[test]
    fn bandwidth_ignores_self_loops_and_out_of_range_edges() {
        let slot: Vec<u32> = (0..4).collect();
        assert_eq!(bandwidth(&slot, &[(0, 0), (2, 2)]), 0);
        assert_eq!(bandwidth(&slot, &[(0, 9), (7, 1)]), 0);
        assert_eq!(bandwidth(&slot, &[(0, 3), (1, 2)]), 3);
    }
    // ---- Task 12: ChainProblem ----

    #[test]
    fn chain_problem_permutes_fields_and_couplings_consistently() {
        let g = IsingGraph::new(
            vec![0.5, -1.5, 0.25, 2.0],
            vec![1.0, -2.0, 0.75],
            vec![(0, 3), (3, 1), (1, 2)],
        );
        let chain = ChainProblem::from_graph(&g);
        assert_eq!(chain.order.len(), 4);
        let slot = inverse(&chain.order);
        for (node, &k) in slot.iter().enumerate() {
            assert_eq!(chain.order[k as usize] as usize, node);
            assert!((chain.h[k as usize] - g.h[node]).abs() < 1e-15);
        }
        assert_eq!(chain.gates.len(), 3);
        // Every gate must name the slots of a real edge, with the low slot first.
        for &(a, b, j) in &chain.gates {
            assert!(a < b, "gate slots must be ordered: ({a}, {b})");
            let u = chain.order[a as usize] as usize;
            let v = chain.order[b as usize] as usize;
            let found = g.edges.iter().zip(&g.j).any(|(&(x, y), &c)| {
                ((x, y) == (u, v) || (x, y) == (v, u)) && (c - j).abs() < 1e-15
            });
            assert!(found, "gate ({a}, {b}, {j}) has no matching edge");
        }
    }

    #[test]
    fn gates_are_sorted_by_ascending_span() {
        // A path graph in given order, so slots follow the chain closely.
        let g = IsingGraph::new(
            vec![0.0; 8],
            vec![1.0, 1.0, 1.0, 1.0, 1.0],
            vec![(0, 7), (0, 1), (2, 6), (3, 4), (1, 3)],
        );
        let chain = ChainProblem::from_graph(&g);
        let spans: Vec<u32> = chain.gates.iter().map(|&(a, b, _)| b - a).collect();
        for w in spans.windows(2) {
            assert!(w[0] <= w[1], "gates not sorted by span: {spans:?}");
        }
        assert_eq!(chain.span_sum(), spans.iter().map(|&x| u64::from(x)).sum());
    }

    #[test]
    fn defensive_edges_are_dropped_exactly_as_the_scoring_drops_them() {
        // Self-loop, out-of-range edge, zero coupling, short `j`, non-finite
        // coupling, non-finite field, and a duplicate edge that must survive.
        let g = IsingGraph::new(
            vec![0.5, f64::NAN, -0.25, 1.0],
            vec![1.0, 2.0, 0.0, f64::INFINITY, -1.0, -1.0],
            vec![(0, 0), (1, 9), (2, 3), (0, 2), (1, 3), (1, 3), (0, 3)],
        );
        let chain = ChainProblem::from_graph(&g);
        // (0,0) self-loop, (1,9) out of range, (2,3) zero coupling,
        // (0,2) non-finite coupling, and (0,3) with a missing `j` all drop.
        // The duplicate (1,3) appears twice, both with coupling -1.
        assert_eq!(chain.gates.len(), 2, "gates: {:?}", chain.gates);
        assert!(chain.gates.iter().all(|&(_, _, j)| (j + 1.0).abs() < 1e-15));
        // A non-finite field becomes 0 so that `tanh` cannot poison the state.
        assert!(chain.h.iter().all(|x| x.is_finite()), "{:?}", chain.h);
        assert!(chain.h.contains(&0.0));
    }

    #[test]
    fn an_empty_graph_produces_an_empty_chain() {
        let g = IsingGraph::new(vec![], vec![], vec![]);
        let chain = ChainProblem::from_graph(&g);
        assert!(chain.order.is_empty());
        assert!(chain.gates.is_empty());
        assert!(chain.h.is_empty());
        assert_eq!(chain.span_sum(), 0);
    }

    #[test]
    fn a_single_node_produces_one_slot_and_no_gates() {
        let g = IsingGraph::new(vec![-2.0], vec![3.0], vec![(0, 0)]);
        let chain = ChainProblem::from_graph(&g);
        assert_eq!(chain.order, vec![0]);
        assert_eq!(inverse(&chain.order), vec![0]);
        assert!(chain.gates.is_empty());
        assert_eq!(chain.h, vec![-2.0]);
    }

}
