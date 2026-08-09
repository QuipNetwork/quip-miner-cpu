//! Graph-to-network mapping for the BP-TNS kernel.
//!
//! Unlike the MPS chain, the tensor network lives directly on the problem
//! graph: one tensor per node, one bond per coupled pair. There is no
//! ordering pass, so no bandwidth cost; the price moves to vertex degree,
//! which is what `select_chi` in `mod.rs` accounts for.

use quip_miner_core::IsingGraph;

/// The problem as a tensor network: cleaned nodes, merged bonds, adjacency.
pub(crate) struct NetGraph {
    /// Bonds as `(u, v, j)` with `u < v`, sorted ascending. Parallel edges
    /// are merged by summing `j`: the imaginary-time gates commute, so
    /// `exp(-eta (j1 + j2) z z)` equals applying both gates, exactly.
    pub(crate) bonds: Vec<(u32, u32, f64)>,
    /// Per node: `(bond_id, neighbour)` in ascending bond order.
    pub(crate) adj: Vec<Vec<(u32, u32)>>,
    /// Fields with non-finite entries replaced by 0, matching the MPS kernel:
    /// `tanh` of a non-finite value would poison the state, and scoring still
    /// sees the original values.
    pub(crate) h: Vec<f64>,
}

impl NetGraph {
    /// Build the network from a received problem.
    ///
    /// The defensive posture matches `ChainProblem::from_graph`: self-loops
    /// and out-of-range edges are skipped, a `j` shorter than the edge list
    /// reads 0, zero and non-finite couplings are dropped from the evolution.
    /// The one divergence is deliberate and documented above: parallel edges
    /// merge into one bond instead of two gates, which is exact here.
    pub(crate) fn from_graph(g: &IsingGraph) -> Self {
        let n = g.h.len();
        let h: Vec<f64> = g
            .h
            .iter()
            .map(|&x| if x.is_finite() { x } else { 0.0 })
            .collect();

        let mut raw: Vec<(u32, u32, f64)> = Vec::with_capacity(g.edges.len());
        for (k, &(u, v)) in g.edges.iter().enumerate() {
            if u >= n || v >= n || u == v {
                continue;
            }
            let j = g.j.get(k).copied().unwrap_or(0.0);
            if j == 0.0 || !j.is_finite() {
                continue;
            }
            let a = u.min(v) as u32;
            let b = u.max(v) as u32;
            raw.push((a, b, j));
        }
        raw.sort_by_key(|&(a, b, _)| (a, b));

        let mut bonds: Vec<(u32, u32, f64)> = Vec::with_capacity(raw.len());
        for (a, b, j) in raw {
            match bonds.last_mut() {
                Some(&mut (pa, pb, ref mut pj)) if pa == a && pb == b => *pj += j,
                _ => bonds.push((a, b, j)),
            }
        }
        // A merged sum of zero is an identity gate and a rank-1 bond carrying
        // nothing; drop it so the network matches what the evolution does.
        bonds.retain(|&(_, _, j)| j != 0.0);

        let mut adj: Vec<Vec<(u32, u32)>> = vec![Vec::new(); n];
        for (e, &(u, v, _)) in bonds.iter().enumerate() {
            adj[u as usize].push((e as u32, v));
            adj[v as usize].push((e as u32, u));
        }
        Self { bonds, adj, h }
    }

    /// Number of nodes.
    pub(crate) fn num_nodes(&self) -> usize {
        self.h.len()
    }

    /// Largest vertex degree, 0 for an edgeless graph. The cost model in
    /// `select_chi` works from per-site degrees directly, so this exists to
    /// hold the mapping to account in the tests, as `order::bandwidth` does.
    #[cfg(test)]
    pub(crate) fn max_degree(&self) -> usize {
        self.adj.iter().map(Vec::len).max().unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_clean_graph_maps_one_bond_per_edge() {
        let g = IsingGraph::new(
            vec![0.5, -0.25, 0.0],
            vec![1.0, -2.0],
            vec![(0, 1), (2, 1)],
        );
        let net = NetGraph::from_graph(&g);
        assert_eq!(net.bonds, vec![(0, 1, 1.0), (1, 2, -2.0)]);
        assert_eq!(net.adj[0], vec![(0, 1)]);
        assert_eq!(net.adj[1], vec![(0, 0), (1, 2)]);
        assert_eq!(net.adj[2], vec![(1, 1)]);
        assert_eq!(net.h, vec![0.5, -0.25, 0.0]);
        assert_eq!(net.max_degree(), 2);
        assert_eq!(net.num_nodes(), 3);
    }

    #[test]
    fn defensive_edges_are_dropped_exactly_as_the_scoring_drops_them() {
        // Self-loop, out-of-range edge, zero coupling, short `j`, non-finite
        // coupling, non-finite field.
        let g = IsingGraph::new(
            vec![0.5, f64::NAN, -0.25, 1.0],
            vec![1.0, 2.0, 0.0, f64::INFINITY, -1.0],
            vec![(0, 0), (1, 9), (2, 3), (0, 2), (1, 3), (0, 3)],
        );
        let net = NetGraph::from_graph(&g);
        assert_eq!(net.bonds, vec![(1, 3, -1.0)]);
        assert!(net.h.iter().all(|x| x.is_finite()), "{:?}", net.h);
        assert_eq!(net.h[1], 0.0);
    }

    #[test]
    fn parallel_edges_merge_by_summing_their_couplings() {
        let g = IsingGraph::new(
            vec![0.0, 0.0],
            vec![-1.0, -1.0, 0.5],
            vec![(0, 1), (0, 1), (1, 0)],
        );
        let net = NetGraph::from_graph(&g);
        assert_eq!(net.bonds, vec![(0, 1, -1.5)]);
    }

    #[test]
    fn parallel_edges_that_cancel_leave_no_bond() {
        let g = IsingGraph::new(vec![0.0, 0.0], vec![1.0, -1.0], vec![(0, 1), (1, 0)]);
        let net = NetGraph::from_graph(&g);
        assert!(net.bonds.is_empty());
        assert!(net.adj.iter().all(Vec::is_empty));
    }

    #[test]
    fn an_empty_graph_produces_an_empty_network() {
        let net = NetGraph::from_graph(&IsingGraph::new(vec![], vec![], vec![]));
        assert!(net.bonds.is_empty());
        assert!(net.adj.is_empty());
        assert_eq!(net.max_degree(), 0);
        assert_eq!(net.num_nodes(), 0);
    }

    #[test]
    fn adjacency_lists_bonds_in_ascending_bond_order() {
        let g = IsingGraph::new(
            vec![0.0; 4],
            vec![1.0, 1.0, 1.0, 1.0],
            vec![(3, 2), (0, 3), (1, 0), (2, 0)],
        );
        let net = NetGraph::from_graph(&g);
        // Sorted bonds: (0,1), (0,2), (0,3), (2,3).
        assert_eq!(net.bonds.len(), 4);
        let ids: Vec<u32> = net.adj[0].iter().map(|&(e, _)| e).collect();
        assert_eq!(ids, vec![0, 1, 2]);
        assert_eq!(net.max_degree(), 3);
    }
}
