//! Graph colouring for chromatic Gibbs.
//!
//! Chromatic Gibbs needs a partition of the nodes into independent sets. Within
//! one set no node is a neighbour of another, so every member's conditional
//! distribution depends only on nodes outside the set. The sampler can then
//! resample a whole set at once and stay a correct Gibbs sampler.
//!
//! The classes themselves must run in sequence. Class `c + 1` reads the values
//! class `c` just wrote. Parallelism lives inside a class, across its members,
//! which is why the worker count is independent of the class count.
//!
//! Welsh-Powell ordering (highest degree first) drives a greedy colouring. It is
//! not minimal. Finding the chromatic number is NP-hard, and the sampler stays
//! correct at any valid colouring, so the extra classes cost sweep time rather
//! than correctness.

use crate::sampler_core::CpuGraph;

/// A valid partition of the graph's nodes into independent sets.
pub(crate) struct Coloring {
    /// Node ids per class. Every class is an independent set.
    classes: Vec<Vec<u32>>,
}

impl Coloring {
    /// Colour `g` greedily in Welsh-Powell order.
    ///
    /// Isolated nodes land in class 0. An empty graph produces no classes.
    pub(crate) fn new(g: &CpuGraph) -> Self {
        let n = g.num_nodes();
        let mut order: Vec<u32> = (0..n as u32).collect();
        order.sort_by_key(|&v| {
            let (nodes, _) = g.neighbors(v as usize);
            std::cmp::Reverse(nodes.len())
        });

        let mut color = vec![u32::MAX; n];
        // Reused across nodes so the colouring stays O(V + E) rather than
        // allocating a neighbour-colour set per node.
        let mut taken: Vec<u32> = vec![u32::MAX; n.max(1)];
        let mut num_classes = 0usize;
        for &v in &order {
            let (nodes, _) = g.neighbors(v as usize);
            for &w in nodes {
                let c = color[w as usize];
                if c != u32::MAX {
                    taken[c as usize] = v;
                }
            }
            let mut c = 0usize;
            while c < n && taken[c] == v {
                c += 1;
            }
            color[v as usize] = c as u32;
            num_classes = num_classes.max(c + 1);
        }

        let mut classes = vec![Vec::new(); num_classes];
        for (v, &c) in color.iter().enumerate() {
            if c != u32::MAX {
                classes[c as usize].push(v as u32);
            }
        }
        Self { classes }
    }

    /// The independent sets, in the order the sampler must visit them.
    pub(crate) fn classes(&self) -> &[Vec<u32>] {
        &self.classes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quip_miner_core::IsingGraph;

    fn colored(g: &IsingGraph) -> (Coloring, CpuGraph) {
        let cpu = CpuGraph::from_base(g);
        (Coloring::new(&cpu), cpu)
    }

    /// Hypothesis: every class is an independent set. This is the property the
    /// whole parallel update rests on. If two neighbours share a class, the
    /// sampler resamples both from stale fields and stops being a Gibbs
    /// sampler, silently and without any test failing elsewhere.
    #[test]
    fn every_class_is_an_independent_set() {
        let edges: Vec<(usize, usize)> = (0..12).map(|i| (i, (i + 1) % 12)).collect();
        let g = IsingGraph::new(vec![0.0; 12], vec![1.0; 12], edges);
        let (c, cpu) = colored(&g);
        for class in c.classes() {
            for &v in class {
                let (nodes, _) = cpu.neighbors(v as usize);
                for &w in nodes {
                    assert!(
                        !class.contains(&w),
                        "nodes {v} and {w} are neighbours in one class"
                    );
                }
            }
        }
    }

    /// Hypothesis: the classes partition the node set exactly once each.
    /// A dropped node would never be resampled, and a duplicated one would be
    /// resampled twice per sweep.
    #[test]
    fn classes_partition_every_node_exactly_once() {
        let g = IsingGraph::new(
            vec![0.0; 6],
            vec![1.0; 5],
            vec![(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)],
        );
        let (c, _) = colored(&g);
        let mut seen: Vec<u32> = c.classes().iter().flatten().copied().collect();
        seen.sort_unstable();
        assert_eq!(seen, (0..6u32).collect::<Vec<_>>());
    }

    /// Hypothesis: a bipartite graph needs two classes, and a triangle needs
    /// three. These pin that the greedy pass is not simply assigning one class
    /// per node.
    #[test]
    fn greedy_uses_the_expected_class_count_on_known_graphs() {
        let path = IsingGraph::new(vec![0.0; 4], vec![1.0; 3], vec![(0, 1), (1, 2), (2, 3)]);
        assert_eq!(colored(&path).0.classes().len(), 2, "a path is bipartite");

        let triangle = IsingGraph::new(vec![0.0; 3], vec![1.0; 3], vec![(0, 1), (1, 2), (0, 2)]);
        assert_eq!(
            colored(&triangle).0.classes().len(),
            3,
            "a triangle needs three"
        );
    }

    /// Hypothesis: a graph with no edges colours in one class, so the sampler
    /// resamples every node at once.
    #[test]
    fn isolated_nodes_share_one_class() {
        let g = IsingGraph::new(vec![0.0; 5], vec![], vec![]);
        let (c, _) = colored(&g);
        assert_eq!(c.classes().len(), 1);
        assert_eq!(c.classes()[0].len(), 5);
    }

    /// Hypothesis: an empty graph produces no classes and does not panic.
    #[test]
    fn empty_graph_has_no_classes() {
        let g = IsingGraph::new(vec![], vec![], vec![]);
        assert_eq!(colored(&g).0.classes().len(), 0);
    }
}
