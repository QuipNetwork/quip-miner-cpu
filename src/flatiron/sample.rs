//! Belief propagation over the double-layer network and conditioned
//! sampling.
//!
//! The published method measures observables by contraction and never draws
//! configurations; drawing them is this kernel's adaptation, and it works by
//! fixing one spin at a time from the BP marginal, projecting the site, and
//! refreshing that site's outgoing messages. On a tree the marginals are
//! exact; on a loopy graph they carry the usual BP approximation.

use super::graph::NetGraph;
use super::state::SiteTensor;
use rand::rngs::SmallRng;
use rand::Rng;

/// BP sweep cap. Convergence is typically exponential in the iteration
/// count, so a run that has not settled by now is dominated by loops that
/// more sweeps do not fix.
const BP_MAX_SWEEPS: usize = 40;

/// Largest message change at which the fixed point counts as reached.
const BP_TOL: f64 = 1e-9;

/// The state prepared for sampling: tensors with the Schmidt weights
/// absorbed, plus one message per directed bond.
#[derive(Clone)]
pub(crate) struct SamplingNet {
    pub(crate) tensors: Vec<SiteTensor>,
    /// `msgs[2e]` flows low-to-high endpoint of bond `e`, `msgs[2e + 1]` the
    /// reverse. Each is a `k x k` row-major matrix over that bond's basis.
    msgs: Vec<Vec<f64>>,
}

/// Direction index of the message on bond `e` that flows *into* `node`.
fn msg_into(net: &NetGraph, e: usize, node: u32) -> usize {
    let (u, _, _) = net.bonds[e];
    if node == u {
        2 * e + 1
    } else {
        2 * e
    }
}

/// Direction index of the message on bond `e` that flows *out of* `node`.
fn msg_out_of(net: &NetGraph, e: usize, node: u32) -> usize {
    msg_into(net, e, node) ^ 1
}

/// Apply the `k x k` matrix `m` along bond `b`: `out[.., c'] = sum_c
/// in[.., c] m[c, c']`.
fn apply_axis(t: &SiteTensor, b: usize, m: &[f64]) -> SiteTensor {
    let (mat, rows, k) = t.matricize(b);
    let mut out = vec![0.0; rows * k];
    for r in 0..rows {
        for (c, &x) in mat[r * k..(r + 1) * k].iter().enumerate() {
            if x == 0.0 {
                continue;
            }
            for cp in 0..k {
                out[r * k + cp] += x * m[c * k + cp];
            }
        }
    }
    let mut result = t.clone();
    result.dematricize(b, &out, k);
    result
}

/// The identity divided by its dimension: the neutral, normalized message.
fn identity_message(k: usize) -> Vec<f64> {
    let mut m = vec![0.0; k * k];
    for c in 0..k {
        m[c * k + c] = 1.0 / k as f64;
    }
    m
}

impl SamplingNet {
    /// Absorb `sqrt(lambda)` into both endpoints of every bond and start all
    /// messages at the identity.
    pub(crate) fn build(net: &NetGraph, tensors: &[SiteTensor], lambdas: &[Vec<f64>]) -> Self {
        let mut tensors = tensors.to_vec();
        for (e, lam) in lambdas.iter().enumerate() {
            let root: Vec<f64> = lam.iter().map(|&x| x.max(0.0).sqrt()).collect();
            let (u, v, _) = net.bonds[e];
            for node in [u, v] {
                let b = net.adj[node as usize]
                    .iter()
                    .position(|&(be, _)| be as usize == e)
                    .unwrap_or(0);
                tensors[node as usize].scale_bond(b, &root);
            }
        }
        let msgs = lambdas
            .iter()
            .flat_map(|lam| {
                let m = identity_message(lam.len());
                [m.clone(), m]
            })
            .collect();
        Self { tensors, msgs }
    }

    /// Double-layer contraction of site `node` with every incoming message
    /// applied except along `skip` (a bond position, or the site's degree to
    /// skip nothing). Returns the transformed copy and the original.
    fn dressed(&self, net: &NetGraph, node: usize, skip: usize) -> SiteTensor {
        let mut k2 = self.tensors[node].clone();
        for (b, &(e, _)) in net.adj[node].iter().enumerate() {
            if b == skip {
                continue;
            }
            let m = &self.msgs[msg_into(net, e as usize, node as u32)];
            // A message on a dimension-1 bond is exactly [1.0] by
            // construction (build starts there and every refresh
            // renormalizes a scalar), so applying it is the identity. This
            // is the whole cost of the mean-field regime, so skip it.
            if m.len() == 1 {
                continue;
            }
            k2 = apply_axis(&k2, b, m);
        }
        k2
    }

    /// Recompute the message out of `node` along bond position `b`.
    /// Returns the largest absolute change.
    fn refresh_message(&mut self, net: &NetGraph, node: usize, b: usize) -> f64 {
        let (e, _) = net.adj[node][b];
        let k2 = self.dressed(net, node, b);
        let (m2, rows, k) = k2.matricize(b);
        let (m1, _, _) = self.tensors[node].matricize(b);
        // rho[a, a'] = sum_r k2[r, a] * k[r, a'].
        let mut rho = vec![0.0; k * k];
        for r in 0..rows {
            for a in 0..k {
                let x = m2[r * k + a];
                if x == 0.0 {
                    continue;
                }
                for ap in 0..k {
                    rho[a * k + ap] += x * m1[r * k + ap];
                }
            }
        }
        let trace: f64 = (0..k).map(|c| rho[c * k + c]).sum();
        if !(trace.is_finite() && trace > 0.0) {
            rho = identity_message(k);
        } else {
            for x in &mut rho {
                *x /= trace;
            }
        }
        let slot = msg_out_of(net, e as usize, node as u32);
        let delta = self.msgs[slot]
            .iter()
            .zip(&rho)
            .fold(0.0f64, |acc, (&old, &new)| acc.max((old - new).abs()));
        self.msgs[slot] = rho;
        delta
    }

    /// Run BP to the fixed point or the sweep cap.
    pub(crate) fn run_bp(&mut self, net: &NetGraph) {
        for _ in 0..BP_MAX_SWEEPS {
            let mut delta = 0.0f64;
            for e in 0..net.bonds.len() {
                // Dimension-1 messages are pinned at [1.0]; see `dressed`.
                if self.msgs[2 * e].len() == 1 {
                    continue;
                }
                let (u, v, _) = net.bonds[e];
                for node in [u as usize, v as usize] {
                    let b = net.adj[node]
                        .iter()
                        .position(|&(be, _)| be as usize == e)
                        .unwrap_or(0);
                    delta = delta.max(self.refresh_message(net, node, b));
                }
            }
            if delta < BP_TOL {
                break;
            }
        }
    }

    /// BP marginal weights of the physical index at `node`, unnormalized.
    pub(crate) fn marginal(&self, net: &NetGraph, node: usize) -> Vec<f64> {
        let deg = net.adj[node].len();
        let k2 = self.dressed(net, node, deg);
        let k1 = &self.tensors[node];
        let bt = k1.bond_total();
        (0..k1.phys)
            .map(|s| {
                k2.data[s * bt..(s + 1) * bt]
                    .iter()
                    .zip(&k1.data[s * bt..(s + 1) * bt])
                    .map(|(&a, &b)| a * b)
                    .sum()
            })
            .collect()
    }

    /// Fix `node` to the physical value `s` and refresh its outgoing
    /// messages so later marginals condition on the choice.
    fn project(&mut self, net: &NetGraph, node: usize, s: usize) {
        let t = &mut self.tensors[node];
        let bt = t.bond_total();
        t.data = t.data[s * bt..(s + 1) * bt].to_vec();
        t.phys = 1;
        for b in 0..net.adj[node].len() {
            // Dimension-1 messages are pinned at [1.0]; see `dressed`.
            if self.tensors[node].dims[b] > 1 {
                self.refresh_message(net, node, b);
            }
        }
    }

    /// Draw one configuration by sequential conditioning. `spins[i]` is +1
    /// or -1; physical index 0 is spin +1.
    pub(crate) fn sample_one(mut self, net: &NetGraph, rng: &mut SmallRng) -> Vec<i8> {
        let n = net.num_nodes();
        let mut spins = vec![1i8; n];
        #[expect(
            clippy::needless_range_loop,
            reason = "the loop body borrows `self` mutably through `project`, so iterating `spins` would hold a conflicting borrow"
        )]
        for node in 0..n {
            let w = self.marginal(net, node);
            let total = w[0] + w[1];
            let p_plus = if total.is_finite() && total > 0.0 && w[0] >= 0.0 {
                w[0] / total
            } else {
                0.5
            };
            let s = if rng.gen::<f64>() < p_plus { 0 } else { 1 };
            spins[node] = if s == 0 { 1 } else { -1 };
            self.project(net, node, s);
        }
        spins
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quip_miner_core::IsingGraph;
    use rand::SeedableRng;

    /// Exact per-site marginals by enumerating physical and bond indices.
    fn brute_marginals(net: &NetGraph, tensors: &[SiteTensor]) -> Vec<[f64; 2]> {
        let n = net.num_nodes();
        let bond_dims: Vec<usize> = net
            .bonds
            .iter()
            .enumerate()
            .map(|(e, &(u, _, _))| {
                let pos = net.adj[u as usize]
                    .iter()
                    .position(|&(be, _)| be as usize == e)
                    .expect("bond listed in adjacency");
                tensors[u as usize].dims[pos]
            })
            .collect();
        let bond_total: usize = bond_dims.iter().product();
        let mut out = vec![[0.0f64; 2]; n];
        for mask in 0..(1u32 << n) {
            let spins: Vec<usize> = (0..n).map(|i| ((mask >> i) & 1) as usize).collect();
            let mut amp = 0.0f64;
            for assign in 0..bond_total {
                // Decode per-bond coordinates.
                let mut rem = assign;
                let mut coord = vec![0usize; net.bonds.len()];
                for (e, &k) in bond_dims.iter().enumerate() {
                    coord[e] = rem % k;
                    rem /= k;
                }
                let mut prod = 1.0f64;
                for (i, t) in tensors.iter().enumerate() {
                    let mut flat = 0usize;
                    for (b, &(e, _)) in net.adj[i].iter().enumerate() {
                        flat = flat * t.dims[b] + coord[e as usize];
                    }
                    prod *= t.data[spins[i] * t.bond_total() + flat];
                }
                amp += prod;
            }
            let p = amp * amp;
            for (i, &s) in spins.iter().enumerate() {
                out[i][s] += p;
            }
        }
        out
    }

    fn path_net(n: usize) -> NetGraph {
        NetGraph::from_graph(&IsingGraph::new(
            vec![0.0; n],
            vec![1.0; n - 1],
            (0..n - 1).map(|i| (i, i + 1)).collect(),
        ))
    }

    /// Deterministic pseudo-random tensors with bond dimension 2 on a path,
    /// which is a tree, so BP must be exact there.
    #[test]
    fn bp_marginals_are_exact_on_a_tree() {
        let net = path_net(4);
        let mut val = 0.3f64;
        let mut next = || {
            val = (val * 137.0 + 0.31).rem_euclid(2.0) - 1.0;
            val
        };
        let tensors: Vec<SiteTensor> = (0..4)
            .map(|i| {
                let dims = vec![2; net.adj[i].len()];
                let len = 2 * dims.iter().product::<usize>();
                SiteTensor {
                    phys: 2,
                    dims,
                    data: (0..len).map(|_| next()).collect(),
                }
            })
            .collect();
        let lambdas: Vec<Vec<f64>> = net.bonds.iter().map(|_| vec![1.0, 1.0]).collect();

        let mut sampling = SamplingNet::build(&net, &tensors, &lambdas);
        sampling.run_bp(&net);

        let exact = brute_marginals(&net, &tensors);
        for (i, truth_pair) in exact.iter().enumerate() {
            let w = sampling.marginal(&net, i);
            let bp = w[0] / (w[0] + w[1]);
            let truth = truth_pair[0] / (truth_pair[0] + truth_pair[1]);
            assert!(
                (bp - truth).abs() < 1e-8,
                "site {i}: BP {bp} vs exact {truth}"
            );
        }
    }

    /// Conditioning must steer later sites: after projecting one end of a
    /// perfectly correlated pair, the other end's marginal follows it.
    #[test]
    fn conditioning_propagates_through_the_messages() {
        let net = path_net(2);
        // A GHZ-like pair: amplitude 1 on (+,+) and on (-,-), 0 across.
        let tensors = vec![
            SiteTensor {
                phys: 2,
                dims: vec![2],
                data: vec![1.0, 0.0, 0.0, 1.0],
            },
            SiteTensor {
                phys: 2,
                dims: vec![2],
                data: vec![1.0, 0.0, 0.0, 1.0],
            },
        ];
        let lambdas = vec![vec![1.0, 1.0]];
        let mut sampling = SamplingNet::build(&net, &tensors, &lambdas);
        sampling.run_bp(&net);

        let mut rng = SmallRng::seed_from_u64(7);
        for _ in 0..8 {
            let spins = sampling.clone().sample_one(&net, &mut rng);
            assert_eq!(spins[0], spins[1], "the pair must sample aligned");
        }
    }

    #[test]
    fn a_graph_with_no_bonds_samples_from_the_site_weights_alone() {
        let net = NetGraph::from_graph(&IsingGraph::new(vec![0.0, 0.0], vec![], vec![]));
        let tensors = vec![
            SiteTensor {
                phys: 2,
                dims: vec![],
                data: vec![1.0, 0.0],
            },
            SiteTensor {
                phys: 2,
                dims: vec![],
                data: vec![0.0, 1.0],
            },
        ];
        let mut sampling = SamplingNet::build(&net, &tensors, &[]);
        sampling.run_bp(&net);
        let mut rng = SmallRng::seed_from_u64(3);
        let spins = sampling.sample_one(&net, &mut rng);
        assert_eq!(spins, vec![1, -1]);
    }

    #[test]
    fn degenerate_marginals_fall_back_to_a_fair_coin() {
        let net = NetGraph::from_graph(&IsingGraph::new(vec![0.0], vec![], vec![]));
        let tensors = vec![SiteTensor {
            phys: 2,
            dims: vec![],
            data: vec![0.0, 0.0],
        }];
        let sampling = SamplingNet::build(&net, &tensors, &[]);
        let mut ups = 0;
        for seed in 0..64 {
            let mut rng = SmallRng::seed_from_u64(seed);
            let spins = sampling.clone().sample_one(&net, &mut rng);
            assert!(spins[0] == 1 || spins[0] == -1);
            if spins[0] == 1 {
                ups += 1;
            }
        }
        assert!((10..=54).contains(&ups), "coin badly biased: {ups} of 64");
    }
}
