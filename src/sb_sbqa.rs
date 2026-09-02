//! Simulated Bifurcation Quantum Annealing (Pawlowski et al., arXiv:2604.01050).
//!
//! Reads are grouped into periodic rings of `replicas` trajectories. Each
//! replica runs discrete SB with a dead zone in the nonlinearity,
//! `f(x) = 0` for `|x| <= 0.7 t/T` and `sign(x)` otherwise, and every step
//! adds a ferromagnetic force from its two ring neighbours,
//! `kring * (x_{q-1} + x_{q+1})`, whose strength rises from weak to
//! dominant as the transverse field `Gamma_x(t)` falls. The ring term is the
//! Suzuki-Trotter coupling of a transverse-field Ising model; the paper's
//! Eq. (4) gives it no `1/R` factor while the SB terms carry one, and its
//! companion data show the SB part advancing at plain dSB's rate per step,
//! which fixes the absorbed form used here: `dt = 1` and `kring = R J_perp`.
//!
//! The paper never states `Gamma_x(0)`; only `theta = beta Gamma_x / R`
//! enters, so this kernel is parameterized by `theta0`. `beta` and `alpha`
//! are drawn once per ring, as the paper does per repetition.
//!
//! Rings of one replica have no ring term. The ring force grows large in the
//! last part of a run, but consensus is not guaranteed: two adjacent replicas
//! that disagree with the rest feel no net ring force, and a ring of two
//! swaps sides every step once the force exceeds the symplectic stability
//! limit at `dt = 1`. Read diversity inside a ring is low, not zero.

use quip_protocol::scoring::energy_milli;
use quip_solver_core::{IsingGraph, SampleParams, SamplerResult};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use crate::sb_core::{
    draw_initial_conditions, gauge_fixed_spins, pump, read_seed, Real, SbGraph, A0, DT,
};

/// Salt on the job seed for the per-ring `(beta, alpha)` draw.
const RING_SALT: u64 = 0x5342_5141_5249_4E47;

/// Replica-ring settings.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SbqaConfig {
    /// Replicas per ring. The paper states 128 for its sensitivity study
    /// only; 16 keeps a job at plain dSB's trajectory count with rings of a
    /// useful size.
    pub replicas: usize,
    /// `theta0 = beta Gamma_x(0) / R`, the only combination the paper's
    /// schedule depends on. 2.5 balances the ring force against the coupling
    /// force at the start of a run for 128 replicas.
    pub theta0: f32,
    /// Inverse temperature, drawn uniformly per ring. Paper: `[0.5, 1.5]`.
    pub beta_range: (f32, f32),
    /// Transverse-field decay exponent, drawn uniformly per ring. Paper:
    /// `[0.5, 1.0]`.
    pub alpha_range: (f32, f32),
    /// Slope of the dead zone, `Delta(t) = dead_zone * t / T`. Paper: 0.7.
    pub dead_zone: f32,
}

impl Default for SbqaConfig {
    fn default() -> Self {
        Self {
            replicas: 16,
            theta0: 2.5,
            beta_range: (0.5, 1.5),
            alpha_range: (0.5, 1.0),
            dead_zone: 0.7,
        }
    }
}

/// Ring force coefficient at normalized time `u`, with the `1/R` factors of
/// the paper's Eq. (7) absorbed: `R * J_perp(t)` where
/// `J_perp = -(1 / 2 beta) ln tanh(theta0 ((1 - u)^alpha + 1e-5))`.
pub(crate) fn ring_coupling(u: Real, theta0: Real, alpha: Real, beta: Real, ring: usize) -> Real {
    let theta = theta0 * ((1.0 - u).powf(alpha) + 1e-5);
    let jperp = -(1.0 / (2.0 * beta)) * theta.tanh().ln();
    ring as Real * jperp
}

/// The paper's Eq. (8): zero inside the dead zone, the sign outside it.
pub(crate) fn dead_zone_sign(x: Real, delta: Real) -> Real {
    if x.abs() <= delta {
        0.0
    } else if x > 0.0 {
        1.0
    } else {
        -1.0
    }
}

/// Consecutive ring sizes for `num_reads` reads.
pub(crate) fn ring_sizes(num_reads: usize, replicas: usize) -> Vec<usize> {
    let num_reads = num_reads.max(1);
    let replicas = replicas.max(1);
    let full = num_reads / replicas;
    let rest = num_reads % replicas;
    let mut sizes = vec![replicas; full];
    if rest > 0 {
        sizes.push(rest);
    }
    sizes
}

/// Momentum update for every replica of one ring from the old positions of
/// every replica: the SB restoring and coupling forces of the replica's own
/// state, plus `kring * (x_{q-1} + x_{q+1})` from its ring neighbours. This
/// reads `xs` and writes only `ys`, so no double buffer is needed; fusing it
/// with the position update would let replica `q` read an advanced position
/// from replica `q - 1`.
fn ring_momentum(
    g: &SbGraph,
    restore: Real,
    kring: Real,
    coupled: &[Vec<Real>],
    xs: &[Vec<Real>],
    ys: &mut [Vec<Real>],
) {
    let n = g.num_nodes();
    let size = xs.len();
    for q in 0..size {
        let qm = if q == 0 { size - 1 } else { q - 1 };
        let qp = if q + 1 == size { 0 } else { q + 1 };
        let cq = &coupled[q];
        for i in 0..n {
            let (nodes, coups) = g.neighbors(i);
            let mut f: Real = 0.0;
            for (&v, &coup) in nodes.iter().zip(coups.iter()) {
                f += coup * cq[v as usize];
            }
            if g.has_bias {
                f += g.h[i] * cq[n];
            }
            let ring = xs[qm][i] + xs[qp][i];
            ys[q][i] += (restore * xs[q][i] - g.c0 * f + kring * ring) * DT;
        }
        if g.has_bias {
            let mut f: Real = 0.0;
            for (&bias, &c) in g.h.iter().zip(cq.iter()) {
                f += bias * c;
            }
            let ring = xs[qm][n] + xs[qp][n];
            ys[q][n] += (restore * xs[q][n] - g.c0 * f + kring * ring) * DT;
        }
    }
}

/// Sample with the replica-ring kernel.
///
/// # Examples
///
/// ```
/// use quip_miner_cpu::{sample_sbqa, IsingGraph, SampleParams, SbqaConfig};
///
/// let graph = IsingGraph::new(vec![0.0, 0.0], vec![-1.0], vec![(0, 1)]);
/// let params = SampleParams { num_reads: 4, num_sweeps: 200, seed: 1, ..Default::default() };
/// let results = sample_sbqa(&graph, &params, SbqaConfig::default());
/// assert_eq!(results.len(), 4);
/// assert!(results.iter().all(|r| r.spins.iter().all(|&s| s == 1 || s == -1)));
/// ```
pub fn sample_sbqa(
    graph: &IsingGraph,
    params: &SampleParams,
    cfg: SbqaConfig,
) -> Vec<SamplerResult> {
    let num_reads = params.num_reads.max(1);
    let g = SbGraph::from_base(graph);
    let m = g.num_particles();
    if m == 0 {
        let energy = energy_milli(&[], &graph.h, &graph.j, &graph.edges);
        return (0..num_reads)
            .map(|_| SamplerResult {
                spins: Vec::new(),
                energy_milli: energy,
            })
            .collect();
    }
    let n_step = params.num_sweeps.max(1);
    let mut out = Vec::with_capacity(num_reads);
    let mut first = 0;

    for (ring_idx, size) in ring_sizes(num_reads, cfg.replicas).into_iter().enumerate() {
        let mut ring_rng = SmallRng::seed_from_u64(read_seed(params.seed ^ RING_SALT, ring_idx));
        let beta: Real = ring_rng.gen_range(cfg.beta_range.0..=cfg.beta_range.1);
        let alpha: Real = ring_rng.gen_range(cfg.alpha_range.0..=cfg.alpha_range.1);

        let mut xs: Vec<Vec<Real>> = Vec::with_capacity(size);
        let mut ys: Vec<Vec<Real>> = Vec::with_capacity(size);
        for q in 0..size {
            let mut rng = SmallRng::seed_from_u64(read_seed(params.seed, first + q));
            let (x, y) = draw_initial_conditions(m, &mut rng);
            xs.push(x);
            ys.push(y);
        }
        let mut coupled: Vec<Vec<Real>> = vec![vec![0.0; m]; size];

        for k in 0..n_step {
            let u = k as Real / n_step as Real;
            let restore = pump(k, n_step) - A0;
            let delta = cfg.dead_zone * u;
            let kring = if size > 1 {
                ring_coupling(u, cfg.theta0, alpha, beta, size)
            } else {
                0.0
            };
            for (cq, xq) in coupled.iter_mut().zip(xs.iter()) {
                for (c, &xi) in cq.iter_mut().zip(xq.iter()) {
                    *c = dead_zone_sign(xi, delta);
                }
            }
            ring_momentum(&g, restore, kring, &coupled, &xs, &mut ys);
            for (xq, yq) in xs.iter_mut().zip(ys.iter_mut()) {
                for (xi, yi) in xq.iter_mut().zip(yq.iter_mut()) {
                    *xi += A0 * *yi * DT;
                    if *xi > 1.0 {
                        *xi = 1.0;
                        *yi = 0.0;
                    } else if *xi < -1.0 {
                        *xi = -1.0;
                        *yi = 0.0;
                    }
                }
            }
        }

        for x in &xs {
            let spins = gauge_fixed_spins(&g, x);
            out.push(SamplerResult {
                energy_milli: energy_milli(&spins, &graph.h, &graph.j, &graph.edges),
                spins,
            });
        }
        first += size;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use quip_solver_core::{IsingGraph, SampleParams};

    fn sb_params(num_reads: usize, num_sweeps: usize, seed: u64) -> SampleParams {
        SampleParams {
            num_reads,
            num_sweeps,
            seed,
            ..Default::default()
        }
    }

    fn ferro_pair() -> IsingGraph {
        IsingGraph::new(vec![0.0, 0.0], vec![-1.0], vec![(0, 1)])
    }

    fn frustrated_triangle() -> IsingGraph {
        IsingGraph::new(
            vec![0.0; 3],
            vec![1.0, 1.0, 1.0],
            vec![(0, 1), (1, 2), (0, 2)],
        )
    }

    fn ring12() -> IsingGraph {
        let edges: Vec<(usize, usize)> = (0..12).map(|i| (i, (i + 1) % 12)).collect();
        let j: Vec<f64> = (0..12)
            .map(|k| if k % 3 == 0 { 1.0 } else { -1.0 })
            .collect();
        IsingGraph::new(vec![0.0; 12], j, edges)
    }

    fn brute_force_min_energy(g: &IsingGraph) -> i64 {
        use quip_protocol::scoring::energy_milli;
        let n = g.h.len();
        (0..1u32 << n)
            .map(|mask| {
                let spins: Vec<i8> = (0..n)
                    .map(|i| if mask & (1 << i) != 0 { 1 } else { -1 })
                    .collect();
                energy_milli(&spins, &g.h, &g.j, &g.edges)
            })
            .min()
            .expect("non-empty")
    }

    /// Hypothesis (paper Eq. 5 and S12): the ring coupling is positive and
    /// rises monotonically over the run, because `Gamma_x` falls and
    /// `-ln tanh` grows as its argument shrinks.
    #[test]
    fn ring_coupling_is_positive_and_increasing() {
        let mut last = 0.0;
        for k in 0..100 {
            let u = k as Real / 100.0;
            let kring = ring_coupling(u, 2.5, 1.0, 1.0, 16);
            assert!(kring > 0.0, "u = {u}: {kring}");
            assert!(kring > last, "u = {u}: {kring} <= {last}");
            last = kring;
        }
    }

    /// Hypothesis: with a zero dead zone the nonlinearity is dSB's sign.
    #[test]
    fn dead_zone_is_plain_sign_at_the_first_step() {
        assert_eq!(dead_zone_sign(0.3, 0.0), 1.0);
        assert_eq!(dead_zone_sign(-0.3, 0.0), -1.0);
        assert_eq!(dead_zone_sign(0.3, 0.5), 0.0);
        assert_eq!(dead_zone_sign(-0.6, 0.5), -1.0);
    }

    /// Hypothesis: reads group consecutively into rings of `replicas`, and
    /// the remainder forms a smaller last ring.
    #[test]
    fn rings_group_reads_with_a_smaller_remainder() {
        assert_eq!(ring_sizes(5, 2), vec![2, 2, 1]);
        assert_eq!(ring_sizes(4, 16), vec![4]);
        assert_eq!(ring_sizes(0, 4), vec![1]);
        assert_eq!(ring_sizes(6, 0), vec![1, 1, 1, 1, 1, 1]);
    }

    /// Hypothesis (paper Eq. 7): the ring force on a replica is
    /// `kring * (x_{q-1} + x_{q+1})`, ferromagnetic, and reads the
    /// neighbours' old positions. On a graph with no couplings and no bias
    /// the SB force vanishes, so a resting replica whose two ring neighbours
    /// are the same replica at `+1` gains exactly `2 kring`, while the
    /// neighbour, whose own neighbours rest at zero, gains nothing.
    #[test]
    fn ring_force_pulls_momentum_toward_the_neighbours() {
        let g = SbGraph::from_base(&IsingGraph::new(vec![0.0, 0.0], vec![], vec![]));
        let xs = vec![vec![0.0, 0.0], vec![1.0, 1.0]];
        let coupled = xs.clone();
        let mut ys = vec![vec![0.0; 2]; 2];
        ring_momentum(&g, 0.0, 0.25, &coupled, &xs, &mut ys);
        assert_eq!(ys[0], vec![0.5, 0.5]);
        assert_eq!(ys[1], vec![0.0, 0.0]);
    }

    /// Hypothesis: the ring term is live end to end. Two reads in one ring
    /// leave a different trajectory than the same two reads run as rings of
    /// one, at the same seeds, on a run too short to converge.
    #[test]
    fn ring_coupling_changes_the_trajectory() {
        let g = ring12();
        let ring = sample_sbqa(
            &g,
            &sb_params(2, 40, 5),
            SbqaConfig {
                replicas: 2,
                ..SbqaConfig::default()
            },
        );
        let alone = sample_sbqa(
            &g,
            &sb_params(2, 40, 5),
            SbqaConfig {
                replicas: 1,
                ..SbqaConfig::default()
            },
        );
        let spins = |v: &[SamplerResult]| v.iter().map(|r| r.spins.clone()).collect::<Vec<_>>();
        assert_ne!(spins(&ring), spins(&alone));
    }

    /// Hypothesis: the ferromagnetic pair is solved by rings of two at the
    /// defaults. Consensus inside the ring is not asserted: with both ring
    /// neighbours the same replica and `dt = 1`, a strong late ring force
    /// can swap the pair every step, which the module doc records.
    #[test]
    fn ring_of_two_solves_the_ferromagnetic_pair() {
        let cfg = SbqaConfig {
            replicas: 2,
            ..SbqaConfig::default()
        };
        for seed in 1..=8 {
            let results = sample_sbqa(&ferro_pair(), &sb_params(2, 512, seed), cfg);
            assert_eq!(results.len(), 2);
            let best = results
                .iter()
                .map(|r| r.energy_milli)
                .min()
                .expect("two reads");
            assert_eq!(best, -1000, "seed {seed}");
        }
    }

    /// Hypothesis: the kernel reaches the true ground state on small
    /// frustrated instances with the defaults.
    #[test]
    fn finds_brute_forced_ground_states() {
        for (name, g, reads, sweeps) in [
            ("triangle", frustrated_triangle(), 32, 512),
            ("ring12", ring12(), 64, 2048),
        ] {
            let want = brute_force_min_energy(&g);
            let results = sample_sbqa(&g, &sb_params(reads, sweeps, 2026), SbqaConfig::default());
            let best = results
                .iter()
                .map(|r| r.energy_milli)
                .min()
                .expect("reads > 0");
            assert_eq!(best, want, "{name}: best {best} != brute-force {want}");
        }
    }

    /// Hypothesis: degenerate inputs return `num_reads` valid spin vectors,
    /// with and without a bias, at ring sizes above and below the read count.
    #[test]
    fn degenerate_graphs_return_valid_spins() {
        let cases = [
            ("empty", IsingGraph::new(vec![], vec![], vec![])),
            ("no edges", IsingGraph::new(vec![0.0; 3], vec![], vec![])),
            (
                "bias only",
                IsingGraph::new(vec![1.0, -1.0], vec![], vec![]),
            ),
            (
                "non-finite coupling",
                IsingGraph::new(vec![0.0, 0.0], vec![f64::INFINITY], vec![(0, 1)]),
            ),
        ];
        for (name, g) in cases {
            for replicas in [1, 2, 8] {
                let cfg = SbqaConfig {
                    replicas,
                    ..SbqaConfig::default()
                };
                let results = sample_sbqa(&g, &sb_params(3, 40, 1), cfg);
                assert_eq!(results.len(), 3, "{name} / {replicas}");
                for r in &results {
                    assert_eq!(r.spins.len(), g.h.len(), "{name} / {replicas}");
                    assert!(
                        r.spins.iter().all(|&s| s == 1 || s == -1),
                        "{name} / {replicas}"
                    );
                }
            }
        }
    }

    /// Hypothesis: one seed, one result through the ring path; and a
    /// different seed moves independent replicas on a run too short to
    /// converge. The seed check uses rings of one because a ring drives its
    /// replicas to a few shared configurations that two seeds can agree on.
    #[test]
    fn same_seed_same_result_different_seed_differs() {
        let g = ring12();
        let a = sample_sbqa(&g, &sb_params(8, 200, 9), SbqaConfig::default());
        let b = sample_sbqa(&g, &sb_params(8, 200, 9), SbqaConfig::default());
        let spins = |v: &[SamplerResult]| v.iter().map(|r| r.spins.clone()).collect::<Vec<_>>();
        assert_eq!(spins(&a), spins(&b));
        let alone = SbqaConfig {
            replicas: 1,
            ..SbqaConfig::default()
        };
        let c = sample_sbqa(&g, &sb_params(8, 20, 9), alone);
        let d = sample_sbqa(&g, &sb_params(8, 20, 10), alone);
        assert_ne!(spins(&c), spins(&d));
    }
}
