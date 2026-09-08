//! Globally guided Simulated Bifurcation (Xiao, Huang, Qiu, Liu, Zhuang,
//! Yung; Physical Review Applied 25, 024014, 2026).
//!
//! Transcribed from the paper's own reference implementation, `GSB.py` in
//! github.com/JugarH/Global-Guided-Simulated-Bifurcation, which the paper
//! cites as its implementation; the paper text is not open. All `num_reads`
//! trajectories advance in lockstep. Each step, every trajectory's extended
//! energy is read off the coupling sum it already computed, the lowest one
//! becomes the leader, and every trajectory is pulled toward the leader's
//! continuous position by a guidance term that is annealed, damped by its
//! distance, sign-flipped when it opposes the trajectory's momentum, and
//! low-pass filtered. After the position update every coordinate is
//! perturbed by a constant `-a` with a probability that falls from 1 to 0.1.
//!
//! Departures from the reference, each deliberate:
//! - The linear pump of this crate replaces the reference's default
//!   exponential pump, so guidance and perturbation are measured on their
//!   own. The reference supports both.
//! - `c0` replaces the reference's coupling scale, whose automatic path is
//!   dead code and whose tuned values are G-set specific.
//! - With a bias, the leader's position is aligned to each trajectory's
//!   ancilla sign before the difference is taken, so a trajectory in the
//!   opposite gauge is pulled toward the equivalent configuration.
//! - The reference compares the swarm minimum against a threshold it never
//!   updates, so its leader is the current-step best after the first step;
//!   this kernel takes the current-step best directly.

use std::f32::consts::PI;

use quip_protocol::scoring::energy_milli;
use quip_solver_core::{IsingGraph, SampleParams, SamplerResult};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use crate::sb_core::{
    draw_initial_conditions, gauge_fixed_spins, pump, read_seed, Coupling, Real, SbGraph,
    SbVariant, A0, DT,
};

/// Globally guided SB settings. Defaults are the reference's discrete-form
/// demo values.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GgsbConfig {
    /// Constant subtracted from a perturbed coordinate. Reference demo: 0.01.
    /// Zero disables the perturbation.
    pub perturb_amplitude: f32,
    /// Cosine threshold below which a trajectory's guidance is negated as a
    /// whole. Reference demo: 0.05 for the discrete form.
    pub flip_threshold: f32,
}

impl Default for GgsbConfig {
    fn default() -> Self {
        Self {
            perturb_amplitude: 0.01,
            flip_threshold: 0.05,
        }
    }
}

/// Perturbation probability, `0.1 + 0.45 (1 + cos(pi u))`: 1 at the start,
/// 0.1 at the end.
pub(crate) fn perturb_probability(u: Real) -> Real {
    0.1 + 0.45 * (1.0 + (PI * u).cos())
}

/// Guidance EMA coefficient, `0.9 - 0.4 u`.
pub(crate) fn ema_coefficient(u: Real) -> Real {
    0.9 - 0.4 * u
}

/// Guidance annealing weight, `1 - u`.
pub(crate) fn guidance_weight(u: Real) -> Real {
    1.0 - u
}

/// Index of the lowest energy; the lowest index on ties.
pub(crate) fn leader(energies: &[Real]) -> usize {
    let mut best = 0;
    for (r, &e) in energies.iter().enumerate().skip(1) {
        if e < energies[best] {
            best = r;
        }
    }
    best
}

/// Fill `guid` with the guidance for one trajectory: `(w / (1 + ||d||)) d`
/// with `d = gauge * gbest - x`, negated as a whole when its cosine with `y`
/// is below `-b`. The cosine is zero when either vector is zero, as in the
/// reference's `cosine_similarity` with its `1e-8` floor.
pub(crate) fn guidance_into(
    guid: &mut [Real],
    gbest: &[Real],
    gauge: Real,
    x: &[Real],
    y: &[Real],
    w: Real,
    b: Real,
) {
    let mut norm2: Real = 0.0;
    for ((gi, &bi), &xi) in guid.iter_mut().zip(gbest.iter()).zip(x.iter()) {
        *gi = gauge * bi - xi;
        norm2 += *gi * *gi;
    }
    let scale = w / (1.0 + norm2.sqrt());
    let mut dot: Real = 0.0;
    let mut gn2: Real = 0.0;
    let mut yn2: Real = 0.0;
    for (gi, &yi) in guid.iter_mut().zip(y.iter()) {
        *gi *= scale;
        dot += *gi * yi;
        gn2 += *gi * *gi;
        yn2 += yi * yi;
    }
    let cs = dot / (gn2.sqrt() * yn2.sqrt()).max(1e-8);
    if cs < -b {
        for gi in guid.iter_mut() {
            *gi = -*gi;
        }
    }
}

/// Coupling sums for one trajectory into `forces`, including the ancilla row.
/// Term order matches `sb_step`, so a trajectory with zero guidance is plain
/// SB bit for bit.
fn coupling_sums(g: &SbGraph, coupled: &[Real], forces: &mut [Real]) {
    let n = g.num_nodes();
    for (i, fi) in forces.iter_mut().take(n).enumerate() {
        let (nodes, coups) = g.neighbors(i);
        let mut f: Real = 0.0;
        for (&v, &coup) in nodes.iter().zip(coups.iter()) {
            f += coup * coupled[v as usize];
        }
        if g.has_bias {
            f += g.h[i] * coupled[n];
        }
        *fi = f;
    }
    if g.has_bias {
        let mut f: Real = 0.0;
        for (&bias, &c) in g.h.iter().zip(coupled.iter()) {
            f += bias * c;
        }
        forces[n] = f;
    }
}

/// Extended-problem energy `0.5 sum_i s_i F_i(s)` of the sign configuration
/// of `x`. For the discrete coupling `F` is the coupling sum already in
/// `forces`; for the continuous coupling it is recomputed from the signs into
/// `scratch`.
fn extended_energy(
    g: &SbGraph,
    variant: SbVariant,
    x: &[Real],
    coupled: &[Real],
    forces: &[Real],
    signs: &mut [Real],
    scratch: &mut [Real],
) -> Real {
    match variant.coupling {
        Coupling::Discrete => {
            0.5 * coupled
                .iter()
                .zip(forces.iter())
                .map(|(s, f)| s * f)
                .sum::<Real>()
        }
        Coupling::Continuous => {
            for (s, &xi) in signs.iter_mut().zip(x.iter()) {
                *s = if xi >= 0.0 { 1.0 } else { -1.0 };
            }
            coupling_sums(g, signs, scratch);
            0.5 * signs
                .iter()
                .zip(scratch.iter())
                .map(|(s, f)| s * f)
                .sum::<Real>()
        }
    }
}

/// Sample with the globally guided kernel. All reads advance together.
///
/// # Examples
///
/// ```
/// use quip_miner_cpu::{sample_ggsb, GgsbConfig, IsingGraph, SampleParams, DSB};
///
/// let graph = IsingGraph::new(vec![0.0, 0.0], vec![-1.0], vec![(0, 1)]);
/// let params = SampleParams { num_reads: 4, num_sweeps: 200, seed: 1, ..Default::default() };
/// let results = sample_ggsb(&graph, &params, DSB, GgsbConfig::default());
/// assert_eq!(results.len(), 4);
/// assert!(results.iter().all(|r| r.spins.iter().all(|&s| s == 1 || s == -1)));
/// ```
pub fn sample_ggsb(
    graph: &IsingGraph,
    params: &SampleParams,
    variant: SbVariant,
    cfg: GgsbConfig,
) -> Vec<SamplerResult> {
    let b = params.num_reads.max(1);
    let g = SbGraph::from_base(graph);
    let m = g.num_particles();
    let n = g.num_nodes();
    if m == 0 {
        let energy = energy_milli(&[], &graph.h, &graph.j, &graph.edges);
        return (0..b)
            .map(|_| SamplerResult {
                spins: Vec::new(),
                energy_milli: energy,
            })
            .collect();
    }
    let n_step = params.num_sweeps.max(1);

    let mut rngs: Vec<SmallRng> = Vec::with_capacity(b);
    let mut xs: Vec<Vec<Real>> = Vec::with_capacity(b);
    let mut ys: Vec<Vec<Real>> = Vec::with_capacity(b);
    for r in 0..b {
        let mut rng = SmallRng::seed_from_u64(read_seed(params.seed, r));
        let (x, y) = draw_initial_conditions(m, &mut rng);
        rngs.push(rng);
        xs.push(x);
        ys.push(y);
    }
    let mut coupled: Vec<Vec<Real>> = vec![vec![0.0; m]; b];
    let mut forces: Vec<Vec<Real>> = vec![vec![0.0; m]; b];
    let mut vm: Vec<Vec<Real>> = vec![vec![0.0; m]; b];
    let mut energies: Vec<Real> = vec![0.0; b];
    let mut gbest: Vec<Real> = vec![0.0; m];
    let mut guid: Vec<Real> = vec![0.0; m];
    let mut signs: Vec<Real> = vec![0.0; m];
    let mut scratch: Vec<Real> = vec![0.0; m];

    for k in 0..n_step {
        let u = k as Real / n_step as Real;
        let restore = pump(k, n_step) - A0;
        let q = perturb_probability(u);
        let ema = ema_coefficient(u);
        let w = guidance_weight(u);

        for r in 0..b {
            match variant.coupling {
                Coupling::Discrete => {
                    for (c, &xi) in coupled[r].iter_mut().zip(xs[r].iter()) {
                        *c = if xi >= 0.0 { 1.0 } else { -1.0 };
                    }
                }
                Coupling::Continuous => coupled[r].copy_from_slice(&xs[r]),
            }
            coupling_sums(&g, &coupled[r], &mut forces[r]);
            energies[r] = extended_energy(
                &g,
                variant,
                &xs[r],
                &coupled[r],
                &forces[r],
                &mut signs,
                &mut scratch,
            );
        }
        gbest.copy_from_slice(&xs[leader(&energies)]);

        for r in 0..b {
            let gauge: Real = if g.has_bias && (gbest[n] < 0.0) != (xs[r][n] < 0.0) {
                -1.0
            } else {
                1.0
            };
            guidance_into(
                &mut guid,
                &gbest,
                gauge,
                &xs[r],
                &ys[r],
                w,
                cfg.flip_threshold,
            );
            for (v, &gi) in vm[r].iter_mut().zip(guid.iter()) {
                *v = ema * *v + (1.0 - ema) * gi;
            }
        }

        for r in 0..b {
            let rng = &mut rngs[r];
            for i in 0..m {
                ys[r][i] += (restore * xs[r][i] - g.c0 * forces[r][i] + vm[r][i]) * DT;
                xs[r][i] += A0 * ys[r][i] * DT;
                if rng.gen::<Real>() < q {
                    xs[r][i] -= cfg.perturb_amplitude;
                }
                if xs[r][i] > 1.0 {
                    xs[r][i] = 1.0;
                    ys[r][i] = 0.0;
                } else if xs[r][i] < -1.0 {
                    xs[r][i] = -1.0;
                    ys[r][i] = 0.0;
                }
            }
        }
    }

    xs.iter()
        .map(|x| {
            let spins = gauge_fixed_spins(&g, x);
            SamplerResult {
                energy_milli: energy_milli(&spins, &graph.h, &graph.j, &graph.edges),
                spins,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sb_core::{sample_sb, DSB};
    use quip_solver_core::{IsingGraph, SampleParams};

    fn sb_params(num_reads: usize, num_sweeps: usize, seed: u64) -> SampleParams {
        SampleParams {
            num_reads,
            num_sweeps,
            seed,
            ..Default::default()
        }
    }

    fn biased_chain() -> IsingGraph {
        IsingGraph::new(vec![2.0, -0.5, 0.25], vec![-1.0, 1.0], vec![(0, 1), (1, 2)])
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

    /// Hypothesis: the three schedules run between the reference's endpoints.
    #[test]
    fn schedules_match_the_reference_endpoints() {
        assert!((perturb_probability(0.0) - 1.0).abs() < 1e-6);
        assert!((perturb_probability(1.0) - 0.1).abs() < 1e-6);
        assert!((ema_coefficient(0.0) - 0.9).abs() < 1e-6);
        assert!((ema_coefficient(1.0) - 0.5).abs() < 1e-6);
        assert!((guidance_weight(0.0) - 1.0).abs() < 1e-6);
        assert!(guidance_weight(1.0).abs() < 1e-6);
    }

    /// Hypothesis: the leader is the lowest energy, lowest index on ties.
    #[test]
    fn leader_is_the_lowest_energy_lowest_index() {
        assert_eq!(leader(&[3.0, -1.0, -1.0, 2.0]), 1);
        assert_eq!(leader(&[0.0]), 0);
    }

    /// Hypothesis: guidance points from the trajectory toward the leader,
    /// scaled by `w / (1 + ||d||)`, and flips as a whole when it opposes the
    /// momentum by more than the threshold.
    #[test]
    fn guidance_points_at_the_leader_and_flips_against_momentum() {
        let gbest = [1.0, 1.0];
        let x = [0.0, 0.0];
        let mut guid = [0.0; 2];
        let expect = 0.5 / (1.0 + 2f32.sqrt());
        guidance_into(&mut guid, &gbest, 1.0, &x, &[1.0, 1.0], 0.5, 0.05);
        assert!((guid[0] - expect).abs() < 1e-6 && (guid[1] - expect).abs() < 1e-6);
        guidance_into(&mut guid, &gbest, 1.0, &x, &[-1.0, -1.0], 0.5, 0.05);
        assert!((guid[0] + expect).abs() < 1e-6 && (guid[1] + expect).abs() < 1e-6);
        // Zero momentum: cosine is zero, no flip.
        guidance_into(&mut guid, &gbest, 1.0, &x, &[0.0, 0.0], 0.5, 0.05);
        assert!(guid[0] > 0.0);
        // Gauge -1 pulls toward the mirrored leader.
        guidance_into(&mut guid, &gbest, -1.0, &x, &[0.0, 0.0], 0.5, 0.05);
        assert!(guid[0] < 0.0);
    }

    /// Hypothesis: one trajectory with no perturbation is plain SB bit for
    /// bit, because its leader is itself and the guidance is zero. Covers a
    /// biased graph so the ancilla path is exercised.
    #[test]
    fn single_trajectory_without_perturbation_matches_plain_sb() {
        let cfg = GgsbConfig {
            perturb_amplitude: 0.0,
            ..GgsbConfig::default()
        };
        for (name, g) in [("ring12", ring12()), ("biased", biased_chain())] {
            let swarm = sample_ggsb(&g, &sb_params(1, 300, 21), DSB, cfg);
            let plain = sample_sb(&g, &sb_params(1, 300, 21), DSB);
            assert_eq!(swarm[0].spins, plain[0].spins, "{name}");
            assert_eq!(swarm[0].energy_milli, plain[0].energy_milli, "{name}");
        }
    }

    #[test]
    fn finds_brute_forced_ground_states() {
        for (name, g, reads, sweeps) in [
            ("triangle", frustrated_triangle(), 32, 512),
            ("ring12", ring12(), 64, 2048),
        ] {
            let want = brute_force_min_energy(&g);
            let results = sample_ggsb(
                &g,
                &sb_params(reads, sweeps, 2026),
                DSB,
                GgsbConfig::default(),
            );
            let best = results
                .iter()
                .map(|r| r.energy_milli)
                .min()
                .expect("reads > 0");
            assert_eq!(best, want, "{name}: best {best} != brute-force {want}");
        }
    }

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
            let results = sample_ggsb(&g, &sb_params(3, 40, 1), DSB, GgsbConfig::default());
            assert_eq!(results.len(), 3, "{name}");
            for r in &results {
                assert_eq!(r.spins.len(), g.h.len(), "{name}");
                assert!(r.spins.iter().all(|&s| s == 1 || s == -1), "{name}");
            }
        }
    }

    /// Hypothesis: one seed, one result, and a different seed moves it on a
    /// run too short to converge.
    #[test]
    fn same_seed_same_result_different_seed_differs() {
        let g = ring12();
        let a = sample_ggsb(&g, &sb_params(8, 20, 9), DSB, GgsbConfig::default());
        let b = sample_ggsb(&g, &sb_params(8, 20, 9), DSB, GgsbConfig::default());
        let c = sample_ggsb(&g, &sb_params(8, 20, 10), DSB, GgsbConfig::default());
        let spins = |v: &[SamplerResult]| v.iter().map(|r| r.spins.clone()).collect::<Vec<_>>();
        assert_eq!(spins(&a), spins(&b));
        assert_ne!(spins(&a), spins(&c));
    }
}
