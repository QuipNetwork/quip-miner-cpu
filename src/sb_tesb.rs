//! Tabu-enhanced Simulated Bifurcation (Tao, Zeng, Huang, Zuo, Liu, Zhuang,
//! Okawa, Yung; Communications Physics 9, 100, 2026).
//!
//! Two phases inside one job. A warm-up runs `warm_replicas` plain SB
//! trajectories over the first `(1 - alpha)` of the step budget and stores
//! each final configuration as one column of a tabu list. The checking phase
//! restarts `num_reads` trajectories from fresh random states over the
//! remaining `alpha` of the budget, and at every step averages `mini_batch`
//! stored columns into a field `tau` that enters the momentum update as
//! `- c0 * beta * tau_i`, a push away from configurations already found.
//! The draw of columns is shared by every replica at a given step, as in
//! Algorithm 1 of the paper.
//!
//! Departures from the paper, each deliberate:
//! - The paper's Eq. (10) also shifts the restoring coefficient by
//!   `+ c0 beta`; both released implementations omit the shift and every
//!   published number comes from them, so this kernel omits it too.
//! - Stored columns are gauge-fixed by their ancilla sign and the ancilla is
//!   dropped, so `tau` lives in the `+1` gauge. Each checking step multiplies
//!   the field by the replica's own ancilla sign, and the ancilla itself
//!   receives no tabu force. With zero fields there is no ancilla and this is
//!   the published algorithm exactly.
//! - The initial draw stays `U(-0.1, 0.1)`; the reference uses a range twenty
//!   times narrower.
//! - The checking replica `r` uses the seed of read `r` of `sample_sb`, so
//!   `beta = 0` reproduces plain SB over the checking steps bit for bit.

use quip_protocol::scoring::energy_milli;
use quip_solver_core::{IsingGraph, SampleParams, SamplerResult};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use crate::sb_core::{
    draw_initial_conditions, gauge_fixed_spins, read_seed, sb_step, Real, SbGraph, SbVariant,
    Trajectory,
};

/// Salt on the job seed for the warm-up replica streams, so they never share
/// a seed with a checking replica.
const WARM_SALT: u64 = 0x5441_4255_5741_524D;
/// Salt on the job seed for the shared mini-batch draw sequence.
const BATCH_SALT: u64 = 0x5441_4255_4241_5443;

/// Tabu-enhanced SB settings. Defaults are the paper's.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TesbConfig {
    /// Fraction of `num_sweeps` spent in the checking phase. Paper: 0.9.
    pub alpha: f32,
    /// Tabu penalty intensity. Paper: 1.0. Zero reduces the checking phase to
    /// plain SB.
    pub beta: f32,
    /// Stored columns averaged into the tabu field at each step. Paper: 2.
    pub mini_batch: usize,
    /// Warm-up replicas, one stored column each. Paper: 100.
    pub warm_replicas: usize,
    /// Multiplier on `c0` for the checking phase. The paper picks the checking
    /// `c0` per instance from `{0.02, 0.05, 0.08, 0.1}` against a warm-up
    /// value near 0.11; 1.0 keeps the warm-up value.
    pub check_c0_scale: f32,
}

impl Default for TesbConfig {
    fn default() -> Self {
        Self {
            alpha: 0.9,
            beta: 1.0,
            mini_batch: 2,
            warm_replicas: 100,
            check_c0_scale: 1.0,
        }
    }
}

/// Split the step budget into `(warm, check)`. The checking phase is
/// `round(alpha * total)` clamped to `[1, total]`; the warm-up gets the rest
/// and may be empty, in which case the tabu list is empty and the checking
/// phase runs plain SB.
pub(crate) fn phase_steps(total: usize, alpha: f32) -> (usize, usize) {
    let total = total.max(1);
    let check = ((alpha * total as f32).round() as usize).clamp(1, total);
    (total - check, check)
}

fn empty_reads(graph: &IsingGraph, num_reads: usize) -> Vec<SamplerResult> {
    let energy = energy_milli(&[], &graph.h, &graph.j, &graph.edges);
    (0..num_reads)
        .map(|_| SamplerResult {
            spins: Vec::new(),
            energy_milli: energy,
        })
        .collect()
}

/// Run the warm-up and return the stored columns, gauge-fixed, ancilla
/// dropped. Empty when the warm-up has no steps or no replicas.
pub(crate) fn warm_up_columns(
    graph: &IsingGraph,
    params: &SampleParams,
    variant: SbVariant,
    cfg: TesbConfig,
) -> Vec<Vec<i8>> {
    let g = SbGraph::from_base(graph);
    let m = g.num_particles();
    let (k_warm, _) = phase_steps(params.num_sweeps, cfg.alpha);
    if m == 0 || k_warm == 0 {
        return Vec::new();
    }
    let heated = variant.gamma != 0.0;
    let controlled = variant.control != 0.0;
    (0..cfg.warm_replicas)
        .map(|l| {
            let mut rng = SmallRng::seed_from_u64(read_seed(params.seed ^ WARM_SALT, l));
            let (x, y) = draw_initial_conditions(m, &mut rng);
            let mut t = Trajectory::new(x, y, heated, controlled);
            for k in 0..k_warm {
                sb_step(&g, k, k_warm, variant, &mut t, None);
            }
            gauge_fixed_spins(&g, &t.x)
        })
        .collect()
}

/// Sample with the two-phase tabu kernel. `variant` selects the base SB form
/// for both phases; the paper's headline form is the discrete one.
///
/// # Examples
///
/// ```
/// use quip_miner_cpu::{sample_tesb, IsingGraph, SampleParams, TesbConfig, DSB};
///
/// let graph = IsingGraph::new(vec![0.0, 0.0], vec![-1.0], vec![(0, 1)]);
/// let params = SampleParams { num_reads: 4, num_sweeps: 200, seed: 1, ..Default::default() };
/// let results = sample_tesb(&graph, &params, DSB, TesbConfig::default());
/// assert_eq!(results.len(), 4);
/// assert!(results.iter().all(|r| r.spins.iter().all(|&s| s == 1 || s == -1)));
/// ```
pub fn sample_tesb(
    graph: &IsingGraph,
    params: &SampleParams,
    variant: SbVariant,
    cfg: TesbConfig,
) -> Vec<SamplerResult> {
    let num_reads = params.num_reads.max(1);
    let g = SbGraph::from_base(graph);
    let m = g.num_particles();
    let n = g.num_nodes();
    if m == 0 {
        return empty_reads(graph, num_reads);
    }
    let (_, k_check) = phase_steps(params.num_sweeps, cfg.alpha);
    let heated = variant.gamma != 0.0;
    let controlled = variant.control != 0.0;

    let tabu = warm_up_columns(graph, params, variant, cfg);
    let mini_batch = cfg.mini_batch.max(1);
    // One shared draw sequence: Algorithm 1 indexes the mini-batch by the step
    // alone, so every replica feels the same field at the same step.
    let draws: Vec<u32> = if tabu.is_empty() {
        Vec::new()
    } else {
        let mut rng = SmallRng::seed_from_u64(read_seed(params.seed ^ BATCH_SALT, 0));
        (0..k_check * mini_batch)
            .map(|_| rng.gen_range(0..tabu.len()) as u32)
            .collect()
    };
    let active = !tabu.is_empty() && cfg.beta != 0.0;
    let mut g_check = g.clone();
    g_check.scale_c0(cfg.check_c0_scale);
    let inv_mb = 1.0 / mini_batch as Real;
    let mut tau: Vec<Real> = vec![0.0; n];

    (0..num_reads)
        .map(|r| {
            let mut rng = SmallRng::seed_from_u64(read_seed(params.seed, r));
            let (x, y) = draw_initial_conditions(m, &mut rng);
            let mut t = Trajectory::new(x, y, heated, controlled);
            for k in 0..k_check {
                if !active {
                    sb_step(&g_check, k, k_check, variant, &mut t, None);
                    continue;
                }
                let idx = &draws[k * mini_batch..(k + 1) * mini_batch];
                for (i, ti) in tau.iter_mut().enumerate() {
                    let sum: i32 = idx.iter().map(|&l| i32::from(tabu[l as usize][i])).sum();
                    *ti = sum as Real * inv_mb;
                }
                // Stored columns live in the +1 gauge; the replica's gauge is
                // its ancilla sign, read before the step like every force.
                let gauge: Real = if g.has_bias && t.x[n] < 0.0 {
                    -1.0
                } else {
                    1.0
                };
                sb_step(
                    &g_check,
                    k,
                    k_check,
                    variant,
                    &mut t,
                    Some((&tau, gauge * cfg.beta)),
                );
            }
            let spins = gauge_fixed_spins(&g_check, &t.x);
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

    /// Hypothesis: the step budget splits into `(warm, check)` with the
    /// checking phase rounded from `alpha` and clamped to `[1, total]`.
    #[test]
    fn phase_steps_splits_the_budget() {
        assert_eq!(phase_steps(1000, 0.9), (100, 900));
        assert_eq!(phase_steps(1, 0.9), (0, 1));
        assert_eq!(phase_steps(10, 0.0), (9, 1));
        assert_eq!(phase_steps(10, 1.0), (0, 10));
        assert_eq!(phase_steps(0, 0.9), (0, 1));
    }

    /// Hypothesis: with `beta = 0` the checking phase is plain SB over
    /// `K_c` steps at the same per-read seeds, bit for bit. This pins the
    /// seed contract and the zero-field path of `sb_step`.
    #[test]
    fn beta_zero_reproduces_plain_sb_over_the_checking_steps() {
        let g = ring12();
        let cfg = TesbConfig {
            beta: 0.0,
            ..TesbConfig::default()
        };
        let tabu = sample_tesb(&g, &sb_params(6, 1000, 77), DSB, cfg);
        let plain = sample_sb(&g, &sb_params(6, 900, 77), DSB);
        assert_eq!(tabu.len(), 6);
        for (a, b) in tabu.iter().zip(plain.iter()) {
            assert_eq!(a.spins, b.spins);
            assert_eq!(a.energy_milli, b.energy_milli);
        }
    }

    /// Hypothesis: the tabu field repels the stored configuration. On a
    /// ferromagnetic pair both aligned states are ground states; with one
    /// warm-up replica stored and `beta` large enough to dominate the
    /// coupling, every checking replica lands on the state the warm-up did
    /// not.
    #[test]
    fn tabu_field_repels_the_stored_configuration() {
        let g = ferro_pair();
        let cfg = TesbConfig {
            beta: 2.0,
            warm_replicas: 1,
            mini_batch: 1,
            ..TesbConfig::default()
        };
        let params = sb_params(8, 200, 3);
        let stored = warm_up_columns(&g, &params, DSB, cfg);
        assert_eq!(stored.len(), 1);
        assert!(stored[0][0] == stored[0][1], "warm-up must align the pair");
        let results = sample_tesb(&g, &params, DSB, cfg);
        for r in &results {
            assert_eq!(r.energy_milli, -1000, "must still be a ground state");
            assert_eq!(r.spins[0], -stored[0][0], "must avoid the stored state");
        }
    }

    /// Hypothesis: the kernel reaches the true ground state on small
    /// frustrated instances with the paper's defaults.
    #[test]
    fn finds_brute_forced_ground_states() {
        for (name, g, reads, sweeps) in [
            ("triangle", frustrated_triangle(), 16, 256),
            ("ring12", ring12(), 32, 2048),
        ] {
            let want = brute_force_min_energy(&g);
            let results = sample_tesb(
                &g,
                &sb_params(reads, sweeps, 2026),
                DSB,
                TesbConfig::default(),
            );
            let best = results
                .iter()
                .map(|r| r.energy_milli)
                .min()
                .expect("reads > 0");
            assert_eq!(best, want, "{name}: best {best} != brute-force {want}");
        }
    }

    /// Hypothesis: degenerate inputs return `num_reads` valid spin vectors.
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
            let results = sample_tesb(&g, &sb_params(3, 40, 1), DSB, TesbConfig::default());
            assert_eq!(results.len(), 3, "{name}");
            for r in &results {
                assert_eq!(r.spins.len(), g.h.len(), "{name}");
                assert!(r.spins.iter().all(|&s| s == 1 || s == -1), "{name}");
            }
        }
    }

    /// Hypothesis: one seed, one result, and a different seed moves it. The
    /// run is short so the reads are far from converged; at a few hundred
    /// steps every read reaches one of the ring's few ground states and two
    /// seeds can legitimately agree.
    #[test]
    fn same_seed_same_result_different_seed_differs() {
        let g = ring12();
        let a = sample_tesb(&g, &sb_params(8, 20, 9), DSB, TesbConfig::default());
        let b = sample_tesb(&g, &sb_params(8, 20, 9), DSB, TesbConfig::default());
        let c = sample_tesb(&g, &sb_params(8, 20, 10), DSB, TesbConfig::default());
        let spins = |v: &[SamplerResult]| v.iter().map(|r| r.spins.clone()).collect::<Vec<_>>();
        assert_eq!(spins(&a), spins(&b));
        assert_ne!(spins(&a), spins(&c));
    }
}
