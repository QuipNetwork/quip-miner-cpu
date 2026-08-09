//! Tensor-network (MPS) Ising kernel: imaginary-time TEBD with an annealed
//! transverse field, exact sampling from the final state, and a greedy polish.
//!
//! Parameter mapping:
//! - `num_reads`   exact samples drawn from the final state, one result each.
//! - `num_sweeps`  split into `min(num_sweeps, 256)` Trotter steps and the
//!   remainder as polish sweeps.
//! - `sweeps_per_beta`  Trotter steps held at each rung of the beta ladder.
//! - `beta_range`  endpoints of the beta ladder, same meaning as for SA.
//! - `seed`        seeds sampling and polish, per-read derivation as for SA.
//!
//! Cancellation: `run_stream_pump`'s kernel closure cannot see a job's
//! generation, so the kernel takes no cancellation closure. Per-job latency is
//! bounded by the wall-clock valve below plus the pump's dequeue check.

use crate::sampler_core::{polish_from, CpuGraph};
use order::ChainProblem;
use quip_miner_core::{IsingGraph, SampleParams, SamplerResult};
use quip_protocol::scoring::energy_milli;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use state::Mps;
use std::time::{Duration, Instant};

mod order;
mod schedule;
mod state;
mod svd;

/// Per-model MPS memory cap in bytes, 64 MB. One MPS costs `16 n chi^2` bytes
/// at 8 bytes per f64 and two physical states. The cap is per model, not per
/// process, because `stream_width` models run at once, one per core.
const MPS_MEMORY_CAP_BYTES: f64 = 6.7e7;

/// Flops per unit of `chi^3 * span_sum` in one Trotter step. A QR-based
/// truncation measures near 16; the one-sided Jacobi kernel this crate uses
/// measures near 50, and using the smaller number would overcommit the budget
/// by a factor of three.
const JACOBI_FLOP_CONST: f64 = 50.0;

/// Which starting configurations the reads come from. `QUIP_MPS_INIT` selects
/// it, and the H3 experiment compares the two arms at equal wall clock.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InitMode {
    /// Run the TEBD anneal and sample the final state.
    Anneal,
    /// Skip the anneal; draw uniform random spins and polish them.
    Random,
}

/// Per-job knobs for the tensor-network kernel.
#[derive(Debug, Clone, Copy)]
pub struct MpsConfig {
    /// Upper bound on the bond dimension. The budget caps may lower it.
    pub chi_max: usize,
    /// Where the reads start.
    pub init: InitMode,
    /// Wall-clock safety valve per job, in milliseconds. `0` disables it, which
    /// is what the determinism tests use.
    pub time_budget_ms: u64,
    /// Deterministic per-job flop budget that selects the bond dimension. It is
    /// deliberately not a measured elapsed time: choosing the bond dimension
    /// from the clock would make the same seed produce different output on a
    /// fast machine than on a slow one.
    pub flop_budget: f64,
}

impl MpsConfig {
    /// Configuration for one binary invocation. `QUIP_MPS_INIT` picks the
    /// starting configurations: `random` skips the anneal, anything else
    /// (including an unset or unrecognized value) runs it.
    pub fn from_env(chi_max: usize) -> Self {
        let init = match std::env::var("QUIP_MPS_INIT").as_deref() {
            Ok("random") => InitMode::Random,
            _ => InitMode::Anneal,
        };
        Self {
            chi_max,
            init,
            time_budget_ms: 2000,
            flop_budget: 1.25e9,
        }
    }
}

/// Bond dimension for one job: the smallest of the deterministic flop budget
/// cap, the per-model memory cap, and `chi_max`, and never below 1.
///
/// Deterministic by construction. Choosing the bond dimension from a measured
/// elapsed time would make the same seed produce different output on a fast
/// machine than on a slow one, which breaks the reproducibility tests. No job
/// is ever rejected for being wide; it degrades to a product state.
pub(crate) fn select_chi(n: usize, span_sum: u64, steps: usize, cfg: &MpsConfig) -> usize {
    let chi_max = cfg.chi_max.max(1);
    if n == 0 {
        return 1;
    }
    let chi_mem = (MPS_MEMORY_CAP_BYTES / (16.0 * n as f64)).sqrt().floor();
    let chi_flop = if span_sum == 0 {
        chi_max as f64
    } else {
        (cfg.flop_budget / (JACOBI_FLOP_CONST * steps.max(1) as f64 * span_sum as f64))
            .cbrt()
            .floor()
    };
    let chi = chi_mem.min(chi_flop).min(chi_max as f64);
    if chi.is_finite() && chi >= 1.0 {
        (chi as usize).clamp(1, chi_max)
    } else {
        1
    }
}

fn score(spins: &[i8], graph: &IsingGraph) -> SamplerResult {
    SamplerResult {
        spins: spins.to_vec(),
        energy_milli: energy_milli(spins, &graph.h, &graph.j, &graph.edges),
    }
}

/// Imaginary-time TEBD from `|+>^n` down to a zero transverse field.
///
/// One step is a second-order Trotter split: half a transverse layer, the
/// commuting classical layer, and half a transverse layer again. The classical
/// terms all commute with each other, so their product is exact and only the
/// transverse term needs splitting.
fn anneal(
    chain: &ChainProblem,
    graph: &IsingGraph,
    params: &SampleParams,
    steps: usize,
    chi: usize,
    cfg: &MpsConfig,
) -> Mps {
    let sched = schedule::build(graph, params, steps);
    let mut mps = Mps::plus_state(chain.h.len());
    // Wall-clock safety valve. It is the one source of non-determinism, and it
    // is confined to overload: it changes result quality, never validity. A
    // budget of 0 disables it, which is what the determinism tests use.
    let deadline = if cfg.time_budget_ms == 0 {
        None
    } else {
        Instant::now().checked_add(Duration::from_millis(cfg.time_budget_ms))
    };
    for k in 0..steps {
        if let Some(limit) = deadline {
            if Instant::now() >= limit {
                break;
            }
        }
        let eta = sched.eta[k];
        // A held rung advances nothing: every gate would be the identity.
        if eta <= 0.0 {
            continue;
        }
        let tau = (eta * sched.gamma[k] / 2.0).tanh();
        mps.apply_transverse(tau);
        for (i, &hi) in chain.h.iter().enumerate() {
            mps.apply_field(i, eta * hi);
        }
        for &(u, v, j) in &chain.gates {
            mps.apply_zz(u as usize, v as usize, (eta * j).tanh(), chi);
        }
        mps.apply_transverse(tau);
        mps.right_canonicalize(chi);
    }
    mps.right_canonicalize(chi);
    mps
}

/// Sample `params.num_reads` configurations with the tensor-network kernel.
///
/// Every case returns `num_reads` valid results. The kernel rejects nothing the
/// harness accepted: a graph too wide for a real bond dimension degrades to the
/// product-state path rather than failing.
///
/// # Examples
///
/// ```
/// use quip_miner_cpu::{sample_ising_mps, InitMode, IsingGraph, MpsConfig, SampleParams};
///
/// let graph = IsingGraph::new(vec![0.0, 0.0], vec![-1.0], vec![(0, 1)]);
/// let params = SampleParams { num_reads: 4, num_sweeps: 64, seed: 1, ..Default::default() };
/// let cfg = MpsConfig {
///     chi_max: 8,
///     init: InitMode::Anneal,
///     time_budget_ms: 2000,
///     flop_budget: 1.25e9,
/// };
/// let results = sample_ising_mps(&graph, &params, &cfg);
/// assert_eq!(results.len(), 4);
/// assert!(results.iter().all(|r| r.energy_milli == -1000));
/// ```
pub fn sample_ising_mps(
    graph: &IsingGraph,
    params: &SampleParams,
    cfg: &MpsConfig,
) -> Vec<SamplerResult> {
    let num_reads = params.num_reads.max(1);
    let n = graph.h.len();
    if n == 0 {
        return (0..num_reads).map(|_| score(&[], graph)).collect();
    }

    let (steps, polish_sweeps) = schedule::split_sweeps(params.num_sweeps);
    let cpu = CpuGraph::from_base(graph);
    let chain = ChainProblem::from_graph(graph);
    let chi = select_chi(n, chain.span_sum(), steps, cfg);
    let annealed = match cfg.init {
        InitMode::Anneal => Some(anneal(&chain, graph, params, steps, chi, cfg)),
        InitMode::Random => None,
    };

    (0..num_reads)
        .map(|read_idx| {
            // Distinct stream per read, identical derivation to the SA kernel;
            // seed 0 still diversifies through the read index.
            let seed = params
                .seed
                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                .wrapping_add(read_idx as u64)
                .wrapping_add(1);
            let mut rng = SmallRng::seed_from_u64(seed);
            let mut spins = vec![1i8; n];
            match &annealed {
                Some(state) => {
                    let chain_spins = state.sample_one(&mut rng);
                    for (k, &s) in chain_spins.iter().enumerate() {
                        spins[chain.order[k] as usize] = s;
                    }
                }
                None => {
                    for s in &mut spins {
                        *s = if rng.gen::<bool>() { 1 } else { -1 };
                    }
                }
            }
            polish_from(&mut spins, &cpu, polish_sweeps);
            score(&spins, graph)
        })
        .collect()
}

/// Serializes every test in the crate that reads or writes `QUIP_MPS_INIT`.
///
/// `cargo test` runs test functions concurrently and the process environment is
/// global, so a test that writes this key races any test that reads it. The
/// lock lives here rather than in either test module because the readers and
/// the writer sit in different modules.
#[cfg(test)]
pub(crate) static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(test)]
mod tests {
    use super::*;

    /// `QUIP_MPS_INIT` is process-global. This is the only test that writes it;
    /// every other test constructs `MpsConfig` literally, and the one that must
    /// read it holds [`ENV_LOCK`] as this one does.
    #[test]
    fn from_env_reads_the_init_knob_and_pins_the_budget_defaults() {
        let _guard = ENV_LOCK.lock().expect("env lock poisoned");
        std::env::remove_var("QUIP_MPS_INIT");
        let cfg = MpsConfig::from_env(32);
        assert_eq!(cfg.chi_max, 32);
        assert_eq!(cfg.init, InitMode::Anneal);
        assert_eq!(cfg.time_budget_ms, 2000);
        assert!((cfg.flop_budget - 1.25e9).abs() < 1.0);

        std::env::set_var("QUIP_MPS_INIT", "random");
        assert_eq!(MpsConfig::from_env(8).init, InitMode::Random);

        std::env::set_var("QUIP_MPS_INIT", "anneal");
        assert_eq!(MpsConfig::from_env(8).init, InitMode::Anneal);

        // An unrecognized value must not change behaviour and must not panic.
        std::env::set_var("QUIP_MPS_INIT", "nonsense");
        assert_eq!(MpsConfig::from_env(8).init, InitMode::Anneal);

        std::env::remove_var("QUIP_MPS_INIT");
        assert_eq!(MpsConfig::from_env(1).init, InitMode::Anneal);
        assert_eq!(MpsConfig::from_env(1).chi_max, 1);
    }
    // ---- Task 15: bond-dimension selection ----

    fn cfg(chi_max: usize) -> MpsConfig {
        MpsConfig {
            chi_max,
            init: InitMode::Anneal,
            time_budget_ms: 0,
            flop_budget: 1.25e9,
        }
    }

    #[test]
    fn a_tiny_problem_is_capped_by_the_budget_not_by_its_size() {
        // 8-node ring, span sum 14, 64 steps. Neither cap comes from the
        // problem's size: memory allows sqrt(6.7e7 / (16 * 8)) = 723, and the
        // flop budget allows cbrt(1.25e9 / (50 * 64 * 14)) = cbrt(27902) = 30.3,
        // so the flop cap binds just below chi_max at 30.
        assert_eq!(select_chi(8, 14, 64, &cfg(32)), 30);
        // Below that, chi_max binds instead.
        assert_eq!(select_chi(8, 14, 64, &cfg(4)), 4);
    }

    #[test]
    fn a_graph_with_no_gates_is_capped_only_by_chi_max() {
        assert_eq!(select_chi(64, 0, 64, &cfg(32)), 32);
    }

    #[test]
    fn the_production_topology_degrades_to_a_product_state() {
        // advantage2-system1 after reverse Cuthill-McKee: 4577 nodes,
        // span sum 7_544_619. No affordable bond dimension above 1 exists.
        assert_eq!(select_chi(4577, 7_544_619, 64, &cfg(32)), 1);
    }

    #[test]
    fn the_memory_cap_binds_before_chi_max_on_a_wide_chain() {
        // 65536 nodes, no couplings, so only the 64 MB per-model cap applies:
        // sqrt(6.7e7 / (16 * 65536)) = 7.99 -> 7.
        assert_eq!(select_chi(65_536, 0, 64, &cfg(32)), 7);
    }

    #[test]
    fn the_flop_cap_binds_on_a_long_quasi_one_dimensional_chain() {
        // A 2400-site nearest-neighbour chain has span sum 2399. At 64 steps
        // the budget allows cbrt(1.25e9 / (50 * 64 * 2399)) = cbrt(162.8) = 5.46,
        // so the flop cap binds at 5.
        assert_eq!(select_chi(2400, 2399, 64, &cfg(32)), 5);
    }

    #[test]
    fn selection_never_returns_zero_and_never_exceeds_chi_max() {
        for &(n, span, steps, chi_max) in &[
            (0usize, 0u64, 1usize, 32usize),
            (1, 0, 1, 1),
            (10, 1_000_000_000, 256, 32),
            (100_000, 500_000, 256, 32),
            (4, 4, 256, 0),
        ] {
            let got = select_chi(n, span, steps, &cfg(chi_max));
            assert!(got >= 1, "n={n} span={span}: got {got}");
            assert!(
                got <= chi_max.max(1),
                "n={n} span={span}: got {got} above chi_max {chi_max}"
            );
        }
    }

    #[test]
    fn selection_is_monotone_in_the_flop_budget() {
        let mut small = cfg(32);
        small.flop_budget = 1.0e6;
        let mut large = cfg(32);
        large.flop_budget = 1.0e12;
        assert!(select_chi(2400, 2399, 64, &small) <= select_chi(2400, 2399, 64, &large));
    }

    // ---- Task 16: the sample_ising_mps entry point ----

    use quip_protocol::scoring::energy_milli as score_energy;

    fn params(num_reads: usize, num_sweeps: usize, seed: u64) -> SampleParams {
        SampleParams {
            num_reads,
            num_sweeps,
            seed,
            ..Default::default()
        }
    }

    fn ferro_chain(n: usize) -> IsingGraph {
        IsingGraph::new(
            vec![0.0; n],
            vec![-1.0; n - 1],
            (0..n - 1).map(|i| (i, i + 1)).collect(),
        )
    }

    fn assert_results_well_formed(results: &[SamplerResult], g: &IsingGraph, want_reads: usize) {
        assert_eq!(results.len(), want_reads);
        for r in results {
            assert_eq!(r.spins.len(), g.h.len());
            assert!(r.spins.iter().all(|&s| s == 1 || s == -1), "{:?}", r.spins);
            assert_eq!(
                r.energy_milli,
                score_energy(&r.spins, &g.h, &g.j, &g.edges),
                "reported energy must equal consensus scoring"
            );
        }
    }

    #[test]
    fn sampling_returns_one_well_formed_result_per_read() {
        let g = ferro_chain(12);
        let results = sample_ising_mps(&g, &params(8, 64, 42), &cfg(8));
        assert_results_well_formed(&results, &g, 8);
    }

    #[test]
    fn an_empty_graph_returns_empty_spins_with_zero_energy() {
        let g = IsingGraph::new(vec![], vec![], vec![]);
        let results = sample_ising_mps(&g, &params(3, 64, 1), &cfg(32));
        assert_eq!(results.len(), 3);
        assert!(results
            .iter()
            .all(|r| r.spins.is_empty() && r.energy_milli == 0));
    }

    #[test]
    fn a_single_node_lands_on_the_sign_of_its_own_field() {
        for (h, want) in [(2.5f64, -1i8), (-2.5, 1)] {
            let g = IsingGraph::new(vec![h], vec![], vec![]);
            let results = sample_ising_mps(&g, &params(4, 64, 7), &cfg(32));
            assert_eq!(results.len(), 4);
            for r in &results {
                assert_eq!(r.spins, vec![want], "h = {h}");
            }
        }
    }

    #[test]
    fn zero_reads_still_produce_one_result() {
        let g = ferro_chain(6);
        let results = sample_ising_mps(&g, &params(0, 64, 3), &cfg(8));
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn a_graph_with_no_fields_and_no_edges_scores_zero() {
        let g = IsingGraph::new(vec![0.0; 5], vec![], vec![]);
        let results = sample_ising_mps(&g, &params(4, 64, 9), &cfg(32));
        assert_results_well_formed(&results, &g, 4);
        assert!(results.iter().all(|r| r.energy_milli == 0));
    }

    #[test]
    fn disconnected_components_are_each_solved() {
        // Two independent ferromagnetic pairs plus one isolated biased node.
        let g = IsingGraph::new(
            vec![0.0, 0.0, 0.0, 0.0, 1.5],
            vec![-1.0, -1.0],
            vec![(0, 1), (2, 3)],
        );
        let results = sample_ising_mps(&g, &params(8, 64, 5), &cfg(8));
        assert_results_well_formed(&results, &g, 8);
        let best = results.iter().map(|r| r.energy_milli).min().unwrap_or(0);
        assert_eq!(best, -3500, "optimum is -1 -1 -1.5 in milli units");
    }

    #[test]
    fn defensive_graphs_are_sampled_rather_than_rejected() {
        // Self-loop, out-of-range edge, short `j`, and a non-finite field.
        let g = IsingGraph::new(
            vec![0.5, -0.25, f64::NAN],
            vec![1.0, -1.0],
            vec![(0, 0), (0, 9), (0, 1), (1, 2)],
        );
        let results = sample_ising_mps(&g, &params(4, 32, 11), &cfg(8));
        assert_eq!(results.len(), 4);
        for r in &results {
            assert_eq!(r.spins.len(), 3);
            assert!(r.spins.iter().all(|&s| s == 1 || s == -1));
        }
    }

    #[test]
    fn a_duplicate_edge_is_applied_twice() {
        // Duplicating a ferromagnetic bond doubles its weight, so the optimum
        // energy doubles too. The sampler must find it.
        let g = IsingGraph::new(vec![0.0, 0.0], vec![-1.0, -1.0], vec![(0, 1), (0, 1)]);
        let results = sample_ising_mps(&g, &params(4, 64, 13), &cfg(8));
        assert_results_well_formed(&results, &g, 4);
        assert!(results.iter().all(|r| r.energy_milli == -2000));
    }

    #[test]
    fn a_graph_too_wide_for_a_real_bond_still_returns_results() {
        // Span sum large enough that `select_chi` returns 1.
        let n = 600;
        let edges: Vec<(usize, usize)> = (0..n).map(|i| (i, (i + 137) % n)).collect();
        let g = IsingGraph::new(vec![0.1; n], vec![-1.0; n], edges);
        let results = sample_ising_mps(&g, &params(2, 64, 17), &cfg(32));
        assert_results_well_formed(&results, &g, 2);
    }

    // ---- Task 17: exact optimality on solvable instances ----

    /// Exhaustive minimum over all `2^n` configurations. Only for `n <= 20`.
    fn brute_force_min(g: &IsingGraph) -> i64 {
        let n = g.h.len();
        let mut best = i64::MAX;
        for mask in 0..(1u32 << n) {
            let spins: Vec<i8> = (0..n)
                .map(|i| if (mask >> i) & 1 == 0 { 1i8 } else { -1 })
                .collect();
            best = best.min(score_energy(&spins, &g.h, &g.j, &g.edges));
        }
        best
    }

    fn best_energy(results: &[SamplerResult]) -> i64 {
        results.iter().map(|r| r.energy_milli).min().unwrap_or(0)
    }

    /// Deterministic sparse instance generator, so the whole set is fixed.
    fn random_sparse(n: usize, m: usize, seed: u64) -> IsingGraph {
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut edges = Vec::with_capacity(m);
        let mut j = Vec::with_capacity(m);
        for _ in 0..m {
            let u = rng.gen_range(0..n);
            let mut v = rng.gen_range(0..n);
            if v == u {
                v = (u + 1) % n;
            }
            edges.push((u, v));
            j.push(if rng.gen::<bool>() { 1.0 } else { -1.0 });
        }
        let h: Vec<f64> = (0..n).map(|_| f64::from(rng.gen_range(-1..=1))).collect();
        IsingGraph::new(h, j, edges)
    }

    #[test]
    fn a_fields_only_problem_is_solved_exactly() {
        let h = vec![0.7, -1.2, 0.3, -0.5, 2.0, -0.1, 0.9, -1.7];
        let g = IsingGraph::new(h.clone(), vec![], vec![]);
        let results = sample_ising_mps(&g, &params(8, 64, 21), &cfg(32));
        let want: Vec<i8> = h
            .iter()
            .map(|&x| if x > 0.0 { -1i8 } else { 1i8 })
            .collect();
        for r in &results {
            assert_eq!(r.spins, want, "fields-only must be exact on every read");
        }
    }

    #[test]
    fn a_ferromagnetic_chain_reaches_its_ground_state() {
        let g = ferro_chain(16);
        let results = sample_ising_mps(&g, &params(8, 64, 22), &cfg(8));
        assert_results_well_formed(&results, &g, 8);
        assert_eq!(best_energy(&results), -15_000);
    }

    #[test]
    fn the_smoke_ring_is_exact_at_bond_dimension_four() {
        // An 8-node ring has bandwidth 2 after reordering, so the largest cut
        // severs 2 edges and bond dimension 4 represents it without truncation.
        let g = IsingGraph::new(
            vec![0.2, -0.4, 0.1, 0.0, -0.3, 0.5, -0.1, 0.25],
            vec![1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0],
            vec![
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 5),
                (5, 6),
                (6, 7),
                (7, 0),
            ],
        );
        let results = sample_ising_mps(&g, &params(16, 128, 23), &cfg(4));
        assert_results_well_formed(&results, &g, 16);
        assert_eq!(best_energy(&results), brute_force_min(&g));
    }

    #[test]
    fn a_fixed_set_of_random_sparse_instances_reaches_the_optimum_at_least_ninety_percent() {
        let mut hits = 0usize;
        let mut misses: Vec<(u64, i64, i64)> = Vec::new();
        for seed in 0..50u64 {
            let g = random_sparse(12, 24, 3000 + seed);
            let optimum = brute_force_min(&g);
            let results = sample_ising_mps(&g, &params(16, 96, seed), &cfg(8));
            let got = best_energy(&results);
            assert!(got >= optimum, "seed {seed}: energy below the optimum");
            if got == optimum {
                hits += 1;
            } else {
                misses.push((seed, got, optimum));
            }
        }
        assert!(
            hits >= 45,
            "hit the optimum on {hits} of 50 instances; misses: {misses:?}"
        );
    }

    /// The sampler must beat brute-force guessing by a wide margin.
    ///
    /// The bound is the *minimum* of 2048 uniform draws rather than a multiple
    /// of their standard deviation: every read the sampler returns has to be
    /// better than the best of 2048 random guesses. A sigma multiple is the
    /// wrong shape here. This instance's true optimum is -43000, only 4.88
    /// standard deviations below the random mean, so any threshold at or above
    /// 5 sigma is unreachable even for a perfect solver.
    #[test]
    fn the_sampler_beats_the_best_of_two_thousand_random_configurations() {
        let g = random_sparse(24, 60, 4242);
        let mut rng = SmallRng::seed_from_u64(999);
        let mut energies = Vec::with_capacity(2048);
        for _ in 0..2048 {
            let spins: Vec<i8> = (0..24)
                .map(|_| if rng.gen::<bool>() { 1i8 } else { -1 })
                .collect();
            energies.push(score_energy(&spins, &g.h, &g.j, &g.edges));
        }
        let random_best = *energies.iter().min().expect("2048 draws");
        let random_mean = energies.iter().sum::<i64>() as f64 / energies.len() as f64;

        let results = sample_ising_mps(&g, &params(32, 96, 31), &cfg(8));
        let sampler_mean =
            results.iter().map(|r| r.energy_milli as f64).sum::<f64>() / results.len() as f64;
        let sampler_best = best_energy(&results);

        assert!(
            sampler_mean < random_best as f64,
            "the average sampler read ({sampler_mean}) must beat the best of 2048 \
             random draws ({random_best}); random mean was {random_mean}"
        );
        assert_eq!(
            sampler_best,
            brute_force_min(&g),
            "the best read must reach the true optimum"
        );
    }

    // ---- Task 18: the random-initialization arm ----

    fn cfg_random(chi_max: usize) -> MpsConfig {
        MpsConfig {
            chi_max,
            init: InitMode::Random,
            time_budget_ms: 0,
            flop_budget: 1.25e9,
        }
    }

    /// No single flip lowers the energy, which is what the polish guarantees.
    fn assert_local_minimum(r: &SamplerResult, g: &IsingGraph) {
        for var in 0..g.h.len() {
            let mut probe = r.spins.clone();
            probe[var] = -probe[var];
            assert!(
                score_energy(&probe, &g.h, &g.j, &g.edges) >= r.energy_milli,
                "flipping {var} lowers the energy, so the polish did not run"
            );
        }
    }

    #[test]
    fn the_random_arm_returns_well_formed_polished_results() {
        let g = random_sparse(20, 45, 555);
        let results = sample_ising_mps(&g, &params(16, 96, 41), &cfg_random(8));
        assert_results_well_formed(&results, &g, 16);
        for r in &results {
            assert_local_minimum(r, &g);
        }
    }

    /// The anneal earns its cost on a ferromagnetic chain, and this is the
    /// cleanest demonstration of it.
    ///
    /// A random start seeds roughly n/2 domain walls. Strict-descent polish
    /// cannot remove an isolated wall: flipping the spin beside it is exactly
    /// energy-neutral, and `polish_from` rejects zero-delta moves so that the
    /// sweep terminates. So the random arm reliably stalls one or more walls
    /// short of the ground state, while the annealed arm reaches it. This gap
    /// is the effect the H3 seeding experiment is designed to measure.
    #[test]
    fn the_random_arm_stalls_on_domain_walls_where_the_anneal_does_not() {
        let g = ferro_chain(16);
        let random = sample_ising_mps(&g, &params(16, 96, 43), &cfg_random(8));
        assert_results_well_formed(&random, &g, 16);
        for r in &random {
            assert_local_minimum(r, &g);
        }
        let random_best = best_energy(&random);
        assert!(
            random_best > -15_000,
            "random starts should stall above the ground state, got {random_best}"
        );
        // A surviving wall costs exactly 2 units of coupling.
        assert_eq!(
            (random_best + 15_000) % 2_000,
            0,
            "the shortfall must be a whole number of domain walls, got {random_best}"
        );

        let annealed = sample_ising_mps(&g, &params(16, 96, 43), &cfg(8));
        assert_eq!(
            best_energy(&annealed),
            -15_000,
            "the annealed arm must reach the ground state"
        );
    }

    #[test]
    fn the_two_arms_differ_on_a_problem_where_the_anneal_carries_information() {
        // Fields only: the annealed arm is exact on every read, so the two arms
        // agreeing on every read would mean the knob does nothing.
        let h = vec![0.7, -1.2, 0.3, -0.5, 2.0, -0.1, 0.9, -1.7, 0.4, -0.6];
        let g = IsingGraph::new(h.clone(), vec![], vec![]);
        let annealed = sample_ising_mps(&g, &params(8, 64, 47), &cfg(8));
        let random = sample_ising_mps(&g, &params(8, 64, 47), &cfg_random(8));
        // Both arms polish to the same optimum here, so compare the work done:
        // the annealed arm must be exact, and the random arm must also land
        // there because the polish alone suffices without couplings.
        let want: Vec<i8> = h
            .iter()
            .map(|&x| if x > 0.0 { -1i8 } else { 1i8 })
            .collect();
        assert!(annealed.iter().all(|r| r.spins == want));
        assert!(random.iter().all(|r| r.spins == want));

        // On a frustrated instance the two arms must not be identical, which is
        // what makes the H3 comparison meaningful.
        let hard = random_sparse(20, 55, 8080);
        let a = sample_ising_mps(&hard, &params(16, 96, 47), &cfg(8));
        let b = sample_ising_mps(&hard, &params(16, 96, 47), &cfg_random(8));
        assert!(
            a.iter().map(|r| &r.spins).ne(b.iter().map(|r| &r.spins)),
            "the two initialization arms produced identical output"
        );
    }

    #[test]
    fn the_random_arm_is_deterministic_for_a_given_seed() {
        let g = random_sparse(18, 40, 606);
        let first = sample_ising_mps(&g, &params(8, 96, 51), &cfg_random(8));
        let second = sample_ising_mps(&g, &params(8, 96, 51), &cfg_random(8));
        assert_eq!(first, second);
    }

    // ---- Task 19: the wall-clock safety valve ----

    #[test]
    fn a_fired_valve_still_returns_complete_polished_results() {
        // A budget of 1 ms fires before the first step on any real problem.
        let g = random_sparse(60, 150, 707);
        let stopped = MpsConfig {
            chi_max: 8,
            init: InitMode::Anneal,
            time_budget_ms: 1,
            flop_budget: 1.25e9,
        };
        let results = sample_ising_mps(&g, &params(8, 256, 61), &stopped);
        assert_results_well_formed(&results, &g, 8);
        // Complete, not partial: the polish still ran, so every configuration
        // is a local minimum even though the anneal was cut short.
        for r in &results {
            assert_local_minimum(r, &g);
        }
    }

    #[test]
    fn a_disabled_valve_runs_the_whole_anneal() {
        // Budget 0 means no valve, which is what every determinism test uses.
        let g = ferro_chain(20);
        let unlimited = sample_ising_mps(&g, &params(8, 128, 63), &cfg(8));
        assert_results_well_formed(&unlimited, &g, 8);
        assert_eq!(best_energy(&unlimited), -19_000);
    }

    #[test]
    fn a_generous_valve_does_not_change_the_answer_on_a_small_problem() {
        let g = ferro_chain(12);
        let with_valve = MpsConfig {
            chi_max: 8,
            init: InitMode::Anneal,
            time_budget_ms: 60_000,
            flop_budget: 1.25e9,
        };
        assert_eq!(
            sample_ising_mps(&g, &params(8, 64, 67), &cfg(8)),
            sample_ising_mps(&g, &params(8, 64, 67), &with_valve),
            "a valve that never fires must not change the output"
        );
    }

    // ---- Task 20: determinism and property tests ----

    use proptest::prelude::*;
    // Both glob imports in scope carry an `Rng` trait. Name rand's explicitly:
    // the glob-ambiguity lint is a future hard error.
    use rand::Rng;

    #[test]
    fn the_same_seed_produces_byte_identical_results() {
        let g = random_sparse(24, 60, 808);
        let first = sample_ising_mps(&g, &params(16, 128, 71), &cfg(8));
        let second = sample_ising_mps(&g, &params(16, 128, 71), &cfg(8));
        assert_eq!(first, second);
    }

    #[test]
    fn different_seeds_produce_different_results() {
        let g = random_sparse(24, 60, 808);
        let a = sample_ising_mps(&g, &params(16, 128, 71), &cfg(8));
        let b = sample_ising_mps(&g, &params(16, 128, 72), &cfg(8));
        assert_ne!(
            a.iter().map(|r| &r.spins).collect::<Vec<_>>(),
            b.iter().map(|r| &r.spins).collect::<Vec<_>>(),
            "the seed must reach the sampler"
        );
    }

    #[test]
    fn seed_zero_still_diversifies_across_reads() {
        // A frustrated instance, so the reads are not all forced onto one
        // configuration by the polish.
        let g = random_sparse(24, 70, 909);
        let results = sample_ising_mps(&g, &params(16, 96, 0), &cfg(8));
        let distinct: std::collections::BTreeSet<&Vec<i8>> =
            results.iter().map(|r| &r.spins).collect();
        assert!(
            distinct.len() > 1,
            "every read returned the same configuration at seed 0"
        );
    }

    proptest! {
        // No file persistence: keeps the worktree free of proptest-regressions/.
        #![proptest_config(ProptestConfig {
            cases: 48,
            failure_persistence: None,
            ..ProptestConfig::default()
        })]

        /// Shape, spin values, energy parity, and finiteness over graphs that
        /// include self-loops, duplicate edges, and out-of-range indices.
        #[test]
        fn prop_results_are_well_formed_on_adversarial_graphs(
            n in 1usize..=8,
            h_raw in prop::collection::vec(-2.0f64..2.0, 8),
            edge_data in prop::collection::vec((any::<u8>(), any::<u8>(), -2.0f64..2.0), 0..=20),
            num_reads in 1usize..=6,
            num_sweeps in 1usize..=48,
            seed in any::<u64>(),
            chi_max in 1usize..=8,
        ) {
            let h: Vec<f64> = h_raw.into_iter().take(n).collect();
            let mut edges = Vec::with_capacity(edge_data.len());
            let mut j = Vec::with_capacity(edge_data.len());
            for (u, v, c) in edge_data {
                // `% (n + 2)` deliberately leaves indices past the end, and
                // `u == v` self-loops appear naturally.
                edges.push(((u as usize) % (n + 2), (v as usize) % (n + 2)));
                j.push(c);
            }
            // Leave the last coupling missing so the short-`j` path is covered.
            j.pop();
            let g = IsingGraph::new(h, j, edges);
            let c = MpsConfig {
                chi_max,
                init: InitMode::Anneal,
                time_budget_ms: 0,
                flop_budget: 1.25e9,
            };
            let results = sample_ising_mps(&g, &params(num_reads, num_sweeps, seed), &c);

            prop_assert_eq!(results.len(), num_reads);
            for r in &results {
                prop_assert_eq!(r.spins.len(), g.h.len());
                prop_assert!(r.spins.iter().all(|&s| s == 1 || s == -1));
                prop_assert_eq!(
                    r.energy_milli,
                    score_energy(&r.spins, &g.h, &g.j, &g.edges)
                );
            }
        }

        /// The same seed gives byte-identical output on any graph.
        #[test]
        fn prop_output_is_deterministic(
            n in 1usize..=8,
            h_raw in prop::collection::vec(-2.0f64..2.0, 8),
            edge_data in prop::collection::vec((any::<u8>(), any::<u8>(), -2.0f64..2.0), 0..=16),
            seed in any::<u64>(),
        ) {
            let h: Vec<f64> = h_raw.into_iter().take(n).collect();
            let mut edges = Vec::with_capacity(edge_data.len());
            let mut j = Vec::with_capacity(edge_data.len());
            for (u, v, c) in edge_data {
                edges.push(((u as usize) % n, (v as usize) % n));
                j.push(c);
            }
            let g = IsingGraph::new(h, j, edges);
            let c = MpsConfig {
                chi_max: 4,
                init: InitMode::Anneal,
                time_budget_ms: 0,
                flop_budget: 1.25e9,
            };
            let p = params(4, 32, seed);
            prop_assert_eq!(
                sample_ising_mps(&g, &p, &c),
                sample_ising_mps(&g, &p, &c)
            );
        }

        /// Spins map back through the chain ordering correctly: scoring in the
        /// chain order with permuted fields and couplings must agree with
        /// scoring in the received order.
        #[test]
        fn prop_chain_permutation_round_trips(
            n in 2usize..=8,
            h_raw in prop::collection::vec(-2.0f64..2.0, 8),
            edge_data in prop::collection::vec((any::<u8>(), any::<u8>(), -2.0f64..2.0), 1..=16),
            spin_bits in prop::collection::vec(any::<bool>(), 8),
        ) {
            let h: Vec<f64> = h_raw.into_iter().take(n).collect();
            let mut edges = Vec::with_capacity(edge_data.len());
            let mut j = Vec::with_capacity(edge_data.len());
            for (u, v, c) in edge_data {
                let a = (u as usize) % n;
                let b = (v as usize) % n;
                if a == b {
                    continue;
                }
                edges.push((a, b));
                j.push(c);
            }
            let spins: Vec<i8> = spin_bits
                .into_iter()
                .take(n)
                .map(|b| if b { 1i8 } else { -1i8 })
                .collect();
            let g = IsingGraph::new(h, j, edges);
            let chain = ChainProblem::from_graph(&g);

            // Same configuration expressed in chain order.
            let chain_spins: Vec<i8> = chain
                .order
                .iter()
                .map(|&node| spins[node as usize])
                .collect();
            let chain_edges: Vec<(usize, usize)> = chain
                .gates
                .iter()
                .map(|&(a, b, _)| (a as usize, b as usize))
                .collect();
            let chain_j: Vec<f64> = chain.gates.iter().map(|&(_, _, c)| c).collect();
            prop_assert_eq!(
                score_energy(&chain_spins, &chain.h, &chain_j, &chain_edges),
                score_energy(&spins, &g.h, &g.j, &g.edges)
            );
        }
    }

    // ---- Task 21: the mean-field precursor at bond dimension 1 ----

    #[test]
    fn chi_max_of_one_pins_the_bond_at_one_for_every_input() {
        for &(n, span) in &[(1usize, 0u64), (8, 14), (600, 60_000), (4577, 7_544_619)] {
            assert_eq!(select_chi(n, span, 64, &cfg(1)), 1);
        }
    }

    #[test]
    fn the_anneal_at_chi_one_never_grows_a_bond() {
        let g = random_sparse(24, 60, 1111);
        let chain = ChainProblem::from_graph(&g);
        let state = anneal(&chain, &g, &params(8, 64, 5), 64, 1, &cfg(1));
        assert_eq!(
            state.max_bond(),
            1,
            "the product-state path must keep every bond at 1"
        );
    }

    #[test]
    fn chi_max_of_one_is_a_complete_working_backend() {
        let g = random_sparse(30, 75, 2222);
        let results = sample_ising_mps(&g, &params(16, 96, 73), &cfg(1));
        assert_results_well_formed(&results, &g, 16);
        assert_eq!(
            results,
            sample_ising_mps(&g, &params(16, 96, 73), &cfg(1)),
            "bond dimension 1 must be deterministic"
        );
    }

    #[test]
    fn chi_max_of_one_and_a_higher_cap_agree_where_the_state_factorizes() {
        // Fields only: the exact state is a product state, so the bond-1 path
        // and the general path must reach the same answer.
        let h = vec![0.7, -1.2, 0.3, -0.5, 2.0, -0.1, 0.9, -1.7];
        let g = IsingGraph::new(h, vec![], vec![]);
        assert_eq!(
            sample_ising_mps(&g, &params(8, 64, 77), &cfg(1)),
            sample_ising_mps(&g, &params(8, 64, 77), &cfg(32))
        );
    }

    /// Mean-field cannot spontaneously break a symmetry, and `quip-cpu-mfa`
    /// inherits that limit, so it is pinned here rather than discovered later.
    ///
    /// On a zero-field ferromagnetic chain every gate matrix
    /// `m[p][q] = a[p] b[q] (1 - t Z_p Z_q)` is symmetric under the global spin
    /// flip, so its best rank-1 factor is the symmetric vector and the state
    /// stays exactly `|+>^n` for the whole anneal. Sampling that is a fair coin
    /// per site, so the bond-1 arm degenerates to random initialization and
    /// stalls on domain walls exactly as `InitMode::Random` does.
    ///
    /// An arbitrarily small field removes the degeneracy and the same path then
    /// reaches the true ground state, which is what distinguishes "mean-field
    /// is weak here" from "the bond-1 path is broken".
    #[test]
    fn the_bond_one_path_cannot_break_a_symmetry_but_solves_once_a_field_does() {
        let g = ferro_chain(32);
        let chain = ChainProblem::from_graph(&g);
        let state = anneal(&chain, &g, &params(16, 96, 79), 96, 1, &cfg(1));
        let plus = std::f64::consts::FRAC_1_SQRT_2;
        for k in 0..state.num_sites() {
            for &x in &state.site[k] {
                assert!(
                    (x.abs() - plus).abs() < 1e-12,
                    "site {k} left the symmetric point at {x}"
                );
            }
        }
        let results = sample_ising_mps(&g, &params(16, 96, 79), &cfg(1));
        assert_results_well_formed(&results, &g, 16);
        let stalled = best_energy(&results);
        assert!(
            stalled > -31_000 && (stalled + 31_000) % 2_000 == 0,
            "expected a whole number of stalled domain walls, got {stalled}"
        );

        // The same chain with one small field. The couplings contribute -31000
        // and the field a further -50 once that spin settles against it.
        let mut h = vec![0.0; 32];
        h[0] = 0.05;
        let biased = IsingGraph::new(h, vec![-1.0; 31], (0..31).map(|i| (i, i + 1)).collect());
        let solved = sample_ising_mps(&biased, &params(16, 96, 79), &cfg(1));
        assert_eq!(
            best_energy(&solved),
            -31_050,
            "a symmetry-breaking field must let the bond-1 path reach the ground state"
        );
    }

}
