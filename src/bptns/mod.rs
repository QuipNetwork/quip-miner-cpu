//! Belief-propagation tensor-network (BP-TNS) Ising kernel: imaginary-time
//! evolution on a tensor network laid directly on the problem graph, with
//! simple-update (BP-gauged) truncation, BP marginals, sequential
//! conditioned sampling, and a greedy polish.
//!
//! The published method (Tindall, Mello, Fishman, Stoudenmire, Sels; Science
//! 392, 868, 2026) evolves real-time quantum annealing dynamics on
//! low-degree logical lattices and measures observables by contraction. This
//! kernel adapts it to classical ground-state search: imaginary time instead
//! of real time, and conditioned sampling instead of contraction, because a
//! miner needs configurations. Both adaptations are ours, not the paper's.
//!
//! Cost model, and why the production topology degenerates: a site tensor
//! holds `2 chi^degree` values and one BP iteration costs
//! `O(N chi^(degree + 1))`, so cost is exponential in vertex degree. The
//! published lattices have degree at most ~6; `advantage2-system1` has mean
//! degree 18.1 and max degree 20, so under the 64 MB per-model cap the only
//! affordable bond dimension is 1 and the kernel runs as mean field. That is
//! measured by `select_chi` per job, never assumed.
//!
//! Parameter mapping matches the MPS kernel: `num_sweeps` splits into
//! Trotter steps and polish sweeps, `beta_range`/`sweeps_per_beta` shape the
//! same ladder, `seed` derives per-read streams exactly as SA does.

use crate::mps::schedule;
use crate::sampler_core::{polish_from, CpuGraph};
use graph::NetGraph;
use quip_miner_core::{IsingGraph, SampleParams, SamplerResult};
use quip_protocol::scoring::energy_milli;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use sample::SamplingNet;
use state::{apply_zz, SiteTensor};
use std::time::{Duration, Instant};

mod graph;
mod sample;
mod state;

/// Per-model memory cap in bytes, matching the MPS kernel's 64 MB.
const MEM_CAP_BYTES: f64 = 6.7e7;

/// Flops per `m * n * min(m, n)` of one Jacobi factorization, matching the
/// constant measured for the MPS kernel's SVD.
const SVD_FLOP_CONST: f64 = 50.0;

/// Per-job knobs for the BP-TNS kernel.
#[derive(Debug, Clone, Copy)]
pub struct BptnsConfig {
    /// Upper bound on the bond dimension. The budget caps may lower it.
    pub chi_max: usize,
    /// Wall-clock safety valve for the anneal, in milliseconds. `0` disables
    /// it, which is what the determinism tests use.
    pub time_budget_ms: u64,
    /// Deterministic flop budget selecting the bond dimension, as for MPS:
    /// a measured elapsed time would make the same seed produce different
    /// output on a fast machine than on a slow one.
    pub flop_budget: f64,
}

impl BptnsConfig {
    /// Documented defaults for one binary invocation.
    pub fn new(chi_max: usize) -> Self {
        Self {
            chi_max,
            time_budget_ms: 2000,
            flop_budget: 1.25e9,
        }
    }
}

/// Bond dimension for one job: the largest value inside both the memory cap
/// and the flop budget, never above `chi_max` and never below 1.
///
/// Memory counts `16 chi^degree` bytes per site (two physical values, 8
/// bytes each). Flops count one truncation per bond per step at the Jacobi
/// constant. When even bond dimension 1 misses the flop budget the job still
/// runs at 1 and relies on the wall-clock valve: no job is ever rejected.
pub(crate) fn select_chi(net: &NetGraph, steps: usize, cfg: &BptnsConfig) -> usize {
    let chi_max = cfg.chi_max.max(1);
    if net.num_nodes() == 0 || net.bonds.is_empty() {
        return chi_max;
    }
    for chi in (2..=chi_max).rev() {
        let c = chi as f64;
        let mem: f64 = net
            .adj
            .iter()
            .map(|a| 16.0 * c.powi(a.len() as i32))
            .sum();
        // `mem` can be infinite at high degree; both infinity and NaN must
        // fail the cap, so compare in the direction that rejects them.
        if mem > MEM_CAP_BYTES || mem.is_nan() {
            continue;
        }
        let mut per_step = 0.0f64;
        for &(u, v, _) in &net.bonds {
            let m = 2.0 * c.powi(net.adj[u as usize].len() as i32 - 1);
            let n = 2.0 * c.powi(net.adj[v as usize].len() as i32 - 1);
            per_step += SVD_FLOP_CONST * m * n * m.min(n);
        }
        if steps.max(1) as f64 * per_step <= cfg.flop_budget {
            return chi;
        }
    }
    1
}

/// Split `tensors` at two distinct indices.
fn two_mut(tensors: &mut [SiteTensor], a: usize, b: usize) -> (&mut SiteTensor, &mut SiteTensor) {
    // Bonds store `u < v`, so `a < b` always holds here.
    let (lo, hi) = tensors.split_at_mut(b);
    (&mut lo[a], &mut hi[0])
}

/// Imaginary-time evolution from `|+>^n` down to zero transverse field.
/// Returns the site tensors and per-bond Schmidt weights.
fn anneal(
    net: &NetGraph,
    graph: &IsingGraph,
    params: &SampleParams,
    steps: usize,
    chi: usize,
    cfg: &BptnsConfig,
) -> (Vec<SiteTensor>, Vec<Vec<f64>>) {
    let sched = schedule::build(graph, params, steps);
    let mut tensors: Vec<SiteTensor> = net.adj.iter().map(|a| SiteTensor::plus(a.len())).collect();
    let mut lambdas: Vec<Vec<f64>> = net.bonds.iter().map(|_| vec![1.0]).collect();
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
        if eta <= 0.0 {
            continue;
        }
        let tau = (eta * sched.gamma[k] / 2.0).tanh();
        let mixer = [[1.0, tau], [tau, 1.0]];
        for t in &mut tensors {
            t.apply_phys_matrix(&mixer);
        }
        for (i, t) in tensors.iter_mut().enumerate() {
            let a = eta * net.h[i];
            t.scale_phys(&[(-a).exp(), a.exp()]);
        }
        for e in 0..net.bonds.len() {
            let (u, v, j) = net.bonds[e];
            let (u, v) = (u as usize, v as usize);
            let t = (eta * j).tanh();
            let mut lam = std::mem::take(&mut lambdas[e]);
            let env = |node: usize| -> Vec<Vec<f64>> {
                net.adj[node]
                    .iter()
                    .map(|&(be, _)| {
                        if be as usize == e {
                            Vec::new()
                        } else {
                            lambdas[be as usize].clone()
                        }
                    })
                    .collect()
            };
            let env_u = env(u);
            let env_v = env(v);
            let env_u_refs: Vec<&[f64]> = env_u.iter().map(Vec::as_slice).collect();
            let env_v_refs: Vec<&[f64]> = env_v.iter().map(Vec::as_slice).collect();
            let bu = net.adj[u]
                .iter()
                .position(|&(be, _)| be as usize == e)
                .unwrap_or(0);
            let bv = net.adj[v]
                .iter()
                .position(|&(be, _)| be as usize == e)
                .unwrap_or(0);
            let (tu, tv) = two_mut(&mut tensors, u, v);
            apply_zz(tu, tv, bu, bv, t, &mut lam, &env_u_refs, &env_v_refs, chi);
            lambdas[e] = lam;
        }
        for t in &mut tensors {
            t.apply_phys_matrix(&mixer);
            t.renormalize();
        }
    }
    (tensors, lambdas)
}

fn score(spins: &[i8], graph: &IsingGraph) -> SamplerResult {
    SamplerResult {
        spins: spins.to_vec(),
        energy_milli: energy_milli(spins, &graph.h, &graph.j, &graph.edges),
    }
}

/// Sample `params.num_reads` configurations with the BP-TNS kernel.
///
/// Every case returns `num_reads` valid results: a graph whose degree makes
/// any bond dimension above 1 unaffordable degrades to the mean-field path
/// rather than failing.
///
/// # Examples
///
/// ```
/// use quip_miner_cpu::{sample_ising_bptns, BptnsConfig, IsingGraph, SampleParams};
///
/// let graph = IsingGraph::new(vec![0.0, 0.0], vec![-1.0], vec![(0, 1)]);
/// let params = SampleParams { num_reads: 4, num_sweeps: 64, seed: 1, ..Default::default() };
/// let results = sample_ising_bptns(&graph, &params, &BptnsConfig::new(8));
/// assert_eq!(results.len(), 4);
/// assert!(results.iter().all(|r| r.energy_milli == -1000));
/// ```
pub fn sample_ising_bptns(
    graph: &IsingGraph,
    params: &SampleParams,
    cfg: &BptnsConfig,
) -> Vec<SamplerResult> {
    let num_reads = params.num_reads.max(1);
    let n = graph.h.len();
    if n == 0 {
        return (0..num_reads).map(|_| score(&[], graph)).collect();
    }

    let (steps, polish_sweeps) = schedule::split_sweeps(params.num_sweeps);
    let net = NetGraph::from_graph(graph);
    let chi = select_chi(&net, steps, cfg);
    let (tensors, lambdas) = anneal(&net, graph, params, steps, chi, cfg);
    let mut sampling = SamplingNet::build(&net, &tensors, &lambdas);
    sampling.run_bp(&net);
    let cpu = CpuGraph::from_base(graph);

    (0..num_reads)
        .map(|read_idx| {
            // Identical per-read seed derivation to the SA and MPS kernels.
            let seed = params
                .seed
                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                .wrapping_add(read_idx as u64)
                .wrapping_add(1);
            let mut rng = SmallRng::seed_from_u64(seed);
            let mut spins = sampling.clone().sample_one(&net, &mut rng);
            polish_from(&mut spins, &cpu, polish_sweeps);
            score(&spins, graph)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use quip_protocol::scoring::energy_milli as score_energy;
    use rand::Rng;

    fn cfg(chi_max: usize) -> BptnsConfig {
        BptnsConfig {
            chi_max,
            time_budget_ms: 0,
            flop_budget: 1.25e9,
        }
    }

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

    fn best_energy(results: &[SamplerResult]) -> i64 {
        results.iter().map(|r| r.energy_milli).min().unwrap_or(0)
    }

    fn assert_results_well_formed(results: &[SamplerResult], g: &IsingGraph, want: usize) {
        assert_eq!(results.len(), want);
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

    // ---- select_chi ----

    #[test]
    fn a_low_degree_ring_affords_a_real_bond_dimension() {
        let g = IsingGraph::new(
            vec![0.0; 8],
            vec![1.0; 8],
            (0..8).map(|i| (i, (i + 1) % 8)).collect(),
        );
        let net = NetGraph::from_graph(&g);
        let got = select_chi(&net, 64, &cfg(8));
        assert!(got >= 2, "an 8-ring at 64 steps affords chi >= 2, got {got}");
        assert_eq!(select_chi(&net, 64, &cfg(1)), 1);
    }

    /// The production mechanism, pinned: at degree 18 to 20 the site-tensor
    /// memory `16 chi^degree` exceeds the 64 MB cap for every `chi >= 2`,
    /// so the kernel runs as mean field on `advantage2-system1`-like graphs.
    #[test]
    fn production_degrees_force_the_bond_dimension_to_one() {
        // A star of degree 20 plus a ring, standing in for the pivot's max
        // degree: 16 * 2^20 = 16.8 MB for one site is already a quarter of
        // the cap, and the flop budget fails long before that.
        let mut edges: Vec<(usize, usize)> = (1..=20).map(|i| (0, i)).collect();
        edges.extend((1..20).map(|i| (i, i + 1)));
        let m = edges.len();
        let g = IsingGraph::new(vec![0.0; 21], vec![1.0; m], edges);
        let net = NetGraph::from_graph(&g);
        assert_eq!(net.max_degree(), 20);
        assert_eq!(select_chi(&net, 64, &cfg(8)), 1);
    }

    #[test]
    fn an_edgeless_graph_is_capped_only_by_chi_max() {
        let net = NetGraph::from_graph(&IsingGraph::new(vec![0.0; 64], vec![], vec![]));
        assert_eq!(select_chi(&net, 64, &cfg(8)), 8);
        assert_eq!(select_chi(&NetGraph::from_graph(&IsingGraph::new(vec![], vec![], vec![])), 64, &cfg(4)), 4);
    }

    #[test]
    fn selection_is_monotone_in_the_flop_budget_and_never_zero() {
        let g = ferro_chain(64);
        let net = NetGraph::from_graph(&g);
        let mut small = cfg(8);
        small.flop_budget = 1.0e3;
        let mut large = cfg(8);
        large.flop_budget = 1.0e12;
        let lo = select_chi(&net, 64, &small);
        let hi = select_chi(&net, 64, &large);
        assert!(lo >= 1 && hi <= 8 && lo <= hi, "lo {lo} hi {hi}");
    }

    // ---- the sampler entry point ----

    #[test]
    fn sampling_returns_one_well_formed_result_per_read() {
        let g = ferro_chain(12);
        let results = sample_ising_bptns(&g, &params(8, 64, 42), &cfg(8));
        assert_results_well_formed(&results, &g, 8);
    }

    #[test]
    fn an_empty_graph_returns_empty_spins_with_zero_energy() {
        let g = IsingGraph::new(vec![], vec![], vec![]);
        let results = sample_ising_bptns(&g, &params(3, 64, 1), &cfg(8));
        assert_eq!(results.len(), 3);
        assert!(results.iter().all(|r| r.spins.is_empty() && r.energy_milli == 0));
    }

    #[test]
    fn zero_reads_still_produce_one_result() {
        let g = ferro_chain(6);
        assert_eq!(sample_ising_bptns(&g, &params(0, 64, 3), &cfg(8)).len(), 1);
    }

    #[test]
    fn a_fields_only_problem_is_solved_exactly() {
        let h = vec![0.7, -1.2, 0.3, -0.5, 2.0, -0.1, 0.9, -1.7];
        let g = IsingGraph::new(h.clone(), vec![], vec![]);
        let results = sample_ising_bptns(&g, &params(8, 64, 21), &cfg(8));
        let want: Vec<i8> = h.iter().map(|&x| if x > 0.0 { -1i8 } else { 1 }).collect();
        for r in &results {
            assert_eq!(r.spins, want, "fields-only must be exact on every read");
        }
    }

    /// The decisive qualitative test: on a zero-field ferromagnetic chain a
    /// mean-field state cannot break the global flip symmetry, but a network
    /// with real bonds carries the correlation, and sequential conditioning
    /// breaks the symmetry spin by spin. Reaching the ground state here is
    /// what separates this kernel from `cpu-mfa`.
    #[test]
    fn a_ferromagnetic_chain_reaches_its_ground_state_through_conditioning() {
        let g = ferro_chain(16);
        let results = sample_ising_bptns(&g, &params(8, 64, 22), &cfg(8));
        assert_results_well_formed(&results, &g, 8);
        assert_eq!(best_energy(&results), -15_000);
    }

    #[test]
    fn a_duplicate_edge_merges_and_still_reaches_the_doubled_optimum() {
        let g = IsingGraph::new(vec![0.0, 0.0], vec![-1.0, -1.0], vec![(0, 1), (0, 1)]);
        let results = sample_ising_bptns(&g, &params(4, 64, 13), &cfg(8));
        assert_results_well_formed(&results, &g, 4);
        assert!(results.iter().all(|r| r.energy_milli == -2000));
    }

    #[test]
    fn defensive_graphs_are_sampled_rather_than_rejected() {
        let g = IsingGraph::new(
            vec![0.5, -0.25, f64::NAN],
            vec![1.0, -1.0],
            vec![(0, 0), (0, 9), (0, 1), (1, 2)],
        );
        let results = sample_ising_bptns(&g, &params(4, 32, 11), &cfg(8));
        assert_eq!(results.len(), 4);
        for r in &results {
            assert_eq!(r.spins.len(), 3);
            assert!(r.spins.iter().all(|&s| s == 1 || s == -1));
        }
    }

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

    #[test]
    fn a_fixed_set_of_random_sparse_instances_reaches_the_optimum_at_least_eighty_percent() {
        let mut hits = 0usize;
        let mut misses: Vec<(u64, i64, i64)> = Vec::new();
        for seed in 0..25u64 {
            let g = random_sparse(12, 24, 3000 + seed);
            let optimum = brute_force_min(&g);
            let results = sample_ising_bptns(&g, &params(16, 96, seed), &cfg(4));
            let got = best_energy(&results);
            assert!(got >= optimum, "seed {seed}: energy below the optimum");
            if got == optimum {
                hits += 1;
            } else {
                misses.push((seed, got, optimum));
            }
        }
        assert!(
            hits >= 20,
            "hit the optimum on {hits} of 25 instances; misses: {misses:?}"
        );
    }

    // ---- determinism and the valve ----

    #[test]
    fn the_same_seed_produces_byte_identical_results() {
        let g = random_sparse(24, 60, 808);
        let first = sample_ising_bptns(&g, &params(16, 96, 71), &cfg(4));
        let second = sample_ising_bptns(&g, &params(16, 96, 71), &cfg(4));
        assert_eq!(first, second);
    }

    #[test]
    fn different_seeds_produce_different_results() {
        let g = random_sparse(24, 60, 808);
        let a = sample_ising_bptns(&g, &params(16, 96, 71), &cfg(4));
        let b = sample_ising_bptns(&g, &params(16, 96, 72), &cfg(4));
        assert_ne!(
            a.iter().map(|r| &r.spins).collect::<Vec<_>>(),
            b.iter().map(|r| &r.spins).collect::<Vec<_>>(),
            "the seed must reach the sampler"
        );
    }

    #[test]
    fn seed_zero_still_diversifies_across_reads() {
        let g = random_sparse(24, 70, 909);
        let results = sample_ising_bptns(&g, &params(16, 96, 0), &cfg(4));
        let distinct: std::collections::BTreeSet<&Vec<i8>> =
            results.iter().map(|r| &r.spins).collect();
        assert!(distinct.len() > 1, "every read identical at seed 0");
    }

    #[test]
    fn a_fired_valve_still_returns_complete_polished_results() {
        let g = random_sparse(60, 150, 707);
        let stopped = BptnsConfig {
            chi_max: 4,
            time_budget_ms: 1,
            flop_budget: 1.25e9,
        };
        let results = sample_ising_bptns(&g, &params(8, 256, 61), &stopped);
        assert_results_well_formed(&results, &g, 8);
        for r in &results {
            for var in 0..g.h.len() {
                let mut probe = r.spins.clone();
                probe[var] = -probe[var];
                assert!(
                    score_energy(&probe, &g.h, &g.j, &g.edges) >= r.energy_milli,
                    "flipping {var} lowers the energy, so the polish did not run"
                );
            }
        }
    }

    #[test]
    fn a_generous_valve_does_not_change_the_answer_on_a_small_problem() {
        let g = ferro_chain(12);
        let with_valve = BptnsConfig {
            chi_max: 8,
            time_budget_ms: 60_000,
            flop_budget: 1.25e9,
        };
        assert_eq!(
            sample_ising_bptns(&g, &params(8, 64, 67), &cfg(8)),
            sample_ising_bptns(&g, &params(8, 64, 67), &with_valve),
            "a valve that never fires must not change the output"
        );
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 32,
            failure_persistence: None,
            ..ProptestConfig::default()
        })]

        /// Shape, spin values, and consensus energy over adversarial graphs.
        #[test]
        fn prop_results_are_well_formed_on_adversarial_graphs(
            n in 1usize..=8,
            h_raw in prop::collection::vec(-2.0f64..2.0, 8),
            edge_data in prop::collection::vec((any::<u8>(), any::<u8>(), -2.0f64..2.0), 0..=16),
            num_reads in 1usize..=4,
            num_sweeps in 1usize..=32,
            seed in any::<u64>(),
            chi_max in 1usize..=4,
        ) {
            let h: Vec<f64> = h_raw.into_iter().take(n).collect();
            let mut edges = Vec::with_capacity(edge_data.len());
            let mut j = Vec::with_capacity(edge_data.len());
            for (u, v, c) in edge_data {
                edges.push(((u as usize) % (n + 2), (v as usize) % (n + 2)));
                j.push(c);
            }
            j.pop();
            let g = IsingGraph::new(h, j, edges);
            let c = BptnsConfig {
                chi_max,
                time_budget_ms: 0,
                flop_budget: 1.25e9,
            };
            let results = sample_ising_bptns(&g, &params(num_reads, num_sweeps, seed), &c);
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
            edge_data in prop::collection::vec((any::<u8>(), any::<u8>(), -2.0f64..2.0), 0..=12),
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
            let c = BptnsConfig {
                chi_max: 3,
                time_budget_ms: 0,
                flop_budget: 1.25e9,
            };
            let p = params(4, 24, seed);
            prop_assert_eq!(
                sample_ising_bptns(&g, &p, &c),
                sample_ising_bptns(&g, &p, &c)
            );
        }
    }
}
