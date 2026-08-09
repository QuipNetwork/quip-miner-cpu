//! Neal-style simulated annealing over a geometric beta ladder.
//!
//! Algorithm notes (match dwave-neal / GPU ports):
//! - Beta schedule: geometric from auto hot/cold range (or explicit range).
//! - SA: per-read random restart; sequential Metropolis flips per sweep.
//! - Gibbs: dispatched to `crate::gibbs_parallel`, which resamples whole
//!   colour classes at once. This module holds no Gibbs kernel.
//! - Solution energies are always scored with
//!   [`quip_protocol::scoring::energy_milli`] (positive sign, trunc toward 0).
//! - Parallelism: model-level (one model per core via the streaming pump);
//!   reads run sequentially and cache-local on a single core.
//!
//! Types (`Algorithm`, `SampleParams`, `SamplerResult`, base `IsingGraph`) and
//! the beta schedule come from `quip-miner-core`; this module keeps only the
//! CPU annealing kernels and a private adjacency (`CpuGraph`) built from the
//! base graph.

use quip_miner_core::beta::{default_ising_beta_range, geometric_beta_schedule};
use quip_miner_core::{Algorithm, IsingGraph, SampleParams, SamplerResult};
use quip_protocol::scoring::energy_milli;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

/// Per-variable neighbor lists in CSR layout for O(degree) local fields.
///
/// A flat CSR (`nbr_start` offsets into contiguous `nbr_node`/`nbr_coup`)
/// keeps each variable's neighbors cache-local in the annealing hot loop,
/// unlike a `Vec<Vec<_>>` whose rows are scattered heap allocations.
///
/// Built from the base [`IsingGraph`] with the same defensive posture as
/// `energy_milli`: edges out of range for `h.len()` are skipped, self-loops
/// `(u, u)` are skipped (they would pollute `heff[u]` with `u`'s own spin and
/// break the ΔE formula), and couplings shorter than the edge list are treated
/// as 0.
pub(crate) struct CpuGraph {
    h: Vec<f64>,
    /// CSR row offsets, length `n + 1`.
    nbr_start: Vec<u32>,
    /// Flattened neighbor node ids.
    nbr_node: Vec<u32>,
    /// Flattened couplings, parallel to `nbr_node`.
    nbr_coup: Vec<f64>,
}

impl CpuGraph {
    pub(crate) fn from_base(g: &IsingGraph) -> Self {
        let n = g.h.len();
        let mut deg = vec![0u32; n];
        for &(u, v) in &g.edges {
            // Range guard + self-loop skip: a self-loop double-increments deg[u]
            // and inserts `u` into its own neighbor list, so effective_field(u)
            // would include u's own spin and apply_field_delta would mutate
            // heff[u], breaking the ΔE = -2 s heff premise (own field excludes
            // own spin). energy_milli scores self-loops as a constant (s_u^2=1),
            // so dropping them does not change reported energies.
            if u >= n || v >= n || u == v {
                continue;
            }
            deg[u] += 1;
            deg[v] += 1;
        }
        let mut nbr_start = vec![0u32; n + 1];
        for i in 0..n {
            nbr_start[i + 1] = nbr_start[i] + deg[i];
        }
        let total = nbr_start[n] as usize;
        let mut nbr_node = vec![0u32; total];
        let mut nbr_coup = vec![0.0f64; total];
        let mut cursor: Vec<u32> = nbr_start[..n].to_vec();
        for (k, &(u, v)) in g.edges.iter().enumerate() {
            if u >= n || v >= n || u == v {
                continue;
            }
            let coup = g.j.get(k).copied().unwrap_or(0.0);
            let pu = cursor[u] as usize;
            nbr_node[pu] = v as u32;
            nbr_coup[pu] = coup;
            cursor[u] += 1;
            let pv = cursor[v] as usize;
            nbr_node[pv] = u as u32;
            nbr_coup[pv] = coup;
            cursor[v] += 1;
        }
        Self {
            h: g.h.clone(),
            nbr_start,
            nbr_node,
            nbr_coup,
        }
    }

    /// Linear bias at `var`.
    #[inline]
    pub(crate) fn bias(&self, var: usize) -> f64 {
        self.h[var]
    }

    pub(crate) fn num_nodes(&self) -> usize {
        self.h.len()
    }

    /// `(neighbor_ids, couplings)` slices for `var`.
    #[inline]
    pub(crate) fn neighbors(&self, var: usize) -> (&[u32], &[f64]) {
        let s = self.nbr_start[var] as usize;
        let e = self.nbr_start[var + 1] as usize;
        (&self.nbr_node[s..e], &self.nbr_coup[s..e])
    }
}

/// Geometric beta schedule for one sample request (f64 for CPU precision).
pub(crate) fn build_beta_schedule(graph: &IsingGraph, params: &SampleParams) -> Vec<f64> {
    let sweeps_per = params.sweeps_per_beta.max(1);
    let num_betas = (params.num_sweeps / sweeps_per).max(1);
    let (hot, cold) = params
        .beta_range
        .unwrap_or_else(|| default_ising_beta_range(graph));
    geometric_beta_schedule(hot, cold, num_betas)
}

fn spin_sign(s: i8) -> f64 {
    if s > 0 {
        1.0
    } else {
        -1.0
    }
}

/// Local field `h_i + Σ_j J_ij s_j` (full recompute; used once to seed the
/// incremental `heff` cache).
pub(crate) fn effective_field(var: usize, spins: &[i8], graph: &CpuGraph) -> f64 {
    let mut heff = graph.h[var];
    let (nodes, coups) = graph.neighbors(var);
    for i in 0..nodes.len() {
        heff += coups[i] * spin_sign(spins[nodes[i] as usize]);
    }
    heff
}

/// Propagate a spin change at `var` (sign delta `ds`) into its neighbors' cached
/// effective fields. `var`'s own field is unaffected (it excludes its own spin).
#[inline]
pub(crate) fn apply_field_delta(graph: &CpuGraph, heff: &mut [f64], var: usize, ds: f64) {
    let (nodes, coups) = graph.neighbors(var);
    for i in 0..nodes.len() {
        heff[nodes[i] as usize] += coups[i] * ds;
    }
}

fn random_spins(n: usize, rng: &mut SmallRng) -> Vec<i8> {
    (0..n)
        .map(|_| if rng.gen::<bool>() { 1i8 } else { -1i8 })
        .collect()
}

fn metropolis_accept(delta: f64, beta: f64, rng: &mut SmallRng) -> bool {
    if delta <= 0.0 {
        return true;
    }
    let accept_prob = (-delta * beta).exp();
    rng.gen::<f64>() < accept_prob
}

fn anneal_one_read(
    graph: &CpuGraph,
    beta_schedule: &[f64],
    sweeps_per_beta: usize,
    rng: &mut SmallRng,
) -> Vec<i8> {
    let n = graph.num_nodes();
    let mut spins = random_spins(n, rng);
    if n == 0 {
        return spins;
    }

    // Incremental effective-field cache: `heff[var]` stays equal to
    // `effective_field(var, spins)` across the whole anneal. Seeded once
    // (O(edges)); each accepted flip updates only its neighbors (O(degree)),
    // so a sweep costs O(n + accepts·degree) instead of O(n·degree) every
    // time. ΔE and the Gibbs conditional both read the cache in O(1).
    //
    // The cache and a from-scratch recompute agree to within IEEE rounding, not
    // exactly: `apply_field_delta` accumulates `+= coup * ds` while
    // `effective_field` rebuilds the sum left-to-right from `h[var]`, so the two
    // associate the same terms differently and drift by ~1 ULP per flip. The
    // property test `prop_field_cache_matches_recompute` pins that bound. Only
    // the incremental path runs in production, so the live accept/reject RNG
    // stream stays self-consistent, but results are not bit-for-bit identical to
    // a recompute-every-flip implementation.
    let mut heff: Vec<f64> = (0..n).map(|v| effective_field(v, &spins, graph)).collect();

    for &beta in beta_schedule {
        for _ in 0..sweeps_per_beta {
            for var in 0..n {
                let s = spin_sign(spins[var]);
                let delta = -2.0 * s * heff[var];
                if metropolis_accept(delta, beta, rng) {
                    spins[var] = -spins[var];
                    apply_field_delta(graph, &mut heff, var, -2.0 * s);
                }
            }
        }
    }
    spins
}

/// Descend from a supplied configuration to a local minimum by single-spin
/// flips, in a fixed sweep order.
///
/// Reuses the same incremental effective-field cache as the annealing kernels:
/// seeded once in `O(edges)`, then each flip updates only its neighbors. Stops
/// after a full pass with no flip, or when `max_sweeps` passes have run.
///
/// Deterministic and RNG-free. The tensor-network backend calls it on every
/// sampled configuration, so this is the stage that guarantees each returned
/// configuration is at least locally optimal.
pub(crate) fn polish_from(spins: &mut [i8], graph: &CpuGraph, max_sweeps: usize) {
    let n = graph.num_nodes();
    if n == 0 || spins.len() != n {
        return;
    }
    let mut heff: Vec<f64> = (0..n).map(|v| effective_field(v, spins, graph)).collect();
    for _ in 0..max_sweeps {
        let mut moved = false;
        for var in 0..n {
            let s = spin_sign(spins[var]);
            // Strictly downhill only: accepting a zero delta would let the
            // sweep cycle between equal-energy configurations forever.
            if -2.0 * s * heff[var] < 0.0 {
                spins[var] = -spins[var];
                apply_field_delta(graph, &mut heff, var, -2.0 * s);
                moved = true;
            }
        }
        if !moved {
            break;
        }
    }
}


fn score_spins(spins: &[i8], graph: &IsingGraph) -> SamplerResult {
    let energy = energy_milli(spins, &graph.h, &graph.j, &graph.edges);
    SamplerResult {
        spins: spins.to_vec(),
        energy_milli: energy,
    }
}

/// Run `num_reads` independent anneals sequentially on one core.
///
/// Reads stay on a single core so the model's arrays (h/j/spins/edges) stay hot
/// in that core's cache — fanning reads across cores bounced those cache lines
/// and measured slower. Model-level parallelism (one model per core) lives in
/// the streaming pump (`CpuSampler::sample_stream`).
///
/// # Examples
///
/// ```
/// use quip_miner_cpu::{sample_ising, Algorithm, IsingGraph, SampleParams};
///
/// let graph = IsingGraph::new(vec![0.0, 0.0], vec![-1.0], vec![(0, 1)]);
/// let params = SampleParams {
///     num_reads: 4,
///     num_sweeps: 16,
///     seed: 1,
///     ..Default::default()
/// };
/// let results = sample_ising(&graph, &params, Algorithm::Sa);
/// assert_eq!(results.len(), params.num_reads);
/// ```
pub fn sample_ising(
    graph: &IsingGraph,
    params: &SampleParams,
    algorithm: Algorithm,
) -> Vec<SamplerResult> {
    if algorithm == Algorithm::Gibbs {
        // Gibbs runs the chromatic kernel at its default worker count. There is
        // no sequential Gibbs: the colour classes are the algorithm, not an
        // optimisation layered over a single-site scan.
        return crate::gibbs_parallel::sample_gibbs_parallel(
            graph,
            params,
            crate::gibbs_parallel::DEFAULT_GIBBS_WORKERS,
        );
    }
    let num_reads = params.num_reads.max(1);
    let cpu = CpuGraph::from_base(graph);
    let beta_schedule = build_beta_schedule(graph, params);
    let sweeps_per = params.sweeps_per_beta.max(1);
    let base_seed = params.seed;

    (0..num_reads)
        .map(|read_idx| {
            // Distinct stream per read; seed 0 still diversifies via read index.
            let seed = base_seed
                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                .wrapping_add(read_idx as u64)
                .wrapping_add(1);
            let mut rng = SmallRng::seed_from_u64(seed);
            let spins = anneal_one_read(&cpu, &beta_schedule, sweeps_per, &mut rng);
            score_spins(&spins, graph)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn ferro2() -> IsingGraph {
        // Prefer aligned spins: J=-1 means E = J s0 s1 is lower when s0==s1.
        IsingGraph::new(vec![0.0, 0.0], vec![-1.0], vec![(0, 1)])
    }

    #[test]
    fn geometric_schedule_endpoints() {
        let s = geometric_beta_schedule(0.1, 10.0, 5);
        assert_eq!(s.len(), 5);
        assert!((s[0] - 0.1).abs() < 1e-12);
        assert!((s[4] - 10.0).abs() < 1e-12);
        // Strictly increasing for hot < cold.
        for w in s.windows(2) {
            assert!(w[0] < w[1]);
        }
    }

    #[test]
    fn energy_of_known_spins_matches_scoring() {
        let g = ferro2();
        let spins = vec![1i8, 1i8];
        let r = score_spins(&spins, &g);
        assert_eq!(r.energy_milli, energy_milli(&spins, &g.h, &g.j, &g.edges));
        // E = -1 * 1 * 1 = -1 → -1000 milli
        assert_eq!(r.energy_milli, -1000);
    }

    #[test]
    fn sa_finds_ground_state_on_ferro() {
        let g = ferro2();
        let params = SampleParams {
            num_reads: 8,
            num_sweeps: 128,
            seed: 42,
            ..Default::default()
        };
        let results = sample_ising(&g, &params, Algorithm::Sa);
        assert_eq!(results.len(), 8);
        // At least one read should land on a ground state (energy -1000).
        assert!(
            results.iter().any(|r| r.energy_milli == -1000),
            "SA failed to find ferro ground state: {:?}",
            results.iter().map(|r| r.energy_milli).collect::<Vec<_>>()
        );
        // Every reported energy must equal consensus scoring of the spins.
        for r in &results {
            assert_eq!(r.energy_milli, energy_milli(&r.spins, &g.h, &g.j, &g.edges));
        }
    }

    #[test]
    fn gibbs_reports_consensus_energies() {
        let g = ferro2();
        let params = SampleParams {
            num_reads: 4,
            num_sweeps: 64,
            seed: 7,
            ..Default::default()
        };
        let results = sample_ising(&g, &params, Algorithm::Gibbs);
        for r in &results {
            assert_eq!(r.energy_milli, energy_milli(&r.spins, &g.h, &g.j, &g.edges));
            assert!(r.spins.iter().all(|&s| s == 1 || s == -1));
        }
    }

    #[test]
    fn empty_graph_is_safe() {
        let g = IsingGraph::new(vec![], vec![], vec![]);
        let params = SampleParams {
            num_reads: 2,
            num_sweeps: 4,
            seed: 1,
            ..Default::default()
        };
        let results = sample_ising(&g, &params, Algorithm::Sa);
        assert_eq!(results.len(), 2);
        assert!(results
            .iter()
            .all(|r| r.spins.is_empty() && r.energy_milli == 0));
    }

    /// Structured 4-node graph with mixed h and asymmetric J (not the 2-node ferro).
    /// Couplings/biases deliberately include non-dyadic values so the cache
    /// path is exercised under real IEEE association (see `heff_close`).
    fn mixed4() -> IsingGraph {
        IsingGraph::new(
            vec![0.5, -0.3, 0.1, 0.0],
            vec![1.0, -0.5, 0.75, -1.25, 0.25],
            vec![(0, 1), (1, 2), (0, 2), (2, 3), (0, 3)],
        )
    }

    /// Incremental `heff += coup * ds` and left-to-right `effective_field`
    /// recompute are algebraically identical but not bit-identical in IEEE
    /// f64 (different add association). Exact `==` fails on the first
    /// non-dyadic fixture (observed 1-ULP and ~1e-14 relative drifts after a
    /// few flips; multi-edge graphs drift further as O(edges·flips)·ε).
    ///
    /// Bound: relative 1e-12 of the larger magnitude, with a 1e-12 absolute
    /// floor near zero. That is ~6 orders tighter than `energy_milli`'s 1e-3
    /// quantum, yet fails if a whole O(1) coupling term is dropped or a
    /// neighbor update is missed.
    fn heff_close(cached: f64, recomputed: f64) -> bool {
        if cached == recomputed {
            return true;
        }
        let diff = (cached - recomputed).abs();
        let scale = cached.abs().max(recomputed.abs()).max(1.0);
        diff <= 1e-12 || diff <= scale * 1e-12
    }

    fn assert_heff_matches(heff: &[f64], spins: &[i8], graph: &CpuGraph) {
        assert_eq!(heff.len(), graph.num_nodes());
        for (i, &cached) in heff.iter().enumerate() {
            let want = effective_field(i, spins, graph);
            assert!(
                heff_close(cached, want),
                "heff[{i}] diverged from effective_field: cached={cached} recompute={want} (spins={spins:?})",
            );
        }
    }

    #[test]
    fn field_cache_tracks_effective_field_across_flips() {
        let g = mixed4();
        let cpu = CpuGraph::from_base(&g);
        let n = cpu.num_nodes();
        let mut spins = vec![1i8, -1, 1, -1];
        let mut heff: Vec<f64> = (0..n).map(|v| effective_field(v, &spins, &cpu)).collect();
        assert_heff_matches(&heff, &spins, &cpu);

        // Fixed flip sequence covering every node more than once.
        let flips = [0usize, 2, 1, 3, 0, 1, 2, 3, 1, 0, 3, 2];
        for &var in &flips {
            let s = spin_sign(spins[var]);
            spins[var] = -spins[var];
            apply_field_delta(&cpu, &mut heff, var, -2.0 * s);
            assert_heff_matches(&heff, &spins, &cpu);
        }
    }

    /// Self-loop hypothesis: without `u == v` skip, from_base double-counts the
    /// loop into deg[u] and inserts `u` twice into its own CSR row. Then
    /// effective_field(u) includes u's own spin and apply_field_delta mutates
    /// heff[u], breaking the ΔE = -2 s heff premise. energy_milli treats the
    /// loop as a constant (s^2 = 1), so the Metropolis delta is wrong while
    /// cache-vs-recompute can still agree (both wrong the same way).
    #[test]
    fn self_loops_are_skipped_and_do_not_corrupt_heff() {
        // Pure self-loop on a single node: field must stay 0 for any spin.
        let pure = IsingGraph::new(vec![0.0], vec![3.0], vec![(0, 0)]);
        let pure_cpu = CpuGraph::from_base(&pure);
        assert!(
            pure_cpu.neighbors(0).0.is_empty(),
            "self-loop must not appear in CSR neighbor list"
        );
        assert_eq!(effective_field(0, &[1i8], &pure_cpu), 0.0);
        assert_eq!(effective_field(0, &[-1i8], &pure_cpu), 0.0);

        // Self-loop mixed with a real edge: only the real neighbor remains.
        let g = IsingGraph::new(vec![0.5, -0.25], vec![2.0, -1.0], vec![(0, 0), (0, 1)]);
        let cpu = CpuGraph::from_base(&g);
        let (n0, c0) = cpu.neighbors(0);
        assert_eq!(n0, &[1]);
        assert_eq!(c0, &[-1.0]);
        let (n1, c1) = cpu.neighbors(1);
        assert_eq!(n1, &[0]);
        assert_eq!(c1, &[-1.0]);

        let mut spins = vec![1i8, -1];
        let mut heff: Vec<f64> = (0..2).map(|v| effective_field(v, &spins, &cpu)).collect();
        // heff[0] must not include own-spin contribution from the self-loop.
        // Coupling on (0,1) is -1.0, so field is h[0] + (-1)*s[1].
        assert_eq!(heff[0], 0.5 - spin_sign(spins[1]));
        assert_heff_matches(&heff, &spins, &cpu);

        for &var in &[0usize, 1, 0, 1, 0] {
            let s = spin_sign(spins[var]);
            spins[var] = -spins[var];
            apply_field_delta(&cpu, &mut heff, var, -2.0 * s);
            // Own field must stay free of self-mutation from the flipped var.
            assert_heff_matches(&heff, &spins, &cpu);
        }
    }

    /// Empirically reconstruct the pre-fix CSR behavior for a self-loop and
    /// show that ΔE and own-field update are wrong while cache-vs-recompute
    /// still matches (silent mining-quality bug).
    #[test]
    fn self_loop_legacy_csr_corrupts_delta_not_cache_equality() {
        // Manual CSR as from_base would build for n=1, edge (0,0), J=1, h=0:
        // deg[0] += 2 → nbr row [0, 0] with coups [1, 1].
        let legacy = CpuGraph {
            h: vec![0.0],
            nbr_start: vec![0, 2],
            nbr_node: vec![0, 0],
            nbr_coup: vec![1.0, 1.0],
        };
        let spins_plus = [1i8];
        let heff0 = effective_field(0, &spins_plus, &legacy);
        // True field excluding own spin is 0; legacy includes 2 * J * s = 2.
        assert_eq!(heff0, 2.0);
        // Metropolis delta used in anneal_one_read: -2 * s * heff.
        let delta_legacy = -2.0 * spin_sign(spins_plus[0]) * heff0;
        assert_eq!(delta_legacy, -4.0);
        // True ΔE for a pure self-loop: energy is J * s * s = J (constant), so 0.
        let delta_true = 0.0;
        assert_ne!(
            delta_legacy, delta_true,
            "legacy self-loop CSR must produce a wrong Metropolis delta"
        );

        // Cache equality can still hold after apply_field_delta (both wrong).
        let mut heff = vec![heff0];
        let mut spins = spins_plus;
        let s = spin_sign(spins[0]);
        spins[0] = -spins[0];
        apply_field_delta(&legacy, &mut heff, 0, -2.0 * s);
        assert_eq!(heff[0], effective_field(0, &spins, &legacy));
    }

    proptest! {
        // No file persistence: keeps the worktree free of proptest-regressions/.
        #![proptest_config(ProptestConfig {
            cases: 128,
            failure_persistence: None,
            ..ProptestConfig::default()
        })]
        #[test]
        fn prop_field_cache_matches_recompute(
            n in 1usize..=6,
            h in prop::collection::vec(-2.0f64..2.0, 1..=6),
            edge_data in prop::collection::vec(
                (any::<u8>(), any::<u8>(), -2.0f64..2.0),
                0..=16,
            ),
            spin_bits in prop::collection::vec(any::<bool>(), 1..=6),
            flip_vars in prop::collection::vec(any::<u8>(), 0..=40),
        ) {
            let n = n.min(h.len()).min(spin_bits.len()).max(1);
            let h: Vec<f64> = h.into_iter().take(n).collect();
            let mut edges = Vec::with_capacity(edge_data.len());
            let mut j = Vec::with_capacity(edge_data.len());
            for (u, v, c) in edge_data {
                edges.push(((u as usize) % n, (v as usize) % n));
                j.push(c);
            }
            let mut spins: Vec<i8> = spin_bits
                .into_iter()
                .take(n)
                .map(|b| if b { 1i8 } else { -1i8 })
                .collect();
            let g = IsingGraph::new(h, j, edges);
            let cpu = CpuGraph::from_base(&g);
            let mut heff: Vec<f64> = (0..n)
                .map(|v| effective_field(v, &spins, &cpu))
                .collect();

            for (i, &cached) in heff.iter().enumerate() {
                prop_assert!(heff_close(cached, effective_field(i, &spins, &cpu)));
            }

            for fv in flip_vars {
                let var = (fv as usize) % n;
                let s = spin_sign(spins[var]);
                spins[var] = -spins[var];
                apply_field_delta(&cpu, &mut heff, var, -2.0 * s);
                for (i, &cached) in heff.iter().enumerate() {
                    // Exact equality does NOT hold: see heff_close docs and the
                    // TEST-4 report (IEEE association of += vs recompute).
                    let want = effective_field(i, &spins, &cpu);
                    prop_assert!(
                        heff_close(cached, want),
                        "heff[{}] after flip {}: cached={} recompute={}",
                        i,
                        var,
                        cached,
                        want,
                    );
                }
            }
        }
    }
    // ---- polish_from ----

    #[test]
    fn polish_descends_a_ferromagnetic_chain_to_a_ground_state() {
        // Ten-site ferromagnetic chain: every ground state is uniform, and the
        // ground energy is -9.
        let n = 10;
        let edges: Vec<(usize, usize)> = (0..n - 1).map(|i| (i, i + 1)).collect();
        let g = IsingGraph::new(vec![0.0; n], vec![-1.0; n - 1], edges);
        let cpu = CpuGraph::from_base(&g);
        let mut spins = vec![1i8, -1, 1, -1, 1, -1, 1, -1, 1, -1];
        polish_from(&mut spins, &cpu, 64);
        assert_eq!(
            energy_milli(&spins, &g.h, &g.j, &g.edges),
            -9000,
            "polish left the chain at {spins:?}"
        );
    }

    #[test]
    fn polish_of_a_fields_only_problem_lands_on_the_exact_optimum() {
        let h = vec![0.7, -1.2, 0.3, -0.5, 2.0];
        let g = IsingGraph::new(h.clone(), vec![], vec![]);
        let cpu = CpuGraph::from_base(&g);
        let mut spins = vec![1i8; 5];
        polish_from(&mut spins, &cpu, 8);
        let want: Vec<i8> = h.iter().map(|&x| if x > 0.0 { -1i8 } else { 1i8 }).collect();
        assert_eq!(spins, want);
    }

    #[test]
    fn polish_never_raises_the_energy_and_stops_at_a_local_minimum() {
        let g = mixed4();
        let cpu = CpuGraph::from_base(&g);
        for start in 0..16u32 {
            let mut spins: Vec<i8> = (0..4)
                .map(|i| if (start >> i) & 1 == 0 { 1i8 } else { -1 })
                .collect();
            let before = energy_milli(&spins, &g.h, &g.j, &g.edges);
            polish_from(&mut spins, &cpu, 32);
            let after = energy_milli(&spins, &g.h, &g.j, &g.edges);
            assert!(after <= before, "start {start}: {before} -> {after}");
            // Local minimum: no single flip lowers the energy.
            for var in 0..4 {
                let mut probe = spins.clone();
                probe[var] = -probe[var];
                assert!(
                    energy_milli(&probe, &g.h, &g.j, &g.edges) >= after,
                    "start {start}: flipping {var} lowers the energy below {after}"
                );
            }
        }
    }

    #[test]
    fn polish_is_deterministic_and_respects_its_sweep_budget() {
        let g = mixed4();
        let cpu = CpuGraph::from_base(&g);
        let run = |budget: usize| {
            let mut spins = vec![1i8, 1, 1, 1];
            polish_from(&mut spins, &cpu, budget);
            spins
        };
        assert_eq!(run(8), run(8));
        // A zero budget must leave the configuration untouched.
        let mut untouched = vec![1i8, 1, 1, 1];
        polish_from(&mut untouched, &cpu, 0);
        assert_eq!(untouched, vec![1i8, 1, 1, 1]);
    }

    #[test]
    fn polish_is_safe_on_an_empty_graph_and_a_mismatched_length() {
        let empty = IsingGraph::new(vec![], vec![], vec![]);
        let cpu = CpuGraph::from_base(&empty);
        let mut none: Vec<i8> = Vec::new();
        polish_from(&mut none, &cpu, 8);
        assert!(none.is_empty());

        let g = mixed4();
        let cpu = CpuGraph::from_base(&g);
        let mut short = vec![1i8, -1];
        polish_from(&mut short, &cpu, 8);
        assert_eq!(short, vec![1i8, -1], "a length mismatch must be a no-op");
    }

}
