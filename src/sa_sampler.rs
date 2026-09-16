//! Tabulated-threshold and multi-spin annealing backends.
//!
//! Two samplers from Isakov, Zintchenko, Rønnow and Troyer (Comput. Phys.
//! Commun. 192, 265 (2015), arXiv:1401.1084), kept apart from
//! [`crate::CpuSampler`] because both change the random stream and are
//! therefore different samplers, not faster spellings of `cpu-sa`:
//!
//! - [`SaVariant::Tabulated`] moves the Metropolis test into its geometric
//!   form and draws the thresholds from a table shared by every read.
//! - [`SaVariant::MultiSpin`] packs 64 reads into each machine word, redraws
//!   one shared threshold row per temperature, and sweeps in color order.
//!
//! Both need discrete couplings, and a problem that does not qualify falls
//! back one step at a time, so either binary is safe to point at any job — it
//! simply stops being fast. A problem the integer representation refuses runs
//! on the ordinary `cpu-sa` kernel. A problem it accepts, but whose field or
//! degree is too large to pack into a machine word, runs on the tabulated
//! kernel of [`SaVariant::Tabulated`], which is still faster than `cpu-sa`.

use quip_protocol::scoring::energy_milli;
use quip_solver_core::adapt::AdaptBounds;
use quip_solver_core::{
    Algorithm, BackendIdentity, CancelToken, IsingGraph, SampleError, SampleParams, Sampler,
    SamplerResult, StreamJob, StreamResult,
};
use rand::rngs::SmallRng;
use rand::SeedableRng;
use std::sync::atomic::AtomicUsize;
use std::sync::{Arc, Mutex};

use crate::coloring::Coloring;
use crate::sa_int::{anneal_from, sweep_offsets, threshold_draws, IntGraph};
use crate::sa_msc::{anneal_words, bond_counts, MscState, LANES};
use crate::sampler_core::{
    build_beta_schedule, sample_ising_cancellable, CpuGraph, SampleCancelled,
};
use crate::{run_stream_pump, CPU_ADAPT, DEFAULT_MAX_EDGES, DEFAULT_MAX_NODES};

/// Backend identity for `quip-cpu-fsa` (tabulated-threshold annealing).
///
/// Experimental track; the binary builds only behind the `experimental` cargo
/// feature. The adapt envelope is `CPU_ADAPT` unchanged, so reads and sweeps
/// stay directly comparable with `quip-cpu-sa`, which is the whole point of
/// this backend: it isolates the acceptance-table trick from the multi-spin
/// packing.
pub const CPU_FSA_IDENTITY: BackendIdentity = BackendIdentity {
    backend: "cpu",
    algorithm: "fsa",
    max_nodes: DEFAULT_MAX_NODES,
    max_edges: DEFAULT_MAX_EDGES,
    features: &[],
    adapt: CPU_ADAPT,
};

/// Backend identity for `quip-cpu-msa` (multi-spin coded annealing).
///
/// Experimental track. The adapt envelope matches CUDA `CUDA_MSA_ADAPT`:
/// 128 reads and 7392..=29568 sweeps. A multiple of 64 is this kernel's
/// natural read count — 64 reads cost what one read costs — so the pinned
/// read count is two replica words rather than the scalar [`CPU_ADAPT`] range.
pub const CPU_MSA_IDENTITY: BackendIdentity = BackendIdentity {
    backend: "cpu",
    algorithm: "msa",
    max_nodes: DEFAULT_MAX_NODES,
    max_edges: DEFAULT_MAX_EDGES,
    features: &[],
    adapt: CPU_MSA_ADAPT,
};

/// Adapt envelope for the multi-spin backend, matching CUDA `CUDA_MSA_ADAPT`.
///
/// Named separately from [`CPU_ADAPT`] so FSA and SA keep the scalar budget
/// while MSA advertises the same mining envelope as `quip-cuda-msa`.
pub(crate) const CPU_MSA_ADAPT: AdaptBounds = AdaptBounds {
    min_sweeps: 7392,
    max_sweeps: 29568,
    min_reads: 128,
    max_reads: 128,
    reads_solution_min_factor: 0,
    reads_solution_max_factor: 0,
    reads_solution_floor_factor: 0,
};

/// Which annealing kernel a binary drives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SaVariant {
    /// Scalar kernel with tabulated acceptance thresholds.
    Tabulated,
    /// Multi-spin coded kernel, 64 reads per machine word.
    MultiSpin,
}

/// One immutable Welsh-Powell colouring, keyed by topology not by weights.
///
/// Fields and coupling values are omitted: `Coloring::new` walks every
/// topology edge, including zero couplings, through `CpuGraph::from_base`.
#[derive(Debug)]
struct CachedColoring {
    nodes: usize,
    edges: Vec<(usize, usize)>,
    colors: Coloring,
}

/// Annealing sampler for the tabulated-threshold and multi-spin kernels.
#[derive(Debug, Clone)]
pub struct SaSampler {
    variant: SaVariant,
    /// Operator core budget from `Configure.backend_toml`. 0 means the host.
    ///
    /// `apply_config` takes `&self`, so the budget needs interior mutability.
    /// Behind an `Arc` rather than a bare atomic so that a clone shares the
    /// stored budget instead of silently reverting to host parallelism, which
    /// is how `CpuSampler` holds the same field. Taking this handle costs
    /// `SaSampler` its `Copy`, `PartialEq` and `Eq` derives.
    num_cpus: Arc<AtomicUsize>,
    /// Single-entry topology cache shared by clones.
    ///
    /// The mutex is held only to clone or replace the `Arc`. Sampling reads
    /// the immutable entry after the guard is dropped.
    coloring: Arc<Mutex<Option<Arc<CachedColoring>>>>,
}

impl SaSampler {
    /// Create a sampler for `variant`.
    ///
    /// # Examples
    ///
    /// ```
    /// use quip_miner_cpu::{IsingGraph, SaSampler, SaVariant, SampleParams};
    /// use quip_solver_core::{SampleError, Sampler};
    ///
    /// # fn main() -> Result<(), SampleError> {
    /// let sampler = SaSampler::new(SaVariant::MultiSpin);
    /// let graph = IsingGraph::new(vec![0.0, 0.0], vec![-1.0], vec![(0, 1)]);
    /// let params = SampleParams {
    ///     num_reads: 3,
    ///     num_sweeps: 64,
    ///     seed: 1,
    ///     ..Default::default()
    /// };
    /// let results = sampler.sample(&graph, &params)?;
    /// assert_eq!(results.len(), 3);
    /// assert!(results.iter().all(|r| r.spins.iter().all(|&s| s == 1 || s == -1)));
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn new(variant: SaVariant) -> Self {
        Self {
            variant,
            num_cpus: Arc::new(AtomicUsize::new(0)),
            coloring: Arc::new(Mutex::new(None)),
        }
    }
}

/// Recover a poisoned cache lock.
///
/// The guard is held only for an `Arc` clone or a single `Option` store, so a
/// panic while it is held cannot leave a half-written entry. `into_inner`
/// keeps later samples running on the still-valid slot.
fn lock_cached_coloring(
    cache: &Mutex<Option<Arc<CachedColoring>>>,
) -> std::sync::MutexGuard<'_, Option<Arc<CachedColoring>>> {
    cache
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Clone a matching cached colouring, or build one outside the lock.
fn cached_coloring(
    cache: &Mutex<Option<Arc<CachedColoring>>>,
    graph: &IsingGraph,
) -> Arc<CachedColoring> {
    let nodes = graph.h.len();
    let cached = {
        let guard = lock_cached_coloring(cache);
        guard.clone()
    };
    if let Some(entry) = cached {
        if entry.nodes == nodes && entry.edges == graph.edges {
            return entry;
        }
    }

    let entry = Arc::new(CachedColoring {
        nodes,
        edges: graph.edges.clone(),
        colors: Coloring::new(&CpuGraph::from_base(graph)),
    });
    let previous = {
        let mut guard = lock_cached_coloring(cache);
        guard.replace(Arc::clone(&entry))
    };
    drop(previous);
    entry
}

/// Run one job on the selected kernel, or on `cpu-sa` when the problem does
/// not qualify.
///
/// Direct callers skip the sampler cache. `SaSampler::sample` and
/// `sample_stream` install a cache on the inner path.
#[cfg(test)]
pub(crate) fn sample_sa_variant(
    graph: &IsingGraph,
    params: &SampleParams,
    variant: SaVariant,
    cancel: Option<(&CancelToken, Option<u64>)>,
) -> Result<Vec<SamplerResult>, SampleCancelled> {
    sample_sa_variant_with_cache(graph, params, variant, cancel, None)
}

fn sample_sa_variant_with_cache(
    graph: &IsingGraph,
    params: &SampleParams,
    variant: SaVariant,
    cancel: Option<(&CancelToken, Option<u64>)>,
    cache: Option<&Mutex<Option<Arc<CachedColoring>>>>,
) -> Result<Vec<SamplerResult>, SampleCancelled> {
    let Some(int) = IntGraph::from_base(graph) else {
        return sample_ising_cancellable(graph, params, Algorithm::Sa, cancel);
    };
    let num_reads = params.num_reads.max(1);
    let betas = build_beta_schedule(graph, params);
    let sweeps_per = params.sweeps_per_beta.max(1);

    let counts = match variant {
        SaVariant::MultiSpin => bond_counts(&int),
        SaVariant::Tabulated => None,
    };

    let mut results = Vec::with_capacity(num_reads);
    match counts {
        Some(counts) => {
            let fresh;
            let hit;
            let colors = match cache {
                Some(cache) => {
                    hit = cached_coloring(cache, graph);
                    &hit.colors
                }
                None => {
                    fresh = Coloring::new(&CpuGraph::from_base(graph));
                    &fresh
                }
            };
            let mut states = Vec::with_capacity(num_reads.div_ceil(LANES));
            for read in (0..num_reads).step_by(LANES) {
                if let Some((guard, watermark)) = cancel {
                    if guard.is_cancelled(watermark) {
                        return Err(SampleCancelled);
                    }
                }
                let mut rng = read_rng(params.seed, read);
                states.push(MscState::random(int.num_nodes(), &mut rng));
            }
            anneal_words(&int, &counts, colors, &betas, params, &mut states, cancel)?;
            for (word, state) in states.iter().enumerate() {
                for lane in 0..LANES.min(num_reads - word * LANES) {
                    results.push(score(&state.lane(lane), graph));
                }
            }
        }
        None => {
            // The scalar tabulated path retains its shared full-ladder table
            // and falls back to ordinary SA if that table exceeds its cap.
            let mut table_rng = SmallRng::seed_from_u64(params.seed ^ 0x5341_5F54_424C_4531);
            let Some(draws) = threshold_draws(&betas, int.max_field(), &mut table_rng) else {
                return sample_ising_cancellable(graph, params, Algorithm::Sa, cancel);
            };
            for read in 0..num_reads {
                if let Some((guard, watermark)) = cancel {
                    if guard.is_cancelled(watermark) {
                        return Err(SampleCancelled);
                    }
                }
                let mut rng = read_rng(params.seed, read);
                let mut spins = crate::sampler_core::random_spins(int.num_nodes(), &mut rng);
                let offsets = sweep_offsets(betas.len(), sweeps_per, &mut rng);
                anneal_from(&int, &draws, sweeps_per, &mut spins, &offsets, cancel)?;
                results.push(score(&spins, graph));
            }
        }
    }
    Ok(results)
}

/// Per-read random stream, seeded the same way `cpu-sa` seeds its reads.
fn read_rng(base: u64, read_idx: usize) -> SmallRng {
    SmallRng::seed_from_u64(
        base.wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add(read_idx as u64)
            .wrapping_add(1),
    )
}

fn score(spins: &[i8], graph: &IsingGraph) -> SamplerResult {
    SamplerResult {
        spins: spins.to_vec(),
        energy_milli: energy_milli(spins, &graph.h, &graph.j, &graph.edges),
    }
}

impl Sampler for SaSampler {
    fn sample(
        &self,
        graph: &IsingGraph,
        params: &SampleParams,
    ) -> Result<Vec<SamplerResult>, SampleError> {
        Ok(
            sample_sa_variant_with_cache(graph, params, self.variant, None, Some(&self.coloring))
                .unwrap_or_default(),
        )
    }

    /// One model per core, the same shape as `CpuSampler::stream_width`.
    ///
    /// `num_cpus` from `Configure.backend_toml` replaces host parallelism here,
    /// exactly as it does for `cpu-sa`. Without this the two binaries would run
    /// different concurrencies under the same operator configuration.
    fn stream_width(&self) -> usize {
        crate::core_budget(&self.num_cpus)
    }

    fn declared_stream_width() -> u32 {
        u32::try_from(crate::host_parallelism()).unwrap_or(u32::MAX)
    }

    fn apply_config(&self, backend_toml: &str) {
        if let Err(e) = crate::apply_num_cpus(&self.num_cpus, backend_toml) {
            tracing::error!(error = %e, "cpu num_cpus rejected");
            crate::reject_num_cpus(e);
        }
    }

    fn sample_stream(
        &self,
        jobs: tokio::sync::mpsc::Receiver<StreamJob>,
        out: tokio::sync::mpsc::Sender<StreamResult>,
        cancel: CancelToken,
    ) {
        let variant = self.variant;
        let coloring = Arc::clone(&self.coloring);
        run_stream_pump(
            || self.stream_width(),
            move |g, p, token, watermark| {
                sample_sa_variant_with_cache(
                    g,
                    p,
                    variant,
                    Some((token, watermark)),
                    Some(&coloring),
                )
                .map_err(|_| crate::StreamKernelError::Cancelled)
            },
            jobs,
            out,
            cancel,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    fn zephyr_like(n: usize, seed: u64, fields: &[f64]) -> IsingGraph {
        let mut rng = SmallRng::seed_from_u64(seed);
        let h: Vec<f64> = (0..n)
            .map(|_| fields[rng.gen_range(0..fields.len())])
            .collect();
        let mut edges = Vec::new();
        let mut j = Vec::new();
        for u in 0..n {
            for v in (u + 1)..n {
                if rng.gen::<f64>() < 0.3 {
                    edges.push((u, v));
                    j.push(if rng.gen::<bool>() { 1.0 } else { -1.0 });
                }
            }
        }
        IsingGraph::new(h, j, edges)
    }

    fn params(num_reads: usize, num_sweeps: usize, seed: u64) -> SampleParams {
        SampleParams {
            num_reads,
            num_sweeps,
            seed,
            ..Default::default()
        }
    }

    fn result_spins(v: &[SamplerResult]) -> Vec<Vec<i8>> {
        v.iter().map(|r| r.spins.clone()).collect()
    }

    fn cached_entry(s: &SaSampler) -> Option<Arc<CachedColoring>> {
        s.coloring.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    #[test]
    fn the_core_budget_tracks_cpu_sa_rather_than_the_bare_host() {
        // An operator setting num_cpus must get the same concurrency from
        // every CPU backend. Without apply_config these binaries would run one
        // model per hardware thread while cpu-sa ran the configured number,
        // oversubscribing the host against an explicit instruction.
        for variant in [SaVariant::Tabulated, SaVariant::MultiSpin] {
            // Both fresh each pass: apply_config stores through a shared
            // atomic, so a reused instance would carry the budget forward.
            let sa = crate::CpuSampler::new(Algorithm::Sa);
            let s = SaSampler::new(variant);
            assert_eq!(s.stream_width(), sa.stream_width(), "default budget");

            // Clamped to the host, so a two-core CI box expects 2, not 3.
            let want = 3.min(crate::host_parallelism());
            s.apply_config("num_cpus = 3");
            sa.apply_config("num_cpus = 3");
            assert_eq!(s.stream_width(), want, "configured budget");
            assert_eq!(s.stream_width(), sa.stream_width(), "configured budget");

            // The budget lives behind a shared handle, so a clone taken after
            // the handshake still sees what the handshake stored.
            assert_eq!(s.clone().stream_width(), want);
        }
    }

    #[test]
    fn msa_matches_scalar_color_sweeps_across_replica_words() {
        // The path's color order is [1, 3], then [0, 2, 4]. Updating in
        // node order, or shifting the second word differently, changes spins.
        let graph = IsingGraph::new(
            vec![1.0, 0.0, -1.0, 0.0, 1.0],
            vec![1.0, -1.0, 1.0, -1.0],
            vec![(0, 1), (1, 2), (2, 3), (3, 4)],
        );
        let params = SampleParams {
            num_reads: 70,
            num_sweeps: 9,
            sweeps_per_beta: 3,
            beta_range: Some((0.05, 0.15)),
            seed: 19,
        };
        let int = IntGraph::from_base(&graph).expect("unit graph");
        let mut rng = SmallRng::seed_from_u64(19 ^ 0x5341_5F54_424C_4531);
        let draws = threshold_draws(
            &build_beta_schedule(&graph, &params),
            int.max_field(),
            &mut rng,
        )
        .expect("three rows");
        // CUDA splitmix64(seed ^ (beta_idx << 20) ^ sweep) & 8191.
        let offsets = [[6724, 4146, 4963], [4285, 4953, 6317], [3132, 4413, 1117]];
        let actual = sample_sa_variant(&graph, &params, SaVariant::MultiSpin, None)
            .expect("sampling succeeds");
        for base in [0, 64] {
            let mut rng = read_rng(params.seed, base);
            let state = MscState::random(5, &mut rng);
            for lane in 0..64.min(params.num_reads - base) {
                let mut spins = state.lane(lane);
                for (row, shifts) in draws.chunks_exact(8192).zip(offsets) {
                    for off in shifts {
                        for var in [1, 3, 0, 2, 4] {
                            let mut field = graph.h[var];
                            for (&(u, v), &j) in graph.edges.iter().zip(&graph.j) {
                                if u == var {
                                    field += j * f64::from(spins[v]);
                                } else if v == var {
                                    field += j * f64::from(spins[u]);
                                }
                            }
                            let delta = -2.0 * f64::from(spins[var]) * field;
                            if delta <= 2.0 * f64::from(row[(var + off) & 8191]) {
                                spins[var] = -spins[var];
                            }
                        }
                    }
                }
                assert_eq!(actual[base + lane].spins, spins, "read {}", base + lane);
            }
        }
    }

    #[test]
    fn msa_sweep_budget_uses_complete_beta_rungs() {
        // With no bonds or fields, every attempted flip is accepted. The
        // final signs expose the sweep count without depending on randomness.
        let graph = IsingGraph::new(vec![0.0; 5], vec![], vec![]);
        let mut rng = read_rng(17, 0);
        let initial = MscState::random(5, &mut rng).lane(0);
        for (sweeps, per_beta, flips) in [(0, 0, 1), (0, 4, 4), (2, 4, 4), (7, 3, 6), (9, 3, 9)] {
            let params = SampleParams {
                num_reads: 1,
                num_sweeps: sweeps,
                sweeps_per_beta: per_beta,
                seed: 17,
                ..Default::default()
            };
            let actual = sample_sa_variant(&graph, &params, SaVariant::MultiSpin, None)
                .expect("sampling succeeds");
            let expected: Vec<i8> = initial
                .iter()
                .map(|&s| if flips % 2 == 0 { s } else { -s })
                .collect();
            assert_eq!(
                actual[0].spins, expected,
                "sweeps={sweeps}, per_beta={per_beta}"
            );
        }
    }

    #[test]
    fn identities_advertise_their_algorithms() {
        assert_eq!(CPU_FSA_IDENTITY.backend, "cpu");
        assert_eq!(CPU_FSA_IDENTITY.algorithm, "fsa");
        assert_eq!(CPU_MSA_IDENTITY.backend, "cpu");
        assert_eq!(CPU_MSA_IDENTITY.algorithm, "msa");
        assert_eq!(CPU_FSA_IDENTITY.adapt.min_reads, CPU_ADAPT.min_reads);
        assert_eq!(CPU_FSA_IDENTITY.adapt.min_sweeps, 64);
        assert_eq!(CPU_FSA_IDENTITY.adapt.max_sweeps, 1024);
        assert_eq!(CPU_MSA_IDENTITY.adapt.min_sweeps, 7392);
        assert_eq!(CPU_MSA_IDENTITY.adapt.max_sweeps, 29568);
        assert_eq!(CPU_MSA_IDENTITY.adapt.min_reads, 128);
        assert_eq!(CPU_MSA_IDENTITY.adapt.max_reads, 128);
        assert_eq!(CPU_MSA_IDENTITY.adapt.reads_solution_min_factor, 0);
        assert_eq!(CPU_MSA_IDENTITY.adapt.reads_solution_max_factor, 0);
        assert_eq!(CPU_MSA_IDENTITY.adapt.reads_solution_floor_factor, 0);
    }

    #[test]
    fn both_variants_return_exactly_num_reads_solutions() {
        let graph = zephyr_like(30, 1, &[0.0]);
        for variant in [SaVariant::Tabulated, SaVariant::MultiSpin] {
            // 1 is below one word, 64 is exactly one, 70 spills into a second.
            for reads in [1usize, 5, 64, 70] {
                let out = SaSampler::new(variant)
                    .sample(&graph, &params(reads, 64, 9))
                    .expect("sampling succeeds");
                assert_eq!(out.len(), reads, "{variant:?} with {reads} reads");
                assert!(out
                    .iter()
                    .all(|r| r.spins.len() == 30 && r.spins.iter().all(|&s| s == 1 || s == -1)));
            }
        }
    }

    #[test]
    fn energies_match_the_protocol_scoring() {
        let graph = zephyr_like(24, 2, &[-1.0, 0.0, 1.0]);
        for variant in [SaVariant::Tabulated, SaVariant::MultiSpin] {
            for r in SaSampler::new(variant)
                .sample(&graph, &params(8, 128, 3))
                .expect("sampling succeeds")
            {
                assert_eq!(
                    r.energy_milli,
                    energy_milli(&r.spins, &graph.h, &graph.j, &graph.edges),
                    "{variant:?}"
                );
            }
        }
    }

    #[test]
    fn same_seed_same_result_and_different_seed_differs() {
        let graph = zephyr_like(28, 5, &[0.0]);
        for variant in [SaVariant::Tabulated, SaVariant::MultiSpin] {
            let s = SaSampler::new(variant);
            let a = s.sample(&graph, &params(4, 32, 11)).expect("sampling");
            let b = s.sample(&graph, &params(4, 32, 11)).expect("sampling");
            let c = s.sample(&graph, &params(4, 32, 12)).expect("sampling");
            let spins = |v: &Vec<SamplerResult>| -> Vec<Vec<i8>> {
                v.iter().map(|r| r.spins.clone()).collect()
            };
            assert_eq!(spins(&a), spins(&b), "{variant:?} is not deterministic");
            assert_ne!(spins(&a), spins(&c), "{variant:?} ignores the seed");
        }
    }

    #[test]
    fn non_discrete_problems_fall_back_to_the_sa_kernel() {
        // Couplings of 0.5 have no integer view, so both variants must still
        // answer, by way of the ordinary cpu-sa kernel.
        let graph = IsingGraph::new(vec![0.0, 0.0, 0.0], vec![0.5, -0.5], vec![(0, 1), (1, 2)]);
        for variant in [SaVariant::Tabulated, SaVariant::MultiSpin] {
            let out = SaSampler::new(variant)
                .sample(&graph, &params(3, 64, 7))
                .expect("sampling succeeds");
            assert_eq!(out.len(), 3);
            let expect = crate::sample_ising(&graph, &params(3, 64, 7), Algorithm::Sa);
            assert_eq!(
                out.iter().map(|r| r.spins.clone()).collect::<Vec<_>>(),
                expect.iter().map(|r| r.spins.clone()).collect::<Vec<_>>(),
                "{variant:?} fallback must be the cpu-sa kernel exactly"
            );
        }
    }

    #[test]
    fn multi_spin_falls_back_when_the_field_is_too_large_for_a_ghost_bond() {
        // |h| = 2 cannot be one ghost bond, so bond_counts refuses and the
        // multi-spin variant runs the scalar tabulated kernel instead. It must
        // still produce well-formed reads.
        let graph = IsingGraph::new(vec![2.0, 0.0, -2.0], vec![1.0, -1.0], vec![(0, 1), (1, 2)]);
        let out = SaSampler::new(SaVariant::MultiSpin)
            .sample(&graph, &params(4, 64, 2))
            .expect("sampling succeeds");
        assert_eq!(out.len(), 4);
        assert!(out.iter().all(|r| r.spins.len() == 3));
    }

    #[test]
    fn a_problem_with_no_bonds_at_all_is_safe() {
        // No couplings and no fields: every configuration has energy 0 and
        // there is no uphill move, which is the edge the threshold table has
        // to survive.
        let graph = IsingGraph::new(vec![0.0; 5], vec![], vec![]);
        for variant in [SaVariant::Tabulated, SaVariant::MultiSpin] {
            let out = SaSampler::new(variant)
                .sample(&graph, &params(3, 32, 1))
                .expect("sampling succeeds");
            assert_eq!(out.len(), 3);
            assert!(out.iter().all(|r| r.energy_milli == 0));
        }
    }

    #[test]
    fn fields_only_problem_lands_on_the_exact_optimum() {
        // No couplings, so every spin independently opposes its field and the
        // optimum is -sum|h|. This is the test that exercises the multi-spin
        // ghost bond: with no neighbours, the field is the only plane.
        let n = 40usize;
        let mut rng = SmallRng::seed_from_u64(6);
        let h: Vec<f64> = (0..n)
            .map(|_| if rng.gen::<bool>() { 1.0 } else { -1.0 })
            .collect();
        let optimum = -(h.iter().map(|v| v.abs()).sum::<f64>() as i64) * 1000;
        let graph = IsingGraph::new(h, vec![], vec![]);
        for variant in [SaVariant::Tabulated, SaVariant::MultiSpin] {
            let best = SaSampler::new(variant)
                .sample(&graph, &params(8, 256, 4))
                .expect("sampling succeeds")
                .iter()
                .map(|r| r.energy_milli)
                .min()
                .expect("reads");
            assert_eq!(best, optimum, "{variant:?}");
        }
    }

    #[test]
    fn ferromagnetic_clique_anneals_to_its_ground_state() {
        // Every pair coupled at -1: the ground states are all-up and all-down
        // at energy -C(n, 2), and single-flip descent reaches them from
        // anywhere, so a kernel that anneals at all must find this.
        let n = 8usize;
        let mut edges = Vec::new();
        for u in 0..n {
            for v in (u + 1)..n {
                edges.push((u, v));
            }
        }
        let j = vec![-1.0; edges.len()];
        let optimum = -((n * (n - 1) / 2) as i64) * 1000;
        let graph = IsingGraph::new(vec![0.0; n], j, edges);
        for variant in [SaVariant::Tabulated, SaVariant::MultiSpin] {
            let best = SaSampler::new(variant)
                .sample(&graph, &params(8, 256, 4))
                .expect("sampling succeeds")
                .iter()
                .map(|r| r.energy_milli)
                .min()
                .expect("reads");
            assert_eq!(best, optimum, "{variant:?}");
        }
    }

    #[test]
    fn public_cached_sample_matches_uncached_reference() {
        let graphs = [
            zephyr_like(20, 1, &[0.0]),
            zephyr_like(16, 2, &[-1.0, 0.0, 1.0]),
            IsingGraph::new(vec![0.0; 5], vec![], vec![]),
            IsingGraph::new(
                vec![1.0, 0.0, -1.0, 0.0, 1.0],
                vec![1.0, -1.0, 1.0, -1.0],
                vec![(0, 1), (1, 2), (2, 3), (3, 4)],
            ),
        ];
        let s = SaSampler::new(SaVariant::MultiSpin);
        for (i, graph) in graphs.iter().enumerate() {
            let p = params(8, 16, 100 + i as u64);
            let cached = s.sample(graph, &p).expect("cached");
            let cold = sample_sa_variant(graph, &p, SaVariant::MultiSpin, None).expect("uncached");
            assert_eq!(result_spins(&cached), result_spins(&cold), "graph {i}");
        }
    }

    #[test]
    fn cache_reuses_one_entry_across_field_and_coupling_changes() {
        let edges = vec![(0, 1), (1, 2), (2, 3), (0, 3)];
        let first = IsingGraph::new(vec![0.0; 4], vec![1.0, -1.0, 1.0, -1.0], edges.clone());
        let second = IsingGraph::new(vec![1.0, -1.0, 0.0, 1.0], vec![-1.0, 0.0, 1.0, 0.0], edges);
        let p = params(8, 16, 3);
        let s = SaSampler::new(SaVariant::MultiSpin);
        let a = s.sample(&first, &p).expect("first sample");
        let ptr = Arc::as_ptr(cached_entry(&s).as_ref().expect("cold fill"));
        let b = s.sample(&second, &p).expect("warm sample");
        let warm = cached_entry(&s).expect("warm entry");
        assert_eq!(
            Arc::as_ptr(&warm),
            ptr,
            "fields and J are not part of the key"
        );
        assert_eq!(warm.nodes, 4);
        assert_eq!(warm.edges, first.edges);
        assert_eq!(
            result_spins(&a),
            result_spins(
                &sample_sa_variant(&first, &p, SaVariant::MultiSpin, None).expect("uncached first")
            )
        );
        assert_eq!(
            result_spins(&b),
            result_spins(
                &sample_sa_variant(&second, &p, SaVariant::MultiSpin, None)
                    .expect("uncached second")
            )
        );
    }

    #[test]
    fn cache_invalidates_when_nodes_or_ordered_edges_change() {
        let s = SaSampler::new(SaVariant::MultiSpin);
        let p = params(4, 8, 11);
        let base = IsingGraph::new(vec![0.0; 4], vec![1.0, 1.0], vec![(0, 1), (1, 2)]);
        s.sample(&base, &p).expect("base");
        let ptr = Arc::as_ptr(cached_entry(&s).as_ref().expect("base entry"));

        let extra_node = IsingGraph::new(vec![0.0; 5], vec![1.0, 1.0], vec![(0, 1), (1, 2)]);
        let extra = s.sample(&extra_node, &p).expect("extra node");
        let ptr_nodes = Arc::as_ptr(cached_entry(&s).as_ref().expect("node entry"));
        assert_ne!(ptr_nodes, ptr, "node count is part of the key");
        assert_eq!(
            result_spins(&extra),
            result_spins(
                &sample_sa_variant(&extra_node, &p, SaVariant::MultiSpin, None)
                    .expect("uncached nodes")
            )
        );

        let reordered = IsingGraph::new(vec![0.0; 4], vec![1.0, 1.0], vec![(1, 2), (0, 1)]);
        let reordered_out = s.sample(&reordered, &p).expect("reordered");
        let ptr_edges = Arc::as_ptr(cached_entry(&s).as_ref().expect("edge entry"));
        assert_ne!(ptr_edges, ptr_nodes, "edge order is part of the key");
        assert_eq!(
            result_spins(&reordered_out),
            result_spins(
                &sample_sa_variant(&reordered, &p, SaVariant::MultiSpin, None)
                    .expect("uncached edges")
            )
        );

        let with_loop = IsingGraph::new(
            vec![0.0; 4],
            vec![1.0, 1.0, 0.0],
            vec![(1, 2), (0, 1), (2, 2)],
        );
        let loop_out = s.sample(&with_loop, &p).expect("self-loop");
        let ptr_loop = Arc::as_ptr(cached_entry(&s).as_ref().expect("loop entry"));
        assert_ne!(ptr_loop, ptr_edges, "self-loops change the raw edge list");
        assert_eq!(
            result_spins(&loop_out),
            result_spins(
                &sample_sa_variant(&with_loop, &p, SaVariant::MultiSpin, None)
                    .expect("uncached loop")
            )
        );
    }

    #[test]
    fn clones_share_the_topology_cache() {
        let s = SaSampler::new(SaVariant::MultiSpin);
        let c = s.clone();
        assert!(Arc::ptr_eq(&s.coloring, &c.coloring));
        let graph = IsingGraph::new(vec![0.0; 3], vec![-1.0], vec![(0, 1)]);
        let p = params(4, 8, 2);
        s.sample(&graph, &p).expect("owner sample");
        let owner = cached_entry(&s).expect("owner cache");
        let clone = cached_entry(&c).expect("clone sees the fill");
        assert!(Arc::ptr_eq(&owner, &clone));
        let warm = c.sample(&graph, &p).expect("clone sample");
        assert_eq!(
            Arc::as_ptr(cached_entry(&s).as_ref().expect("still cached")),
            Arc::as_ptr(&owner)
        );
        assert_eq!(
            result_spins(&warm),
            result_spins(
                &sample_sa_variant(&graph, &p, SaVariant::MultiSpin, None).expect("uncached")
            )
        );
    }

    #[test]
    fn concurrent_cached_samples_match_the_uncached_reference() {
        let graph = zephyr_like(18, 9, &[0.0, 1.0, -1.0]);
        let p = params(8, 16, 21);
        let expect = sample_sa_variant(&graph, &p, SaVariant::MultiSpin, None).expect("uncached");
        let sampler = SaSampler::new(SaVariant::MultiSpin);
        std::thread::scope(|scope| {
            let joins: Vec<_> = (0..8)
                .map(|_| scope.spawn(|| sampler.sample(&graph, &p).expect("concurrent sample")))
                .collect();
            for join in joins {
                let got = join.join().expect("thread");
                assert_eq!(result_spins(&got), result_spins(&expect));
            }
        });
        let entry = cached_entry(&sampler).expect("cache filled");
        assert_eq!(entry.nodes, graph.h.len());
        assert_eq!(entry.edges, graph.edges);
    }

    #[test]
    fn tabulated_and_fallback_cached_paths_match_uncached() {
        let discrete = zephyr_like(12, 4, &[0.0, 1.0]);
        let fractional =
            IsingGraph::new(vec![0.0, 0.0, 0.0], vec![0.5, -0.5], vec![(0, 1), (1, 2)]);
        let too_large_field =
            IsingGraph::new(vec![2.0, 0.0, -2.0], vec![1.0, -1.0], vec![(0, 1), (1, 2)]);
        let p = params(4, 32, 7);
        let tab = SaSampler::new(SaVariant::Tabulated);
        let msa = SaSampler::new(SaVariant::MultiSpin);
        for (s, graph, variant) in [
            (&tab, &discrete, SaVariant::Tabulated),
            (&tab, &fractional, SaVariant::Tabulated),
            (&msa, &fractional, SaVariant::MultiSpin),
            (&msa, &too_large_field, SaVariant::MultiSpin),
        ] {
            let cached = s.sample(graph, &p).expect("cached path");
            let uncached = sample_sa_variant(graph, &p, variant, None).expect("uncached path");
            assert_eq!(
                result_spins(&cached),
                result_spins(&uncached),
                "{variant:?}"
            );
        }
    }

    #[test]
    #[expect(clippy::panic)]
    fn poisoned_coloring_lock_recovers_for_later_samples() {
        let s = SaSampler::new(SaVariant::MultiSpin);
        let graph = IsingGraph::new(vec![0.0; 3], vec![1.0], vec![(0, 1)]);
        let p = params(4, 8, 1);
        let cache = Arc::clone(&s.coloring);
        let join = std::thread::spawn(move || {
            let _guard = cache.lock().unwrap_or_else(|e| e.into_inner());
            panic!("poison the coloring cache");
        });
        assert!(join.join().is_err());
        let got = s.sample(&graph, &p).expect("recovered sample");
        let expect = sample_sa_variant(&graph, &p, SaVariant::MultiSpin, None).expect("uncached");
        assert_eq!(result_spins(&got), result_spins(&expect));
    }

    #[tokio::test]
    #[expect(clippy::panic)]
    async fn cached_sample_stream_matches_uncached_reference() {
        use quip_solver_core::StreamOutcome;
        use std::time::Duration;

        let sampler = SaSampler::new(SaVariant::MultiSpin);
        let probe = sampler.clone();
        let graph = IsingGraph::new(
            vec![1.0, 0.0, -1.0, 0.0],
            vec![1.0, -1.0, 1.0],
            vec![(0, 1), (1, 2), (2, 3)],
        );
        let p = params(4, 16, 5);
        let expect = sample_sa_variant(&graph, &p, SaVariant::MultiSpin, None).expect("uncached");

        let (job_tx, job_rx) = tokio::sync::mpsc::channel::<StreamJob>(1);
        let (out_tx, mut out_rx) = tokio::sync::mpsc::channel::<StreamResult>(1);
        job_tx
            .send(StreamJob {
                job_id: b"cached-stream".to_vec(),
                graph,
                params: p,
                watermark: None,
            })
            .await
            .expect("send StreamJob");
        drop(job_tx);

        let pump = tokio::task::spawn_blocking(move || {
            sampler.sample_stream(job_rx, out_tx, CancelToken::default());
        });

        let got = tokio::time::timeout(Duration::from_secs(30), out_rx.recv())
            .await
            .expect("timeout waiting for StreamResult")
            .expect("output channel closed without a result");
        let results = match got.outcome {
            StreamOutcome::Completed(result) => result.expect("stream job should succeed"),
            StreamOutcome::Cancelled => {
                panic!("expected Completed outcome, got Cancelled")
            }
        };
        assert_eq!(result_spins(&results), result_spins(&expect));
        assert!(cached_entry(&probe).is_some(), "stream fills the cache");

        tokio::time::timeout(Duration::from_secs(30), pump)
            .await
            .expect("timeout waiting for sample_stream to exit")
            .expect("spawn_blocking join");
    }
}
