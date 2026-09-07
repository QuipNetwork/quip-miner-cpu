//! Tabulated-threshold and multi-spin annealing backends.
//!
//! Two samplers from Isakov, Zintchenko, Rønnow and Troyer (Comput. Phys.
//! Commun. 192, 265 (2015), arXiv:1401.1084), kept apart from
//! [`crate::CpuSampler`] because both change the random stream and are
//! therefore different samplers, not faster spellings of `cpu-sa`:
//!
//! - [`SaVariant::Tabulated`] moves the Metropolis test into its geometric
//!   form and draws the thresholds from a table shared by every read.
//! - [`SaVariant::MultiSpin`] does the same and packs 64 reads into the bits
//!   of a machine word.
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
use std::sync::Arc;

use crate::sa_int::{anneal_from, sweep_offsets, threshold_draws, IntGraph};
use crate::sa_msc::{anneal_word, bond_counts, MscState, LANES};
use crate::sampler_core::{build_beta_schedule, sample_ising_cancellable, SampleCancelled};
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
/// Experimental track. The reads envelope is deliberately `CPU_ADAPT` as well.
/// A multiple of 64 is this kernel's natural read count — 64 reads cost what
/// one read costs — but leaving the envelope alone keeps every CPU binary
/// comparable on the same axes until the campaign says otherwise.
pub const CPU_MSA_IDENTITY: BackendIdentity = BackendIdentity {
    backend: "cpu",
    algorithm: "msa",
    max_nodes: DEFAULT_MAX_NODES,
    max_edges: DEFAULT_MAX_EDGES,
    features: &[],
    adapt: CPU_MSA_ADAPT,
};

/// Adapt envelope for the multi-spin backend, identical to [`CPU_ADAPT`].
///
/// Named separately so a later campaign can raise `min_reads` to a multiple of
/// [`LANES`] without disturbing the other CPU binaries.
pub(crate) const CPU_MSA_ADAPT: AdaptBounds = CPU_ADAPT;

/// Which annealing kernel a binary drives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SaVariant {
    /// Scalar kernel with tabulated acceptance thresholds.
    Tabulated,
    /// Multi-spin coded kernel, 64 reads per machine word.
    MultiSpin,
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
        }
    }
}

/// Run one job on the selected kernel, or on `cpu-sa` when the problem does
/// not qualify.
pub(crate) fn sample_sa_variant(
    graph: &IsingGraph,
    params: &SampleParams,
    variant: SaVariant,
    cancel: Option<(&CancelToken, Option<u64>)>,
) -> Result<Vec<SamplerResult>, SampleCancelled> {
    let Some(int) = IntGraph::from_base(graph) else {
        return sample_ising_cancellable(graph, params, Algorithm::Sa, cancel);
    };
    let num_reads = params.num_reads.max(1);
    let betas = build_beta_schedule(graph, params);
    let sweeps_per = params.sweeps_per_beta.max(1);

    // The threshold table is drawn once and shared by every read, which is
    // what removes the per-attempt draw. Seeding it from the job seed keeps
    // the whole sampler deterministic.
    let mut table_rng = SmallRng::seed_from_u64(params.seed ^ 0x5341_5F54_424C_4531);
    let Some(draws) = threshold_draws(&betas, int.max_field(), &mut table_rng) else {
        // A ladder long enough to price the threshold table out is the last
        // precondition these kernels can fail, and it falls back the same way
        // as the others.
        return sample_ising_cancellable(graph, params, Algorithm::Sa, cancel);
    };

    let counts = match variant {
        SaVariant::MultiSpin => bond_counts(&int),
        SaVariant::Tabulated => None,
    };

    let mut results = Vec::with_capacity(num_reads);
    match counts {
        Some(counts) => {
            let mut read = 0usize;
            while read < num_reads {
                if let Some((guard, watermark)) = cancel {
                    if guard.is_cancelled(watermark) {
                        return Err(SampleCancelled);
                    }
                }
                let mut rng = read_rng(params.seed, read);
                let mut state = MscState::random(int.num_nodes(), &mut rng);
                let offsets = sweep_offsets(betas.len(), sweeps_per, &mut rng);
                anneal_word(
                    &int, &counts, &draws, sweeps_per, &mut state, &offsets, cancel,
                )?;
                for lane in 0..LANES.min(num_reads - read) {
                    results.push(score(&state.lane(lane), graph));
                }
                read += LANES;
            }
        }
        None => {
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
        Ok(sample_sa_variant(graph, params, self.variant, None).unwrap_or_default())
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
        run_stream_pump(
            || self.stream_width(),
            move |g, p, token, watermark| {
                sample_sa_variant(g, p, variant, Some((token, watermark)))
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
    fn identities_advertise_their_algorithms() {
        assert_eq!(CPU_FSA_IDENTITY.backend, "cpu");
        assert_eq!(CPU_FSA_IDENTITY.algorithm, "fsa");
        assert_eq!(CPU_MSA_IDENTITY.backend, "cpu");
        assert_eq!(CPU_MSA_IDENTITY.algorithm, "msa");
        assert_eq!(CPU_MSA_IDENTITY.adapt.min_reads, CPU_ADAPT.min_reads);
        assert_eq!(CPU_FSA_IDENTITY.adapt.min_reads, CPU_ADAPT.min_reads);
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
}
