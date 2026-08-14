//! Tensor-network sampler backend and its coordinator-facing identity.
//!
//! Kept out of `lib.rs` for the same reason as [`crate::sb_sampler`]: the new
//! algorithms stay isolated from the existing SA and Gibbs path. This module
//! and [`crate::mps`] are the whole of the tensor-network backend, and the only
//! thing they borrow from the rest of the crate is the shared streaming pump
//! and the greedy polish.

use quip_miner_core::adapt::AdaptBounds;
use quip_miner_core::{
    BackendIdentity, CancelGuard, IsingGraph, SampleParams, Sampler, SamplerResult, StreamJob,
    StreamResult,
};
use quip_proto::v1::RejectReason;

use crate::mps::{sample_ising_mps, MpsConfig};
use crate::run_stream_pump;

/// Tensor-network adapt envelope. Sweeps are capped below the SA ceiling
/// because a Trotter step costs far more than a Metropolis sweep, and the
/// sweep budget is split with the polish stage anyway. Reads match the other
/// CPU backends because sampling the final state is cheap.
///
/// Shared by the `mps` and `mfa` identities: the two differ only in the
/// algorithm string.
pub(crate) const CPU_MPS_ADAPT: AdaptBounds = AdaptBounds {
    min_sweeps: 64,
    max_sweeps: 1024,
    min_reads: 64,
    max_reads: 512,
    reads_solution_min_factor: 4,
    reads_solution_max_factor: 8,
    reads_solution_floor_factor: 0,
};

/// Backend identity for `quip-cpu-mps`.
///
/// `max_nodes` and `max_edges` are memory bounds, not quality bounds.
/// Advertising a limit low enough to exclude `advantage2-system1` was
/// considered and rejected: the coordinator defaults to that preset, so a miner
/// that rejects it mines nothing. Degrading to bond dimension 1 and returning
/// valid configurations is the better failure mode, and it is documented rather
/// than hidden.
pub const CPU_MPS_IDENTITY: BackendIdentity = BackendIdentity {
    backend: "cpu",
    algorithm: "mps",
    max_nodes: 65_536,
    max_edges: 524_288,
    adapt: CPU_MPS_ADAPT,
};

/// Backend identity for `quip-cpu-mfa`: the tensor-network kernel with bond
/// dimension fixed at 1 (mean-field annealing). Shares `CPU_MPS_ADAPT` with
/// `quip-cpu-mps`; only the algorithm string differs.
///
/// The two binaries agree exactly whenever `select_chi` would have returned 1
/// anyway, which is every graph as wide as the production topology. `mfa` names
/// that regime so a campaign can measure it without inferring the bond
/// dimension from the results.
pub const CPU_MFA_IDENTITY: BackendIdentity = BackendIdentity {
    backend: "cpu",
    algorithm: "mfa",
    max_nodes: 65_536,
    max_edges: 524_288,
    adapt: CPU_MPS_ADAPT,
};

/// Tensor-network sampler backend: imaginary-time TEBD, exact sampling, greedy
/// polish.
#[derive(Debug, Clone, Copy)]
pub struct MpsSampler {
    cfg: MpsConfig,
}

impl MpsSampler {
    /// Create a tensor-network sampler with the given configuration.
    ///
    /// # Examples
    ///
    /// ```
    /// use quip_miner_cpu::{InitMode, IsingGraph, MpsConfig, MpsSampler, SampleParams};
    /// use quip_miner_core::Sampler;
    /// use quip_proto::v1::RejectReason;
    ///
    /// # fn main() -> Result<(), RejectReason> {
    /// let cfg = MpsConfig {
    ///     chi_max: 8,
    ///     init: InitMode::Anneal,
    ///     // The clock takes no part: the work budget below bounds the anneal.
    ///     time_budget_ms: 0,
    ///     flop_budget: 1.25e9,
    ///     anneal_work_budget: 1.2e11,
    /// };
    /// let sampler = MpsSampler::new(cfg);
    /// let graph = IsingGraph::new(vec![0.0, 0.0], vec![-1.0], vec![(0, 1)]);
    /// let params = SampleParams { num_reads: 2, num_sweeps: 64, seed: 1, ..Default::default() };
    /// let results = sampler.sample(&graph, &params)?;
    /// assert_eq!(results.len(), 2);
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(cfg: MpsConfig) -> Self {
        Self { cfg }
    }
}

impl Sampler for MpsSampler {
    fn sample(
        &self,
        graph: &IsingGraph,
        params: &SampleParams,
    ) -> Result<Vec<SamplerResult>, RejectReason> {
        Ok(sample_ising_mps(graph, params, &self.cfg))
    }

    /// One model per core, matching the other CPU backends. A single model's
    /// reads share the annealed state, so they are cheap and stay sequential.
    fn stream_width(&self) -> usize {
        std::thread::available_parallelism().map_or(1, |n| n.get())
    }

    fn sample_stream(
        &self,
        jobs: tokio::sync::mpsc::Receiver<StreamJob>,
        out: tokio::sync::mpsc::Sender<StreamResult>,
        cancel: CancelGuard,
    ) {
        let cfg = self.cfg;
        run_stream_pump(
            || self.stream_width(),
            move |graph, params, _, _| Ok(sample_ising_mps(graph, params, &cfg)),
            jobs,
            out,
            cancel,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mps::InitMode;
    use quip_miner_core::StreamOutcome;
    use quip_protocol::scoring::energy_milli;
    use std::time::Duration;

    fn mps_cfg() -> MpsConfig {
        MpsConfig {
            chi_max: 8,
            init: InitMode::Anneal,
            time_budget_ms: 0,
            flop_budget: 1.25e9,
            anneal_work_budget: crate::mps::ANNEAL_WORK_BUDGET,
        }
    }

    fn tiny_ferro() -> IsingGraph {
        IsingGraph::new(vec![0.0, 0.0], vec![-1.0], vec![(0, 1)])
    }

    fn tiny_params(num_reads: usize) -> SampleParams {
        SampleParams {
            num_reads,
            num_sweeps: 64,
            seed: 42,
            ..Default::default()
        }
    }

    #[test]
    fn mps_identity_advertises_the_documented_envelope() {
        assert_eq!(CPU_MPS_IDENTITY.backend, "cpu");
        assert_eq!(CPU_MPS_IDENTITY.algorithm, "mps");
        // Memory bounds, not quality bounds: at 65536 nodes a bond-4 model is
        // 16.8 MB, and 16 of them fit in 268 MB.
        assert_eq!(CPU_MPS_IDENTITY.max_nodes, 65_536);
        assert_eq!(CPU_MPS_IDENTITY.max_edges, 524_288);
        assert_eq!(CPU_MPS_IDENTITY.adapt.min_sweeps, 64);
        assert_eq!(CPU_MPS_IDENTITY.adapt.max_sweeps, 1024);
        assert_eq!(CPU_MPS_IDENTITY.adapt.min_reads, 64);
        assert_eq!(CPU_MPS_IDENTITY.adapt.max_reads, 512);
        assert_eq!(CPU_MPS_IDENTITY.adapt.reads_solution_min_factor, 4);
        assert_eq!(CPU_MPS_IDENTITY.adapt.reads_solution_max_factor, 8);
        assert_eq!(CPU_MPS_IDENTITY.adapt.reads_solution_floor_factor, 0);
    }

    #[test]
    fn mps_sampler_samples_and_reports_consensus_energies() {
        let sampler = MpsSampler::new(mps_cfg());
        let graph = tiny_ferro();
        let results = sampler
            .sample(&graph, &tiny_params(4))
            .expect("MpsSampler::sample must not reject");
        assert_eq!(results.len(), 4);
        for r in &results {
            assert_eq!(r.spins.len(), 2);
            assert!(r.spins.iter().all(|&s| s == 1 || s == -1));
            assert_eq!(r.energy_milli, -1000, "ferro pair must align");
        }
    }

    #[test]
    fn mps_stream_width_is_at_least_one() {
        assert!(MpsSampler::new(mps_cfg()).stream_width() >= 1);
    }

    #[tokio::test]
    async fn mps_sample_stream_one_job_round_trip() {
        let sampler = MpsSampler::new(mps_cfg());
        let (job_tx, job_rx) = tokio::sync::mpsc::channel::<StreamJob>(1);
        let (out_tx, mut out_rx) = tokio::sync::mpsc::channel::<StreamResult>(1);

        let job_id = b"job-mps-1".to_vec();
        job_tx
            .send(StreamJob {
                job_id: job_id.clone(),
                graph: tiny_ferro(),
                params: tiny_params(2),
                generation: 0,
            })
            .await
            .expect("send StreamJob");
        drop(job_tx);

        let pump = tokio::task::spawn_blocking(move || {
            sampler.sample_stream(job_rx, out_tx, CancelGuard::default());
        });

        let got = tokio::time::timeout(Duration::from_secs(30), out_rx.recv())
            .await
            .expect("timeout waiting for StreamResult")
            .expect("output channel closed without a result");
        assert_eq!(got.job_id, job_id);
        // `clippy::panic` is denied crate-wide, so fail through an assertion
        // rather than a panic, matching the sb_sampler stream test.
        let StreamOutcome::Completed(result) = got.outcome else {
            assert_eq!("got", "Completed outcome");
            return;
        };
        let results = result.expect("stream job should succeed");
        assert_eq!(results.len(), 2);

        tokio::time::timeout(Duration::from_secs(30), pump)
            .await
            .expect("timeout waiting for sample_stream to exit")
            .expect("spawn_blocking join");
        assert!(out_rx.recv().await.is_none());
    }

    #[tokio::test]
    async fn mps_sample_stream_skips_a_cancelled_generation_without_sampling() {
        let sampler = MpsSampler::new(mps_cfg());
        let (job_tx, job_rx) = tokio::sync::mpsc::channel::<StreamJob>(1);
        let (out_tx, mut out_rx) = tokio::sync::mpsc::channel::<StreamResult>(1);

        let cancel = CancelGuard::default();
        cancel.cancel_through(7);

        job_tx
            .send(StreamJob {
                job_id: b"job-mps-cancelled".to_vec(),
                graph: tiny_ferro(),
                params: tiny_params(1),
                generation: 7,
            })
            .await
            .expect("send StreamJob");
        drop(job_tx);

        let pump = tokio::task::spawn_blocking(move || {
            sampler.sample_stream(job_rx, out_tx, cancel);
        });

        let got = tokio::time::timeout(Duration::from_secs(30), out_rx.recv())
            .await
            .expect("timeout waiting for StreamResult")
            .expect("output channel closed without a result");
        assert_eq!(got.job_id, b"job-mps-cancelled".to_vec());
        assert!(
            matches!(got.outcome, StreamOutcome::Cancelled),
            "an abandoned generation must be dropped before sampling"
        );
        assert_eq!(
            got.device_access_time_us, 0,
            "a cancelled job must not report sampling time"
        );

        tokio::time::timeout(Duration::from_secs(30), pump)
            .await
            .expect("timeout waiting for sample_stream to exit")
            .expect("spawn_blocking join");
    }
    #[test]
    fn cpu_mfa_identity_fields() {
        assert_eq!(CPU_MFA_IDENTITY.backend, "cpu");
        assert_eq!(CPU_MFA_IDENTITY.algorithm, "mfa");
        assert_eq!(CPU_MFA_IDENTITY.max_nodes, 65_536);
        assert_eq!(CPU_MFA_IDENTITY.max_edges, 524_288);
    }

    // ---- Task 3 and 4: the mfa construction path ----

    /// Five-node ring plus one chord, mixed-sign fields and couplings.
    fn mfa_graph() -> IsingGraph {
        IsingGraph::new(
            vec![0.4, -0.2, 0.1, -0.5, 0.3],
            vec![0.6, -0.9, 0.25, -0.4, 0.75],
            vec![(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)],
        )
    }

    /// No edges: the ground state factorizes exactly, `s_i = -sign(h_i)`.
    fn fields_only_graph() -> IsingGraph {
        IsingGraph::new(vec![0.7, -0.3, 1.2, -2.5], vec![], vec![])
    }

    /// `quip-cpu-mfa` is exactly `quip-cpu-mps` pinned to bond dimension 1, and
    /// nothing else. If this ever diverges, the two binaries are measuring
    /// different algorithms and the campaign comparison between them is void.
    ///
    /// Holds [`crate::mps::ENV_LOCK`] because it reads `QUIP_MPS_INIT` through
    /// `from_env`, and another test writes that key.
    #[test]
    fn mfa_equals_mps_at_chi_1() {
        let _guard = crate::mps::ENV_LOCK.lock().expect("env lock poisoned");

        let graph = mfa_graph();
        let params = SampleParams {
            num_reads: 4,
            num_sweeps: 96,
            seed: 11,
            ..Default::default()
        };
        // 2000 ms and 1.25e9 flops are the documented defaults. Neither changes
        // the result at chi_max = 1 on a graph this small: chi stays pinned to 1
        // whatever the flop budget, and the anneal finishes far inside the time
        // budget.
        let direct = MpsConfig {
            chi_max: 1,
            init: InitMode::Anneal,
            time_budget_ms: 2000,
            flop_budget: 1.25e9,
            anneal_work_budget: crate::mps::ANNEAL_WORK_BUDGET,
        };
        let via_from_env = MpsConfig::from_env(1);
        assert_eq!(via_from_env.chi_max, 1);
        assert_eq!(via_from_env.init, InitMode::Anneal);

        let mps_at_chi_1 = sample_ising_mps(&graph, &params, &direct);
        let mfa_path = sample_ising_mps(&graph, &params, &via_from_env);
        assert_eq!(
            mps_at_chi_1, mfa_path,
            "the mfa construction must match the mps kernel pinned to chi_max = 1"
        );
    }

    #[test]
    fn mfa_fields_only_is_exact() {
        let graph = fields_only_graph();
        let cfg = MpsConfig {
            chi_max: 1,
            init: InitMode::Anneal,
            time_budget_ms: 2000,
            flop_budget: 1.25e9,
            anneal_work_budget: crate::mps::ANNEAL_WORK_BUDGET,
        };
        let params = SampleParams {
            num_reads: 6,
            num_sweeps: 64,
            seed: 5,
            ..Default::default()
        };
        let results = sample_ising_mps(&graph, &params, &cfg);
        assert_eq!(results.len(), params.num_reads);
        let expected: Vec<i8> = graph
            .h
            .iter()
            .map(|&h| if h > 0.0 { -1 } else { 1 })
            .collect();
        for r in &results {
            assert_eq!(r.spins, expected, "fields-only optimum is s_i = -sign(h_i)");
            assert_eq!(
                r.energy_milli,
                energy_milli(&r.spins, &graph.h, &graph.j, &graph.edges)
            );
        }
    }

    #[test]
    fn mfa_random_init_mode_differs_and_stays_valid() {
        let graph = mfa_graph();
        let params = SampleParams {
            num_reads: 8,
            num_sweeps: 96,
            seed: 11,
            ..Default::default()
        };
        let anneal_cfg = MpsConfig {
            chi_max: 1,
            init: InitMode::Anneal,
            time_budget_ms: 2000,
            flop_budget: 1.25e9,
            anneal_work_budget: crate::mps::ANNEAL_WORK_BUDGET,
        };
        let random_cfg = MpsConfig {
            init: InitMode::Random,
            ..anneal_cfg
        };

        let anneal_results = sample_ising_mps(&graph, &params, &anneal_cfg);
        let random_results = sample_ising_mps(&graph, &params, &random_cfg);
        let random_results_again = sample_ising_mps(&graph, &params, &random_cfg);

        assert_eq!(anneal_results.len(), params.num_reads);
        assert_eq!(random_results.len(), params.num_reads);
        for r in anneal_results.iter().chain(random_results.iter()) {
            assert!(r.spins.iter().all(|&s| s == 1 || s == -1));
            assert_eq!(
                r.energy_milli,
                energy_milli(&r.spins, &graph.h, &graph.j, &graph.edges)
            );
        }
        assert_eq!(
            random_results, random_results_again,
            "the same seed must reproduce the random-init path byte-identically"
        );
        assert_ne!(
            anneal_results, random_results,
            "random seeding must reach a different sample set than the annealed path here"
        );
    }
}
