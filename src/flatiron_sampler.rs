//! BP-TNS sampler backend and its coordinator-facing identity.
//!
//! Kept out of `lib.rs` for the same reason as [`crate::mps_sampler`]: the
//! experimental kernels stay isolated from the SA and Gibbs path, borrowing
//! only the shared streaming pump and the greedy polish.

use quip_miner_core::{
    BackendIdentity, CancelGuard, IsingGraph, SampleParams, Sampler, SamplerResult, StreamJob,
    StreamResult,
};
use quip_proto::v1::RejectReason;

use crate::flatiron::{sample_ising_flatiron, FlatironConfig};
use crate::mps_sampler::CPU_MPS_ADAPT;
use crate::run_stream_pump;

/// Backend identity for `quip-cpu-flatiron`.
///
/// The envelope matches the MPS identity: both are tensor-network kernels
/// whose Trotter step costs far more than a Metropolis sweep, and the memory
/// bounds are the same 64 MB per-model cap. Only the algorithm string
/// differs, so a campaign can tell the two apart.
pub const CPU_FLATIRON_IDENTITY: BackendIdentity = BackendIdentity {
    backend: "cpu",
    algorithm: "flatiron",
    max_nodes: 65_536,
    max_edges: 524_288,
    adapt: CPU_MPS_ADAPT,
};

/// BP-TNS sampler backend: imaginary-time simple update on the problem
/// graph, BP marginals, conditioned sampling, greedy polish.
#[derive(Debug, Clone, Copy)]
pub struct FlatironSampler {
    cfg: FlatironConfig,
}

impl FlatironSampler {
    /// Create a BP-TNS sampler with the given configuration.
    ///
    /// # Examples
    ///
    /// ```
    /// use quip_miner_cpu::{FlatironConfig, FlatironSampler, IsingGraph, SampleParams};
    /// use quip_miner_core::Sampler;
    /// use quip_proto::v1::RejectReason;
    ///
    /// # fn main() -> Result<(), RejectReason> {
    /// let sampler = FlatironSampler::new(FlatironConfig::new(8));
    /// let graph = IsingGraph::new(vec![0.0, 0.0], vec![-1.0], vec![(0, 1)]);
    /// let params = SampleParams { num_reads: 2, num_sweeps: 64, seed: 1, ..Default::default() };
    /// let results = sampler.sample(&graph, &params)?;
    /// assert_eq!(results.len(), 2);
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(cfg: FlatironConfig) -> Self {
        Self { cfg }
    }
}

impl Sampler for FlatironSampler {
    fn sample(
        &self,
        graph: &IsingGraph,
        params: &SampleParams,
    ) -> Result<Vec<SamplerResult>, RejectReason> {
        Ok(sample_ising_flatiron(graph, params, &self.cfg))
    }

    /// One model per core, matching the other CPU backends.
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
            self.stream_width(),
            move |graph, params, _, _| Ok(sample_ising_flatiron(graph, params, &cfg)),
            jobs,
            out,
            cancel,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quip_miner_core::StreamOutcome;
    use std::time::Duration;

    fn test_cfg() -> FlatironConfig {
        FlatironConfig {
            chi_max: 8,
            time_budget_ms: 0,
            flop_budget: 1.25e9,
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
    fn flatiron_identity_advertises_the_documented_envelope() {
        assert_eq!(CPU_FLATIRON_IDENTITY.backend, "cpu");
        assert_eq!(CPU_FLATIRON_IDENTITY.algorithm, "flatiron");
        assert_eq!(CPU_FLATIRON_IDENTITY.max_nodes, 65_536);
        assert_eq!(CPU_FLATIRON_IDENTITY.max_edges, 524_288);
        assert_eq!(CPU_FLATIRON_IDENTITY.adapt.min_sweeps, 64);
        assert_eq!(CPU_FLATIRON_IDENTITY.adapt.max_sweeps, 1024);
    }

    #[test]
    fn flatiron_sampler_samples_and_reports_consensus_energies() {
        let sampler = FlatironSampler::new(test_cfg());
        let results = sampler
            .sample(&tiny_ferro(), &tiny_params(4))
            .expect("FlatironSampler::sample must not reject");
        assert_eq!(results.len(), 4);
        for r in &results {
            assert_eq!(r.spins.len(), 2);
            assert_eq!(r.energy_milli, -1000, "ferro pair must align");
        }
    }

    #[test]
    fn flatiron_stream_width_is_at_least_one() {
        assert!(FlatironSampler::new(test_cfg()).stream_width() >= 1);
    }

    #[tokio::test]
    async fn flatiron_sample_stream_one_job_round_trip() {
        let sampler = FlatironSampler::new(test_cfg());
        let (job_tx, job_rx) = tokio::sync::mpsc::channel::<StreamJob>(1);
        let (out_tx, mut out_rx) = tokio::sync::mpsc::channel::<StreamResult>(1);

        let job_id = b"job-flatiron-1".to_vec();
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
    async fn flatiron_sample_stream_skips_a_cancelled_generation_without_sampling() {
        let sampler = FlatironSampler::new(test_cfg());
        let (job_tx, job_rx) = tokio::sync::mpsc::channel::<StreamJob>(1);
        let (out_tx, mut out_rx) = tokio::sync::mpsc::channel::<StreamResult>(1);

        let cancel = CancelGuard::default();
        cancel.cancel_through(7);

        job_tx
            .send(StreamJob {
                job_id: b"job-flatiron-cancelled".to_vec(),
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
        assert_eq!(got.job_id, b"job-flatiron-cancelled".to_vec());
        assert!(
            matches!(got.outcome, StreamOutcome::Cancelled),
            "an abandoned generation must be dropped before sampling"
        );
        assert_eq!(got.device_access_time_us, 0);

        tokio::time::timeout(Duration::from_secs(30), pump)
            .await
            .expect("timeout waiting for sample_stream to exit")
            .expect("spawn_blocking join");
    }
}
