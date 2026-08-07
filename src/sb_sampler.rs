//! Simulated Bifurcation sampler backend and its coordinator-facing identity.
//!
//! Kept separate from `lib.rs` so the SB work stays isolated from the existing
//! SA and Gibbs backend: this module and [`crate::sb_core`] are the whole of
//! it. The only thing it borrows from the rest of the crate is the shared
//! streaming pump, so cancellation and panic propagation cannot drift between
//! binaries.

use quip_miner_core::adapt::AdaptBounds;
use quip_miner_core::{
    BackendIdentity, CancelGuard, IsingGraph, SampleParams, Sampler, SamplerResult, StreamJob,
    StreamResult,
};
use quip_proto::v1::RejectReason;

use crate::sb_core::{sample_sb, SbVariant};
use crate::{run_stream_pump, DEFAULT_MAX_EDGES, DEFAULT_MAX_NODES};

/// SB adapt envelope, shared by all four SB identities.
///
/// The reads bounds match `CPU_ADAPT` exactly, so the reads dimension stays
/// comparable across every CPU binary. The sweeps bounds differ: `min_sweeps`
/// rises because a short linear ramp gives the bifurcation almost no time to
/// separate and returns near-random signs, and `max_sweeps` rises because an SB
/// step costs less than a Metropolis sweep, so the envelope is matched on
/// estimated cost rather than on step count.
///
/// Both sweeps numbers are provisional. The benchmark campaign re-tunes them.
/// No golden fixture pins them: `adapt_params` takes `AdaptBounds` as a
/// parameter and the conformance golden `adapt_params_cpu_sa` pins only the SA
/// bounds.
pub(crate) const CPU_SB_ADAPT: AdaptBounds = AdaptBounds {
    min_sweeps: 256,
    max_sweeps: 8192,
    min_reads: 64,
    max_reads: 512,
    reads_solution_min_factor: 4,
    reads_solution_max_factor: 8,
    reads_solution_floor_factor: 0,
};

/// Backend identity for `quip-cpu-sb` (discrete Simulated Bifurcation).
pub const CPU_SB_IDENTITY: BackendIdentity = BackendIdentity {
    backend: "cpu",
    algorithm: "sb",
    max_nodes: DEFAULT_MAX_NODES,
    max_edges: DEFAULT_MAX_EDGES,
    adapt: CPU_SB_ADAPT,
};

/// Simulated Bifurcation sampler backend. No device, no governor, uncapped
/// reads. The variant selects the coupling form and the heating rate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SbSampler {
    variant: SbVariant,
}

impl SbSampler {
    /// Create an SB sampler for `variant`.
    ///
    /// # Examples
    ///
    /// ```
    /// use quip_miner_cpu::{IsingGraph, SampleParams, SbSampler, DSB};
    /// use quip_miner_core::Sampler;
    /// use quip_proto::v1::RejectReason;
    ///
    /// # fn main() -> Result<(), RejectReason> {
    /// let sampler = SbSampler::new(DSB);
    /// let graph = IsingGraph::new(vec![0.0, 0.0], vec![-1.0], vec![(0, 1)]);
    /// let params = SampleParams {
    ///     num_reads: 2,
    ///     num_sweeps: 256,
    ///     seed: 1,
    ///     ..Default::default()
    /// };
    /// let results = sampler.sample(&graph, &params)?;
    /// assert_eq!(results.len(), 2);
    /// assert!(results.iter().all(|r| r.spins.iter().all(|&s| s == 1 || s == -1)));
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(variant: SbVariant) -> Self {
        Self { variant }
    }
}

impl Sampler for SbSampler {
    fn sample(
        &self,
        graph: &IsingGraph,
        params: &SampleParams,
    ) -> Result<Vec<SamplerResult>, RejectReason> {
        Ok(sample_sb(graph, params, self.variant))
    }

    /// One model per core, the same shape as [`crate::CpuSampler::stream_width`].
    fn stream_width(&self) -> usize {
        std::thread::available_parallelism().map_or(1, |n| n.get())
    }

    fn sample_stream(
        &self,
        jobs: tokio::sync::mpsc::Receiver<StreamJob>,
        out: tokio::sync::mpsc::Sender<StreamResult>,
        cancel: CancelGuard,
    ) {
        let variant = self.variant;
        run_stream_pump(
            self.stream_width(),
            move |g, p| sample_sb(g, p, variant),
            jobs,
            out,
            cancel,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sb_core::DSB;
    use crate::CPU_ADAPT;
    use quip_miner_core::StreamOutcome;
    use std::time::Duration;

    fn tiny_ferro() -> IsingGraph {
        IsingGraph::new(vec![0.0, 0.0], vec![-1.0], vec![(0, 1)])
    }

    fn tiny_params(num_reads: usize) -> SampleParams {
        SampleParams {
            num_reads,
            num_sweeps: 16,
            seed: 42,
            ..Default::default()
        }
    }

    /// Hypothesis: the sb identity advertises the values the design fixed. The
    /// `algorithm` string reaches the coordinator in the Hello handshake and in
    /// `--capabilities`, and nothing in `quip-miner-core` validates it, so this
    /// test is the only guard on it.
    #[test]
    fn sb_identity_advertises_the_sb_algorithm_and_adapt_envelope() {
        assert_eq!(CPU_SB_IDENTITY.backend, "cpu");
        assert_eq!(CPU_SB_IDENTITY.algorithm, "sb");
        assert_eq!(CPU_SB_IDENTITY.max_nodes, 100_000);
        assert_eq!(CPU_SB_IDENTITY.max_edges, 1_000_000);
        assert_eq!(CPU_SB_IDENTITY.adapt.min_sweeps, 256);
        assert_eq!(CPU_SB_IDENTITY.adapt.max_sweeps, 8192);
        assert_eq!(CPU_SB_IDENTITY.adapt.min_reads, 64);
        assert_eq!(CPU_SB_IDENTITY.adapt.max_reads, 512);
        assert_eq!(CPU_SB_IDENTITY.adapt.reads_solution_min_factor, 4);
        assert_eq!(CPU_SB_IDENTITY.adapt.reads_solution_max_factor, 8);
        assert_eq!(CPU_SB_IDENTITY.adapt.reads_solution_floor_factor, 0);
    }

    /// Hypothesis: the reads envelope matches `CPU_ADAPT` exactly, so the reads
    /// dimension stays comparable across every CPU binary. Only the sweeps
    /// bounds differ, and they are provisional pending the benchmark.
    #[test]
    fn sb_reads_envelope_matches_the_sa_envelope() {
        assert_eq!(CPU_SB_ADAPT.min_reads, CPU_ADAPT.min_reads);
        assert_eq!(CPU_SB_ADAPT.max_reads, CPU_ADAPT.max_reads);
        assert_eq!(
            CPU_SB_ADAPT.reads_solution_min_factor,
            CPU_ADAPT.reads_solution_min_factor
        );
        assert_eq!(
            CPU_SB_ADAPT.reads_solution_max_factor,
            CPU_ADAPT.reads_solution_max_factor
        );
        assert_eq!(
            CPU_SB_ADAPT.reads_solution_floor_factor,
            CPU_ADAPT.reads_solution_floor_factor
        );
    }

    #[test]
    fn sb_sampler_returns_num_reads_of_pm1_spins() {
        let sampler = SbSampler::new(DSB);
        let results = sampler
            .sample(&tiny_ferro(), &tiny_params(4))
            .expect("SbSampler::sample should not reject");
        assert_eq!(results.len(), 4);
        for r in &results {
            assert_eq!(r.spins.len(), 2);
            assert!(
                r.spins.iter().all(|&s| s == 1 || s == -1),
                "spins must be ±1, got {:?}",
                r.spins
            );
        }
    }

    #[test]
    fn sb_stream_width_is_at_least_one() {
        assert!(SbSampler::new(DSB).stream_width() >= 1);
    }

    #[tokio::test]
    async fn sb_sample_stream_one_job_round_trip() {
        let sampler = SbSampler::new(DSB);
        let (job_tx, job_rx) = tokio::sync::mpsc::channel::<StreamJob>(1);
        let (out_tx, mut out_rx) = tokio::sync::mpsc::channel::<StreamResult>(1);

        let job_id = b"job-sb-stream-1".to_vec();
        job_tx
            .send(StreamJob {
                job_id: job_id.clone(),
                graph: tiny_ferro(),
                params: tiny_params(1),
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
        let StreamOutcome::Completed(result) = got.outcome else {
            assert_eq!("got", "Completed outcome");
            return;
        };
        let results = result.expect("stream job should succeed");
        assert_eq!(results.len(), 1);
        assert!(results[0].spins.iter().all(|&s| s == 1 || s == -1));

        tokio::time::timeout(Duration::from_secs(30), pump)
            .await
            .expect("timeout waiting for sample_stream to exit")
            .expect("spawn_blocking join");

        assert!(
            out_rx.recv().await.is_none(),
            "exactly one StreamResult expected"
        );
    }
}
