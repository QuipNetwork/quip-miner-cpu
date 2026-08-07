//! CPU Ising samplers.
//!
//! Three binaries share this library:
//! - `quip-cpu-sa` — neal-style geometric SA (Metropolis)
//! - `quip-cpu-gibbs` — heat-bath single-site Gibbs over the same ladder
//! - `quip-cpu-sb` — discrete Simulated Bifurcation
//!
//! The coordinator session loop lives in `quip-miner-core`; this crate provides
//! the [`CpuSampler`] and [`SbSampler`] backends and the three binaries. All of
//! them stream jobs through one shared pump, so cancellation and panic
//! propagation cannot drift between them.

pub mod sampler_core;
pub mod sb_core;
pub mod sb_sampler;

pub use quip_miner_core::{Algorithm, IsingGraph, SampleParams, SamplerResult};
pub use sampler_core::sample_ising;
pub use sb_core::{sample_sb, Coupling, SbVariant, BSB, DSB, HBSB, HDSB};
pub use sb_sampler::{SbSampler, CPU_SB_IDENTITY};

use quip_miner_core::adapt::AdaptBounds;
use quip_miner_core::{
    BackendIdentity, CancelGuard, Sampler, StreamJob, StreamOutcome, StreamResult,
};
use quip_proto::v1::RejectReason;

const DEFAULT_MAX_NODES: u32 = 100_000;
const DEFAULT_MAX_EDGES: u32 = 1_000_000;

/// CPU adapt envelope (from `CPU/sa_miner.py`).
const CPU_ADAPT: AdaptBounds = AdaptBounds {
    min_sweeps: 64,
    max_sweeps: 4096,
    min_reads: 64,
    max_reads: 512,
    reads_solution_min_factor: 4,
    reads_solution_max_factor: 8,
    reads_solution_floor_factor: 0,
};

/// Backend identity for `quip-cpu-sa`.
pub const CPU_SA_IDENTITY: BackendIdentity = BackendIdentity {
    backend: "cpu",
    algorithm: "sa",
    max_nodes: DEFAULT_MAX_NODES,
    max_edges: DEFAULT_MAX_EDGES,
    adapt: CPU_ADAPT,
};

/// Backend identity for `quip-cpu-gibbs`.
pub const CPU_GIBBS_IDENTITY: BackendIdentity = BackendIdentity {
    backend: "cpu",
    algorithm: "gibbs",
    max_nodes: DEFAULT_MAX_NODES,
    max_edges: DEFAULT_MAX_EDGES,
    adapt: CPU_ADAPT,
};

/// Shared streaming pump for every CPU sampler.
///
/// Runs `width` worker threads over an MPMC hand-off: this thread pulls from
/// the async job channel and each worker takes one model at a time. Cancelled
/// generations are dropped here, before a worker ever touches the graph.
///
/// Both `CpuSampler` and `SbSampler` call this, so the cancellation and
/// panic-propagation semantics cannot drift between binaries.
fn run_stream_pump<K>(
    width: usize,
    kernel: K,
    mut jobs: tokio::sync::mpsc::Receiver<StreamJob>,
    out: tokio::sync::mpsc::Sender<StreamResult>,
    cancel: CancelGuard,
) where
    K: Fn(&IsingGraph, &SampleParams) -> Vec<SamplerResult> + Send + Sync + Clone + 'static,
{
    let (work_tx, work_rx) = crossbeam_channel::bounded::<StreamJob>(width);
    let workers: Vec<_> = (0..width)
        .map(|_| {
            let work_rx = work_rx.clone();
            let out = out.clone();
            let kernel = kernel.clone();
            std::thread::spawn(move || {
                for j in work_rx.iter() {
                    let t0 = std::time::Instant::now();
                    let result = Ok(kernel(&j.graph, &j.params));
                    let device_access_time_us = t0.elapsed().as_micros() as u64;
                    if out
                        .blocking_send(StreamResult {
                            job_id: j.job_id,
                            outcome: StreamOutcome::Completed(result),
                            device_access_time_us,
                        })
                        .is_err()
                    {
                        break;
                    }
                }
            })
        })
        .collect();
    drop(work_rx);

    while let Some(j) = jobs.blocking_recv() {
        // Abandoned generations are dropped here, before a worker ever
        // touches the graph: a reseed can leave the queue full of stale
        // nonces, and sampling one would waste the round for nothing.
        if cancel.is_cancelled(j.generation) {
            if out
                .blocking_send(StreamResult {
                    job_id: j.job_id,
                    outcome: StreamOutcome::Cancelled,
                    device_access_time_us: 0,
                })
                .is_err()
            {
                break;
            }
            continue;
        }
        if work_tx.send(j).is_err() {
            break;
        }
    }
    drop(work_tx); // close -> workers drain and exit
    drop(out);
    for w in workers {
        // A panicking worker never emits a StreamResult for its in-flight
        // job, so swallowing the join error would silently shrink the pump
        // width for the rest of the session. The worker's own panic message
        // already reached stderr via the default hook; re-raise here so the
        // failure propagates instead of degrading throughput unnoticed.
        if let Err(payload) = w.join() {
            std::panic::resume_unwind(payload);
        }
    }
}

/// CPU sampler backend. No device, no governor, uncapped reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CpuSampler {
    algorithm: Algorithm,
}

impl CpuSampler {
    /// Create a CPU sampler for `algorithm` (SA or Gibbs).
    ///
    /// # Examples
    ///
    /// ```
    /// use quip_miner_cpu::{Algorithm, CpuSampler, IsingGraph, SampleParams};
    /// use quip_miner_core::Sampler;
    /// use quip_proto::v1::RejectReason;
    ///
    /// # fn main() -> Result<(), RejectReason> {
    /// let sampler = CpuSampler::new(Algorithm::Sa);
    /// let graph = IsingGraph::new(vec![0.0, 0.0], vec![-1.0], vec![(0, 1)]);
    /// let params = SampleParams {
    ///     num_reads: 2,
    ///     num_sweeps: 16,
    ///     seed: 1,
    ///     ..Default::default()
    /// };
    /// let results = sampler.sample(&graph, &params)?;
    /// assert_eq!(results.len(), 2);
    /// assert!(results.iter().all(|r| r.spins.iter().all(|&s| s == 1 || s == -1)));
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(algorithm: Algorithm) -> Self {
        Self { algorithm }
    }
}

impl Sampler for CpuSampler {
    fn sample(
        &self,
        graph: &IsingGraph,
        params: &SampleParams,
    ) -> Result<Vec<SamplerResult>, RejectReason> {
        Ok(sample_ising(graph, params, self.algorithm))
    }

    /// One model per core: `sample`'s reads are sequential and cache-local, so
    /// throughput comes from running `stream_width` models concurrently, each
    /// pinned to a worker thread. Fanning a single model's reads across cores
    /// bounced the shared arrays' cache lines and measured slower.
    fn stream_width(&self) -> usize {
        std::thread::available_parallelism().map_or(1, |n| n.get())
    }

    fn sample_stream(
        &self,
        jobs: tokio::sync::mpsc::Receiver<StreamJob>,
        out: tokio::sync::mpsc::Sender<StreamResult>,
        cancel: CancelGuard,
    ) {
        let algorithm = self.algorithm;
        run_stream_pump(
            self.stream_width(),
            move |g, p| sample_ising(g, p, algorithm),
            jobs,
            out,
            cancel,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quip_miner_core::{StreamJob, StreamOutcome, StreamResult};
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

    #[test]
    fn new_and_sample_returns_num_reads_of_pm1_spins() {
        let sampler = CpuSampler::new(Algorithm::Sa);
        let graph = tiny_ferro();
        let params = tiny_params(4);
        let results = sampler
            .sample(&graph, &params)
            .expect("CpuSampler::sample should not reject");
        assert_eq!(results.len(), params.num_reads);
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
    fn stream_width_is_at_least_one() {
        let sampler = CpuSampler::new(Algorithm::Gibbs);
        assert!(sampler.stream_width() >= 1);
    }

    #[tokio::test]
    async fn sample_stream_one_job_round_trip() {
        let sampler = CpuSampler::new(Algorithm::Sa);
        let (job_tx, job_rx) = tokio::sync::mpsc::channel::<StreamJob>(1);
        let (out_tx, mut out_rx) = tokio::sync::mpsc::channel::<StreamResult>(1);

        let job_id = b"job-stream-1".to_vec();
        job_tx
            .send(StreamJob {
                job_id: job_id.clone(),
                graph: tiny_ferro(),
                params: tiny_params(1),
                generation: 0,
            })
            .await
            .expect("send StreamJob");
        // Close input so sample_stream drains workers and returns.
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

        // Pump finishes after the closed input is fully drained.
        tokio::time::timeout(Duration::from_secs(30), pump)
            .await
            .expect("timeout waiting for sample_stream to exit")
            .expect("spawn_blocking join");

        assert!(
            out_rx.recv().await.is_none(),
            "exactly one StreamResult expected"
        );
    }

    /// Hypothesis: the shared pump drops an abandoned generation before any
    /// worker touches the graph, and reports it as `Cancelled` with zero device
    /// time. A reseed can leave the queue full of stale nonces, and sampling one
    /// would waste the round.
    #[tokio::test]
    async fn run_stream_pump_cancels_abandoned_generations_before_sampling() {
        let (job_tx, job_rx) = tokio::sync::mpsc::channel::<StreamJob>(1);
        let (out_tx, mut out_rx) = tokio::sync::mpsc::channel::<StreamResult>(1);
        let cancel = CancelGuard::default();
        cancel.cancel_through(7);

        let job_id = b"job-stale".to_vec();
        job_tx
            .send(StreamJob {
                job_id: job_id.clone(),
                graph: tiny_ferro(),
                params: tiny_params(1),
                generation: 7,
            })
            .await
            .expect("send StreamJob");
        drop(job_tx);

        let pump = tokio::task::spawn_blocking(move || {
            run_stream_pump(
                2,
                |g, p| sample_ising(g, p, Algorithm::Sa),
                job_rx,
                out_tx,
                cancel,
            );
        });

        let got = tokio::time::timeout(Duration::from_secs(30), out_rx.recv())
            .await
            .expect("timeout waiting for StreamResult")
            .expect("output channel closed without a result");
        assert_eq!(got.job_id, job_id);
        assert!(
            matches!(got.outcome, StreamOutcome::Cancelled),
            "a cancelled generation must not be sampled"
        );
        assert_eq!(got.device_access_time_us, 0);

        tokio::time::timeout(Duration::from_secs(30), pump)
            .await
            .expect("timeout waiting for run_stream_pump to exit")
            .expect("spawn_blocking join");
    }
}
