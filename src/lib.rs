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

mod coloring;
pub mod flatiron;
pub mod flatiron_sampler;
pub mod gibbs_parallel;
pub mod mps;
pub mod mps_sampler;
pub mod sampler_core;
pub mod sb_core;
pub mod sb_sampler;
mod spin_barrier;

pub use flatiron::{sample_ising_flatiron, FlatironConfig};
pub use flatiron_sampler::{FlatironSampler, CPU_FLATIRON_IDENTITY};
pub use gibbs_parallel::{ConfigError, GibbsConfig, GibbsParallelism};
pub use mps::{sample_ising_mps, InitMode, MpsConfig};
pub use mps_sampler::{MpsSampler, CPU_MFA_IDENTITY, CPU_MPS_IDENTITY};
pub use quip_miner_core::{Algorithm, IsingGraph, SampleParams, SamplerResult};
pub use sampler_core::sample_ising;
pub use sb_core::{sample_sb, sample_sb_with_workers, Coupling, SbVariant, BSB, DSB, HBSB, HDSB};
pub use sb_sampler::{
    SbSampler, CPU_BSB_IDENTITY, CPU_HBSB_IDENTITY, CPU_HDSB_IDENTITY, CPU_SB_IDENTITY,
};

use quip_miner_core::adapt::AdaptBounds;
use quip_miner_core::{
    BackendIdentity, CancelGuard, Sampler, StreamJob, StreamOutcome, StreamResult,
};
use quip_proto::v1::RejectReason;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

const DEFAULT_MAX_NODES: u32 = 100_000;
const DEFAULT_MAX_EDGES: u32 = 1_000_000;

/// CPU adapt envelope.
///
/// Sized so one SA attempt on an Advantage2-scale graph finishes inside a
/// typical cancelled round. The previous bounds came from the Python GPU
/// miner. They asked for about four times the Metal work on a CPU. One
/// read of 3324 sweeps took 571 ms on that topology. 415 reads at that
/// depth is about 4 minutes serial, and longer once 16 workers share the
/// machine. A 6-minute reseed then cancels the attempt before it returns.
const CPU_ADAPT: AdaptBounds = AdaptBounds {
    min_sweeps: 64,
    max_sweeps: 1024,
    min_reads: 64,
    max_reads: 128,
    reads_solution_min_factor: 0,
    reads_solution_max_factor: 0,
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
/// generations are dropped here, before a worker ever touches the graph. The
/// kernel also receives the guard so a long in-flight attempt can abort at
/// its own checkpoints.
///
/// The pump waits for the first job before it reads `width`. The session
/// applies `Configure.backend_toml` before any job, so a `num_cpus` setting
/// is visible when the workers start.
///
/// Both `CpuSampler` and `SbSampler` call this, so the cancellation and
/// panic-propagation semantics cannot drift between binaries.
fn run_stream_pump<K, F>(
    width: F,
    kernel: K,
    mut jobs: tokio::sync::mpsc::Receiver<StreamJob>,
    out: tokio::sync::mpsc::Sender<StreamResult>,
    cancel: CancelGuard,
) where
    F: FnOnce() -> usize,
    K: Fn(
            &IsingGraph,
            &SampleParams,
            &CancelGuard,
            u64,
        ) -> Result<Vec<SamplerResult>, sampler_core::SampleCancelled>
        + Send
        + Sync
        + Clone
        + 'static,
{
    let Some(first) = jobs.blocking_recv() else {
        return;
    };
    let width = width().max(1);
    let (work_tx, work_rx) = crossbeam_channel::bounded::<StreamJob>(width);
    let workers: Vec<_> = (0..width)
        .map(|_| {
            let work_rx = work_rx.clone();
            let out = out.clone();
            let kernel = kernel.clone();
            let cancel = cancel.clone();
            std::thread::spawn(move || {
                for j in work_rx.iter() {
                    let t0 = std::time::Instant::now();
                    let outcome = match kernel(&j.graph, &j.params, &cancel, j.generation) {
                        Ok(samples) => StreamOutcome::Completed(Ok(samples)),
                        Err(sampler_core::SampleCancelled) => StreamOutcome::Cancelled,
                    };
                    let device_access_time_us = t0.elapsed().as_micros() as u64;
                    if out
                        .blocking_send(StreamResult {
                            job_id: j.job_id,
                            outcome,
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

    let mut pending = Some(first);
    loop {
        let j = match pending.take() {
            Some(j) => j,
            None => match jobs.blocking_recv() {
                Some(j) => j,
                None => break,
            },
        };
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

/// Why a `num_cpus` setting cannot be applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NumCpusError {
    /// Operator set zero or a negative count.
    NonPositive {
        /// The value from `backend_toml`.
        requested: i64,
    },
    /// The TOML is not a table this backend can read.
    InvalidToml,
}

impl std::fmt::Display for NumCpusError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NonPositive { requested } => {
                write!(f, "num_cpus must be a positive integer, got {requested}")
            }
            Self::InvalidToml => write!(f, "backend_toml is not valid TOML"),
        }
    }
}

/// CPU `[cpu]` subsection of `Configure.backend_toml`.
#[derive(serde::Deserialize, Default)]
struct CpuBackendConfig {
    num_cpus: Option<i64>,
    #[serde(flatten)]
    unknown: BTreeMap<String, toml::Value>,
}

fn host_parallelism() -> usize {
    std::thread::available_parallelism().map_or(1, |n| n.get())
}

/// Cap a requested core budget at the host parallelism. Never returns 0.
fn resolve_core_budget(requested: usize, available: usize) -> usize {
    requested.min(available).max(1)
}

/// Read `num_cpus` from `backend_toml`. `None` means the key is absent.
fn parse_num_cpus(backend_toml: &str) -> Result<Option<usize>, NumCpusError> {
    if backend_toml.trim().is_empty() {
        return Ok(None);
    }
    let cfg: CpuBackendConfig =
        toml::from_str(backend_toml).map_err(|_| NumCpusError::InvalidToml)?;
    quip_miner_core::config::warn_unknown_fields("cpu", cfg.unknown.keys());
    match cfg.num_cpus {
        None => Ok(None),
        Some(n) if n <= 0 => Err(NumCpusError::NonPositive { requested: n }),
        Some(n) => Ok(Some(usize::try_from(n).unwrap_or(usize::MAX))),
    }
}

/// CPU sampler backend. No device, no governor, uncapped reads.
#[derive(Debug, Clone)]
pub struct CpuSampler {
    algorithm: Algorithm,
    gibbs: GibbsConfig,
    /// 0 means use host parallelism.
    num_cpus: Arc<AtomicUsize>,
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
        Self {
            algorithm,
            gibbs: GibbsConfig::default(),
            num_cpus: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Replace the chromatic Gibbs settings. Ignored by the SA path.
    ///
    /// Validate the configuration before calling this. The sampler cannot
    /// refuse a bad worker count once a job is in flight, so the binary checks
    /// it at startup and exits with a configuration error instead.
    pub fn with_gibbs_config(self, gibbs: GibbsConfig) -> Self {
        Self { gibbs, ..self }
    }

    fn core_budget(&self) -> usize {
        match self.num_cpus.load(Ordering::Acquire) {
            0 => host_parallelism(),
            n => n,
        }
    }

    fn apply_num_cpus(&self, backend_toml: &str) -> Result<(), NumCpusError> {
        let Some(requested) = parse_num_cpus(backend_toml)? else {
            return Ok(());
        };
        let available = host_parallelism();
        let budget = resolve_core_budget(requested, available);
        if budget < requested {
            tracing::debug!(
                requested,
                available,
                budget,
                "num_cpus exceeds host parallelism; clamped"
            );
        }
        self.num_cpus.store(budget, Ordering::Release);
        Ok(())
    }
}

/// Refuse a bad `num_cpus` at handshake. A typo must not look like it worked.
#[expect(
    clippy::exit,
    clippy::print_stderr,
    reason = "a zero or negative num_cpus is an operator typo; refuse to start \
              rather than silently fall back"
)]
fn reject_num_cpus(e: NumCpusError) -> ! {
    eprintln!("configuration error: {e}");
    std::process::exit(64);
}

impl Sampler for CpuSampler {
    fn sample(
        &self,
        graph: &IsingGraph,
        params: &SampleParams,
    ) -> Result<Vec<SamplerResult>, RejectReason> {
        if self.algorithm == Algorithm::Gibbs {
            // A colour budget the graph cannot meet is a property of the job,
            // not of the miner, so it rejects that job rather than exiting.
            return gibbs_parallel::sample_gibbs_with(graph, params, &self.gibbs)
                .map_err(|_| RejectReason::Malformed);
        }
        Ok(sample_ising(graph, params, self.algorithm))
    }

    /// Models to run concurrently.
    ///
    /// SA reads are sequential and cache-local, so one model per core is right:
    /// fanning a single model's reads across cores bounced the shared arrays'
    /// cache lines and measured slower. Chromatic Gibbs already spends
    /// `gibbs.workers` cores inside one model, so it runs proportionally fewer
    /// models to keep the machine from oversubscribing itself.
    ///
    /// `num_cpus` from `Configure.backend_toml` replaces host parallelism as
    /// this core budget. Absent, the host value is used. A value larger than
    /// the host is clamped.
    fn stream_width(&self) -> usize {
        let cores = self.core_budget();
        if self.algorithm == Algorithm::Gibbs {
            (cores / self.gibbs.workers.max(1)).max(1)
        } else {
            cores
        }
    }

    fn apply_config(&self, backend_toml: &str) {
        if let Err(e) = self.apply_num_cpus(backend_toml) {
            tracing::error!(error = %e, "cpu num_cpus rejected");
            reject_num_cpus(e);
        }
    }

    fn sample_stream(
        &self,
        jobs: tokio::sync::mpsc::Receiver<StreamJob>,
        out: tokio::sync::mpsc::Sender<StreamResult>,
        cancel: CancelGuard,
    ) {
        let algorithm = self.algorithm;
        let gibbs = self.gibbs;
        run_stream_pump(
            || self.stream_width(),
            move |g, p, guard, generation| {
                if algorithm == Algorithm::Gibbs {
                    Ok(gibbs_parallel::sample_gibbs_with(g, p, &gibbs).unwrap_or_default())
                } else {
                    sampler_core::sample_ising_cancellable(
                        g,
                        p,
                        algorithm,
                        Some((guard, generation)),
                    )
                }
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
    use quip_miner_core::{Sampler, StreamJob, StreamOutcome, StreamResult};
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

    #[test]
    fn apply_config_num_cpus_sets_stream_width() {
        let sampler = CpuSampler::new(Algorithm::Sa);
        sampler.apply_config("num_cpus = 1");
        assert_eq!(sampler.stream_width(), 1);
    }

    #[test]
    fn apply_config_absent_num_cpus_keeps_default_width() {
        let sampler = CpuSampler::new(Algorithm::Sa);
        let default_width = sampler.stream_width();
        sampler.apply_config("");
        assert_eq!(sampler.stream_width(), default_width);
        sampler.apply_config("num_sweeps = 128");
        assert_eq!(sampler.stream_width(), default_width);
    }

    #[test]
    fn parse_num_cpus_rejects_zero_and_negative() {
        assert_eq!(
            parse_num_cpus("num_cpus = 0"),
            Err(NumCpusError::NonPositive { requested: 0 })
        );
        assert_eq!(
            parse_num_cpus("num_cpus = -3"),
            Err(NumCpusError::NonPositive { requested: -3 })
        );
    }

    #[test]
    fn parse_num_cpus_absent_is_none() {
        assert_eq!(parse_num_cpus("").unwrap(), None);
        assert_eq!(parse_num_cpus("num_sweeps = 128").unwrap(), None);
    }

    #[test]
    fn resolve_core_budget_clamps_oversized() {
        assert_eq!(resolve_core_budget(32, 8), 8);
        assert_eq!(resolve_core_budget(4, 8), 4);
        assert_eq!(resolve_core_budget(1, 8), 1);
    }

    #[test]
    fn apply_config_oversized_num_cpus_clamps_to_default_width() {
        let sampler = CpuSampler::new(Algorithm::Sa);
        let default_width = sampler.stream_width();
        sampler.apply_config("num_cpus = 1000000");
        assert_eq!(sampler.stream_width(), default_width);
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
                || 2,
                |g, p, _, _| Ok(sample_ising(g, p, Algorithm::Sa)),
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

    /// An in-flight SA job must abort at a sweep checkpoint instead of
    /// running to completion. The job returns `Cancelled` so the coordinator
    /// credit path stays the same as the dequeue cancel.
    #[tokio::test]
    async fn in_flight_sa_attempt_cancels_at_sweep_checkpoint() {
        let sampler = CpuSampler::new(Algorithm::Sa);
        let (job_tx, job_rx) = tokio::sync::mpsc::channel::<StreamJob>(1);
        let (out_tx, mut out_rx) = tokio::sync::mpsc::channel::<StreamResult>(1);
        let cancel = CancelGuard::default();
        let cancel_for_pump = cancel.clone();

        // A 400-node chain with many sweeps cannot finish in the cancel
        // window. A two-node job can, and that would hide a missing
        // mid-sweep check behind a Completed result.
        let n = 400;
        let h = vec![0.0; n];
        let j = vec![-1.0; n - 1];
        let edges: Vec<(usize, usize)> = (0..n - 1).map(|i| (i, i + 1)).collect();
        let graph = IsingGraph::new(h, j, edges);

        job_tx
            .send(StreamJob {
                job_id: b"job-long".to_vec(),
                graph,
                params: SampleParams {
                    num_reads: 1,
                    num_sweeps: 2_000_000,
                    seed: 7,
                    ..Default::default()
                },
                generation: 3,
            })
            .await
            .expect("send StreamJob");
        drop(job_tx);

        let pump = tokio::task::spawn_blocking(move || {
            sampler.sample_stream(job_rx, out_tx, cancel_for_pump);
        });

        // Leave dequeue and enter the sweep loop, then abandon the generation.
        tokio::time::sleep(Duration::from_millis(50)).await;
        cancel.cancel_through(3);

        let got = tokio::time::timeout(Duration::from_secs(2), out_rx.recv())
            .await
            .expect("in-flight cancel did not abort the job promptly")
            .expect("output channel closed without a result");
        assert!(
            matches!(got.outcome, StreamOutcome::Cancelled),
            "abandoned job must emit Cancelled, got {:?}",
            std::mem::discriminant(&got.outcome)
        );

        tokio::time::timeout(Duration::from_secs(5), pump)
            .await
            .expect("timeout waiting for sample_stream to exit")
            .expect("spawn_blocking join");
    }

    /// A live CancelGuard that never fires must not change the RNG stream or
    /// the flip order. Results stay bit-identical to [`sample_ising`].
    #[test]
    fn uncancelled_guard_matches_sample_ising_bit_for_bit() {
        let graph = tiny_ferro();
        let params = SampleParams {
            num_reads: 4,
            num_sweeps: 64,
            seed: 99,
            ..Default::default()
        };
        let baseline = sample_ising(&graph, &params, Algorithm::Sa);
        let guard = CancelGuard::default();
        let live = sampler_core::sample_ising_cancellable(
            &graph,
            &params,
            Algorithm::Sa,
            Some((&guard, 4)),
        )
        .expect("a live generation must not cancel");
        assert_eq!(baseline.len(), live.len());
        for (a, b) in baseline.iter().zip(live.iter()) {
            assert_eq!(a.spins, b.spins);
            assert_eq!(a.energy_milli, b.energy_milli);
        }
    }

    /// A CPU attempt on the logged Advantage2 problem must finish inside a
    /// cancelled round. One read of 3324 sweeps took 571 ms on that
    /// topology. Sixteen workers slowed a small batch by about 3x. Serial
    /// time must stay under 120 s so a 6-minute reseed still sees a
    /// completed attempt after that contention.
    #[test]
    fn cpu_adapt_attempt_fits_a_cancelled_round() {
        use quip_miner_core::adapt::adapt_params;

        let p = adapt_params(-14_518_191, 1, 4577, 41515, &[0], &CPU_ADAPT);
        let product = u64::from(p.num_reads) * u64::from(p.num_sweeps);
        const MS_PER_SWEEP_READ: f64 = 571.0 / 3324.0;
        let serial_ms = product as f64 * MS_PER_SWEEP_READ;
        assert!(
            serial_ms < 120_000.0,
            "CPU adapt asks for {product} sweep-reads ({serial_ms:.0} ms serial) \
             at the logged problem; that misses a 6-minute cancelled round \
             once 16 workers share the machine"
        );
    }
}
