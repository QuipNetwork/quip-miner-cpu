//! Measure CPU sampler stream throughput at different worker counts.

use clap::Parser;
use quip_miner_cpu::{Algorithm, CpuSampler, IsingGraph, SampleParams, Sampler};
use quip_protocol::scoring::{energy_milli, ENERGY_MILLI_NON_FINITE};
use quip_solver_core::{CancelToken, StreamJob, StreamOutcome, StreamResult};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

#[derive(Parser)]
#[command(about = "Measure CPU sampler stream throughput by worker count")]
struct Args {
    /// Comma-separated requested worker counts.
    #[arg(long, value_delimiter = ',', default_value = "1,2,4,8,10,14,16,28,32")]
    widths: Vec<usize>,
    /// Jobs to submit per requested worker.
    #[arg(long, default_value_t = 4)]
    jobs_per_width: usize,
    /// Reads per job.
    #[arg(long, default_value_t = 105)]
    reads: usize,
    /// Sweeps per read.
    #[arg(long, default_value_t = 846)]
    sweeps: usize,
    /// Seed for the graph and sampler jobs.
    #[arg(long, default_value_t = 1)]
    seed: u64,
    /// Run one job alone at width one.
    #[arg(long)]
    single: bool,
    /// Topology spec JSON with `nodes` and `edges`, such as
    /// `isingmark/fixtures/advantage2-system1.spec.json`.
    #[arg(long)]
    topology: PathBuf,
}

#[derive(Debug)]
enum BenchError {
    InvalidArgument(&'static str),
    JobCountOverflow {
        width: usize,
        jobs_per_width: usize,
    },
    InputClosed,
    OutputClosed {
        received: usize,
        expected: usize,
    },
    UnknownJob,
    Cancelled {
        job: usize,
    },
    Sample {
        job: usize,
        message: String,
    },
    ResultCount {
        job: usize,
        expected: usize,
        actual: usize,
    },
    SpinCount {
        job: usize,
        expected: usize,
        actual: usize,
    },
    InvalidSpin {
        job: usize,
        spin: i8,
    },
    NonFiniteEnergy {
        job: usize,
    },
    EnergyMismatch {
        job: usize,
        reported: i64,
        scored: i64,
    },
    Pump(String),
    Topology(String),
}

impl fmt::Display for BenchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidArgument(message) => write!(f, "{message}"),
            Self::JobCountOverflow {
                width,
                jobs_per_width,
            } => write!(
                f,
                "job count overflows for width {width} and jobs-per-width {jobs_per_width}"
            ),
            Self::InputClosed => write!(f, "stream input closed before all jobs were submitted"),
            Self::OutputClosed { received, expected } => write!(
                f,
                "stream output closed after {received} of {expected} results"
            ),
            Self::UnknownJob => write!(f, "stream returned an unknown or duplicate job id"),
            Self::Cancelled { job } => write!(f, "stream cancelled job {job}"),
            Self::Sample { job, message } => {
                write!(f, "sampling job {job} failed: {message}")
            }
            Self::ResultCount {
                job,
                expected,
                actual,
            } => write!(f, "job {job} returned {actual} reads, expected {expected}"),
            Self::SpinCount {
                job,
                expected,
                actual,
            } => {
                write!(f, "job {job} returned {actual} spins, expected {expected}")
            }
            Self::InvalidSpin { job, spin } => {
                write!(f, "job {job} returned invalid spin value {spin}")
            }
            Self::NonFiniteEnergy { job } => {
                write!(f, "job {job} returned a non-finite energy")
            }
            Self::EnergyMismatch {
                job,
                reported,
                scored,
            } => write!(
                f,
                "job {job} reported energy {reported}, but scoring returned {scored}"
            ),
            Self::Pump(message) => write!(f, "stream pump failed: {message}"),
            Self::Topology(message) => write!(f, "cannot load topology: {message}"),
        }
    }
}

impl std::error::Error for BenchError {}

struct Row {
    width: usize,
    jobs: usize,
    wall_s: f64,
    jobs_per_s: f64,
    median_job_s: f64,
    p90_job_s: f64,
    host_parallelism: usize,
    stream_width: usize,
}

struct StreamMetrics {
    started_at: Instant,
    latencies: Vec<Duration>,
}

fn make_graph(topology: &Path, seed: u64) -> Result<IsingGraph, BenchError> {
    // The production topology: node ids and edges from the spec fixture, with
    // random +-1 couplings and zero fields, as a miner job uses them.
    let raw = std::fs::read_to_string(topology)
        .map_err(|e| BenchError::Topology(format!("{}: {e}", topology.display())))?;
    let spec: TopologySpec =
        serde_json::from_str(&raw).map_err(|e| BenchError::Topology(e.to_string()))?;
    let index: HashMap<u32, usize> = spec
        .nodes
        .iter()
        .enumerate()
        .map(|(i, &node)| (node, i))
        .collect();
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut edges = Vec::with_capacity(spec.edges.len());
    let mut couplings = Vec::with_capacity(spec.edges.len());
    for [u, v] in spec.edges {
        let (Some(&u), Some(&v)) = (index.get(&u), index.get(&v)) else {
            return Err(BenchError::Topology(format!(
                "edge ({u}, {v}) names an unknown node"
            )));
        };
        edges.push((u, v));
        couplings.push(if rng.gen_bool(0.5) { 1.0 } else { -1.0 });
    }
    Ok(IsingGraph::new(
        vec![0.0; spec.nodes.len()],
        couplings,
        edges,
    ))
}

#[derive(serde::Deserialize)]
struct TopologySpec {
    nodes: Vec<u32>,
    edges: Vec<[u32; 2]>,
}

fn record_result(
    result: StreamResult,
    graph: &IsingGraph,
    reads: usize,
    submitted: &mut BTreeMap<Vec<u8>, (usize, Instant)>,
    latencies: &mut Vec<Duration>,
) -> Result<(), BenchError> {
    let StreamResult {
        job_id, outcome, ..
    } = result;
    let Some((job, started_at)) = submitted.remove(&job_id) else {
        return Err(BenchError::UnknownJob);
    };

    let samples = match outcome {
        StreamOutcome::Completed(Ok(samples)) => samples,
        StreamOutcome::Completed(Err(error)) => {
            return Err(BenchError::Sample {
                job,
                message: format!("{error:?}"),
            });
        }
        StreamOutcome::Cancelled => return Err(BenchError::Cancelled { job }),
    };

    if samples.len() != reads {
        return Err(BenchError::ResultCount {
            job,
            expected: reads,
            actual: samples.len(),
        });
    }

    for sample in samples {
        if sample.spins.len() != graph.h.len() {
            return Err(BenchError::SpinCount {
                job,
                expected: graph.h.len(),
                actual: sample.spins.len(),
            });
        }
        if let Some(&spin) = sample.spins.iter().find(|&&spin| spin != -1 && spin != 1) {
            return Err(BenchError::InvalidSpin { job, spin });
        }

        let scored = energy_milli(&sample.spins, &graph.h, &graph.j, &graph.edges);
        if scored == ENERGY_MILLI_NON_FINITE {
            return Err(BenchError::NonFiniteEnergy { job });
        }
        if scored != sample.energy_milli {
            return Err(BenchError::EnergyMismatch {
                job,
                reported: sample.energy_milli,
                scored,
            });
        }
    }

    latencies.push(started_at.elapsed());
    Ok(())
}

async fn measure_stream(
    job_tx: tokio::sync::mpsc::Sender<StreamJob>,
    mut result_rx: tokio::sync::mpsc::Receiver<StreamResult>,
    graph: &IsingGraph,
    jobs: usize,
    reads: usize,
    sweeps: usize,
    seed: u64,
) -> Result<StreamMetrics, BenchError> {
    let started_at = Instant::now();
    let mut next_job = 0;
    let mut received = 0;
    let mut submitted = BTreeMap::new();
    let mut latencies = Vec::with_capacity(jobs);
    let mut job_tx = Some(job_tx);

    while received < jobs {
        if next_job == jobs {
            drop(job_tx.take());
            let Some(result) = result_rx.recv().await else {
                return Err(BenchError::OutputClosed {
                    received,
                    expected: jobs,
                });
            };
            record_result(result, graph, reads, &mut submitted, &mut latencies)?;
            received += 1;
            continue;
        }

        let job_id = format!("stream-scaling-{next_job}").into_bytes();
        let job_seed = seed
            .checked_add(u64::try_from(next_job).map_err(|_| {
                BenchError::InvalidArgument("job index does not fit in the seed range")
            })?)
            .ok_or(BenchError::InvalidArgument("job seed overflows u64"))?;
        let job = StreamJob {
            job_id: job_id.clone(),
            graph: graph.clone(),
            params: SampleParams {
                num_reads: reads,
                num_sweeps: sweeps,
                seed: job_seed,
                ..Default::default()
            },
            watermark: None,
        };
        let submitted_at = Instant::now();
        let sender = job_tx.as_ref().ok_or(BenchError::InputClosed)?;

        tokio::select! {
            send_result = sender.send(job) => {
                send_result.map_err(|_| BenchError::InputClosed)?;
                submitted.insert(job_id, (next_job, submitted_at));
                next_job += 1;
            }
            result = result_rx.recv() => {
                let Some(result) = result else {
                    return Err(BenchError::OutputClosed { received, expected: jobs });
                };
                record_result(result, graph, reads, &mut submitted, &mut latencies)?;
                received += 1;
            }
        }
    }

    if !submitted.is_empty() {
        return Err(BenchError::OutputClosed {
            received,
            expected: jobs,
        });
    }

    Ok(StreamMetrics {
        started_at,
        latencies,
    })
}

async fn run_width(
    graph: &IsingGraph,
    width: usize,
    jobs: usize,
    reads: usize,
    sweeps: usize,
    seed: u64,
    host_parallelism: usize,
) -> Result<Row, BenchError> {
    let sampler = CpuSampler::new(Algorithm::Sa);
    let backend_toml = format!("num_cpus = {width}");
    sampler.apply_config(&backend_toml);
    let stream_width = sampler.stream_width();

    let (job_tx, job_rx) = tokio::sync::mpsc::channel(stream_width);
    let (result_tx, result_rx) = tokio::sync::mpsc::channel(jobs);
    let pump = tokio::task::spawn_blocking(move || {
        sampler.sample_stream(job_rx, result_tx, CancelToken::default());
    });

    let metrics = measure_stream(job_tx, result_rx, graph, jobs, reads, sweeps, seed).await;
    let pump_result = pump
        .await
        .map_err(|error| BenchError::Pump(error.to_string()));
    let mut metrics = match (metrics, pump_result) {
        (Ok(metrics), Ok(())) => metrics,
        (Err(_), Err(error)) => return Err(error),
        (Err(error), Ok(())) => return Err(error),
        (Ok(_), Err(error)) => return Err(error),
    };

    let wall_s = metrics.started_at.elapsed().as_secs_f64();
    if !wall_s.is_finite() || wall_s <= 0.0 {
        return Err(BenchError::InvalidArgument(
            "measured wall time must be finite and positive",
        ));
    }
    metrics.latencies.sort_unstable();
    let median_job_s = median_seconds(&metrics.latencies);
    let p90_job_s = p90_seconds(&metrics.latencies);

    Ok(Row {
        width,
        jobs,
        wall_s,
        jobs_per_s: jobs as f64 / wall_s,
        median_job_s,
        p90_job_s,
        host_parallelism,
        stream_width,
    })
}

fn median_seconds(sorted: &[Duration]) -> f64 {
    let middle = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        (sorted[middle - 1].as_secs_f64() + sorted[middle].as_secs_f64()) / 2.0
    } else {
        sorted[middle].as_secs_f64()
    }
}

fn p90_seconds(sorted: &[Duration]) -> f64 {
    let index = sorted.len() - sorted.len() / 10 - 1;
    sorted[index].as_secs_f64()
}

fn validate_args(args: &Args) -> Result<(), BenchError> {
    if args.widths.is_empty() {
        return Err(BenchError::InvalidArgument("widths must not be empty"));
    }
    if args.widths.contains(&0) {
        return Err(BenchError::InvalidArgument("widths must be positive"));
    }
    if args.jobs_per_width == 0 {
        return Err(BenchError::InvalidArgument(
            "jobs-per-width must be positive",
        ));
    }
    if args.reads == 0 {
        return Err(BenchError::InvalidArgument("reads must be positive"));
    }
    if args.sweeps == 0 {
        return Err(BenchError::InvalidArgument("sweeps must be positive"));
    }
    Ok(())
}

#[expect(
    clippy::print_stdout,
    reason = "the benchmark writes parseable CSV rows to stdout"
)]
fn emit_csv(rows: &[Row]) {
    println!("width,jobs,wall_s,jobs_per_s,median_job_s,p90_job_s,host_parallelism,stream_width");
    for row in rows {
        println!(
            "{},{},{:.6},{:.6},{:.6},{:.6},{},{}",
            row.width,
            row.jobs,
            row.wall_s,
            row.jobs_per_s,
            row.median_job_s,
            row.p90_job_s,
            row.host_parallelism,
            row.stream_width
        );
    }
}

#[tokio::main]
async fn main() -> Result<(), BenchError> {
    let args = Args::parse();
    validate_args(&args)?;

    let graph = make_graph(&args.topology, args.seed)?;
    let host_parallelism = std::thread::available_parallelism().map_or(1, |n| n.get());
    let cases = if args.single {
        vec![(1, 1)]
    } else {
        args.widths
            .iter()
            .map(|&width| {
                let jobs =
                    width
                        .checked_mul(args.jobs_per_width)
                        .ok_or(BenchError::JobCountOverflow {
                            width,
                            jobs_per_width: args.jobs_per_width,
                        })?;
                Ok((width, jobs))
            })
            .collect::<Result<Vec<_>, BenchError>>()?
    };

    let mut rows = Vec::with_capacity(cases.len());
    for (width, jobs) in cases {
        rows.push(
            run_width(
                &graph,
                width,
                jobs,
                args.reads,
                args.sweeps,
                args.seed,
                host_parallelism,
            )
            .await?,
        );
    }
    emit_csv(&rows);
    Ok(())
}
