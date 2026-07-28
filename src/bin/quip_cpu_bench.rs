//! CPU sampler benchmark harness (`quip-cpu-bench`).

use std::io::Write as _;
use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, ValueEnum};
use quip_miner_cpu::bench::{run_bench, BenchArgs, SourceKind};
use quip_miner_cpu::Algorithm;

#[derive(Copy, Clone, ValueEnum)]
enum AlgoArg {
    Sa,
    Gibbs,
}

impl From<AlgoArg> for Algorithm {
    fn from(a: AlgoArg) -> Self {
        match a {
            AlgoArg::Sa => Self::Sa,
            AlgoArg::Gibbs => Self::Gibbs,
        }
    }
}

#[derive(Parser)]
#[command(version = concat!(env!("CARGO_PKG_VERSION"), " protocol 1"))]
struct Cli {
    /// Sampler algorithm.
    #[arg(long, value_enum, default_value = "sa")]
    algorithm: AlgoArg,
    /// Synthetic node count (mutually exclusive with --source).
    #[arg(long)]
    nodes: Option<usize>,
    /// Synthetic edge count.
    #[arg(long, default_value_t = 0)]
    edges: usize,
    /// Corpus JSONL path (mutually exclusive with --nodes).
    #[arg(long)]
    source: Option<PathBuf>,
    /// Coordinator `topology.spec.json` supplying topology for nonce-ref
    /// corpus entries.
    #[arg(long)]
    topology: Option<PathBuf>,
    /// Bench only the first K corpus models.
    #[arg(long)]
    limit: Option<usize>,
    /// Reads per sample call.
    #[arg(long, default_value_t = 64)]
    num_reads: usize,
    /// Sweeps per read.
    #[arg(long, default_value_t = 1024)]
    num_sweeps: usize,
    /// Sweeps per beta rung.
    #[arg(long, default_value_t = 1)]
    sweeps_per_beta: usize,
    /// PRNG seed.
    #[arg(long, default_value_t = 0)]
    seed: u64,
    /// Warm-up iterations discarded before measurement.
    #[arg(long, default_value_t = 1)]
    warmup: usize,
    /// Measured iterations per model.
    #[arg(long, default_value_t = 5)]
    iters: usize,
    /// Output directory for per-model JSON + folded stacks.
    #[arg(long, default_value = "bench-out")]
    out_dir: PathBuf,
}

fn source_kind(cli: &Cli) -> Result<SourceKind, String> {
    match (cli.nodes, &cli.source) {
        (Some(n), None) => Ok(SourceKind::Synthetic {
            n_nodes: n,
            n_edges: cli.edges,
        }),
        (None, Some(p)) => Ok(SourceKind::Corpus {
            path: p.clone(),
            topology: cli.topology.clone(),
        }),
        (None, None) => Err("provide --nodes (synthetic) or --source (corpus)".into()),
        (Some(_), Some(_)) => Err("--nodes and --source are mutually exclusive".into()),
    }
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let source = match source_kind(&cli) {
        Ok(s) => s,
        Err(e) => {
            let _ = writeln!(std::io::stderr(), "quip-cpu-bench: {e}");
            return ExitCode::FAILURE;
        }
    };
    let args = BenchArgs {
        algorithm: cli.algorithm.into(),
        source,
        num_reads: cli.num_reads,
        num_sweeps: cli.num_sweeps,
        sweeps_per_beta: cli.sweeps_per_beta,
        seed: cli.seed,
        warmup: cli.warmup,
        iters: cli.iters,
        out_dir: cli.out_dir,
        limit: cli.limit,
    };
    match run_bench(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            let _ = writeln!(std::io::stderr(), "quip-cpu-bench: {e}");
            ExitCode::FAILURE
        }
    }
}
