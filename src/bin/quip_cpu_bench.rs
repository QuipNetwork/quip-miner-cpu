//! CPU sampler benchmark harness (`quip-cpu-bench`).

use std::io::Write as _;
use std::process::ExitCode;

use clap::Parser;
use quip_miner_cpu::bench::{run_bench, BenchArgs};

#[derive(Parser)]
#[command(version = concat!(env!("CARGO_PKG_VERSION"), " protocol 1"))]
struct Cli {
    /// Warm-up iterations discarded before measurement.
    #[arg(long, default_value_t = 1)]
    warmup: usize,
    /// Measured iterations per model.
    #[arg(long, default_value_t = 5)]
    iters: usize,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let args = BenchArgs {
        warmup: cli.warmup,
        iters: cli.iters,
    };
    match run_bench(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            // stderr via writeln (print-stderr lint forbids eprintln!).
            let _ = writeln!(std::io::stderr(), "quip-cpu-bench: {e}");
            ExitCode::FAILURE
        }
    }
}
